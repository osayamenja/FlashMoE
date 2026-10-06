//
// Created by osayamen on 1/12/26.
//

#ifndef FLASHMOE_STRUCTURES_CUH
#define FLASHMOE_STRUCTURES_CUH
#include <cuda/std/cstddef>
namespace flashmoe
{
    enum class DropTokens {
        yes,
        no
    };

    /// Quantization recipe of the expert MLPs.
    /// none: weights and activations are used in the element type of the layer (fp16, bf16, ...).
    /// mxfp8: W8A8 with OCP microscaling, i.e., E4M3 weights and E4M3 activations carrying one UE8M0 scale per
    ///        32 consecutive elements along the reduction dimension. Bias, router and combine stay in the element type.
    /// mxfp4: W4A4 with OCP microscaling, i.e., E2M1 weights and activations carrying one UE8M0 scale per 32 consecutive
    ///        elements along the reduction dimension. Bias, router and combine stay in the element type.
    /// nvfp4: W4A4 with E2M1 weights and activations carrying one UE4M3 scale per 16 consecutive elements along the
    ///        reduction dimension and one FP32 scale per tensor. Bias, router and combine stay in the element type.
    enum class QuantRecipe {
        none,
        mxfp8,
        mxfp4,
        nvfp4
    };

    /// mxfp4 or nvfp4
    __host__ __device__ __forceinline__
    constexpr bool isFP4(const QuantRecipe& r) {
        return r == QuantRecipe::mxfp4 || r == QuantRecipe::nvfp4;
    }

    /// Any recipe whose operands carry block scale factors in the canonical 128x4 layout, see infra/mx.cuh
    __host__ __device__ __forceinline__
    constexpr bool isBlockScaled(const QuantRecipe& r) {
        return r != QuantRecipe::none;
    }

    /// Bits of one quantized element: 8 (mxfp8) or 4 (mxfp4, nvfp4). 0 for none.
    __host__ __device__ __forceinline__
    constexpr int quantElementBits(const QuantRecipe& r) {
        return r == QuantRecipe::mxfp8 ? 8 : (isFP4(r) ? 4 : 0);
    }

    /// Number of consecutive elements along K that share one scale factor: 32 (mxfp8, mxfp4) or 16 (nvfp4). 0 for none.
    __host__ __device__ __forceinline__
    constexpr int quantSFVecSize(const QuantRecipe& r) {
        return r == QuantRecipe::nvfp4 ? 16 : (r == QuantRecipe::none ? 0 : 32);
    }

    /// Packet Encoding Lookup info, retrievable in a single memory lookup
    /// Key is global expert index
    struct __align__(16) PEL {
        cuda::std::byte* remoteSHeap;
        uint64_t* remoteSFlags;
        uint eC;
        uint16_t pTTt;
        uint16_t expertLocalIdx;
        uint16_t peer;
        uint16_t pe;
        uint16_t isRemote;
        uint16_t nLocalExperts;

        __host__ __device__ __forceinline__
        void dump() const {
            printf("{\n\t"
                   "this: %p,\n\t"
                   "remoteSHeap: %p,\n\t"
                   "remoteSFlags: %p,\n\t"
                   "eC: %u,\n\t"
                   "pTTt: %u,\n\t"
                   "expertLocalIndex: %u,\n\t"
                   "peer: %u,\n\t"
                   "pe: %u,\n\t"
                   "isRemote: %s,\n\t"
                   "nLocalExperts: %u"
                   "\n}\n",
                   this, remoteSHeap, remoteSFlags, eC, pTTt, expertLocalIdx,
                   peer, pe, isRemote ? "True" : "False", nLocalExperts);
        }
    };

    /// Expert lookup info: key is global expert index
    struct __align__(8) ELI {
        uint epRank; // host peer
        uint16_t localExpertIndex;
        uint16_t isRemote;

        __host__ __device__ __forceinline__
        void dump() const {
            printf("{\n\t"
                   "this: %p,\n\t"
                   "epRank: %u,\n\t"
                   "localExpertIndex: %u,\n\t"
                   "isRemote: %s"
                   "\n}\n",
                   this,
                   epRank, localExpertIndex, isRemote ? "True" : "False");
        }
    };

    /// Local expert lookup: key is local expert index
    struct __align__(4) LXI {
        uint expertIndex;
        __host__ __device__ __forceinline__
        void dump() const {
            printf("{\n\t"
                   "expertIndex: %u\n}\n", expertIndex);
        }
    };

    /// Peer lookup info: key is ep rank
    struct __align__(8) PLI {
        cuda::std::byte* remoteSHeap;
        uint64_t* remoteSFlags;
        uint pe;
        uint isRemote;

        __host__ __device__ __forceinline__
        void dump() const {
            printf("{\n\t"
                   "this: %p,\n\t"
                   "remoteSHeap: %p,\n\t"
                   "remoteSFlags: %p,\n\t"
                   "pe: %u,\n\t"
                   "isRemote: %s"
                   "\n}\n",
                   this, remoteSHeap, remoteSFlags, pe, isRemote ? "True" : "False");
        }
    };
}
#endif //FLASHMOE_STRUCTURES_CUH