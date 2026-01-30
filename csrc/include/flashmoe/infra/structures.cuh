//
// Created by osayamen on 1/12/26.
//

#ifndef FLASHMOE_STRUCTURES_CUH
#define FLASHMOE_STRUCTURES_CUH
namespace flashmoe
{
    enum class DropTokens {
        yes,
        no
    };

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