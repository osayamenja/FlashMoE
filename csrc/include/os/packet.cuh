//
// Created by oja7 on 12/2/24.
//

#ifndef PACKET_CUH
#define PACKET_CUH
#include <nvshmem.h>

#include "../types.cuh"
#include "../atomics.cuh"

namespace aristos::packet {
    template<
        unsigned int blocks,
        DropTokens d = DropTokens::yes,
        unsigned int superBlockSize = ARISTOS_SUPER_BLOCK_SIZE,
        unsigned int H = ACC::H::value,
        unsigned int E = ACC::E::value,
        unsigned int EC = ACC::EC::value,
        unsigned int threads = ACC::PeakHardware::OS::threads::value,
        typename Activations
    >
    requires (isTensor<Activations>::value)
    __forceinline__ __device__
    void encode(const Activations& activations, unsigned int* const& __restrict__ workspace, const uint16_t& rSeqBit) {
        static_assert(sizeof(SignalPayload<>) == sizeof(ull_t) && alignof(SignalPayload<>) == alignof(ull_t));
        using Element = typename Activations::value_type;
        using NativeElement = typename ToCDx<Element>::T;
        // Below is always true, but we assert to ensure
        static_assert(sizeof(NativeElement) == sizeof(Element) && alignof(NativeElement) == alignof(Element));
        static_assert(blocks % superBlockSize == 0);
        // Map a static set of blocks to an expert and stride as thus
        constexpr auto numSuperBlocks = blocks / superBlockSize;
        const auto superBlockIdx = blockIdx.x / superBlockSize;
        const auto lBid = blockIdx.x % superBlockSize;
        const bool isLeader = !lBid && !threadIdx.x;

        // cache
        const auto* __restrict__ tP = bookkeeping.tP();
        const auto world = bookkeeping.world;
        auto* __restrict__ pSA = bookkeeping.pSA();
        const auto* __restrict__ pDs = bookkeeping.pDs();
        auto* __restrict__ sHeap = bookkeeping.sHeap;
        auto* __restrict__ flags = bookkeeping.flags;
        const auto expertSlots = bookkeeping.xs;
        const auto rank = bookkeeping.rank;

        const auto tokenIds = make_tensor(cute::make_gmem_ptr(tP),
            cute::Layout<cute::Shape<cute::Int<E>, cute::Int<EC>>,
                cute::Stride<cute::Int<EC>, cute::_1>>{});

        // readonly arrays
        // copy only what we need
        // Note that the length of these arrays is rather small,
        // which is why shared memory can accommodate them
        constexpr auto sF = sizeof(cuda::std::byte*) / sizeof(uint);
        const auto dSl = 3 * E + world * (2 * sF + 1);
        const auto* __restrict__ fP = CAST_TO(cuda::std::byte*, workspace);
        const auto* __restrict__ hP = fP + world;
        auto* __restrict__ pDsP = workspace + sF * 2 * world;
        const auto* __restrict__ expertCounts = pDsP;
        const auto* __restrict__ pS = expertCounts + E;
        const auto* __restrict__ pT = pS + E;
        const auto* __restrict__ xLs = pT + world;
        const auto* __restrict__ peerTotalTokens = xLs + E;

        auto* __restrict__ pTTx = workspace + dSl;
        // Populate above data structures
        constexpr auto vF = sizeof(uint4) / sizeof(uint);
        const auto vDsL = dSl / vF;
        const auto* __restrict__ vPDs = static_cast<const uint4*>(static_cast<const void*>(pDs));
        for (uint i = threadIdx.x; i < vDsL; i += threads) {
            CAST_TO(uint4, pDsP)[i] = vPDs[i];
        }
        for (uint i = threadIdx.x + vDsL * vF; i < dSl; i += threads) {
            pDsP[i] = pDs[i];
        }
        // wipe clean first
        for (uint i = threadIdx.x; i < world; i += threads) {
            // done this way to preserve const attribute
            pTTx[i] = 0U;
        }
        __syncthreads();
        // symmetric heap pointer
        auto* __restrict__ sFp = CAST_TO(cuda::std::byte*, workspace);
        auto* __restrict__ sHp = sFp + world;
        for (uint i = 0; i < world; i += threads) {
            // get peer idx
            const auto peer = pT[i];
            if (const auto rSHeap = nvshmem_ptr(sHeap, peer); rSHeap == nullptr) {
                sFp[i] = nullptr;
                sHp[i] = nullptr;
            }
            else {
                sFp[i] = CAST_TO(cuda::std::byte, nvshmem_ptr(flags, peer));
                sHp[i] = CAST_TO(cuda::std::byte, rSHeap);
            }
        }
        #pragma unroll
        for (uint i = threadIdx.x; i < E; i += threads) {
            const auto peer = __ldg(pS + i);
            atomicAdd_block(pTTx + peer, expertCounts[i]);
        }
        __syncthreads();
        #pragma unroll
        for (uint expertIdx = superBlockIdx; expertIdx < E; expertIdx += numSuperBlocks) {
            const auto pLIdx = xLs[expertIdx]; // local index for current expert on peer
            const auto routedTokens = d == DropTokens::yes ?
                cute::min(expertCounts[expertIdx], EC) : expertCounts[expertIdx];
            const auto peer = pS[expertIdx];
            const auto pe = pT[peer];
            const auto pTT = cute::min(peerTotalTokens[peer], EC);
            auto* __restrict__ pH = hP[peer];
            auto* __restrict__ pF = fP[peer];
            const auto isRemote = pH == nullptr;
            auto* __restrict__ peerHeap = isRemote ?
                heap::cAdvance<0, 0>(sHeap, peer, pLIdx) :
            heap::cAdvance<0, 1>(pH, peer, pLIdx);

            if (!routedTokens) {
                if (isLeader && !pTT) {
                    // single thread sends a noop packet to unblock the remote peer
                    // Pack payload into single signal word
                    const auto sigPayload = SignalPayload{
                        routedTokens,
                        rSeqBit,
                        static_cast<uint16_t>(Bookkeeping::tiles<BLOCK_M>(pTT)) // this should be safe
                    };
                    if (isRemote) {
                        // transmit signal
                        nvshmemx_signal_op(flags + rank * expertSlots + pLIdx,
                            *CAST_TO(flagsType, &sigPayload), NVSHMEM_SIGNAL_SET, pe);
                    }
                    else {
                        // Better to use below than the volatile write operation used in the public-facing NVSHMEM API
                        atomicExch_system(CAST_TO(ull_t, pF + rank * expertSlots + pLIdx),
                            *CONST_CAST_TO(ull_t, sigPayload));
                    }
                }
                continue;
            }
            // copy tokens: not padded
            for (uint j = lBid; j < routedTokens; j += superBlockSize) {
                const auto [tokenIdx, _] = tokenIds(expertIdx, j);
                auto* __restrict__ localPH = peerHeap + j * H * sizeof(Element);
                const auto* __restrict__ aP = CONST_CAST_TO(NativeElement, &activations(tokenIdx, 0));
                const auto* __restrict__ vAP = static_cast<const uint4*>(static_cast<const void*>(aP));
                constexpr auto vTokenSize = H / (sizeof(uint4) / sizeof(Element));
                // Use high-throughput vector copy
                #pragma unroll
                for (uint k = threadIdx.x; k < vTokenSize; k += threads) {
                    CAST_TO(uint4, localPH)[k] = __ldg(vAP + k);
                }
                const auto rIdx = vTokenSize * (sizeof(uint4) / sizeof(Element));
                localPH += sizeof(uint4) * vTokenSize;
                #pragma unroll
                for (uint k = threadIdx.x + rIdx; k < H; k += threads) {
                    CAST_TO(NativeElement, localPH)[k] = __ldg(aP + k);
                }
            }
            __syncthreads();
            if (!threadIdx.x) {
                if (isRemote) {
                    __threadfence();
                }
                else {
                    __threadfence_system();
                }
                if (atomicIncrement(pSA + expertIdx) + 1 == superBlockSize) {
                    // I am the last block, let's finalize this transfer.
                    const auto sigPayload = SignalPayload{
                        routedTokens,
                        rSeqBit,
                        static_cast<uint16_t>(Bookkeeping::tiles<BLOCK_M>(pTT)) // this should be safe
                    };
                    if (isRemote) {
                        // do RDMA transfer + signal
                        nvshmem_putmem_signal_nbi(
                            heap::cAdvance<0, 1>(sHeap, peer, pLIdx),
                            peerHeap,
                            sizeof(Element) * routedTokens * H,
                            flags + rank * expertSlots + pLIdx,
                            *CAST_TO(flagsType, &sigPayload),
                            NVSHMEM_SIGNAL_SET,
                            pe);
                    }
                    else {
                        // we've done the DMA transfer already, so we set the signal instead
                        atomicExch_system(CAST_TO(ull_t, pF + rank * expertSlots + pLIdx),
                            *CONST_CAST_TO(ull_t, sigPayload));
                    }
                }
            }
        }
    }

    // Resident in registers
    struct __align__(16) DecoderArg {
        cuda::std::byte* sHeap;
        Task* tQ;
        const unsigned int tPs;
        __device__
        DecoderArg(
            cuda::std::byte* const& _sHeap,
            Task* const& _tQ,
            unsigned int const& _tPs) :
        sHeap(_sHeap), tQ(_tQ), tPs(_tPs) {}
    };

    /// Decodes a single packet from the initial stage
    template<
        PacketStage s,
        PeerConnectivity p,
        typename Element = void
    >
    struct Decoder {
        static_assert(aristos::TensorValueType<Element>);
        static_assert(s == PacketStage::initial);
        __device__ __forceinline__
        void operator()(const DecoderArg& dA,
            const cuda::std::byte* const& packet,
            unsigned int* __restrict__ const& status,
            unsigned int* __restrict__ const& taskCount,
            uint const& routedTokens, uint16_t const& globalTaskTiles,
            unsigned int const& localExpertIdx,
            cuda::std::byte* __restrict__ const& pGB, //postGEMM buffer
            const cuda::std::array<const cuda::std::byte*, GEMMs>& weights,
            const cuda::std::array<const cuda::std::byte*, GEMMs>& bias,
            unsigned int const& peer, // relative to the EP group
            unsigned int const& gPeer, // relative to the global group, needed for network operations
            unsigned int& lTQHead,
            unsigned int* const& tQHead) const {

            constexpr auto tN = ACC::TN::value;
            constexpr auto tNx = ACC::TNx::value;
            constexpr auto H = ACC::H::value;
            constexpr auto eCap = ACC::EC::value;
            constexpr auto E = ACC::E::value;

            auto* __restrict__ tQ = dA.tQ + (threadIdx.x * dA.tPs + lTQHead);
            const auto fTilesM = routedTokens / BLOCK_M;
            const auto padM = Bookkeeping::pad<BLOCK_M>(routedTokens);

            if (!atomicTAS<cuda::thread_scope_block>(status + peer)) {
                // atomically reduce taskCount
                const auto superfluous = (tN + tNx) * (Bookkeeping::tiles<BLOCK_M>(eCap) -
                    globalTaskTiles);
                atomicSub_block(taskCount, superfluous);
            }

            // expert, peer offset
            const auto pXo = eCap * (peer * E + localExpertIdx);
            cuda::std::array<cuda::std::byte*, GEMMs> taskResults{};
            // Staging buffer for results of preGEMM
            taskResults[0] = pGB + pXo * H * sizeof(Element);
            // Egress packet buffer
            taskResults[1] = p == PeerConnectivity::remote ?
                heap::cAdvance<1, 0>(dA.sHeap, peer, localExpertIdx) :
            heap::cAdvance<1, 1>(dA.sHeap, peer, localExpertIdx);
            for (uint i = 0; i < fTilesM; ++i) {
                #pragma unroll
                for (uint j = 0; j < tN; ++j) {
                    const auto tileIdx = j + i * tN;
                    tQ[tileIdx] = Task{
                        TaskType::preGEMM,
                        packet,
                        weights,
                        taskResults,
                        bias,
                        pXo + i,
                        tileIdx,
                        padM,
                        pXo + (p == PeerConnectivity::remote ? i : tileIdx),
                        static_cast<uint16_t>(BLOCK_M),
                        gPeer,
                        i,
                        p == PeerConnectivity::remote
                    };
                }
            }

            // residue tile
            if (const auto residue = routedTokens - fTilesM * BLOCK_M; residue) {
                #pragma unroll
                for (uint j = 0; j < tN; ++j) {
                    const auto tileIdx = j + fTilesM * tN;
                    tQ[tileIdx] = Task{
                        TaskType::preGEMM,
                        packet,
                        weights,
                        taskResults,
                        bias,
                        pXo + fTilesM,
                        tileIdx,
                        padM,
                        pXo + (p == PeerConnectivity::remote ? fTilesM : tileIdx),
                        static_cast<uint16_t>(residue),
                        gPeer,
                        fTilesM,
                        p == PeerConnectivity::remote
                    };
                }
            }

            if (routedTokens) {
                __threadfence();
                const auto totalTasks = Bookkeeping::tiles<BLOCK_M>(routedTokens) * tN;
                lTQHead += totalTasks;
                // notifies scheduler of work
                atomicAdd_block(tQHead, totalTasks);
            }
        }
    };


    template<>
    struct Decoder<PacketStage::last, PeerConnectivity::p2p> {
        __device__ __forceinline__
        void operator()(Task* __restrict__ tQ,
            const cuda::std::byte* const& packet,
            const cuda::std::byte* const& tokenIndices,
            cuda::std::byte* __restrict__ const& moeOutput,
            const unsigned int& nTokens,
            const unsigned int& tileIdx,
            unsigned int* const& tQHead,
            const unsigned int& expertIdx) const {
            // now let's decode this single tile
            *tQ = Task{
                TaskType::combine,
                tokenIndices,
                cuda::std::array<const cuda::std::byte*, GEMMs>{packet},
                cuda::std::array<cuda::std::byte*, GEMMs>{moeOutput},
                nTokens,
                tileIdx,
                BLOCK_M,
                expertIdx
            };
            __threadfence();
            // notifies scheduler of work
            atomicIncrement<cuda::thread_scope_block>(tQHead);
        }
    };

    template<>
    struct Decoder<PacketStage::last, PeerConnectivity::remote> {
        __device__ __forceinline__
        void operator()(const DecoderArg& dA,
            const cuda::std::byte* const& packet,
            const cuda::std::byte* const& tokenIndices,
            cuda::std::byte* __restrict__ const& moeOutput,
            const unsigned int& nTokens,
            unsigned int& lTQHead,
            unsigned int* const& tQHead,
            const unsigned int& expertIdx) const {
            auto* __restrict__ tQ = dA.tQ + (threadIdx.x * dA.tPs + lTQHead);
            constexpr auto tN = ACC::TN::value;
            #pragma unroll
            for (uint i = 0; i < tN; ++i) {
                tQ[i] = Task{
                    TaskType::combine,
                    tokenIndices,
                    cuda::std::array<const cuda::std::byte*, GEMMs>{packet},
                    cuda::std::array<cuda::std::byte*, GEMMs>{moeOutput},
                    nTokens,
                    i,
                    BLOCK_M,
                    expertIdx
                };
            }
            __threadfence();
            lTQHead += tN;
            // notifies scheduler
            atomicAdd_block(tQHead, tN);
        }
    };
}
#endif //PACKET_CUH
