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
    void encode(const Activations& activations, cuda::std::byte* __restrict__ const& workspace, const uint16_t& rSeqBit) {
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
        auto* __restrict__ pSA = bookkeeping.pSA();
        auto* __restrict__ sHeap = bookkeeping.sHeap;
        auto* __restrict__ flags = bookkeeping.flags;
        const auto fpO = bookkeeping.rank * bookkeeping.xs;

        const auto tokenIds = make_tensor(cute::make_gmem_ptr(tP),
            cute::Layout<cute::Shape<cute::Int<E>, cute::Int<EC>>,
                cute::Stride<cute::Int<EC>, cute::_1>>{});

        /// Populate Data Structures
        const auto* __restrict__ enL = CAST_TO(PEL, workspace);
        const auto* __restrict__ eC = bookkeeping.eC();
        const auto* __restrict__ eL = bookkeeping.pEL();
        constexpr auto oT = E * sizeof(PEL);
        const auto* __restrict__ seC = CAST_TO(uint, workspace + oT);

        #pragma unroll
        for (uint i = threadIdx.x; i < E; i += threads) {
            CAST_TO(PEL, workspace)[i] = eL[i];
            CAST_TO(uint, workspace + oT)[i] = eC[i];
        }
        constexpr auto oT2 = oT + E * sizeof(uint);
        const auto* __restrict__ sPTT = CAST_TO(uint, workspace + oT2);
        __syncthreads();
        #pragma unroll
        for (uint i = threadIdx.x; i < E; i += threads) {
            const auto peer = enL[i].peer;
            atomicAdd_block(CAST_TO(uint, workspace + oT2) + peer, seC[i]);
        }
        __syncthreads();
        // Update encoding lookup table
        #pragma unroll
        for (uint i = threadIdx.x; i < E; i += threads) {
            auto* __restrict__ peL = CAST_TO(PEL, workspace) + i;
            const auto peer = peL->peer;
            peL->eC = seC[i];
            peL->pTT = sPTT[peer];
        }
        __syncthreads();
        #pragma unroll
        for (uint expertIdx = superBlockIdx; expertIdx < E; expertIdx += numSuperBlocks) {
            const auto lI = enL[expertIdx];
            const auto flagOffset = fpO + lI.expertLocalIdx;
            const auto routedTokens = d == DropTokens::yes ?
                cute::min(lI.eC, EC) : lI.eC;
            auto* __restrict__ peerHeap = lI.isRemote ?
                heap::cAdvance<0, 0>(sHeap, lI.peer, lI.expertLocalIdx) :
            heap::cAdvance<0, 1>(lI.remoteSHeap, lI.peer, lI.expertLocalIdx);

            if (!routedTokens) {
                if (isLeader && !lI.pTT) {
                    // single thread sends a noop packet to unblock the remote peer
                    // Pack payload into single signal word
                    const auto sigPayload = SignalPayload{
                        routedTokens,
                        static_cast<uint16_t>(Bookkeeping::tiles<BLOCK_M>(lI.pTT)), // this should be safe
                        rSeqBit
                    };
                    if (lI.isRemote) {
                        // transmit signal
                        nvshmemx_signal_op(flags + flagOffset,
                            *CAST_TO(flagsType, &sigPayload), NVSHMEM_SIGNAL_SET, lI.pe);
                    }
                    else {
                        // Better to use below than the volatile write operation used in the public-facing API
                        atomicExch_system(CAST_TO(ull_t, lI.remoteSFlags), *CONST_CAST_TO(ull_t, sigPayload));
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
                if (lI.isRemote) {
                    __threadfence();
                }
                else {
                    __threadfence_system();
                }
                if (atomicIncrement(pSA + expertIdx) + 1 == superBlockSize) {
                    // I am the last block, let's finalize this transfer.
                    const auto sigPayload = SignalPayload{
                        routedTokens,
                        static_cast<uint16_t>(Bookkeeping::tiles<BLOCK_M>(lI.pTT)), // this should be safe
                        rSeqBit
                    };
                    if (lI.isRemote) {
                        // do RDMA transfer + signal
                        nvshmem_putmem_signal_nbi(
                            heap::cAdvance<0, 1>(sHeap, lI.peer, lI.expertLocalIdx),
                            peerHeap,
                            sizeof(Element) * routedTokens * H,
                            flags + flagOffset,
                            *CAST_TO(flagsType, &sigPayload),
                            NVSHMEM_SIGNAL_SET,
                            lI.pe);
                    }
                    else {
                        // we've done the DMA transfer already, so we set the signal instead
                        atomicExch_system(CAST_TO(ull_t, lI.remoteSFlags), *CONST_CAST_TO(ull_t, sigPayload));
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
