//
// Created by oja7 on 12/2/24.
//

#ifndef PACKET_CUH
#define PACKET_CUH

#include <nvshmemx.h>
#include <nvshmem.h>
#include <host/nvshmemx_api.h>

#include "../types.cuh"
#include "../atomics.cuh"

namespace aristos::packet {
    namespace signal {
        template<PacketStage p = PacketStage::final>
        __device__ __forceinline__
        // {{batchIdx, numTokens}, sequence number}
        // buffer is an 8-byte array, which we split into the following:
        // 2-byte integer denoting batch index
        // 2-byte integer blockM dimension
        // 4-byte sequence number.
        void encode(cuda::std::byte* __restrict__ const& buffer, const uint2& s) {
            static_assert(p == PacketStage::final);
            *CAST_TO(uint2, buffer) = uint2{s.x, s.y + 1};
        }

        template<PacketStage p = PacketStage::final>
        __device__ __forceinline__
        auto decode(cuda::std::byte* __restrict__ const& buffer) {
            static_assert(p == PacketStage::final);
            return *CAST_TO(uint2, buffer);
        }
    }

    template<unsigned int blocks,
    DropTokens d = DropTokens::yes,
    unsigned int superBlockSize = ARISTOS_SUPER_BLOCK_SIZE,
    typename Activations>
    requires aristos::Matrix<Activations>
    __forceinline__ __device__
    void encode(const Activations& activations, unsigned int* const& __restrict__ workspace) {
        // assert(blocks <= gridDim.x - 1)
        static_assert(blocks % superBlockSize == 0);
        using ElementAct = typename Activations::value_type;
        using ElementGate = maxPrecision;

        // Map a static set of blocks to an expert and stride as thus
        constexpr auto numSuperBlocks = blocks / superBlockSize;
        const auto superBlockIdx = blockIdx.x / superBlockSize;
        const auto lBid = blockIdx.x % superBlockSize;
        const auto tIdx =  lBid * THREADS + threadIdx.x;
        const bool isLeader = !lBid && !threadIdx.x;
        const auto cap = moeConfig.expertCapacity;

        auto tokenIds = make_tensor(cute::make_gmem_ptr(moeConfig.tIdx()),
            make_layout(cute::make_shape(moeConfig.numExperts, cap), cute::LayoutRight{}));

        // readonly arrays
        // copy only what we need
        // Note that the length of these arrays is rather small,
        // which is why we can fit then in shared memory
        // TODO construct data structures
        const auto aX = cute::ceil_div(moeConfig.numExperts, numSuperBlocks);
        const auto* expertCounts = workspace + aX;
        const auto* batchIdx = expertCounts + aX;
        const auto* pRT = batchIdx + aX;
        const auto* iPS = pRT + aX;
        const auto* nX = iPS + aX;

        const auto* pS = nX + moeConfig.worldSize;
        const auto* pT = pS + moeConfig.worldSize;
        const auto* peerTotalTokens = pT + moeConfig.worldSize;

        const auto* ptr = moeConfig.getExpertCounts();
        for (unsigned int i = threadIdx.x; i < moeConfig.numExperts; i += THREADS) {
            workspace[i] = ptr[i];
        }

        for (uint expertIdx = superBlockIdx; expertIdx < moeConfig.numExperts; expertIdx += numSuperBlocks) {
            // prefix array of routed tokens
            // logical expert index
            const auto lIdx = expertIdx / numSuperBlocks;
            const auto pLIdx = iPS[lIdx];
            const auto routedTokens = d == DropTokens::yes ?
                cute::min(expertCounts[lIdx], cap) : expertCounts[lIdx];
            const auto peer = pS[lIdx];
            const auto pTT = cute::min(peerTotalTokens[peer], moeConfig.expertCapacity);
            const auto padded = Config::pad<BLOCK_M>(routedTokens);
            const auto isRemote = nvshmem_ptr(moeConfig.sHeap, peer) == nullptr;
            auto* peerHeap = static_cast<cuda::std::byte*>(isRemote ?
                heap::advance<0, 0>(peer, pLIdx):
                nvshmem_ptr(heap::advance<0, 1>(peer, pLIdx), peer));
            if (!routedTokens) {
                if (isLeader && !pTT) {
                    // single thread sends a noop packet to unblock the remote peer
                    // header first
                    *CAST_TO(uint2, peerHeap) = uint2{pTT, routedTokens};
                    if (isRemote) {
                        // do RDMA transfer
                        nvshmem_putmem_signal_nbi(heap::advance<0, 1>(peer, pLIdx),
                            peerHeap,
                            sizeof(uint2),
                            moeConfig.flags + moeConfig.rank * nX[peer] + pLIdx,
                            constructSignal(PacketStage::initial),
                            NVSHMEM_SIGNAL_SET,
                            pT[peer]);
                    }
                    else {
                        __threadfence_system();
                        // we've done the DMA transfer already, so we set the signal instead
                        nvshmemx_signal_op(moeConfig.flags + moeConfig.rank * nX[peer] + pLIdx,
                            constructSignal(PacketStage::initial),
                            NVSHMEM_SIGNAL_SET,
                            pT[peer]);
                    }
                }
                continue;
            }
            if (isLeader) {
                // write header -> {total tokens from me to peer, number of tokens in this packet}
                *CAST_TO(uint2, peerHeap) = uint2{pTT, routedTokens};
            }
            peerHeap += sizeof(uint2);
            // write packet Ids
            const auto startIdx = batchIdx[lIdx];
            const auto batchIdxLength = cute::ceil_div(padded, BLOCK_M);
            for (uint j = tIdx; j < batchIdxLength; j += superBlockSize * THREADS) {
                CAST_TO(unsigned int, peerHeap)[j] = startIdx + j;
            }
            peerHeap += sizeof(unsigned int) * batchIdxLength;
            for (uint j = tIdx; j < routedTokens; j += superBlockSize * THREADS) {
                CAST_TO(ElementGate, peerHeap)[j] = tokenIds(expertIdx, j).second;
            }
            peerHeap += sizeof(ElementGate) * routedTokens;
            // Not needed for P2P connectivity due to non-contiguous access
            if (isRemote) {
                // We pad below rather than tokens as it reduces the packet size by pad amount * embedDim
                // Technically, we could optimize the below by sending a single zero byte per element,
                // rather than the materialized element 0.
                // This jeopardizes decoding as the decoder expects all data to be available in the
                // packet.
                const auto padResidue = padded - routedTokens;
                const auto vPr = padResidue / (sizeof(uint4) / sizeof(ElementGate));
                for (uint j = tIdx; j < vPr; j += superBlockSize * THREADS) {
                    CAST_TO(uint4, peerHeap)[j] = uint4{0UL, 0UL, 0UL, 0UL};
                }
                peerHeap += sizeof(uint4) * vPr;
                for (uint j = tIdx; j < padResidue; j += superBlockSize * THREADS) {
                    CAST_TO(ElementGate, peerHeap)[j] = ElementGate{0};
                }
            }

            peerHeap += sizeof(ElementGate) * (padded - routedTokens);

            // copy tokens: not padded
            for (uint j = lBid; j < routedTokens; j += superBlockSize) {
                auto* localPH = peerHeap + j * moeConfig.embedDim * sizeof(ElementAct);
                const auto tokenIdx = tokenIds(expertIdx, j);
                const auto vTokenSize = moeConfig.embedDim / (sizeof(uint4) / sizeof(ElementAct));
                // Use high-throughput vector copy
                for (uint k = threadIdx.x; k < vTokenSize; k += THREADS) {
                    CAST_TO(uint4, localPH)[k] = activations(tokenIdx, k);
                }
                const auto residue = moeConfig.embedDim - vTokenSize * (sizeof(uint4) / sizeof(ElementAct));
                localPH += sizeof(uint4) * vTokenSize;
                for (uint k = threadIdx.x; k < residue; k += THREADS) {
                    CAST_TO(ElementAct, localPH)[k] = activations(tokenIdx, k);
                }
            }
            __syncthreads();
            if (!threadIdx.x) {
                if (atomicIncrement(moeConfig.xSync() + expertIdx) + 1 == superBlockSize) {
                    // I am the last block, let's finalize this transfer.
                    if (isRemote) {
                        const auto tB = sizeof(uint2) +
                            sizeof(unsigned int) * batchIdxLength +
                            sizeof(ElementGate) * padded +
                            sizeof(ElementAct) * routedTokens * moeConfig.embedDim;
                        // do RDMA transfer
                        nvshmem_putmem_signal_nbi(heap::advance<0, 1>(peer, pLIdx),
                            peerHeap,
                            tB,
                            moeConfig.flags + moeConfig.rank * nX[peer] + pLIdx,
                            constructSignal(PacketStage::initial),
                            NVSHMEM_SIGNAL_SET,
                            pT[peer]);
                    }
                    else {
                        __threadfence_system();
                        // we've done the DMA transfer already, so we set the signal instead
                        nvshmemx_signal_op(moeConfig.flags + moeConfig.rank * nX[peer] + pLIdx,
                            constructSignal(PacketStage::initial),
                            NVSHMEM_SIGNAL_SET,
                            pT[peer]);
                    }
                }
            }
        }
    }

    /// Decodes a single packet from the initial stage
    template<
        PacketStage s,
        PeerConnectivity p,
        typename Element = void,
        typename ElementScale = Element
    >
    struct Decoder {
        static_assert(aristos::TensorValueType<Element> && aristos::TensorValueType<ElementScale>);
        static_assert(s == PacketStage::initial);
        __device__ __forceinline__
        void operator()(cuda::std::byte* __restrict__ const& packet,
        unsigned int* __restrict__ const& status,
        unsigned int* __restrict__ const& taskCount,
        unsigned int const& localExpertIdx,
        cuda::std::byte* __restrict__ const& pGB, //postGEMM buffer
        const cuda::std::array<cuda::std::byte*, GEMMs>& weights,
        const cuda::std::array<cuda::std::byte*, GEMMs>& bias,
        unsigned int const& peer, // relative to the EP group
        unsigned int& lTQHead,
        unsigned int* __restrict__ const& gTQHead) const {
            // TODO put config in shared memory allocated to this block
            // process header
            const auto [totalTokens, routedTokens] = *CAST_TO(uint2, packet);
            const auto globalTaskTiles = Config::tiles<BLOCK_M>(totalTokens);
            const auto fTilesM = routedTokens / BLOCK_M;
            const auto padM = Config::pad<BLOCK_M>(routedTokens);
            if (!atomicTAS<cuda::thread_scope_block>(status + peer)) {
                // atomically reduce taskCount
                const auto superfluous = (moeConfig.tilesN + moeConfig.tilesNx) *
                    (Config::tiles<BLOCK_M>(moeConfig.expertCapacity) - globalTaskTiles);
                atomicSub_block(taskCount, superfluous);
            }
            // process payload
            auto* __restrict__ scaleWeights = packet + sizeof(uint2);
            auto* __restrict__ tokens = scaleWeights + sizeof(ElementScale) * padM;
            const auto tokenSize = moeConfig.embedDim;
            auto* __restrict__ tQ = schedulerState.taskQ + lTQHead;

            // expert, peer offset
            const auto pXo = moeConfig.expertCapacity * (peer * moeConfig.numExperts + localExpertIdx);
            cuda::std::array<cuda::std::byte*, GEMMs> taskResults{};
            // Staging buffer for results of preGEMM
            taskResults[0] = pGB + pXo * tokenSize * sizeof(Element);
            // Egress packet buffer
            taskResults[1] = p == PeerConnectivity::remote ? heap::advance<1, 0>(peer, localExpertIdx) :
            heap::advance<1, 1>(peer, localExpertIdx);
            const auto tilesN = moeConfig.tilesN;
            const auto totalTasks = Config::tiles<BLOCK_M>(routedTokens) * tilesN;

            // TODO unroll these loops
            for (uint i = 0; i < fTilesM; ++i) {
                for (uint j = 0; j < tilesN; ++j) {
                    const auto tileIdx = j + i * tilesN;
                    tQ[tileIdx] = Task{
                        TaskType::preGEMM,
                        tokens,
                        weights,
                        taskResults,
                        bias,
                        scaleWeights,
                        pXo + i,
                        tileIdx,
                        padM,
                        pXo + (p == PeerConnectivity::remote ? i : tileIdx),
                        BLOCK_M,
                        peer,
                        i
                    };
                }
            }

            // residue tile
            const auto residue = routedTokens - fTilesM * BLOCK_M;
            // TODO unroll these loops
            for (uint j = 0; j < tilesN; ++j) {
                const auto tileIdx = j + fTilesM * tilesN;
                tQ[tileIdx] = Task{
                    TaskType::preGEMM,
                    tokens,
                    weights,
                    taskResults,
                    bias,
                    scaleWeights,
                    pXo + fTilesM,
                    tileIdx,
                    padM,
                    pXo + (p == PeerConnectivity::remote ? fTilesM : tileIdx),
                    residue,
                    peer,
                    fTilesM
                };
            }

            if (routedTokens) {
                __threadfence();
                lTQHead += totalTasks;
                // notifies scheduler of work
                atomicAdd_block(gTQHead, totalTasks);
            }
        }
    };


    template<>
    struct Decoder<PacketStage::final, PeerConnectivity::p2p> {
        __device__ __forceinline__
        void operator()(cuda::std::byte* __restrict__ const& packet,
            cuda::std::byte* __restrict__ const& tokenIndices,
            cuda::std::byte* __restrict__ const& activations,
            const unsigned int& nTokens,
            unsigned int& lTQHead,
            const unsigned int& tileIdx,
            unsigned int* __restrict__ const& gTQHead) const {
            // now let's decode this single tile
            auto* __restrict__ tQ = schedulerState.taskQ + lTQHead++;
            *tQ = Task{
                TaskType::combine,
                tokenIndices,
                cuda::std::array<cuda::std::byte*, GEMMs>{packet},
                cuda::std::array<cuda::std::byte*, GEMMs>{activations},
                nTokens,
                tileIdx,
                BLOCK_M
            };
            __threadfence();
            // notifies scheduler of work
            atomicIncrement<cuda::thread_scope_block>(gTQHead);
        }
    };

    template<>
    struct Decoder<PacketStage::final, PeerConnectivity::remote> {
        __device__ __forceinline__
        void operator()(cuda::std::byte* __restrict__ const& packet,
            cuda::std::byte* __restrict__ const& tokenIndices,
            cuda::std::byte* __restrict__ const& activations,
            const unsigned int& nTokens,
            unsigned int& lTQHead,
            unsigned int* __restrict__ const& gTQHead) const {
            auto* __restrict__ tQ = schedulerState.taskQ + lTQHead;
            const auto tilesN = moeConfig.tilesN;
            constexpr auto tB = 8;
            const auto tBs = tilesN / tB;
            for (uint i = 0; i < tBs; ++i) {
                #pragma unroll
                for (uint j = 0; j < tB; ++j) {
                    tQ[j + i * tB] = Task{
                        TaskType::combine,
                        tokenIndices,
                        cuda::std::array<cuda::std::byte*, GEMMs>{packet},
                        cuda::std::array<cuda::std::byte*, GEMMs>{activations},
                        nTokens,
                        j + i * tB,
                        BLOCK_M
                    };
                }
            }

            for (uint j = tBs * tB; j < tilesN; ++j) {
                tQ[j] = Task{
                    TaskType::combine,
                    tokenIndices,
                    cuda::std::array<cuda::std::byte*, GEMMs>{packet},
                    cuda::std::array<cuda::std::byte*, GEMMs>{activations},
                    nTokens,
                    j,
                    BLOCK_M
                };
            }

            __threadfence();
            lTQHead += tilesN;
            // notifies scheduler
            atomicIncrement<cuda::thread_scope_block>(gTQHead);
        }
    };
}
#endif //PACKET_CUH
