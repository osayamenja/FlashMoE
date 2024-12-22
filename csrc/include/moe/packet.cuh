//
// Created by oja7 on 12/2/24.
//

#ifndef PACKET_CUH
#define PACKET_CUH

#include <nvshmemx.h>
#include <nvshmem.h>
#include <host/nvshmemx_api.h>

#include "../definition/types.cuh"
#include "../definition/memory_layout.cuh"
#include "../util/atomics.cuh"

namespace aristos::packet {
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
            const auto pTT = cute::min(peerTotalTokens[peer], moeConfig.capacity);
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

    // Decodes a single packet,
    // TODO make a struct and parametrize for remote peer
    template<PacketStage p = PacketStage::initial,
    typename Element,
    typename ElementScale>
    __device__ __forceinline__
    void decode(cuda::std::byte* __restrict__ const& packet,
        unsigned int* __restrict__ const& status,
        unsigned int* __restrict__ const& taskCount,
        unsigned int const& localExpertIdx,
        cuda::std::byte* __restrict__& cInitialBase,
        unsigned int* const& cBOffset,
        cuda::std::byte* __restrict__& peerHeapBase,
        unsigned long int* pHOffset,
        const cuda::std::array<cuda::std::byte*, GEMMs>& weights,
        const cuda::std::array<cuda::std::byte*, GEMMs>& bias,
        unsigned int const& peer) {
        static_assert(p == PacketStage::initial);
        // process header
        const auto [totalTokens, routedTokens] = *CAST_TO(uint2, packet);
        const auto globalTaskTiles = Config::tiles<BLOCK_M>(totalTokens);
        const auto taskTiles = routedTokens / BLOCK_M;
        const auto padM = Config::pad<BLOCK_M>(routedTokens);
        if (!atomicTAS<cuda::thread_scope_block>(status + peer)) {
            // atomically reduce taskCount
            const auto superfluous = (moeConfig.tilesN + moeConfig.tilesNx) *
                (Config::tiles<BLOCK_M>(moeConfig.capacity) - globalTaskTiles);
            atomicSub_block(taskCount, superfluous);
        }
        // process payload
        auto* tokenIndices = packet + sizeof(uint2);
        auto* scaleWeights = tokenIndices + sizeof(unsigned int) * routedTokens;
        auto* tokens = scaleWeights + sizeof(ElementScale) * padM;
        const auto tokenSize = moeConfig.embedDim;
        auto* schedulerDB = schedulerState.taskQSignals;

        auto* tQ = schedulerState.taskQ;
        auto* tQHead = schedulerState.taskQSignals + 1;

        const auto syncIdx = moeConfig.expertCapacity * (peer * moeConfig.numExperts + localExpertIdx);
        cuda::std::array<cuda::std::byte*, GEMMs> taskData{};
        taskData[0] = cInitialBase +
            atomicAdd_block(cBOffset, padM) * BLOCK_M * BLOCK_N * sizeof(Element);
        auto* packetBuffer = peerHeapBase + atomicAdd_block(pHOffset, Config::finalPacketSize<Element>(routedTokens));
        taskData[1] = packetBuffer + sizeof(unsigned int) * (1 + routedTokens);

        if (routedTokens) {
            *CAST_TO(unsigned int, packetBuffer) = routedTokens;
            packetBuffer += sizeof(unsigned int);
            const auto vRt = routedTokens / (sizeof(uint4) / sizeof(unsigned int));
            for (uint i = 0; i < vRt; ++i) {
                CAST_TO(uint4, packetBuffer)[i] = CAST_TO(uint4, tokenIndices)[i];
            }
            packetBuffer += sizeof(uint4) * vRt;
            const auto residue = routedTokens - vRt * (sizeof(uint4) / sizeof(unsigned int));
            for (uint i = 0; i < residue; ++i) {
                CAST_TO(unsigned int, packetBuffer)[i] = CAST_TO(unsigned int, tokenIndices)[i];
            }
        }

        const auto tilesN = moeConfig.tilesN;
        const auto flagIdx = moeConfig.rank * moeConfig.numExperts * moeConfig.capacity + localExpertIdx * moeConfig.capacity;

        if (nvshmem_ptr(moeConfig.sHeap, peer) != nullptr)[[likely]] {
            // TODO unroll this loop
            // Below batch notifies the scheduler
            for (uint i = 0; i < taskTiles; ++i) {
                for (uint j = 0; j < tilesN; ++j) {
                    tQ[atomicIncrement(tQHead)] = Task{
                        TaskType::preGEMM,
                        tokens,
                        weights,
                        taskData,
                        bias,
                        scaleWeights,
                        syncIdx + i,
                        i * tilesN + j,
                        padM,
                        flagIdx + i,
                        BLOCK_M,
                        peer,
                    };
                }
            }

            // Residual task
            if (routedTokens % BLOCK_M != 0) {
                for (uint j = 0; j < tilesN; ++j) {
                    tQ[atomicIncrement(tQHead)] = Task{
                        TaskType::preGEMM,
                        tokens,
                        weights,
                        taskData,
                        bias,
                        scaleWeights,
                        syncIdx + taskTiles,
                        taskTiles * tilesN + j,
                        padM,
                        flagIdx + taskTiles,
                        routedTokens - taskTiles * BLOCK_M,
                        peer,
                    };
                }
            }
        }
        else {
            // P2P peer
            for (uint i = 0; i < taskTiles; ++i) {
                for (uint j = 0; j < tilesN; ++j) {
                    tQ[atomicIncrement(tQHead)] = Task{
                        TaskType::preGEMM,
                        tokens ,
                        weights,
                        taskData,
                        bias,
                        scaleWeights,
                        syncIdx + i,
                        i * tilesN + j,
                        padM,
                        flagIdx + i * tilesN + j,
                        BLOCK_M,
                        peer,
                    };
                }
            }

            // Residual task
            if (routedTokens % BLOCK_M != 0) {
                for (uint j = 0; j < tilesN; ++j) {
                    tQ[atomicIncrement(tQHead)] = Task{
                        TaskType::preGEMM,
                        tokens,
                        weights,
                        taskData,
                        bias,
                        scaleWeights,
                        syncIdx + taskTiles,
                        taskTiles * tilesN + j,
                        padM,
                        flagIdx + taskTiles * tilesN + j,
                        routedTokens - taskTiles * BLOCK_M,
                        peer,
                    };
                }
            }
        }

        if (routedTokens) {
            __threadfence();
            // notify scheduler
            atomicAdd(schedulerDB, taskTiles);
        }
    }
}
#endif //PACKET_CUH
