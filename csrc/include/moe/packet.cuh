//
// Created by oja7 on 12/2/24.
//

#ifndef PACKET_CUH
#define PACKET_CUH
#include "../definition/types.cuh"
#include "../definition/memory_layout.cuh"
#include "../util/atomics.cuh"

namespace aristos::packet {
    template<unsigned int blocks,
    DropTokens d = DropTokens::yes,
    unsigned int superBlockSize = ARISTOS_SUPER_BLOCK_SIZE,
    typename Activations, typename GateProb>
    requires aristos::Matrix<Activations> && aristos::Matrix<GateProb>
    __forceinline__ __device__
    void encode(const Activations& activations, const GateProb& gateOutput,
        unsigned int* const& __restrict__ workspace) {
        // assert(blocks <= gridDim.x - 1)
        static_assert(blocks % superBlockSize == 0);
        using ElementAct = typename Activations::value_type;
        using ElementGate = typename GateProb::value_type;

        // Map a static set of blocks to an expert and stride as thus
        constexpr auto numSuperBlocks = blocks / superBlockSize;
        const auto superBlockIdx = blockIdx.x / superBlockSize;
        const auto lBid = blockIdx.x % superBlockSize;
        const auto tIdx =  lBid * THREADS + threadIdx.x;
        const bool isLeader = !blockIdx.x && !threadIdx.x;
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
            // TODO below is not needed
            const auto prt = pRT[expertIdx];
            // logical expert index
            const auto lIdx = expertIdx / numSuperBlocks;
            const auto pLIdx = iPS[lIdx];
            const auto routedTokens = d == DropTokens::yes ?
                cute::min(expertCounts[lIdx], cap) : expertCounts[lIdx];
            if (!routedTokens) {
                // nothing to do here, folks
                continue;
            }
            const auto padded = Config::pad<BLOCK_M>(routedTokens);
            const auto peer = pS[lIdx];
            const auto pTT = cute::min(peerTotalTokens[peer], moeConfig.capacity);
            const auto isRemote = nvshmem_ptr(moeConfig.sHeap, peer) == nullptr;
            auto* peerHeap = static_cast<cuda::std::byte*>(isRemote ?
                heap::advance<0, 0>(peer) + prt :
                nvshmem_ptr(heap::advance<0, 1>(peer) + prt, peer));
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
            // copy scale probabilities
            for (uint j = tIdx; j < routedTokens; j += superBlockSize * THREADS) {
                peerHeap[j] = gateOutput(tokenIds(expertIdx, j), expertIdx);
            }
            peerHeap += sizeof(ElementGate) * routedTokens;

            // Not needed for P2P connectivity due to non-contiguous access
            if (isRemote) {
                // Padding the below rather than tokens reduces the packet size by pad amount * embedDim
                // Technically, we could optimize the below by sending a single zero byte per element,
                // rather than the materialized element 0.
                // This jeopardizes decoding as the decoder expects all data to be available in the
                // packet.
                for (uint j = tIdx; j < padded; j += superBlockSize * THREADS) {
                    peerHeap[j] = ElementGate{0};
                }
            }

            peerHeap += sizeof(ElementGate) * (padded - routedTokens);

            // copy tokens: not padded
            for (uint j = lBid; j < routedTokens; j += superBlockSize) {
                auto* localPH = peerHeap + j * moeConfig.embedDim * sizeof(ElementAct);
                const auto tokenIdx = tokenIds(expertIdx, j);
                const auto vTokenSize = moeConfig.embedDim / (sizeof(cute::uint128_t) / sizeof(ElementAct));
                // Use high-throughput vector copy
                for (uint k = threadIdx.x; k < vTokenSize; k += THREADS) {
                    CAST_TO(cute::uint128_t, localPH)[k] = activations(tokenIdx, k);
                }
                localPH += sizeof(cute::uint128_t) * vTokenSize;
                for (uint k = threadIdx.x; k < moeConfig.embedDim; k += THREADS) {
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
                        nvshmem_putmem_signal_nbi(heap::advance<0, 1>(peer) + prt,
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
    template<PacketStage p = PacketStage::initial,
    typename ElementData,
    typename ElementScale>
    __device__ __forceinline__
    void decode(cuda::std::byte* __restrict__ const& packet,
        unsigned short int* __restrict__ const& status,
        unsigned int* __restrict__ const& taskCount,
        unsigned int const& localExpertIdx,
        cuda::std::byte* __restrict__& cInitial,
        cuda::std::byte** __restrict__& weights,
        cuda::std::byte** __restrict__& bias,
        unsigned int const& peer) {
        static_assert(p == PacketStage::initial);
        // process header
        const auto [totalTokens, routedTokens] = *CAST_TO(uint2, packet);
        const auto globalTaskTiles = Config::tiles<BLOCK_M>(totalTokens);
        const auto taskTiles = routedTokens / BLOCK_M;
        if (!atomicTAS(status + peer)) {
            // atomically reduce taskCount
            const auto superfluous = (moeConfig.tilesN + moeConfig.tilesNx) *
                (Config::tiles<BLOCK_M>(moeConfig.capacity) - globalTaskTiles);
            atomicSub(taskCount, superfluous);
        }
        // process payload
        auto* tokenIndices = packet + sizeof(uint2);
        auto* scaleWeights = tokenIndices + sizeof(unsigned int) * routedTokens;
        auto* tokens = scaleWeights + sizeof(ElementScale) * Config::pad<BLOCK_M>(routedTokens);
        const auto tokenSize = moeConfig.embedDim;
        auto* schedulerDB = schedulerState.taskQSignals;

        // TODO abstract the below using structured bindings
        auto* tQ = schedulerState.taskQ;
        auto* tQHead = schedulerState.taskQSignals + 1;

        const auto syncIdx = moeConfig.expertCapacity * (peer * moeConfig.numExperts + localExpertIdx);

        // Below batch notifies the scheduler
        for (uint i = 0; i < taskTiles; ++i) {
            tQ[atomicIncrement(tQHead)] = Task{
                TaskType::preGEMM,
                tokens + i * BLOCK_M * tokenSize,
                weights,
                nullptr,
                bias,
                scaleWeights + i * BLOCK_M,
                syncIdx + i,
                i,
                Config::pad<BLOCK_M>(routedTokens),
                BLOCK_M,
                peer,
            };
            // Populate ids in egress packet buffer
        }

        // Residual task
        if (routedTokens % BLOCK_M != 0) {
            tQ[atomicIncrement(tQHead)] = Task{
                TaskType::preGEMM,
                tokens + taskTiles * BLOCK_M * tokenSize,
                weights,
                nullptr,
                bias,
                scaleWeights + taskTiles * BLOCK_M,
                syncIdx + taskTiles,
                taskTiles,
                Config::pad<BLOCK_M>(routedTokens),
                routedTokens - taskTiles * BLOCK_M,
                peer,
            };
        }
        // Populate ids in egress packet buffer

        __threadfence();
        // notify scheduler
        atomicAdd(schedulerDB, taskTiles);
    }
}
#endif //PACKET_CUH
