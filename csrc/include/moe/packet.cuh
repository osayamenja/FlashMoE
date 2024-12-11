//
// Created by oja7 on 12/2/24.
//

#ifndef PACKET_CUH
#define PACKET_CUH
#include "../definition/memory_layout.cuh"
#include "../util/atomics.cuh"

namespace aristos::packet {
    template<unsigned int blocks, typename Activations, typename GateProb>
    requires aristos::Matrix<Activations> && aristos::Matrix<GateProb>
    __forceinline__ __device__
    void constructSend(const Activations& activations, const GateProb& gateOutput,
        cuda::std::byte* __restrict__ workspace) {
        __shared__ unsigned int packetTokenIdx;
        // assumption that simplifies pointer arithmetic
        static_assert(sizeof(unsigned int) == sizeof(maxPrecision));
        using ElementAct = typename Activations::value_type;
        using ElementGate = typename GateProb::value_type;
        using TrailerPair = cuda::std::pair<unsigned int, ElementAct>;

        // Map 64 blocks to a peer and stride as thus
        constexpr auto residual = blocks / ARISTOS_SUPER_BLOCK_SIZE * ARISTOS_SUPER_BLOCK_SIZE;
        constexpr auto numSuperBlocks = blocks / ARISTOS_SUPER_BLOCK_SIZE + (blocks - residual > 32);
        const auto superBlockIdx = blockIdx.x / ARISTOS_SUPER_BLOCK_SIZE -
            (blockIdx.x >= residual && blocks - residual < 32);
        auto* superBlockBarrier = moeConfig.packetBarriers[superBlockIdx];

        const auto numExperts = moeConfig.numExperts;
        const auto seqLen = moeConfig.seqLen;
        const auto world = moeConfig.worldSize;
        const auto len = 2 * (numExperts + world);
        const auto frameSize = moeConfig.frameSize<ElementAct>();
        const auto tokenSize = moeConfig.embedDim;
        auto* scratch = CAST_TO(unsigned int, workspace);
        auto* flags = moeConfig.flags;
        const auto* expertCounts = scratch;
        const auto* prefixExperts = scratch + numExperts;
        const auto* experts = prefixExperts + world + 1; // sentinel at the end of this array
        const auto* peerTranslation = experts + numExperts;
        // below is above + world; done as such to maintain const for the above
        auto* expertOffsets = CAST_TO(unsigned long int, scratch + numExperts + world + 1 + numExperts + world);
        for (unsigned int i = threadIdx.x; i < numExperts; i += THREADS) {
            scratch[i] = moeConfig.getExpertCounts()[i];
        }
        scratch += numExperts;
        for (unsigned int i = threadIdx.x; i < len; i += THREADS) {
            scratch[i] = moeConfig.getPeerXLookup()[i];
        }
        __syncthreads();

        // Build expert offsets using a prefix sum
        for (uint i = threadIdx.x; i < world; ++i) {
            // get offset from prefix array
            const auto offset = prefixExperts[i];
            const auto length = prefixExperts[i + 1] - offset;
            auto* slice = experts + offset;
            expertOffsets[slice[0]] = 0;
            auto prev = 0;
            for (uint j = 1; j < length; ++j) {
                auto actualExpertCounts = expertCounts[slice[j - 1]];
                // frame length = padded(tokens) + padded(probabilities) + ids + single integer
                prev += Config::frameSize<ElementAct>(actualExpertCounts);
                expertOffsets[slice[j]] = prev;
            }
        }
        // Build and send a packet for this peer
        for (unsigned int i = superBlockIdx; i < world; i += numSuperBlocks) {
            const auto numPeerExperts = prefixExperts[i + 1] - prefixExperts[i];
            const auto counts = numPeerExperts * seqLen;
            const bool isRemote = nvshmem_ptr(moeConfig.sHeap, i) == nullptr;
            auto* peerExperts = experts + prefixExperts[i];
            // If the peer is P2P connected, then all memory writes below are NVLink transfers
            // TODO benchmark the below vs staging and sending one giant packet instead
            auto* peerHeap = static_cast<cuda::std::byte*>(isRemote? heap::advanceRemote<0, 0>(i) :
                nvshmem_ptr(heap::advanceP2P<0>(i), i));
            // Iterate through experts
            for (unsigned int j = superBlockIdx; j < counts; j += ARISTOS_SUPER_BLOCK_SIZE) {
                const auto expertOffset = j / seqLen;
                const auto expertIdx = peerExperts[expertOffset];
                const auto tokenIdx = j % seqLen;
                auto* packetBuffer = peerHeap + expertOffsets[expertOffset];
                auto* packetPayload = CAST_TO(ElementAct, packetBuffer + 1);
                auto* packetTrailer = static_cast<TrailerPair*>(static_cast<void*>(packetPayload + tokenSize));
                auto expertProbability = gateOutput(tokenIdx, expertIdx);
                if (expertProbability > ElementGate(0)) {
                    // insert into packet
                    if (!threadIdx.x) {
                        // Increment token count, needed later for packet transfer
                        // Note that using device scope rather than system is valid here
                        // because the remote peer will not access this buffer
                        // until we signal its availability.
                        packetTokenIdx = atomicIncrement(CAST_TO(unsigned int, packetBuffer));
                        // Tag on a subset of this packet's trailer
                        // Hopefully the pair struct here motivates the compiler
                        // to use a 64-bit store.
                        packetTrailer[packetTokenIdx] = TrailerPair{tokenIdx, expertProbability};
                    }
                    __syncthreads();
                    // single block cooperates to transfer token to the packet's buffer
                    // we could do a 128-bit transfer here,
                    // but that would require assertions to ensure corrections
                    for (unsigned int k = threadIdx.x; k < tokenSize; ++k) {
                        packetPayload[k] = activations(tokenIdx, k);
                    }
                }
            }
            __syncthreads();
            if (!threadIdx.x) {
                superBlockBarrier->arrive_and_wait();
                const auto actualCount = expertCounts[peerExperts[numPeerExperts - 1]];
                const auto totalBytes = expertOffsets[numPeerExperts - 1] + Config::frameSize<ElementAct>(actualCount);
                if (!superBlockIdx) {
                    if (isRemote) {
                        // Only one thread is needed for RDMA transfer
                        nvshmem_putmem_signal_nbi(peerHeap, peerHeap,
                            totalBytes,
                            flags + i,
                            constructSignal(shouldProcess), NVSHMEM_SIGNAL_SET, peerTranslation[i]);
                    }
                    else {
                        // We have already done the transfer; thus, we set the signal only
                        nvshmemx_signal_op(flags + i,
                            constructSignal(shouldProcess), NVSHMEM_SIGNAL_SET, peerTranslation[i]);
                    }
                }
            }
        }
    }
}
#endif //PACKET_CUH
