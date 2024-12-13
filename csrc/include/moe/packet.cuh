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

        const auto numExperts = moeConfig.numExperts;
        const auto seqLen = moeConfig.seqLen;
        const auto world = moeConfig.worldSize;
        const auto len = 2 * (numExperts + world);
        const auto tokenSize = moeConfig.embedDim;
        auto* scratch = CAST_TO(unsigned int, workspace);
        auto* flags = moeConfig.flags;
        // readonly arrays
        const auto* expertCounts = scratch;
        const auto* prefixExperts = scratch + numExperts;
        const auto* experts = prefixExperts + world + 1; // sentinel at the end of this array
        const auto* peerTranslation = experts + numExperts;
        for (unsigned int i = threadIdx.x; i < numExperts; i += THREADS) {
            scratch[i] = moeConfig.getExpertCounts()[i];
        }
        scratch += numExperts;
        for (unsigned int i = threadIdx.x; i < len; i += THREADS) {
            scratch[i] = moeConfig.getPeerXLookup()[i];
        }
        __syncthreads();

        scratch += world + 1 + numExperts + world;
        auto* expertOffsets = CAST_TO(unsigned long long int, scratch);
        auto* tokenOffsets = CAST_TO(unsigned int, expertOffsets + numExperts);
        // Build expert and token offsets using a prefix sum
        for (uint i = threadIdx.x; i < world; ++i) {
            // get offset from prefix array
            const auto offset = prefixExperts[i];
            const auto length = prefixExperts[i + 1] - offset;
            const auto* slice = experts + offset;
            expertOffsets[slice[0]] = 0;
            tokenOffsets[slice[0]] = 0;
            auto p = 0UL;
            auto tP = 0U;
            for (uint j = 1; j < length; ++j) {
                auto actualExpertCounts = expertCounts[slice[j - 1]];
                p += Config::frameSize<ElementAct>(actualExpertCounts, tokenSize);
                tP += cute::ceil_div(actualExpertCounts, THREADS);
                expertOffsets[slice[j]] = p;
                tokenOffsets[slice[j]] = tP;
            }
        }
        __syncthreads();
        auto* staging = CAST_TO(ElementGate, scratch + 2 * numExperts + world);
        // Account for subscriber block currently not here
        constexpr auto blockStride = blocks - 1;
        // upper bound metadata at <= 1024 elements
        constexpr auto stagingSize = SHARED_SIZE / sizeof(ElementGate) - 1024;
        static_assert(stagingSize % THREADS == 0);
        constexpr auto prefetchStages = stagingSize / THREADS;
        const auto trips = seqLen / stagingSize;
        const auto residue = seqLen - trips * stagingSize;

        // data structures
        const unsigned int* epSpec = nullptr;
        for (unsigned int i = blockIdx.x; i < numExperts; i += blockStride) {
            const auto peer = epSpec[i];
            auto* peerHeap = static_cast<cuda::std::byte*>(nvshmem_ptr(moeConfig.sHeap, peer) == nullptr ?
                heap::advanceRemote<0, 0>(peer) : nvshmem_ptr(heap::advanceP2P<0>(peer), peer));
            auto* packetBuffer = peerHeap + expertOffsets[i];
            const auto routedTokens = expertCounts[i];
            auto* packetPayload = CAST_TO(ElementAct, packetBuffer + 1);
            auto* packetPayProb = packetPayload + Config::pad<BLOCK_M>(routedTokens) * tokenSize;
            auto* packetIds = CAST_TO(unsigned int, packetPayProb + Config::pad<BLOCK_M>(routedTokens));
            unsigned int tokenIdx = 0U;

            if (!threadIdx.x) {
                // header -> number of tokens in this packet
                *CAST_TO(unsigned int, packetBuffer) = routedTokens;
                const auto startIdx = tokenOffsets[i];
                const auto nIds = cute::ceil_div(expertCounts[i], BLOCK_M);
                packetIds += startIdx;
                packetIds[0] = startIdx;
                for (unsigned int j = 0; j < nIds; ++j) {
                    packetIds[j] = startIdx + j;
                }
            }
            // Split into two loop types: a static loop below and dynamic loop subsequently
            // We unroll the static one.
            if (trips) {
                // prefetch subset of token context to shared memory
                #pragma unroll
                for (unsigned int k = threadIdx.x; k < stagingSize; ++k) {
                    staging[k] = gateOutput(k, i);
                }
            }

            for (unsigned int j = 0; j < trips; ++j) {
                __syncthreads();
                #pragma unroll
                for (unsigned int k = 0; k < stagingSize; ++k) {
                    const auto probability = staging[i];
                    if (k < prefetchStages && j + 1 < trips) {
                        // overlap prefetch of the next batch
                        staging[threadIdx.x + k * THREADS] = gateOutput(k, i);
                    }
                    if (probability > ElementGate(0)) {
                        if (!threadIdx.x) {
                            // write trailer
                            packetPayProb[tokenIdx] = probability;
                        }
                        packetPayload += tokenSize * tokenIdx++;
                        for (int l = threadIdx.x; l < tokenSize; ++l) {
                            packetPayload[l] = activations(k, l);
                        }
                    }
                }
            }
            for (unsigned int k = threadIdx.x; k < residue; ++k) {
                staging[k] = gateOutput(k, i);
            }
            // Subset of token sequence now resides in shared memory
            for (unsigned int k = 0; k < stagingSize; ++k) {
                const auto probability = staging[i];
                if (probability > ElementGate(0)) {
                    if (!threadIdx.x) {
                        // write trailer
                        packetPayProb[tokenIdx] = probability;
                    }
                    packetPayload += tokenSize * tokenIdx++;
                    for (int l = threadIdx.x; l < tokenSize; ++l) {
                        packetPayload[l] = activations(k, l);
                    }
                }
            }
            // ensures data persistence before network transfer
            __syncthreads();
            // send the packet or signal
            if (!threadIdx.x) {
                const auto totalBytes = sizeof(unsigned int) *
                    (1 + cute::ceil_div(expertCounts[i], BLOCK_M))
                + Config::pad<BLOCK_M>(routedTokens) * (tokenSize * sizeof(ElementAct) + sizeof(ElementGate));
                if (nvshmem_ptr(moeConfig.sHeap, peer) == nullptr) {
                    // Only one thread is needed for RDMA transfer
                    nvshmem_putmem_signal_nbi(peerHeap, peerHeap,
                        totalBytes,
                        flags + i,
                        sent, NVSHMEM_SIGNAL_ADD, peerTranslation[i]);
                }
                else {
                    __threadfence_system();
                    // We have already done the transfer; thus, we set the signal only
                    nvshmemx_signal_op(flags + i,
                        sent, NVSHMEM_SIGNAL_ADD, peerTranslation[i]);
                }
            }
        }
    }
}
#endif //PACKET_CUH
