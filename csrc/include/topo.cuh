//
// Created by oja7 on 9/19/24.
//

#ifndef TOPO_CUH
#define TOPO_CUH

#include <cuda/cmath>
#include "types.cuh"
#include "atomics.cuh"

namespace aristos::topology{
    struct __align__(8) TopologySignal{
        unsigned int throughput;
        unsigned int signal;
    };
    __device__ __forceinline__
    auto* advancePtr(cuda::std::byte* __restrict__ const& buffer, const unsigned int& slot) {
        return buffer + slot * BETA_BUFFER;
    }

    __device__ __forceinline__
    void awaitResponses(uint64_t* __restrict__ const& flags,
        uint* __restrict__ const& syncArray, uint* __restrict__ const& rates,
        const uint& rank, const uint& n, const uint& seqNo) {
        using Payload = TopologySignal;
        auto result = Payload{};
        for (int i = threadIdx.x; i < n; i += ARISTOS_BLOCK_SIZE) {
            if (i != rank) {
                awaitPayload<cuda::thread_scope_system>(CAST_TO(unsigned long long int, flags + i), &result, seqNo);
                rates[i] = result.throughput;
                // Enforce consistency
                nvshmem_uint_test(&CAST_TO(Payload, flags + i)->signal, NVSHMEM_CMP_EQ, seqNo);
            }
        }
        if (!threadIdx.x) {
            *syncArray = seqNo;
        }
    }

    template<size_t betaBuf = BETA_BUFFER, size_t alphaBuf = ALPHA_BUFFER, typename Put>
    requires (cuda::std::is_invocable_r_v<void, Put, void*, const void*, size_t, int> && betaBuf > 0 && alphaBuf > 0)
    __device__ __forceinline__
    void measureTransfer(const unsigned int& rank, cuda::std::byte* __restrict__ const& sHeap,
        floatPair* __restrict__ const& remoteDurations, const int& peer,
        const unsigned int& id, const Put& put, const unsigned int& peerIdx,
        const unsigned int& lBid = 0, const unsigned int& nb = 1) {
        ull_t start, end;
        float duration = 0.0;
        bool isResidual = alphaBuf % nb != 0 && alphaBuf > nb && lBid == nb - 1 ;
        size_t buf = (lBid * cute::ceil_div(alphaBuf, nb) < alphaBuf) *
                (!isResidual * cute::ceil_div(alphaBuf, nb)
                + isResidual * (alphaBuf - (cute::ceil_div(alphaBuf, nb) * (nb - 1))));
        /// Alpha cost: ms
        #pragma unroll
        for (unsigned int j = 0; j < TOPO_LOOP_TRIP; ++j) {
            asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
            put(advancePtr(sHeap, rank) + (lBid * buf), advancePtr(sHeap, rank) + (lBid * buf), buf, peer);
            asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
            duration += static_cast<float>(end - start) / static_cast<float>(TOPO_LOOP_TRIP*NANO_TO_MILLI);
        }
        if(!id) {
            remoteDurations[peerIdx].alpha = duration;
        }
        duration = 0.0;
        isResidual = betaBuf % nb != 0 && betaBuf > nb && lBid == nb - 1;
        buf = (lBid * cute::ceil_div(betaBuf, nb) < betaBuf) * (!isResidual * cute::ceil_div(betaBuf, nb)
               + isResidual * (betaBuf - (cute::ceil_div(betaBuf, nb) * (nb - 1))));
        ///Beta Cost: ms/MB
        #pragma unroll
        for (unsigned int j = 0; j < TOPO_LOOP_TRIP; ++j) {
            asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
            put(advancePtr(sHeap, rank) + (lBid * buf), advancePtr(sHeap, rank) + (lBid * buf), buf, peer);
            asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
            duration += static_cast<float>(end - start) / static_cast<float>(TOPO_LOOP_TRIP*NANO_TO_MILLI);
        }
        if(!id) {
            // Compute beta using slope intercept equation
            remoteDurations[peerIdx].beta = ((duration - remoteDurations[peerIdx].alpha) / (TO_MB(betaBuf) - TO_MB(alphaBuf)));
            // Compute alpha and apply abs for spurious negatives
            remoteDurations[peerIdx].alpha = fabs(remoteDurations[peerIdx].alpha - (TO_MB(alphaBuf) * remoteDurations[peerIdx].beta));
        }
    }

    __device__ __forceinline__
    void singularBuilder(floatPair* __restrict__ const& scratchpad, const unsigned int& n,
        const unsigned int& rank, const unsigned int& processingRate,
        cuda::std::byte* __restrict__ const& sHeap, floatPair* __restrict__ const& results,
        uint64_t* __restrict__ const& flags, uint* __restrict__ const& syncArray, uint* __restrict__ const& rates) {
        const auto seqNo = (__ldg(syncArray) + 1) % 2; // Equivalent to a bit flip
        for (unsigned int i = 1U; i < n; ++i) {
            const auto peer = (rank + i) % n;
            measureTransfer(rank, sHeap, scratchpad, peer, threadIdx.x, nvshmemx_putmem_block, peer);
        }
        __syncthreads();

        /// Stage my row on the symmetric heap
        for (unsigned int i = threadIdx.x; i < n; i += ARISTOS_BLOCK_SIZE) {
            results[i] = scratchpad[i];
        }

        auto signal = TopologySignal{processingRate, seqNo};
        // Signal my vector, including FLOPs, to others
        for (unsigned int i = 1U; i < n; ++i) {
            nvshmemx_putmem_signal_nbi_block(results, results, n * sizeof(floatPair), flags + rank,
                *CAST_TO(uint64_t, &signal), NVSHMEM_SIGNAL_SET, (rank + i) % n);
        }

        // await responses from other GPUs
        awaitResponses(flags, syncArray, rates, rank, n, seqNo);
    }

    __device__ __forceinline__
    void pluralRemoteBuilder(floatPair* __restrict__ const& scratchpad, const unsigned int* __restrict__& peers,
        const unsigned int& n, const unsigned int& rank,
        const unsigned int& processingRate, const unsigned int numPeers, cuda::std::byte* __restrict__ const& sHeap,
        floatPair* __restrict__ const& results, uint64_t* __restrict__ const& flags,
        uint* __restrict__ const& syncArray) {
        const auto seqNo = (__ldg(syncArray) + 1) % 2; // Equivalent to a bit flip
        auto* blockade = syncArray + 1;
        for (unsigned int i = 0U; i < numPeers; ++i) {
            const auto idx = (i + rank) % numPeers;
            measureTransfer(rank, sHeap, scratchpad, peers[idx], threadIdx.x, nvshmemx_putmem_block, idx);
        }
        __syncthreads();

        /// Stage my row to the symmetric heap
        for (unsigned int i = threadIdx.x; i < numPeers; i += ARISTOS_BLOCK_SIZE) {
            results[peers[i]] = scratchpad[i];
        }
        __syncthreads();

        // Ensures our vector is complete before sending to neighbors.
        // if num remote peers < n - 1, then we must await the contribution of our p2p siblings
        if (!threadIdx.x && numPeers < n - 1) {
            __threadfence();
            atomicAdd(blockade, 1);
            while (atomicLoad(blockade) % gridDim.x != 0) {}
            __threadfence();
        }
        __syncthreads();
        auto signal = TopologySignal{processingRate, seqNo};
        // Signal our vector, including FLOPs, to others
        for (unsigned int i = threadIdx.x; i < numPeers; i += ARISTOS_BLOCK_SIZE) {
            nvshmem_putmem_signal_nbi(results, results, n * sizeof(floatPair), flags + rank,
                    *CAST_TO(uint64_t, &signal), NVSHMEM_SIGNAL_SET, peers[i]);
        }
    }

    __device__ __forceinline__
    void pluralP2PBuilder(floatPair* __restrict__ const& scratchpad, const unsigned int* __restrict__ peers,
        const unsigned int& n, const unsigned int& rank,
        const unsigned int& processingRate, const unsigned int numPeers, const bool& remotePresent,
        cuda::std::byte* __restrict__ const& sHeap, floatPair* __restrict__ const& results,
        uint64_t* __restrict__ const& flags, uint* __restrict__ const& syncArray, uint* __restrict__ const& rates) {
        const auto seqNo = (__ldg(syncArray) + 1) % 2; // Equivalent to a bit flip
        auto* blockade = syncArray + 1; // barrier
        // If num of other P2P peers == 0, then we adjourn early after conditional subscription
        if (numPeers <= 1)[[unlikely]] {
            if (blockIdx.x == (gridDim.x - 1)) {
                awaitResponses(flags, syncArray, rates, rank, n, seqNo);
            }
            return;
        }
        const unsigned int localBlockIdx = blockIdx.x - remotePresent;
        const unsigned int numP2PBlocks = gridDim.x - remotePresent;

        for (unsigned int i = 1U; i < numPeers; ++i) {
            const auto idx = (i + rank) % numPeers;
            measureTransfer(rank, sHeap, scratchpad, peers[idx],
                threadIdx.x, nvshmemx_putmem_block, idx, localBlockIdx, numP2PBlocks);
        }
        __syncthreads();

        /// All-Reduce to get max transfer time across blocks
        /// Update the global buffer with my values via max reduction
        /// Intra-block slicing
        for (unsigned int i = threadIdx.x; i < numPeers; i += ARISTOS_BLOCK_SIZE) {
            cuda::std::ignore = cuda::atomic_ref<floatPair, cuda::thread_scope_device>{results[peers[i]]}
                .fetch_max(scratchpad[i]);
        }
        __syncthreads();

        // Synchronize across all blocks
        // We do not use a block-wide barrier immediately afterward,
        // because there is already one at the beginning of the succeeding cooperative put API.
        if (!threadIdx.x) {
            __threadfence();
            atomicAdd(blockade, 1);
            while (atomicLoad(blockade) % gridDim.x != 0) {}
            __threadfence();
        }

        // Signal our vector, including FLOPs, to others
        // Inter-block slicing
        // pack payload and signal into single word
        auto signal = TopologySignal{processingRate, seqNo};
        for(unsigned int i = localBlockIdx; i < numPeers; i += numP2PBlocks){
            nvshmemx_putmem_signal_nbi_block(results, results,
                                             (peers[i] != rank) * sizeof(floatPair) * n, flags + rank,
                                             *CAST_TO(uint64_t, &signal),
                                             NVSHMEM_SIGNAL_SET, peers[i]);
        }

        // The last block awaits results
        // Most likely this block will not partake in the above thus, they would do the below in parallel
        // Could potentially enlist more blocks if n > THREADS, but that's unlikely
        if (blockIdx.x == gridDim.x - 1) {
            awaitResponses(flags, syncArray, rates, rank, n, seqNo);
        }
    }

    /// Build Adjacency Matrix
    __global__ void discover(__grid_constant__ const int n, __grid_constant__ const int rank,
        __grid_constant__ const bool remotePresent, __grid_constant__ const unsigned int processingRate,
        cuda::std::byte* __restrict__ sHeap, uint64_t* flags,
        floatPair* __restrict__ results, uint* __restrict__ syncArray, uint* __restrict__ rates) {
        assert(blockDim.x == ARISTOS_BLOCK_SIZE);
        assert(blockDim.y * blockDim.z == 1);
        assert(gridDim.x <= ARISTOS_SUPER_BLOCK_SIZE + remotePresent);
        assert(gridDim.y * gridDim.z == 1);

        // Align to 16 bytes for optimal copy performance.
        // However, empirical results show identical performance (1.024 𝜇s) for 128 threads copying 256 floats,
        // which is a likely practical upper bound for n.
        extern __shared__ __align__(16) floatPair scratchpad[];
        for (uint i = threadIdx.x; i < n; i += ARISTOS_BLOCK_SIZE) {
            scratchpad[i] = floatPair{0, 0};
        }
        if(gridDim.x == 1){
            /// number of blocks is insufficient for remote specialization
            singularBuilder(scratchpad, n, rank, processingRate, sHeap, results, flags, syncArray, rates);
        }
        else{
            // Specialization
            auto nP = 0;
            __shared__ unsigned int numPeers;
            if (!threadIdx.x) {
                auto* __restrict__ peersX = CAST_TO(unsigned int, scratchpad + n);
                /// Block 0 gets remote peers, if present; otherwise, joins the others in getting proximal peers
                for(unsigned int i = 0U; i < n; ++i) {
                    if (const bool b = nvshmem_ptr(results, i) == nullptr;
                        (!b && blockIdx.x > 0) || ((!remotePresent && !b ) || (remotePresent && (b && blockIdx.x == 0)))) {
                        peersX[nP++] = i;
                    }
                }
                numPeers = nP;
            }
            __syncthreads();
            nP = numPeers;
            const auto* __restrict__ peers = CAST_TO(unsigned int, scratchpad + n);

            /// local publisher block 0 will service remote communication requests
            /// while other blocks will further specialize to serve parallel P2P, à la NVLink, transfers
            if(!blockIdx.x && remotePresent){
                /// remote publisher
                pluralRemoteBuilder(scratchpad, peers, n, rank, processingRate, nP, sHeap, results, flags,
                    syncArray);
            }
            else{
                /// P2P publisher only at most one super block
                pluralP2PBuilder(scratchpad, peers, n, rank, processingRate, nP, remotePresent, sHeap,
                    results, flags, syncArray, rates);
            }
        }
    }
}
#endif //TOPO_CUH
