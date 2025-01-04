//
// Created by oja7 on 9/19/24.
//

#ifndef TOPO_CUH
#define TOPO_CUH

#include <cuda/cmath>
#include "types.cuh"
#include "util/atomics.cuh"

namespace aristos::topology{
    __device__ __inline__ unsigned int blockade = 0U;
    template<typename T>
    requires(!cuda::std::is_same_v<T, void>)
    __device__ __forceinline__
    T* advancePtr(T* buffer, const unsigned int& slot) {
        return buffer + (slot * BETA_BUFFER);
    }

    template<size_t betaBuf = BETA_BUFFER, size_t alphaBuf = ALPHA_BUFFER, typename Put>
    requires (cuda::std::is_invocable_r_v<void, Put, void*, const void*, size_t, int>)
    __device__ __forceinline__
    void measureTransfer(const unsigned int& rank, cuda::std::byte* sHeap, floatPair* remoteDurations,
        const int& peer, const unsigned int& id, const Put& put, const unsigned int& peerIdx,
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
    void singularBuilder(floatPair* scratchpad, const unsigned int& n, const unsigned int& rank, const unsigned long& processingRate,
        cuda::std::byte* sHeap, floatPair* results, uint64_t* flags) {
        for (unsigned int i = 1U; i < n; ++i) {
            measureTransfer(rank, sHeap, scratchpad, (rank + i) % n,
                threadIdx.x, nvshmemx_putmem_block, (rank + i) % n);
        }
        __syncthreads();

        /// Stage my row on the symmetric heap
        for (unsigned int i = threadIdx.x; i < n; i += ARISTOS_BLOCK_SIZE) {
            results[i] = scratchpad[i];
        }

        // Signal my vector, including FLOPs, to others
        for (unsigned int i = 1U; i < n; ++i) {
            nvshmemx_putmem_signal_nbi_block(results, results, n*sizeof(floatPair), flags + rank,
                constructSignal(sent, processingRate), NVSHMEM_SIGNAL_SET, (rank + i) % n);
        }

        // await responses from other GPUs
        for (int i = threadIdx.x + 1; i < n; i += ARISTOS_BLOCK_SIZE) {
            nvshmem_signal_wait_until(flags + (rank + i) % n, NVSHMEM_CMP_GT, sent + seqNo);
            flags[(rank + i) % n] -= sent + seqNo;
        }
    }

    __device__ __forceinline__
    void pluralRemoteBuilder(floatPair* scratchpad, const unsigned int* peers, const unsigned int& n, const unsigned int& rank,
        const unsigned long& processingRate, const unsigned int numPeers, cuda::std::byte* sHeap, floatPair* results, uint64_t* flags) {
        for (unsigned int i = 0U; i < numPeers; ++i) {
            measureTransfer(rank, sHeap, scratchpad, peers[(i + rank) % numPeers],
                threadIdx.x, nvshmemx_putmem_block, (i + rank) % numPeers);
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
            atomicAdd(&blockade, 1);
            while (atomicLoad(&blockade) % gridDim.x != 0) {}
            __threadfence();
        }
        __syncthreads();

        // Signal our vector, including FLOPs, to others
        for (unsigned int i = threadIdx.x; i < numPeers; i += ARISTOS_BLOCK_SIZE) {
            nvshmem_putmem_signal_nbi(results, results, n * sizeof(floatPair), flags + rank,
                    constructSignal(sent, processingRate), NVSHMEM_SIGNAL_SET, peers[i]);
        }
    }

    __device__ __forceinline__
    void pluralP2PBuilder(floatPair* scratchpad,const unsigned int* peers, const unsigned int& n, const unsigned int& rank,
        const unsigned long& processingRate, const unsigned int numPeers, const bool& remotePresent,
        cuda::std::byte* sHeap, floatPair* results, uint64_t* flags) {
        // If num of other P2P peers == 0, then we adjourn early after conditional subscription
        if (numPeers <= 1)[[unlikely]] {
            if (blockIdx.x == (gridDim.x - 1)) {
                for (int i = threadIdx.x + 1; i < n; i += ARISTOS_BLOCK_SIZE) {
                    nvshmem_signal_wait_until(flags + (rank + i) % n, NVSHMEM_CMP_GT, sent + seqNo);
                    flags[(rank + i) % n] -= sent + seqNo;
                }
            }
            return;
        }
        const unsigned int localBlockIdx = blockIdx.x - remotePresent;
        const unsigned int numP2PBlocks = gridDim.x - remotePresent;

        for (unsigned int i = 1U; i < numPeers; ++i) {
            measureTransfer(rank, sHeap, scratchpad, peers[(i + rank) % numPeers],
                threadIdx.x, nvshmemx_putmem_block,
                (i + rank) % numPeers, localBlockIdx, numP2PBlocks);
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
            atomicAdd(&blockade, 1);
            while (atomicLoad(&blockade) % gridDim.x != 0) {}
            __threadfence();
        }

        // Signal our vector, including FLOPs, to others
        // Inter-block slicing
        for(unsigned int i = localBlockIdx; i < numPeers; i += numP2PBlocks){
            nvshmemx_putmem_signal_nbi_block(results, results,
                                             (peers[i] != rank) * sizeof(floatPair) * n, flags + rank,
                                             constructSignal(sent, processingRate),
                                             NVSHMEM_SIGNAL_SET, peers[i]);
        }

        // The last block awaits results
        // Most likely this block will not partake in the above thus, they would do the below in parallel
        // Could potentially enlist more blocks if n > THREADS, but that's unlikely
        if (blockIdx.x == gridDim.x - 1) {
            for (int i = threadIdx.x + 1; i < n; i += ARISTOS_BLOCK_SIZE) {
                nvshmem_signal_wait_until(flags + (rank + i) % n, NVSHMEM_CMP_GT, sent + seqNo);
                flags[(rank + i) % n] -= sent + seqNo;
            }
        }
    }

    /// Build Adjacency Matrix
    __global__ void discover(CUTE_GRID_CONSTANT const int n, CUTE_GRID_CONSTANT const int rank,
        CUTE_GRID_CONSTANT const bool remotePresent, const unsigned long processingRate,
        cuda::std::byte* sHeap, uint64_t* flags, floatPair* results) {
        assert(blockDim.x == ARISTOS_BLOCK_SIZE);
        assert(blockDim.y * blockDim.z == 1);
        assert(gridDim.x <= ARISTOS_SUPER_BLOCK_SIZE + remotePresent);
        assert(gridDim.y * gridDim.z == 1);

        // Align to 16 bytes for optimal copy performance.
        // However, empirical results show identical performance (1.024 𝜇s) for 128 threads copying 256 floats,
        // which is a likely practical upper bound for n.
        extern __shared__ __align__(16) floatPair scratchpad[];

        if(gridDim.x == 1){
            /// number of blocks is insufficient for remote specialization
            singularBuilder(scratchpad, n, rank, processingRate, sHeap, results, flags);
        }
        else{
            // Specialization
            __shared__ unsigned int numPeers;
            __shared__ unsigned int* peers;
            if (!threadIdx.x) {
                numPeers = 0;
                peers = static_cast<unsigned int*>(static_cast<void*>(scratchpad + n));
                /// Block 0 gets remote peers, if present; otherwise, joins the others in getting proximal peers
                for(unsigned int i = 0U; i < n; ++i) {
                    if (const bool b = nvshmem_ptr(results, i) == nullptr;
                        (!b && blockIdx.x > 0) || ((!remotePresent && !b ) || (remotePresent && (b && blockIdx.x == 0)))) {
                        peers[numPeers++] = i;
                    }
                }
            }
            __syncthreads();
            /// local publisher block 0 will service remote communication requests
            /// while other blocks will further specialize to serve parallel P2P, à la NVLink, transfers
            if(!blockIdx.x && remotePresent){
                /// remote publisher
                pluralRemoteBuilder(scratchpad, peers, n, rank, processingRate, numPeers, sHeap, results, flags);
            }
            else{
                /// P2P publisher only at most one super block
                pluralP2PBuilder(scratchpad, peers, n, rank, processingRate, numPeers, remotePresent, sHeap, results, flags);
            }
        }
    }
}
#endif //TOPO_CUH
