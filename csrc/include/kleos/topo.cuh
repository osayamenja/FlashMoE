/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Kleos Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 */

//
// Created by oja7 on 9/19/24.
//

#ifndef TOPO_CUH
#define TOPO_CUH

#include <cuda/cmath>
#include "types.cuh"
#include "atomics.cuh"

namespace kleos::topology{
    __device__ __forceinline__
    auto* advancePtr(cuda::std::byte* __restrict__ const& buffer, const unsigned int& slot) {
        return buffer + slot * BETA_BUFFER;
    }

    // Only a single block executes the below
    __device__ __forceinline__
    void awaitResponses(uint64_t* __restrict__ const& flags, WorkerAttribute* __restrict__ const& workerAttributes,
        const uint& rank, const uint& n, const uint& seqNo, const WorkerAttribute& self) {
        using Payload = TopologySignal;
        auto result = Payload{};
        if (!threadIdx.x) {
            workerAttributes[rank] = self;
        }
        for (int i = threadIdx.x; i < n; i += KLEOS_BLOCK_SIZE) {
            if (i != rank) {
                awaitBarrier<cuda::thread_scope_system>(CAST_TO(Payload, flags + i), &result, seqNo);
                workerAttributes[i] = result.wA;
            }
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
        float t0 = 0.0f;
        float t1 = 0.0f;
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
            t0 += static_cast<float>(end - start) / static_cast<float>(TOPO_LOOP_TRIP*NANO_TO_MILLI);
        }
        isResidual = betaBuf % nb != 0 && betaBuf > nb && lBid == nb - 1;
        buf = (lBid * cute::ceil_div(betaBuf, nb) < betaBuf) * (!isResidual * cute::ceil_div(betaBuf, nb)
               + isResidual * (betaBuf - (cute::ceil_div(betaBuf, nb) * (nb - 1))));
        ///Beta Cost: ms/MB
        #pragma unroll
        for (unsigned int j = 0; j < TOPO_LOOP_TRIP; ++j) {
            asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
            put(advancePtr(sHeap, rank) + (lBid * buf), advancePtr(sHeap, rank) + (lBid * buf), buf, peer);
            asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
            t1 += static_cast<float>(end - start) / static_cast<float>(TOPO_LOOP_TRIP*NANO_TO_MILLI);
        }
        if(!id) {
            // Compute beta using slope intercept equation
            const auto beta = ((t1 - t0) / (TO_MB(betaBuf) - TO_MB(alphaBuf)));
            remoteDurations[peerIdx].beta = beta;
            remoteDurations[peerIdx].alpha = t0 - (TO_MB(alphaBuf) * beta);
        }
    }

    /// Expository purposes
    __device__ __forceinline__
    void singularBuilder(floatPair* __restrict__ const& scratchpad, const unsigned int& n,
        const unsigned int& rank,
        cuda::std::byte* __restrict__ const& sHeap, floatPair* __restrict__ const& results,
        uint64_t* __restrict__ const& flags,
        WorkerAttribute* __restrict__ const& workerAttributes, const WorkerAttribute& self, const uint seqNo) {
        for (unsigned int i = 1U; i < n; ++i) {
            const auto peer = (rank + i) % n;
            measureTransfer(rank, sHeap, scratchpad, peer, threadIdx.x, nvshmemx_putmem_block, peer);
        }
        __syncthreads();

        /// Stage my row on the symmetric heap
        for (unsigned int i = threadIdx.x; i < n; i += KLEOS_BLOCK_SIZE) {
            results[i] = scratchpad[i];
        }

        auto signal = TopologySignal{seqNo, self};
        // Signal my vector, including FLOPs, to others
        for (unsigned int i = 1U; i < n; ++i) {
            nvshmemx_putmem_signal_nbi_block(results, results, n * sizeof(floatPair), flags + rank,
                *CAST_TO(uint64_t, &signal), NVSHMEM_SIGNAL_SET, (rank + i) % n);
        }

        // await responses from other GPUs
        awaitResponses(flags, workerAttributes, rank, n, seqNo, self);
    }

    __device__ __forceinline__
    void pluralRemoteBuilder(floatPair* __restrict__ const& scratchpad, const unsigned int* __restrict__& peers,
        const unsigned int& n, const unsigned int& rank,
        const unsigned int numPeers, cuda::std::byte* __restrict__ const& sHeap,
        floatPair* __restrict__ const& results, uint64_t* __restrict__ const& flags,
        cuda::barrier<cuda::thread_scope_device>* const& dvB, const WorkerAttribute& self, const uint seqNo) {
        constexpr auto nW = 4;
        const auto warpId = threadIdx.x / WARP_SIZE;
        const auto laneId = threadIdx.x % WARP_SIZE;
        for (unsigned int i = warpId; i < numPeers; i += nW) {
            const auto idx = (i + rank) % numPeers;
            // use warp to get peak performance when the transport is IBGDA
            measureTransfer(rank, sHeap, scratchpad, peers[idx], laneId, nvshmemx_putmem_warp, idx);
        }
        __syncthreads();

        /// Stage my row to the symmetric heap
        for (unsigned int i = threadIdx.x; i < numPeers; i += KLEOS_BLOCK_SIZE) {
            results[peers[i]] = scratchpad[i];
        }
        __syncthreads();

        // Ensures our vector is complete before sending to neighbors.
        // if num remote peers < n - 1, then we must await the contribution of our p2p siblings
        if (!threadIdx.x && numPeers < n - 1) {
            __threadfence();
            dvB->arrive_and_wait();
        }
        __syncthreads();
        auto signal = TopologySignal{seqNo, self};
        // Signal our vector, including FLOPs, to others
        for (unsigned int i = threadIdx.x; i < numPeers; i += KLEOS_BLOCK_SIZE) {
            nvshmem_putmem_signal_nbi(results, results, n * sizeof(floatPair), flags + rank,
                    *CAST_TO(uint64_t, &signal), NVSHMEM_SIGNAL_SET, peers[i]);
        }
    }

    __device__ __forceinline__
    void pluralP2PBuilder(floatPair* __restrict__ const& scratchpad, const unsigned int* __restrict__ peers,
        const unsigned int& n, const unsigned int& rank, const unsigned int& numPeers, const bool& remotePresent,
        cuda::std::byte* __restrict__ const& sHeap, floatPair* __restrict__ const& results,
        uint64_t* __restrict__ const& flags, cuda::barrier<cuda::thread_scope_device>* const& dvB,
        WorkerAttribute* __restrict__ const& workerAttributes, const WorkerAttribute& self, const uint seqNo) {
        // If num of other P2P peers == 0, then we adjourn early after conditional subscription
        if (numPeers <= 1)[[unlikely]] {
            if (blockIdx.x == gridDim.x - 1) {
                awaitResponses(flags, workerAttributes, rank, n, seqNo, self);
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
        for (unsigned int i = threadIdx.x; i < numPeers; i += KLEOS_BLOCK_SIZE) {
            cuda::std::ignore = cuda::atomic_ref<floatPair, cuda::thread_scope_device>{results[peers[i]]}
                .fetch_max(scratchpad[i]);
        }
        __syncthreads();
        // Synchronize across all blocks
        if (!threadIdx.x) {
            __threadfence();
            dvB->arrive_and_wait();
        }
        __syncthreads();

        // Signal our vector, including FLOPs, to others
        // Inter-block slicing
        // pack payload and signal into single word
        auto signal = TopologySignal{seqNo, self};
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
            awaitResponses(flags, workerAttributes, rank, n, seqNo, self);
        }
    }

    /// Build Adjacency Matrix
    template<unsigned int blocks = KLEOS_STATIC_SBZ> requires(blocks > 0)
    __global__ void discover(__grid_constant__ const int n, __grid_constant__ const int rank,
        __grid_constant__ const bool remotePresent, __grid_constant__ const WorkerAttribute self,
        cuda::std::byte* __restrict__ sHeap, uint64_t* flags,
        floatPair* __restrict__ results, cuda::barrier<cuda::thread_scope_device>* dvB,
        WorkerAttribute* __restrict__ workerAttributes, const __grid_constant__ uint seqNo) {
        assert(blockDim.x == KLEOS_BLOCK_SIZE);
        assert(blockDim.y * blockDim.z == 1);
        assert(gridDim.x <= blocks + remotePresent);
        assert(gridDim.y * gridDim.z == 1);

        // Align to 16 bytes for optimal copy performance.
        // However, empirical results show identical performance (1.024 𝜇s) for 128 threads copying 256 floats,
        // which is a likely practical upper bound for n.
        extern __shared__ __align__(16) floatPair scratchpad[];
        for (uint i = threadIdx.x; i < n; i += KLEOS_BLOCK_SIZE) {
            scratchpad[i] = floatPair{0, 0};
        }
        __syncthreads();
        if(gridDim.x == 1){
            /// number of blocks is insufficient for remote specialization
            singularBuilder(scratchpad, n, rank, sHeap, results, flags, workerAttributes, self, seqNo);
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
                pluralRemoteBuilder(scratchpad, peers, n, rank, nP, sHeap, results, flags,
                    dvB, self, seqNo);
            }
            else{
                /// P2P publisher only at most one super block
                pluralP2PBuilder(scratchpad, peers, n, rank, nP, remotePresent, sHeap,
                    results, flags, dvB, workerAttributes, self, seqNo);
            }
        }
    }
}
#endif //TOPO_CUH
