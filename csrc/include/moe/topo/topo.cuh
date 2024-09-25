//
// Created by oja7 on 9/19/24.
//

#ifndef TOPO_CUH
#define TOPO_CUH

#define TOPO_LOOP_TRIP 4 // may be too much
#define BETA_BUFFER 64 * 1024
#define BETA_BUFFER_MB BETA_BUFFER / (1024*1024)
#define ALPHA_BUFFER 16
#include <cuda/std/__algorithm/max.h>

using Nano = cuda::std::chrono::duration<double, cuda::std::nano>;
using Milli = cuda::std::chrono::duration<double, cuda::std::milli>;
namespace aristos::topology{
    __device__ __inline__ unsigned int barrier = 0;
    template<typename T=void>
    CUTE_DEVICE
    T* advancePtr(T* buffer, const unsigned int& peer) {
        return buffer + (peer * BETA_BUFFER);
    }


    template<unsigned int betaBuf = BETA_BUFFER, unsigned int alphaBuf = ALPHA_BUFFER, typename ID, typename Put>
    requires cuda::std::is_invocable_r_v<unsigned int, ID>
    && cuda::std::is_invocable_r_v<void, Put, void*, const void*, size_t, int>
    CUTE_DEVICE
    void measureTransfer(const unsigned int& rank, double* sHeap, double* durations,
        const unsigned int& peer, const ID& id, const Put& put) {
        Nano transferDuration;
        uint64_t start, end;
        /// Alpha cost: ms
        CUTE_UNROLL
        for (unsigned int j = 0; j < TOPO_LOOP_TRIP; ++j) {
            asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
            put(advancePtr(sHeap, rank), advancePtr(sHeap, rank), alphaBuf, peer);
            asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
            transferDuration += static_cast<Nano>(end - start);
        }
        if(id() == 0) {
            durations[peer*2] = cuda::std::chrono::duration_cast<Milli>(transferDuration / TOPO_LOOP_TRIP).count();
        }
        ///Beta Cost: ms/MB
        CUTE_UNROLL
        for (unsigned int j = 0; j < TOPO_LOOP_TRIP; ++j) {
            asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
            put(advancePtr(sHeap, rank), advancePtr(sHeap, rank), betaBuf, peer);
            asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
            transferDuration += static_cast<Nano>(end - start);
        }
        if(id() == 0) {
            durations[2*peer + 1] = cuda::std::chrono::duration_cast<Milli>((transferDuration / TOPO_LOOP_TRIP)/(betaBuf/(1024*1024))).count();
        }
    }

    CUTE_DEVICE
    void singularBuilder(const unsigned int& n, const unsigned int& rank, const unsigned long& processingRate,
        double* sHeap, uint64_t* flags, unsigned long* rates) {
        __shared__ extern size_t scratchpad[];
        __shared__ extern double durations[];
        __shared__ unsigned short int* baton;
        unsigned short int shouldTransfer = 0U;
        if (block::threadID() == 0) {
            baton = static_cast<unsigned short int *>(static_cast<void*>(scratchpad + 2*n));
            for(unsigned int i = 0; i < n; ++i) {
                baton[i] = 1U;
            }
        }
        __threadfence_block();
        __syncthreads();

        for (unsigned int i = 0; i < n; ++i) {
            const auto peer = (((i - rank) % n) + n) % n; // (i-rank) mod n
            if (nvshmem_ptr(sHeap, peer) == nullptr) {
                if (!warp::laneID()) {
                    shouldTransfer = atomicCAS(baton + i, 1U, 2U);
                }
                if (__shfl_sync(0xffffffff, shouldTransfer, 0) == 1U) {
                    // We do the transfer
                    measureTransfer(rank, sHeap, durations, peer, warp::laneID, nvshmemx_putmem_warp);
                }
                continue;
            }
            /// Below is not needed for correctness, but to avoid protracting duration measurements
            __syncthreads();
            /// NVLink-connected peers
            measureTransfer(rank, sHeap, durations, peer, block::threadID, nvshmemx_putmem_block);
        }
        /// Stage my row on the symmetric heap
        auto slice = cute::ceil_div(2*n, ARISTOS_BLOCK_SIZE);
        for (unsigned int i = block::threadID() * slice; i < slice * (block::threadID() + 1) && i < n; ++i) {
            advancePtr(sHeap, rank)[i] = durations[i];
        }

        // Signal my vector, including FLOPs, to others
        for (unsigned int i = 0; i < n; ++i) {
            if (const auto peer = (((i - rank) % n) + n) % n; // (i-rank) mod n
                nvshmem_ptr(sHeap, peer) == nullptr) {
                if (!warp::laneID()) {
                    shouldTransfer = atomicCAS(baton + i, 2U, 3U);
                }
                if (__shfl_sync(0xffffffff, shouldTransfer, 0) == 2U) {
                    nvshmemx_putmem_signal_nbi_warp(advancePtr(sHeap, rank),
                    advancePtr(sHeap, rank),
                    n, flags,
                    constructSignal(processingRate, sent), NVSHMEM_SIGNAL_SET, peer);
                }
            }
            else {
                nvshmemx_putmem_signal_nbi_block(advancePtr(sHeap, rank),
                    advancePtr(sHeap, rank),
                    n, flags,
                    constructSignal(processingRate, sent), NVSHMEM_SIGNAL_SET, peer);
            }
        }

        // await responses
        slice = cute::ceil_div(n, ARISTOS_BLOCK_SIZE);
        auto expecting = slice;
        auto const offset = slice* block::threadID();
        size_t* indices = scratchpad + (offset * 2);
        auto* status = static_cast<int*>(static_cast<void*>(indices + slice));
        if (offset < n) {
            do {
                __nanosleep(4);
                const auto nReceived = nvshmem_uint64_test_some(flags + offset, slice,
                    indices, status, NVSHMEM_CMP_GT, sent);

                for (unsigned int i = 0; i < nReceived; ++i) {
                    status[indices[i]] = 1;
                    rates[indices[i] + offset] = flags[indices[i]] - sent;
                }
                expecting -= nReceived;
            } while (expecting > 0);
        }
    }

    CUTE_DEVICE
    void pluralRemoteBuilder(const unsigned int& n, const unsigned int& rank, const unsigned long& processingRate,
        double* sHeap, uint64_t* flags, unsigned long* rates) {
        __shared__ extern size_t scratchpad[];
        __shared__ extern double durations[];
        __shared__ unsigned int* remotePeers;
        __shared__ unsigned int numRemotePeers;
        __shared__ unsigned short int* baton;
        unsigned short int shouldTransfer = 0U;
        if (!block::threadID()) {
            remotePeers = static_cast<unsigned int *>(static_cast<void*>(scratchpad + 2*n));
            numRemotePeers = 0;
            for(unsigned int i = 0; i < n; ++i) {
                if (nvshmem_ptr(sHeap, i) == nullptr) {
                    remotePeers[numRemotePeers++] = i;
                }
            }
            baton = static_cast<unsigned short int *>(static_cast<void*>((scratchpad + 2*n) + numRemotePeers));
            for(unsigned int i = 0; i < n; ++i) {
                baton[i] = 1U;
            }
        }
        __threadfence_block();
        __syncthreads();
        for (unsigned int i = 0; i < numRemotePeers; ++i) {
            if (!warp::laneID()) {
                shouldTransfer = atomicCAS(baton + i, 1U, 2U);
            }
            if (__shfl_sync(0xffffffff, shouldTransfer, 0) == 1U) {
                measureTransfer(rank, sHeap, durations, remotePeers[i], warp::laneID, nvshmemx_putmem_warp);
            }
        }

        /// Stage my row to the symmetric heap
        auto slice = cute::ceil_div(numRemotePeers, ARISTOS_BLOCK_SIZE);
        for (unsigned int i = block::threadID() * slice; i < slice * (block::threadID() + 1) && i < n; ++i) {
            advancePtr(sHeap, rank)[2*remotePeers[i]] = durations[2*i];
            advancePtr(sHeap, rank)[2*remotePeers[i] + 1] = durations[2*i + 1];
        }
        // Barrier ensures our vector is complete prior to sending to neighbors.
        if (!block::threadID()) {
            atomicAdd(&barrier, 1);
            while (atomicLoad(&barrier) != 2) {
                __nanosleep(2);
            }
        }
        __syncthreads();
        // Signal our vector, including FLOPs, to others
        for (unsigned int i = 0; i < numRemotePeers; ++i) {
            if (!warp::laneID()) {
                shouldTransfer = atomicCAS(baton + i, 2U, 3U);
            }
            if (__shfl_sync(0xffffffff, shouldTransfer, 0) == 2U) {
                nvshmemx_putmem_signal_nbi_warp(advancePtr(sHeap, rank),
                    advancePtr(sHeap, rank),
                    n, flags,
                    constructSignal(processingRate, sent), NVSHMEM_SIGNAL_SET, remotePeers[i]);
            }
        }

        // This block will await responses.
        slice = cute::ceil_div(numRemotePeers, ARISTOS_BLOCK_SIZE);
        auto expecting = slice;
        auto const offset = slice * block::threadID();
        size_t* indices = scratchpad + (offset * 2);
        auto* status = static_cast<int*>(static_cast<void*>(indices + slice));
        if (offset < n) {
            do {
                __nanosleep(4);
                const auto nReceived = nvshmem_uint64_test_some(flags + offset, slice,
                    indices, status, NVSHMEM_CMP_GT, sent);

                for (unsigned int i = 0; i < nReceived; ++i) {
                    status[indices[i]] = 1;
                    rates[indices[i] + offset] = flags[indices[i]] - sent;
                }
                expecting -= nReceived;
            } while (expecting > 0);
        }
    }

    CUTE_DEVICE
    void pluralP2PBuilder(const unsigned int& n, const unsigned int& rank, const unsigned long& processingRate, double* sHeap, uint64_t* flags) {
        __shared__ extern size_t scratchpad[];
        __shared__ extern double durations[];
        __shared__ unsigned int* peers;
        __shared__ unsigned int numPeers;
        __shared__ PublisherConfig publisherConfig;
        __shared__ unsigned int stages;

        if (!block::threadID()) {
            peers = static_cast<unsigned int *>(static_cast<void*>(scratchpad + 2*n));
            numPeers = 0;
            for(unsigned int i = 0; i < n; ++i) {
                if (nvshmem_ptr(sHeap, i) != nullptr) {
                    peers[numPeers++] = i;
                }
            }
            publisherConfig = PublisherConfig(moeConfig);
        }
        __threadfence_block();
        for (unsigned int i = 0; i < numPeers; ++i) {
            measureTransfer<BETA_BUFFER/superBlockSize, superBlockSize>(rank, sHeap, durations,
                peers[i], block::threadID, nvshmemx_putmem_block);
        }
        /// All-Reduce to get max transfer time across blocks
        /// Reusing concepts from publisher namespace
        // Wait our turn
        if (!block::threadID()) {
            publisher::awaitBaton(publisherConfig.localBlockID);
        }
        __syncthreads();
        /// Update the global buffer with my values via max reduction
        const auto slice = cute::ceil_div(numPeers, ARISTOS_BLOCK_SIZE);
        for (unsigned int i = block::threadID() * slice; i < slice * (block::threadID() + 1) && i < n; ++i) {
            advancePtr(sHeap, rank)[2*peers[i]] = cuda::std::max(durations[2*i], advancePtr(sHeap, rank)[2*peers[i]]);
            advancePtr(sHeap, rank)[2*peers[i] + 1] = cuda::std::max(durations[2*i + 1], advancePtr(sHeap, rank)[2*peers[i] + 1]);
        }
        // Ensures all threads are done
        __syncthreads();
        if (!block::threadID()) {
            // pass baton
            atomicExch(&publisher::baton, (publisherConfig.localBlockID + 1) % superBlockSize);
            // Release or await remote builder
            atomicAdd(&barrier, 1);
            while (atomicLoad(&barrier) != 2) {
                __nanosleep(2);
            }
        }
        __syncthreads();
        if (!publisherConfig.localBlockID) {
            // Signal our vector, including FLOPs, to others
            for (unsigned int i = 0; i < numPeers; ++i) {
                nvshmemx_putmem_signal_nbi_block(advancePtr(sHeap, rank),
                        advancePtr(sHeap, rank),
                        n, flags,
                        constructSignal(processingRate, sent), NVSHMEM_SIGNAL_SET, peers[i]);
            }
        }
    }

    /// Build Adjacency Matrix
    __global__ __forceinline__ void discover(CUTE_GRID_CONSTANT const unsigned int n,
        CUTE_GRID_CONSTANT const unsigned int rank, const unsigned long& processingRate,
        double* sHeap, uint64_t* flags) {
        /// Persist relevant data in cache
        associate_access_property(advancePtr(sHeap, rank), cuda::access_property(advancePtr(sHeap, rank),
            BETA_BUFFER, BETA_BUFFER, cuda::access_property::persisting{}));
        associate_access_property(flags, cuda::access_property(flags,
            n, n, cuda::access_property::persisting{}));
        cuda::associate_access_property(&barrier, cuda::access_property::persisting{});

        if(moeConfig.numPublisherBlocks < ((NVLinkThreshold / blockSize) + 1)){
            /// number of threads is insufficient for remote specialization
            //one block does all
            singularBuilder(n, rank, processingRate, sHeap, flags, flags + n);
        }
        else{
            /// local publisher block 0 will service remote communication requests
            /// while other blocks will further specialize to serve parallel P2P, à la NVLink, transfers
            if(!PublisherConfig::getLocalBlockID(moeConfig.numPublisherBlocks)){
                /// remote publisher
                pluralRemoteBuilder(n, rank, processingRate, sHeap, flags, flags + n);
            }
            else{
                /// P2P publisher
                pluralP2PBuilder(n, rank, processingRate, sHeap, flags);
            }
        }
    }
}
#endif //TOPO_CUH
