//
// Created by oja7 on 9/19/24.
//

#ifndef TOPO_CUH
#define TOPO_CUH

#define TOPO_LOOP_TRIP 4 // may be too much
#define BETA_BUFFER 64 * 1024
#define ALPHA_BUFFER 16

using Nano = cuda::std::chrono::duration<float, cuda::std::nano>;
using Milli = cuda::std::chrono::duration<float, cuda::std::milli>;
namespace aristos::topology{
    template<typename T=void>
    CUTE_DEVICE
    T* advancePtr(T* buffer, const unsigned int& peer) {
        return buffer + (peer * BETA_BUFFER);
    }


    template<typename ID, typename Put>
    requires cuda::std::is_invocable_r_v<unsigned int, ID>
    && cuda::std::is_invocable_v<Put, void*, const void*, size_t, int>
    CUTE_DEVICE
    void measureTransfer(const unsigned int& rank, float* sHeap, float* durations,
        const unsigned int& peer, const ID& id, const Put& put) {
        Nano transferDuration;
        uint64_t start, end;
        /// Alpha cost: ms
        CUTE_UNROLL
        for (unsigned int j = 0; j < TOPO_LOOP_TRIP; ++j) {
            asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
            put(advancePtr(sHeap, rank), advancePtr(sHeap, rank), ALPHA_BUFFER, peer);
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
            put(advancePtr(sHeap, rank), advancePtr(sHeap, rank), BETA_BUFFER, peer);
            asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
            transferDuration += static_cast<Nano>(end - start);
        }
        if(id() == 0) {
            durations[(peer*2) + 1] = cuda::std::chrono::duration_cast<Milli>(transferDuration / TOPO_LOOP_TRIP).count();
        }
    }

    CUTE_DEVICE
    void singularBuilder(const unsigned int& n, const unsigned int& rank, const unsigned long& processingRate,
        float* sHeap, uint64_t* flags, unsigned long* rates) {
        __shared__ extern size_t scratchpad[];
        __shared__ extern float durations[];
        __shared__ unsigned int baton;
        if (block::threadID() == 0) {
            baton = 0;
        }
        __threadfence_block();

        for (unsigned int i = 0; i < n; ++i) {
            const auto peer = (((i - rank) % n) + n) % n; // (i-rank) mod n
            if (nvshmem_ptr(sHeap, peer) == nullptr) {
                if (atomicCAS_block(&baton, block::warpID(), ((block::warpID() + 1) % blockSize)) == block::warpID()) {
                    // I do the transfer
                    measureTransfer(rank, sHeap, durations, peer, warp::laneID, nvshmemx_putmem_warp);
                }
                continue;
            }
            /// Below is not needed for correctness, but to avoid protracting duration measurements
            __syncthreads();
            /// NVLink-connected peers
            measureTransfer(rank, sHeap, durations, peer, block::threadID, nvshmemx_putmem_block);
        }
        /// Stage my row to the symmetric heap
        const auto slice = cute::ceil_div(n, blockSize);
        for (unsigned int i = block::threadID() * slice; i < slice * (block::threadID() + 1) && i < n; ++i) {
            advancePtr(sHeap, rank)[i] = durations[i];
        }

        // Signal my vector, including FLOPs, to others
        for (unsigned int i = 0; i < n; ++i) {
            const auto peer = (((i - rank) % n) + n) % n; // (i-rank) mod n
            if (nvshmem_ptr(sHeap, peer) == nullptr) {
                if (atomicCAS_block(&baton, block::warpID(), ((block::warpID() + 1) % blockSize)) == block::warpID()) {
                    nvshmemx_putmem_signal_nbi_warp(advancePtr(sHeap, rank),
                    advancePtr(sHeap, rank),
                    n, flags,
                    constructSignal(sent, processingRate), NVSHMEM_SIGNAL_SET, peer);
                }
            }
            else {
                nvshmemx_putmem_signal_nbi_block(advancePtr(sHeap, rank),
                    advancePtr(sHeap, rank),
                    n, flags,
                    constructSignal(sent, processingRate), NVSHMEM_SIGNAL_SET, peer);
            }
        }

        // await responses
        auto expecting = slice;
        auto const offset = slice* block::threadID();
        size_t* indices = scratchpad + (offset * 2);
        auto* status = static_cast<int*>(static_cast<void*>(indices + slice));
        if (offset < n) {
            do {
                __nanosleep(4);
                auto nReceived = nvshmem_uint64_test_some(flags + offset, slice,
                    indices, status, NVSHMEM_CMP_GT, sent);

                for (unsigned int i = 0; i < nReceived; ++i) {
                    status[indices[i]] = 1;
                    rates[indices[i] + offset] = flags[indices[i]] - sent;
                }
                expecting -= nReceived;
            } while (expecting > 0);
        }
    }

    /// Build Adjacency Matrix
    __global__ __forceinline__ ARISTOS_LAUNCH_BOUNDS void discover(CUTE_GRID_CONSTANT const unsigned int n,
        CUTE_GRID_CONSTANT const unsigned int rank, const unsigned long& processingRate,
        float* sHeap, uint64_t* flags) {
        /// Persist relevant data in cache
        associate_access_property(advancePtr(sHeap, rank), cuda::access_property(advancePtr(sHeap, rank),
            BETA_BUFFER, BETA_BUFFER, cuda::access_property::persisting{}));
        associate_access_property(flags, cuda::access_property(flags,
            n, n, cuda::access_property::persisting{}));

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
            }
            else{
                /// P2P publisher
            }
        }
    }
}
#endif //TOPO_CUH
