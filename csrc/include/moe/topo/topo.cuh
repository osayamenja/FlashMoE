//
// Created by oja7 on 9/19/24.
//

#ifndef TOPO_CUH
#define TOPO_CUH

#define TOPO_LOOP_TRIP 4 // this may be too much
#define BETA_BUFFER (64 * 1024)
#define BETA_BUFFER_MB (64.0f / (1024.0f))
#define ALPHA_BUFFER 32
#define ALPHA_BUFFER_MB ((32.0f/ 1024.0f)/1024.0f)
#define NANO_TO_MILLI (cuda::std::nano::den / cuda::std::milli::den)

#include <cuda/std/__algorithm/max.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

using Nano = cuda::std::chrono::duration<float, cuda::std::nano>;
using Milli = cuda::std::chrono::duration<float, cuda::std::milli>;
using ull_t = unsigned long long int;
namespace cg = cooperative_groups;
namespace aristos::topology{
    template<typename T=void>
    CUTE_DEVICE
    T* advancePtr(T* buffer, const unsigned int& peer) {
        return buffer + (peer * BETA_BUFFER);
    }

    template<unsigned int betaBuf = BETA_BUFFER, unsigned int alphaBuf = ALPHA_BUFFER, typename ID, typename Put>
    requires cuda::std::is_invocable_r_v<unsigned int, ID>
    && cuda::std::is_invocable_r_v<void, Put, void*, const void*, size_t, int>
    CUTE_DEVICE
    void measureTransfer(const unsigned int& rank, float* sHeap, float* remoteDurations,
        const unsigned int& peer, const ID& id, const Put& put, unsigned int lBid = 0, unsigned int nb = 1) {
        ull_t start, end, duration = 0;
        bool isResidual = alphaBuf % nb != 0 && lBid == nb - 1;
        unsigned int buf = (lBid * cute::ceil_div(alphaBuf, nb) < alphaBuf) *
                (!isResidual * cute::ceil_div(alphaBuf, nb)
                + isResidual * (alphaBuf - (cute::ceil_div(alphaBuf, nb) * (nb - 1))));
        /// Alpha cost: ms
        CUTE_UNROLL
        for (unsigned int j = 0; j < TOPO_LOOP_TRIP; ++j) {
            asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
            put(advancePtr(sHeap, rank) + (lBid * buf), advancePtr(sHeap, rank) + (lBid * buf), buf, peer);
            asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
            duration += end - start;
            nvshmem_fence();
        }
        if(id() == 0) {
            remoteDurations[peer*2] = static_cast<float>(duration / TOPO_LOOP_TRIP) / NANO_TO_MILLI;
        }
        duration = 0;
        isResidual = betaBuf % nb != 0 && lBid == nb - 1;
        buf = (lBid * cute::ceil_div(betaBuf, nb) < betaBuf) * (!isResidual * cute::ceil_div(betaBuf, nb)
               + isResidual * (betaBuf - (cute::ceil_div(betaBuf, nb) * (nb - 1))));
        ///Beta Cost: ms/MB
        CUTE_UNROLL
        for (unsigned int j = 0; j < TOPO_LOOP_TRIP; ++j) {
            asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
            put(advancePtr(sHeap, rank) + (lBid * buf), advancePtr(sHeap, rank) + (lBid * buf), buf, peer);
            asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
            duration += end - start;
            nvshmem_fence();
        }
        if(id() == 0) {
            // Compute beta using slope intercept equation
            remoteDurations[2*peer + 1] = (((static_cast<float>(duration / TOPO_LOOP_TRIP)/ NANO_TO_MILLI) - remoteDurations[2*peer]) / (BETA_BUFFER_MB - ALPHA_BUFFER_MB));
            // Compute alpha
            remoteDurations[2*peer] = remoteDurations[2*peer] - (ALPHA_BUFFER_MB * remoteDurations[2*peer + 1]);
        }
    }

    CUTE_DEVICE
    void singularBuilder(float* scratchpad, const unsigned int& n, const unsigned int& rank, const unsigned long& processingRate,
        float* sHeap, uint64_t* flags, unsigned long* rates) {
        for (unsigned int i = 0; i < n; ++i) {
            /// We scramble the communicating peer to prevent congestion
            measureTransfer(rank, sHeap, scratchpad, (((i - rank) % n) + n) % n, block::threadID, nvshmemx_putmem_block);
        }
        /// Stage my row on the symmetric heap
        memcpy_async(cg::this_thread_block(), advancePtr(sHeap, rank), scratchpad, sizeof(float)*2*n);
        cooperative_groups::wait_prior<1>(cooperative_groups::this_thread_block());

        // Signal my vector, including FLOPs, to others
        for (unsigned int i = 0U; i < n; ++i) {
            nvshmemx_putmem_signal_nbi_block(advancePtr(sHeap, rank),
                    advancePtr(sHeap, rank),
                    n, flags,
                    constructSignal(processingRate, sent), NVSHMEM_SIGNAL_SET, (((i - rank) % n) + n) % n);
        }

        // await responses from other GPUs
        if (!block::threadID()) {
            __shared__ __align__(16) extern int status[];
            nvshmem_uint64_wait_until_all(flags, n, status, NVSHMEM_CMP_GT, sent);
            for (unsigned int i = 0U; i < n; ++i) {
                rates[i] = flags[i] - sent;
            }
        }
    }

    CUTE_DEVICE
    void pluralRemoteBuilder(float* scratchpad, const unsigned int& n, const unsigned int& rank, const unsigned long& processingRate,
        float* sHeap, uint64_t* flags, unsigned long* rates) {
        __shared__ float* remoteDurations;
        __shared__ unsigned int* remotePeers;
        __shared__ unsigned int numRemotePeers;
        __shared__ unsigned int warpSlice;
        __shared__ int* status;
        if (!block::threadID()) {
            // Avoid overlapping with remoteDurations
            remoteDurations = scratchpad;
            remotePeers = static_cast<unsigned int *>(static_cast<void*>(scratchpad + 2 * n));
            numRemotePeers = 0;
            for(unsigned int i = 0U; i < n; ++i) {
                // Nifty branch-less technique that may be unnecessary
                // Converts the branch if(b) remotePeers[numRemotePeers++] = i; to the below
                // if b == 0
                // >> numRemotePeers += 0
                // >> remotePeers[numRemotePeers] = remotePeers[numRemotePeers] + 0
                // if b == 1
                // >> numRemotePeers += 1
                // >> remotePeers[numRemotePeers] = 0 + i
                const bool b = nvshmem_ptr(sHeap, i) == nullptr;
                numRemotePeers += b;
                remotePeers[numRemotePeers] = !b * remotePeers[numRemotePeers] + i * b;
            }
            warpSlice = cute::ceil_div(numRemotePeers, ARISTOS_BLOCK_SIZE_WARP);
        }
        __threadfence_block();
        for (unsigned int i = 0; i < numRemotePeers; ++i) {
            measureTransfer(rank, sHeap, remoteDurations, remotePeers[i], block::threadID, nvshmemx_putmem_block);
        }

        /// Stage my row to the symmetric heap
        const auto slice = cute::ceil_div(numRemotePeers, ARISTOS_BLOCK_SIZE);
        for (unsigned int i = block::threadID() * slice; i < slice * (block::threadID() + 1) && i < n; ++i) {
            advancePtr(sHeap, rank)[2*remotePeers[i]] = remoteDurations[2 * i];
            advancePtr(sHeap, rank)[2*remotePeers[i] + 1] = remoteDurations[2 * i + 1];
        }
        // Barrier ensures our vector is complete before sending to neighbors.
        if (!block::threadID()) {
            atomicAdd(&publisher::blockade, 1);
            while (atomicLoad(&publisher::blockade) != 2) {
                __nanosleep(2);
            }
            // Reset blockade to ensure functional correctness for subsequent usage in the publisher
            atomicExch(&publisher::blockade, 0U);
        }
        __syncthreads();
        // Signal our vector, including FLOPs, to others
        for (unsigned int i = block::warpID() * warpSlice; i < warpSlice * (block::warpID() + 1); ++i) {
            nvshmemx_putmem_signal_nbi_warp(advancePtr(sHeap, rank),
                    advancePtr(sHeap, rank),
                    n, flags,
                    constructSignal(processingRate, sent), NVSHMEM_SIGNAL_SET, remotePeers[i]);
        }

        // This block will await all responses.
        if (!block::threadID()) {
            status = static_cast<int*>(static_cast<void*>(scratchpad));
            nvshmem_uint64_wait_until_all(flags, n, status, NVSHMEM_CMP_GT, sent);
            for (unsigned int i = 0U; i < n; ++i) {
                rates[i] = flags[i] - sent;
            }
        }
    }

    CUTE_DEVICE
    void pluralP2PBuilder(float* scratchpad, const unsigned int& n, const unsigned int& rank, const unsigned long& processingRate,
                          float* sHeap, uint64_t* flags, const unsigned int& numBlocks) {
        __shared__ float* p2pDurations;
        __shared__ unsigned int* proximalPeers;
        __shared__ unsigned int numProximalPeers;
        __shared__ unsigned int localBlockId;

        if (!block::threadID()) {
            p2pDurations = scratchpad;
            proximalPeers = static_cast<unsigned int *>(static_cast<void*>(p2pDurations + 2*n));
            numProximalPeers = 0;
            for(unsigned int i = 0; i < n; ++i) {
                const bool b = nvshmem_ptr(sHeap, i) != nullptr;
                numProximalPeers += b;
                proximalPeers[numProximalPeers] = !b * proximalPeers[numProximalPeers] + i * b;
            }
            localBlockId = PublisherConfig::getLocalBlockID(moeConfig.numP2PPublisherBlocks);
        }
        __threadfence_block();
        for (unsigned int i = 0U; i < numProximalPeers; ++i) {
            measureTransfer<BETA_BUFFER>(rank, sHeap, p2pDurations,
                proximalPeers[i], block::threadID, nvshmemx_putmem_block, localBlockId, numBlocks);
        }
        /// All-Reduce to get max transfer time across blocks
        /// Reusing concepts from publisher
        // Wait our turn
        if (!block::threadID()) {
            publisher::awaitBaton(localBlockId);
        }
        __syncthreads();
        /// Update the global buffer with my values via max reduction
        auto slice = cute::ceil_div(numProximalPeers, ARISTOS_BLOCK_SIZE);
        if (block::threadID() * slice < numProximalPeers) {
            for (unsigned int i = block::threadID() * slice; i < slice * (block::threadID() + 1); ++i) {
                advancePtr(sHeap, rank)[2*proximalPeers[i]] = cuda::std::max(p2pDurations[2*i], advancePtr(sHeap, rank)[2*proximalPeers[i]]);
                advancePtr(sHeap, rank)[2*proximalPeers[i] + 1] = cuda::std::max(p2pDurations[2*i + 1], advancePtr(sHeap, rank)[2*proximalPeers[i] + 1]);
            }
        }
        if (!block::threadID()) {
            // pass the baton
            atomicExch(&publisher::baton, (localBlockId + 1) % ARISTOS_SUPER_BLOCK_SIZE);
            // Await baton to circle back, at this point the reduction is complete, and we are free to go
            publisher::awaitBaton(localBlockId);
            /// Critical that the initiating process has id BATON_LEADER for the below to be correct
            if(localBlockId == BATON_LEADER){
                // unblock remote publisher
                atomicAdd(&publisher::blockade, 1);
                while (atomicLoad(&publisher::blockade) != 2) {
                    __nanosleep(2);
                }
            }
            // Unblock sibling blocks
            atomicExch(&publisher::baton, (localBlockId + 1) % ARISTOS_SUPER_BLOCK_SIZE);
        }
        // Ensures all threads are done
        __syncthreads();
        // Signal our vector, including FLOPs, to others
        slice = cute::ceil_div(numProximalPeers, ARISTOS_SUPER_BLOCK_SIZE);
        if(slice * localBlockId < numProximalPeers){
            for(unsigned int i = slice * localBlockId; i < slice * (localBlockId + 1); ++i){
                nvshmemx_putmem_signal_nbi_block(advancePtr(sHeap, rank),
                                                 advancePtr(sHeap, rank),
                                                 n, flags,
                                                 constructSignal(processingRate, sent),
                                                 NVSHMEM_SIGNAL_SET, proximalPeers[i]);
            }
        }
    }

    /// Build Adjacency Matrix
    __global__ void discover(CUTE_GRID_CONSTANT const unsigned int n, CUTE_GRID_CONSTANT const unsigned int rank,
        CUTE_GRID_CONSTANT const unsigned int nb, const unsigned long processingRate, float* sHeap, uint64_t* flags) {
        /// Persist relevant data in cache
        /// Parition the below among threads
        /*associate_access_property(advancePtr(sHeap, rank), cuda::access_property(advancePtr(sHeap, rank),
            BETA_BUFFER, BETA_BUFFER, cuda::access_property::persisting{}));*/
        associate_access_property(flags, cuda::access_property(flags,
            n*sizeof(uint64_t), n*sizeof(uint64_t), cuda::access_property::persisting{}));
        associate_access_property(&publisher::blockade, cuda::access_property::persisting{});

        // Align to 16 bytes for optimal copy performance.
        // However, empirical results show identical performance (1.024 𝜇s) for 128 threads copying 256 floats,
        // which is a likely practical upper bound for 2*n.
        __shared__ __align__(16) extern float scratchpad[];

        if(nb == 1){
            /// number of threads is insufficient for remote specialization
            singularBuilder(scratchpad, n, rank, processingRate, sHeap, flags, flags + n);
        }
        else{
            /// local publisher block 0 will service remote communication requests
            /// while other blocks will further specialize to serve parallel P2P, à la NVLink, transfers
            if(!PublisherConfig::getLocalBlockID(nb)){
                /// remote publisher
                pluralRemoteBuilder(scratchpad, n, rank, processingRate, sHeap, flags, flags + n);
            }
            else{
                /// P2P publisher only at most one super block
                pluralP2PBuilder(scratchpad, n, rank, processingRate, sHeap, flags, nb);
            }
        }
    }
}
#endif //TOPO_CUH
