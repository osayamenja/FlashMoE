//
// Created by oja7 on 9/19/24.
//

#ifndef TOPO_CUH
#define TOPO_CUH

#define TOPO_LOOP_TRIP 4U // this may be too much
#define BETA_BUFFER (64UL * 1024UL)
#define BETA_BUFFER_MB (64.0f / (1024.0f))
#define ALPHA_BUFFER 1024UL
#define ALPHA_BUFFER_MB (1/1024.0f)
#define NANO_TO_MILLI (cuda::std::nano::den / cuda::std::milli::den)
#define NANO_TO_MICRO (cuda::std::nano::den / cuda::std::micro::den)
#define BYTE_MAX cuda::std::numeric_limits<cuda::std::underlying_type_t<cuda::std::byte>>::max()

#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <math_functions.h>

using Nano = cuda::std::chrono::duration<float, cuda::std::nano>;
using Milli = cuda::std::chrono::duration<float, cuda::std::milli>;
using ull_t = unsigned long long int;
using BlockReduce = cub::BlockReduce<double, ARISTOS_BLOCK_SIZE>;
namespace cg = cooperative_groups;
namespace aristos::topology{
    template<typename T=void>
    CUTE_DEVICE
    T* advancePtr(T* buffer, const unsigned int& slot) {
        return buffer + (slot * BETA_BUFFER);
    }

    /// We scramble the communicating peer to prevent congestion
    CUTE_DEVICE
    decltype(auto) getPeer(const int& currentPeer, const int& rank, const int& n) {
        return (((currentPeer - rank) % n) + n) % n;
    }

    /// Fused memcpy and memset.
    /// Note, M cannot be any of function types, incomplete types, or bit-field.
    template<cuda::std::byte setValue=cuda::std::byte{0}, typename M>
    requires (!cuda::std::is_function_v<M> && cuda::std::to_integer<unsigned int>(setValue) <= BYTE_MAX)
    CUTE_DEVICE
    void memCopySet(M* dst, M* src, const unsigned int& beginIdx, const unsigned int& endIdx) {
        for (unsigned int i = beginIdx; i < endIdx; ++i) {
            dst[i] = src[i];
            CUTE_UNROLL
            for (unsigned int j = 0; j < sizeof(M); ++j) {
                static_cast<cuda::std::byte*>(static_cast<void*>(src))[j + i*sizeof(M)] = setValue;
            }
        }
    }

    template<size_t betaBuf = BETA_BUFFER, size_t alphaBuf = ALPHA_BUFFER, typename ID, typename Put>
    requires (cuda::std::is_invocable_r_v<unsigned int, ID>
        && cuda::std::is_invocable_r_v<void, Put, void*, const void*, size_t, int>)
    CUTE_DEVICE
    void measureTransfer(const int& rank, double* sHeap, double* remoteDurations,
        const int& peer, const ID& id, const Put& put, unsigned int lBid = 0, unsigned int nb = 1) {
        ull_t start, end;
        double duration = 0.0;
        bool isResidual = alphaBuf % nb != 0 && lBid == nb - 1;
        size_t buf = (lBid * cute::ceil_div(alphaBuf, nb) < alphaBuf) *
                (!isResidual * cute::ceil_div(alphaBuf, nb)
                + isResidual * (alphaBuf - (cute::ceil_div(alphaBuf, nb) * (nb - 1))));
        /// Alpha cost: ms
        CUTE_UNROLL
        for (unsigned int j = 0; j < TOPO_LOOP_TRIP; ++j) {
            asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
            put(advancePtr(sHeap, rank) + (lBid * buf), advancePtr(sHeap, rank) + (lBid * buf), (rank != peer) * buf, peer);
            asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
            duration += (end - start) / static_cast<double>(TOPO_LOOP_TRIP*NANO_TO_MILLI);
        }
        if(id() == 0) {
            remoteDurations[peer*2] = duration;
        }
        duration = 0.0;
        isResidual = betaBuf % nb != 0 && lBid == nb - 1;
        buf = (lBid * cute::ceil_div(betaBuf, nb) < betaBuf) * (!isResidual * cute::ceil_div(betaBuf, nb)
               + isResidual * (betaBuf - (cute::ceil_div(betaBuf, nb) * (nb - 1))));
        ///Beta Cost: ms/MB
        CUTE_UNROLL
        for (unsigned int j = 0; j < TOPO_LOOP_TRIP; ++j) {
            asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
            put(advancePtr(sHeap, rank) + (lBid * buf), advancePtr(sHeap, rank) + (lBid * buf), (rank != peer) * buf, peer);
            asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
            duration += (end - start)/ static_cast<double>(TOPO_LOOP_TRIP*NANO_TO_MILLI);
        }
        if(id() == 0) {
            // Compute beta using slope intercept equation
            remoteDurations[2*peer + 1] = ((duration - remoteDurations[2*peer]) / (BETA_BUFFER_MB - ALPHA_BUFFER_MB));
            // Compute alpha and apply abs for spurious negatives
            remoteDurations[2*peer] = fabs(remoteDurations[2*peer] - (ALPHA_BUFFER_MB * remoteDurations[2*peer + 1]));
        }
    }

    CUTE_DEVICE
    void singularBuilder(void* scratchpad, const int& n, const int& rank, const unsigned long& processingRate,
        double* sHeap, double* results, uint64_t* flags) {
        for (int i = 0; i < n; ++i) {
            measureTransfer(rank, sHeap, static_cast<double*>(scratchpad), getPeer(i, rank, n),
                block::threadID, nvshmemx_putmem_block);
        }
        __threadfence_block();
        __syncthreads();

        /// Stage my row on the symmetric heap
        const auto slice = cute::ceil_div(2*n, blockDim.x);
        memCopySet(results + (2*rank*n),static_cast<double*>(scratchpad),block::threadID() * slice,
            min(slice * (block::threadID() + 1), 2U*n));
        __threadfence_block();

        // Signal my vector, including FLOPs, to others
        for (int i = 0; i < n; ++i) {
            nvshmemx_double_put_signal_nbi_block(results + (2*rank*n),
                    results + (2*rank*n),
                    (getPeer(i, rank, n) != rank) * 2*n, flags + rank,
                    constructSignal(processingRate, sent), NVSHMEM_SIGNAL_SET, getPeer(i, rank, n));
        }

        // await responses from other GPUs
        if (block::threadID() == 0) {
            nvshmem_uint64_wait_until_all(flags, n, static_cast<int*>(scratchpad), NVSHMEM_CMP_GT, sent);
            for (unsigned int i = 0U; i < n; ++i) {
                flags[i] -= sent;
            }
        }
    }

    CUTE_DEVICE
    void pluralRemoteBuilder(void* scratchpad, const unsigned int& n, const unsigned int& rank, const unsigned long& processingRate,
        double* sHeap, uint64_t* flags) {
        __shared__ double* remoteDurations;
        __shared__ unsigned int* remotePeers;
        __shared__ unsigned int numRemotePeers;
        __shared__ unsigned int warpSlice;
        __shared__ int* status;
        if (!block::threadID()) {
            // Avoid overlapping with remoteDurations
            remoteDurations = static_cast<double*>(scratchpad);
            remotePeers = static_cast<unsigned int *>(static_cast<void*>(remoteDurations + 2 * n));
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
            (sHeap + (2*rank*n))[2*remotePeers[i]] = remoteDurations[2 * i];
            (sHeap + (2*rank*n))[2*remotePeers[i] + 1] = remoteDurations[2 * i + 1];
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
            nvshmemx_double_put_signal_nbi_warp(sHeap + (2*rank*n),
                    sHeap + (2*rank*n),
                    (i != rank) * 2*n, flags + rank,
                    constructSignal(processingRate, sent), NVSHMEM_SIGNAL_SET, remotePeers[i]);
        }

        // This block will await all responses.
        if (!block::threadID()) {
            status = static_cast<int*>(static_cast<void*>(scratchpad));
            nvshmem_uint64_wait_until_all(flags, n, status, NVSHMEM_CMP_GT, sent);
            for (unsigned int i = 0U; i < n; ++i) {
                flags[i] = flags[i] - sent;
            }
        }
    }

    CUTE_DEVICE
    void pluralP2PBuilder(void* scratchpad, const unsigned int& n, const unsigned int& rank, const unsigned long& processingRate,
                          double* sHeap, uint64_t* flags, const unsigned int& numBlocks) {
        __shared__ double* p2pDurations;
        __shared__ unsigned int* proximalPeers;
        __shared__ unsigned int numProximalPeers;
        __shared__ unsigned int localBlockId;
        __shared__ double* row;

        if (!block::threadID()) {
            row = sHeap + (2*rank*n);
            p2pDurations = static_cast<double*>(scratchpad);
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
                row[2*proximalPeers[i]] = fmax(p2pDurations[2*i], row[2*proximalPeers[i]]);
                row[2*proximalPeers[i] + 1] = fmax(p2pDurations[2*i + 1], row[2*proximalPeers[i] + 1]);
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
                nvshmemx_double_put_signal_nbi_block(sHeap + (2*rank*n),
                                                 sHeap + (2*rank*n),
                                                 (i != rank)*2*n, flags + rank,
                                                 constructSignal(processingRate, sent),
                                                 NVSHMEM_SIGNAL_SET, proximalPeers[i]);
            }
        }
    }

    /// Build Adjacency Matrix
    __global__ void discover(CUTE_GRID_CONSTANT const int n, CUTE_GRID_CONSTANT const int rank,
        const unsigned long processingRate, double* sHeap, uint64_t* flags,
        double* results) {
        assert(blockDim.x <= ARISTOS_BLOCK_SIZE);
        assert(blockDim.y * blockDim.z == 1);
        /// Persist relevant data in cache
        /// Parition the below among threads
        /*associate_access_property(advancePtr(sHeap, rank), cuda::access_property(advancePtr(sHeap, rank),
            BETA_BUFFER, BETA_BUFFER, cuda::access_property::persisting{}));*/
        /*associate_access_property(flags, cuda::access_property(flags,
            n*sizeof(uint64_t), n*sizeof(uint64_t), cuda::access_property::persisting{}));
        associate_access_property(&publisher::blockade, cuda::access_property::persisting{});*/

        // Align to 16 bytes for optimal copy performance.
        // However, empirical results show identical performance (1.024 𝜇s) for 128 threads copying 256 floats,
        // which is a likely practical upper bound for 2*n.
        extern __shared__ __align__(16) double scratchpad[];

        if(gridDim.x == 1){
            /// number of threads is insufficient for remote specialization
            singularBuilder(scratchpad, n, rank, processingRate, sHeap, results, flags);
        }
        else{
            // Specialization
            __shared__ unsigned int numPeers;
            __shared__ unsigned int* peers;
            if (block::threadID() == 0) {
                numPeers = 0;
                peers = static_cast<unsigned int*>(static_cast<void*>(scratchpad + 2*n));
                /// Block 0 gets remote peers while others get proximal peers
                for(unsigned int i = 0U; i < n; ++i) {
                    bool b = nvshmem_ptr(sHeap, i) == nullptr;
                    b = (blockIdx.x > 0) * !b + b * (blockIdx.x == 0);
                    peers[numPeers] = !b * peers[numPeers] + i * b;
                    numPeers += b;
                }
            }
            __threadfence_block();
            /// local publisher block 0 will service remote communication requests
            /// while other blocks will further specialize to serve parallel P2P, à la NVLink, transfers
            /// TODO if no remote then everyone does P2P
            if(!PublisherConfig::getLocalBlockID(blockDim.x)){
                /// remote publisher
                pluralRemoteBuilder(scratchpad, n, rank, processingRate, sHeap, flags);
            }
            else{
                /// P2P publisher only at most one super block
                pluralP2PBuilder(scratchpad, n, rank, processingRate, sHeap, flags, blockDim.x);
            }
        }
    }
}
#endif //TOPO_CUH
