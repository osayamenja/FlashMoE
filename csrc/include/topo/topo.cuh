//
// Created by oja7 on 9/19/24.
//

#ifndef TOPO_CUH
#define TOPO_CUH

#define TOPO_LOOP_TRIP 4U // this may be too much
#define BETA_BUFFER (1024UL * 1024UL) // 1MB
#define ALPHA_BUFFER 1024UL // 1KB
#define NANO_TO_MILLI (cuda::std::nano::den / cuda::std::milli::den)
#define NANO_TO_MICRO (cuda::std::nano::den / cuda::std::micro::den)
#define BYTE_MAX cuda::std::numeric_limits<cuda::std::underlying_type_t<cuda::std::byte>>::max()
#define TO_MB(b) (static_cast<double>(b) / (1024.0f*1024.0f))

using Nano = cuda::std::chrono::duration<float, cuda::std::nano>;
using Milli = cuda::std::chrono::duration<float, cuda::std::milli>;
using ull_t = unsigned long long int;
namespace aristos::topology{
    template<typename T=void>
    CUTE_DEVICE
    T* advancePtr(T* buffer, const unsigned int& slot) {
        return buffer + (slot * BETA_BUFFER);
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
        const int& peer, const ID& id, const Put& put, const unsigned int& peerIdx,
        const unsigned int& lBid = 0, const unsigned int& nb = 1) {
        ull_t start, end;
        double duration = 0.0;
        bool isResidual = alphaBuf % nb != 0 && lBid == nb - 1;
        size_t splitAlphaBuf = (lBid * cute::ceil_div(alphaBuf, nb) < alphaBuf) *
                (!isResidual * cute::ceil_div(alphaBuf, nb)
                + isResidual * (alphaBuf - (cute::ceil_div(alphaBuf, nb) * (nb - 1))));
        /// Alpha cost: ms
        CUTE_UNROLL
        for (unsigned int j = 0; j < TOPO_LOOP_TRIP; ++j) {
            asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
            put(advancePtr(sHeap, rank) + (lBid * splitAlphaBuf), advancePtr(sHeap, rank) + (lBid * splitAlphaBuf), splitAlphaBuf, peer);
            asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
            duration += (end - start) / static_cast<double>(TOPO_LOOP_TRIP*NANO_TO_MILLI);
        }
        if(id() == 0) {
            remoteDurations[peerIdx*2] = duration;
        }
        duration = 0.0;
        isResidual = betaBuf % nb != 0 && lBid == nb - 1;
        size_t splitBetaBuf = (lBid * cute::ceil_div(betaBuf, nb) < betaBuf) * (!isResidual * cute::ceil_div(betaBuf, nb)
               + isResidual * (betaBuf - (cute::ceil_div(betaBuf, nb) * (nb - 1))));
        ///Beta Cost: ms/MB
        CUTE_UNROLL
        for (unsigned int j = 0; j < TOPO_LOOP_TRIP; ++j) {
            asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
            put(advancePtr(sHeap, rank) + (lBid * splitBetaBuf), advancePtr(sHeap, rank) + (lBid * splitBetaBuf), splitBetaBuf, peer);
            asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
            duration += (end - start)/ static_cast<double>(TOPO_LOOP_TRIP*NANO_TO_MILLI);
        }
        if(id() == 0) {
            // Compute beta using slope intercept equation
            remoteDurations[2*peerIdx + 1] = ((duration - remoteDurations[2*peerIdx]) / (TO_MB(splitBetaBuf) - TO_MB(splitAlphaBuf)));
            // Compute alpha and apply abs for spurious negatives
            remoteDurations[2*peerIdx] = fabs(remoteDurations[2*peerIdx] - (TO_MB(splitAlphaBuf) * remoteDurations[2*peerIdx + 1]));
        }
    }

    CUTE_DEVICE
    void singularBuilder(void* scratchpad, const unsigned int& n, const unsigned int& rank, const unsigned long& processingRate,
        double* sHeap, double* results, uint64_t* flags) {
        for (unsigned int i = 1U; i < n; ++i) {
            measureTransfer(rank, sHeap, static_cast<double*>(scratchpad), (rank + i) % n,
                block::threadID, nvshmemx_putmem_block, (rank + i) % n);
        }
        __threadfence_block();
        __syncthreads();

        /// Stage my row on the symmetric heap
        const auto slice = cute::ceil_div(2*n, ARISTOS_BLOCK_SIZE);
        memCopySet(results,static_cast<double*>(scratchpad),block::threadID() * slice,
            min(slice * (block::threadID() + 1), 2U*n));
        __threadfence_block();

        // Signal my vector, including FLOPs, to others
        for (unsigned int i = 1U; i < n; ++i) {
            nvshmemx_double_put_signal_nbi_block(results, results, 2*n, flags + rank,
                constructSignal(processingRate, sent), NVSHMEM_SIGNAL_SET, (rank + i) % n);
        }

        // await responses from other GPUs
        if (block::threadID() == 0) {
            static_cast<int*>(scratchpad)[rank] = 1; // Do not wait for me
            nvshmem_uint64_wait_until_all(flags, n, static_cast<int*>(scratchpad), NVSHMEM_CMP_GT, sent);
            for (unsigned int i = 1U; i < n; ++i) {
                flags[(rank  + i) % n] -= sent;
            }
        }
    }

    CUTE_DEVICE
    void pluralRemoteBuilder(void* scratchpad, const unsigned int* peers, const unsigned int& n, const unsigned int& rank,
        const unsigned long& processingRate, const unsigned int numPeers, double* sHeap, double* results, uint64_t* flags) {
        for (unsigned int i = 0U; i < numPeers; ++i) {
            measureTransfer(rank, sHeap, static_cast<double*>(scratchpad), peers[(i + rank) % numPeers],
                block::threadID, nvshmemx_putmem_block, (i + rank) % numPeers);
        }
        __threadfence_block();
        __syncthreads();

        /// Stage my row to the symmetric heap
        auto slice = cute::ceil_div(numPeers, ARISTOS_BLOCK_SIZE);
        auto bookend = min(slice * (block::threadID() + 1), numPeers);
        for (unsigned int i = block::threadID() * slice; i < bookend; ++i) {
            results[2*peers[i]] = static_cast<double*>(scratchpad)[2*i];
            results[2*peers[i] + 1] = static_cast<double*>(scratchpad)[2*i + 1];
        }
        __threadfence_block();

        // Barrier ensures our vector is complete before sending to neighbors.
        if (!block::threadID()) {
            atomicAdd(&publisher::blockade, 1);
            while (atomicLoad(&publisher::blockade) != 2) {
                __nanosleep(2);
            }
        }
        __syncthreads();
        slice = cute::ceil_div(numPeers, ARISTOS_BLOCK_SIZE_WARP);
        bookend = min(slice * (block::warpID() + 1), numPeers);
        // Signal our vector, including FLOPs, to others
        for (unsigned int i = block::warpID() * slice; i < bookend; ++i) {
            nvshmemx_double_put_signal_nbi_warp(results, results, 2*n, flags + rank,
                    constructSignal(processingRate, sent), NVSHMEM_SIGNAL_SET, peers[i]);
        }
    }

    CUTE_DEVICE
    void pluralP2PBuilder(void* scratchpad,const unsigned int* peers, const unsigned int& n, const unsigned int& rank,
        const unsigned long& processingRate, const unsigned int numPeers, double* sHeap, double* results, uint64_t* flags) {
        __shared__ unsigned int localBlockIdx;
        for (unsigned int i = 1U; i < numPeers; ++i) {
            measureTransfer(rank, sHeap, static_cast<double*>(scratchpad),
                peers[(i + rank) % numPeers], block::threadID, nvshmemx_putmem_block,
                (i + rank) % numPeers, blockIdx.x - (numPeers < n), gridDim.x - (numPeers < n));
        }
        if (!block::threadID()) {
            localBlockIdx = blockIdx.x - (numPeers < n);
        }
        __threadfence_block();
        __syncthreads();
        /// All-Reduce to get max transfer time across blocks
        /// Reusing concepts from publisher
        // Wait our turn
        if (!block::threadID()) {
            publisher::awaitBaton(localBlockIdx);
        }
        __syncthreads();
        /// Update the global buffer with my values via max reduction
        auto slice = cute::ceil_div(numPeers, ARISTOS_BLOCK_SIZE);
        auto bookend = min(slice * (block::threadID() + 1), numPeers);
        for (unsigned int i = block::threadID() * slice; i < bookend; ++i) {
            results[2*peers[i]] = fmax(static_cast<double*>(scratchpad)[2*i], results[2*peers[i]]);
            results[2*peers[i] + 1] = fmax(static_cast<double*>(scratchpad)[2*i + 1], results[2*peers[i] + 1]);
        }

        if (!block::threadID()) {
            // pass the baton
            atomicExch(&publisher::baton, (localBlockIdx + 1) % (gridDim.x - (numPeers < n)));
            // Await baton to circle back, at this point the reduction is complete, and we are free to go
            publisher::awaitBaton(localBlockIdx);
            /// Critical that the initiating process has id GRAND_MASTER for the below to be correct
            if(localBlockIdx == GRAND_MASTER){
                // unblock remote publisher
                atomicAdd(&publisher::blockade, 1);
                while (atomicLoad(&publisher::blockade) != 2) {
                    __nanosleep(2);
                }
            }
            // Unblock sibling blocks
            atomicExch(&publisher::baton, (localBlockIdx + 1) % (gridDim.x - (numPeers < n)));
        }
        // Ensures all threads are done
        __threadfence_block();
        __syncthreads();
        // Signal our vector, including FLOPs, to others
        slice = cute::ceil_div(numPeers, gridDim.x - (numPeers < n));
        bookend = min(slice * (localBlockIdx + 1), numPeers);
        for(unsigned int i = slice * localBlockIdx; i < bookend; ++i){
            nvshmemx_double_put_signal_nbi_block(results, results,
                                             (peers[i] != rank) * 2*n, flags + rank,
                                             constructSignal(processingRate, sent),
                                             NVSHMEM_SIGNAL_SET, peers[i]);
        }

        if (localBlockIdx == 0 && !block::threadID()) {
            memset(scratchpad, 0, sizeof(int)*n);
            static_cast<int*>(scratchpad)[rank] = 1;
            nvshmem_uint64_wait_until_all(flags, n, static_cast<int*>(scratchpad), NVSHMEM_CMP_GT, sent);
            for (unsigned int i = 1U; i < n; ++i) {
                flags[(rank  + i) % n] -= sent;
            }
        }
    }

    /// Build Adjacency Matrix
    __global__ void discover(CUTE_GRID_CONSTANT const int n, CUTE_GRID_CONSTANT const int rank,
        CUTE_GRID_CONSTANT const bool remotePresent, const unsigned long processingRate,
        double* sHeap, uint64_t* flags, double* results) {
        assert(blockDim.x == ARISTOS_BLOCK_SIZE);
        assert(blockDim.y * blockDim.z == 1);
        assert(gridDim.x <= ARISTOS_SUPER_BLOCK_SIZE + remotePresent);
        assert(gridDim.y * gridDim.z == 1);

        // Align to 16 bytes for optimal copy performance.
        // However, empirical results show identical performance (1.024 𝜇s) for 128 threads copying 256 floats,
        // which is a likely practical upper bound for 2*n.
        extern __shared__ __align__(16) double scratchpad[];

        if(gridDim.x == 1){
            /// number of blocks is insufficient for remote specialization
            singularBuilder(scratchpad, n, rank, processingRate, sHeap, results, flags);
        }
        else{
            // Specialization
            __shared__ unsigned int numPeers;
            __shared__ unsigned int* peers;
            if (block::threadID() == 0) {
                apply_access_property(&publisher::blockade, sizeof(decltype(publisher::blockade)),
                    cuda::access_property::persisting{});
                numPeers = 0;
                peers = static_cast<unsigned int*>(static_cast<void*>(scratchpad + 2*n));
                /// Block 0 gets remote peers, if present; otherwise; joins the others in getting proximal peers
                for(unsigned int i = 0U; i < n; ++i) {
                    if (const bool b = nvshmem_ptr(results, i);
                        (!b && blockIdx.x > 0) || ((!remotePresent && !b ) || (remotePresent && (b && blockIdx.x == 0)))) {
                        peers[numPeers++] = i;
                    }
                }
            }
            assert(0);
            __threadfence_block();
            __syncthreads();
            /// local publisher block 0 will service remote communication requests
            /// while other blocks will further specialize to serve parallel P2P, à la NVLink, transfers
            if(!PublisherConfig::getLocalBlockID(blockDim.x) && remotePresent){
                /// remote publisher
                pluralRemoteBuilder(scratchpad, peers, n, rank, processingRate, numPeers, sHeap, results, flags);
            }
            else{
                /// P2P publisher only at most one super block
                pluralP2PBuilder(scratchpad, peers, n, rank, processingRate, numPeers, sHeap, results, flags);
            }
        }
    }
}
#endif //TOPO_CUH
