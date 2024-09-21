//
// Created by oja7 on 9/19/24.
//

#ifndef TOPO_CUH
#define TOPO_CUH

#define TOPO_LOOP_TRIP 4 // may be too much
namespace aristos::topology{
    /// Build Adjacency Matrix
    using Micro = cuda::std::chrono::duration<double, cuda::std::micro>;
    __global__ __forceinline__ ARISTOS_LAUNCH_BOUNDS void build(CUTE_GRID_CONSTANT const unsigned int n,
        CUTE_GRID_CONSTANT const unsigned int rank) {
        if(moeConfig.numPublisherBlocks < ((NVLinkThreshold / blockSize) + 1)){
            /// number of threads is insufficient for remote specialization
            //one block does all
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

    CUTE_DEVICE void singularBuilder(const unsigned int& n, const unsigned int& rank) {
        using Nano = cuda::std::chrono::duration<double, cuda::std::nano>;
        using Micro = cuda::std::chrono::duration<double, cuda::std::micro>;
        __shared__ Nano transferDuration;
        __shared__ uint64_t start, end;
        __shared__ extern double durations[];
        if (block::threadID() == 0) {
            transferDuration = Nano::zero();
            start = end = 0;
        }
        __threadfence_block();
        for (unsigned int i = 0; i < n; ++i) {
            if (const auto peer = (((i - rank) % n) + n) % n;
                /// NVLink-connected peers
                nvshmem_ptr(moeConfig.sHeap, peer) != nullptr) {
                /// Alpha cost
                CUTE_UNROLL
                for (unsigned int j = 0; j < TOPO_LOOP_TRIP; ++j) {
                    asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
                    //nvshmemx_putmem_block(getTokenPointer(0, 0, 0, 0), )
                    asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
                    transferDuration += static_cast<Nano>(end - start);
                }
                if(block::threadID() == 0) {
                    durations[peer] = cuda::std::chrono::duration_cast<Micro>(transferDuration / TOPO_LOOP_TRIP).count();
                }
                ///Beta Cost
                CUTE_UNROLL
                for (unsigned int j = 0; j < TOPO_LOOP_TRIP; ++j) {
                    asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
                    //nvshmemx_putmem_block(getTokenPointer(0, 0, 0, 0), )
                    asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
                    transferDuration += static_cast<Nano>(end - start);
                }
                if(block::threadID() == 0) {
                    durations[peer] = cuda::std::chrono::duration_cast<Micro>(transferDuration / TOPO_LOOP_TRIP).count();
                }
            }
        }
        // parallelize writes to gmem
    }
}
#endif //TOPO_CUH
