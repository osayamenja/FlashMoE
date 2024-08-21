//
// Created by Jonathan on 7/18/24.
//

#ifndef ARISTOS_COMMUNICATOR_CUH
#define ARISTOS_COMMUNICATOR_CUH

#include <cooperative_groups.h>
#include <cuda/atomic>
#include <cuda/annotated_ptr>
#include "../util/atomics.cuh"
#include <cuda/barrier>
#include "../definition/memory_layout.cuh"
#include "../definition/values.cuh"

namespace aristos{
    __device__ unsigned int doorbell = 0U;
    __device__ unsigned int blockade = 1U;

    CUTE_DEVICE
    void tryUntilSignal(){
        while(atomicLoad(&doorbell) == 0 && atomicLoad(&stillExecuting)){}
    }

    CUTE_DEVICE
    void batchSend(unsigned int* heap_iter, // heap iterator
                                 unsigned int rank,
                    const unsigned long seq_no,
                    uint64_t* flags,
                    unsigned int* sync_grid,
                    std::byte* symmetric_heap,
                    const unsigned int peer_offset,
                    const size_t blocks_to_peers, // r
                                 const uint peer_stripe_len, // l
                                 const unsigned int cap,
                    const unsigned int embed_bytes,
                    const unsigned int k,
                    const uint intra_peer_index,
                    const uint n_experts,
                    const size_t payload_bytes,
                    const int first_peer,
                    const unsigned int* checkpoints){
        // TODO avoid copy for self-send?
        // TODO move to shared?
        for(int i = first_peer; i < peer_stripe_len; ++i){
            if(auto n_k = atomicAdd((heap_iter + (peer_offset * i)), 0); n_k > 0){ // atomicAdd == load
                // TODO __syncwarp() somewhere here?
                auto chunk_size = cute::ceil_div(payload_bytes, blocks_to_peers);
                auto data_ptr = symmetric_heap + (intra_peer_index * chunk_size);
                // dest is wrong below
                if(intra_peer_index == (blocks_to_peers - 1) && payload_bytes % blocks_to_peers != 0){
                    // residual chunk
                    nvshmemx_putmem_nbi_block(data_ptr,
                                              (data_ptr + (payload_bytes % chunk_size)),
                                              chunk_size,
                                              i);
                }
                else{
                    // send complete chunk
                    nvshmemx_putmem_nbi_block(data_ptr,
                                              (data_ptr + chunk_size),
                                              chunk_size,
                                              i);
                }
            }
        }
        __syncthreads(); // needed to ensure quiet() encompasses transfers by all threads
        if(block_tid() == 0){
            nvshmem_quiet();
            unsigned int j = first_peer + peer_stripe_len;
            for(int i = first_peer; i < peer_stripe_len; ++i, ++j){
                if(checkpoints[j] > 0){
                    auto expert_i = packet_trailer_index(send_cell(1),
                                                         checkpoints[j],
                                                         cap,
                                                         embed_bytes,
                                                         k);
                    auto old = atomicAdd((sync_grid + ((n_experts*i) + heap_iter[(peer_offset * i) + expert_i])), 1U);
                    if(old == blocks_to_peers){
                        nvshmemx_signal_op((flags + rank),
                                           constructSignal(seq_no, aristos::processed),
                                           NVSHMEM_SIGNAL_SET,
                                           i);
                        doorbell--;
                    }
                }
            }
        }
        __syncthreads();
    }

    CUTE_DEVICE
    void startPublisher(){
        __shared__ SenderConfig s;
        if(aristos::block_tid() == 0){
            s = SenderConfig(moeConfig);
        }
        __threadfence_block();
        __syncthreads();

        // broadcast()
        while(atomicLoad(&stillExecuting)){
            tryUntilSignal();
            while(atomicLoad( &stillExecuting) && atomicLoad(&doorbell) > 0){

            }
        }
    }

    CUTE_DEVICE
    void batchNotifyPublisher(){
        if(atomicAdd(&blockade, 1U) % moeConfig.numResultChunks == 0){
            atomicAdd(&doorbell, 1U);
        }
    }

}
#endif //ARISTOS_COMMUNICATOR_CUH
