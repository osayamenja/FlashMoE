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

namespace aristos{
    //TODO use global mem + atomicAdd
    __device__ unsigned int doorbell = 0U;
    __device__ unsigned int blockade = 1U;

    CUTE_DEVICE
    void try_until_signal(){
        while(atomicLoad(&doorbell) == 0 && atomicLoad(&stillExecuting)){} //TODO maybe sleep?
    }

    template<bool isPrimingStage = false>
    CUTE_DEVICE
    void communicator_batch_send(unsigned int* heap_iter, // heap iterator
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
                auto chunk_size = cuda::ceil_div(payload_bytes, blocks_to_peers);
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
    void communicator_start(void* symmetric_heap,
                            unsigned int rank,
                            uint64_t* flags,
                            unsigned int* sync_grid,
                            unsigned int seq_no,
                            const uint n_peers,
                            const uint cap,
                            const uint k,
                            const uint embed_dim,
                            const uint embed_p,
                            const uint n_experts){
        // TODO annotated ptr @ startup instead
        // TODO shared_mem config object for all func parameters
        cuda::access_property accessProperty(cuda::access_property::persisting{});
        cuda::associate_access_property(symmetric_heap, accessProperty);

        __shared__ unsigned long stripe_len, blocksToPeers; //TODO force register use instead of shared?
        extern __shared__ unsigned int checkpoints[];
        if(aristos::block_tid() == 0){
            // 1 < gridDim.z <= 2
            // RHS is a recommendation
            blocksToPeers = size_t(max((gridDim.z - 1) * (gridDim.x * gridDim.y) / n_peers, 1));
            stripe_len = cuda::ceil_div(1UL, blocksToPeers);
            for(unsigned int i = 0; i < 2*stripe_len; ++i){
                checkpoints[i] = 0;
            }
        }
        __threadfence_block();
        __syncthreads();

        // broadcast()
        // Most likely == 1
        while(atomicLoad(&stillExecuting)){
            try_until_signal();
            while(atomicLoad(&stillExecuting) && atomicLoad(&doorbell) > 0){
                //TODO please cache the arguments outside the loop
                communicator_batch_send(
                        static_cast<unsigned int*>(symmetric_heap),
                        rank,
                        seq_no,
                        flags,
                        sync_grid,
                        static_cast<std::byte*>(symmetric_heap),
                        (symmetric_heap_peer_offset(cap, k,(embed_dim * embed_p)) / micro_header_bytes),
                        blocksToPeers,
                        stripe_len,
                        cap,
                        (embed_dim * embed_p),
                        k,
                        aristos::bid() - ((aristos::bid() / blocksToPeers) * blocksToPeers),
                        n_experts,
                        payload_bytes((embed_dim * embed_p), 1), // singular packets regime
                        static_cast<int>(aristos::bid() / blocksToPeers),
                        checkpoints); // int due to API
            }
        }
    }

    CUTE_DEVICE
    void communicator_batch_enqueue(const int n_requests){
        if(atomicAdd(&blockade, 1U) % n_requests == 0){
            atomicAdd(&doorbell, 1U);
        }
    }

}
#endif //ARISTOS_COMMUNICATOR_CUH
