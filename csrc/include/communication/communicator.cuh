//
// Created by Jonathan on 7/18/24.
//

#ifndef ARISTOS_COMMUNICATOR_CUH
#define ARISTOS_COMMUNICATOR_CUH

#include <cooperative_groups.h>
#include <cuda/atomic>
#include <cuda/annotated_ptr>
#include <cuda/barrier>
#include "../definition/memory_layout.cuh"

namespace aristos{
    //TODO use global mem + atomicAdd
    __device__ cuda::atomic<medium_int, cuda::thread_scope_device> doorbell{0};
    __device__ cuda::atomic<unsigned int, cuda::thread_scope_device> last{1};

    using DeviceAtomicRef = cuda::atomic_ref<n_bytes_repr, cuda::thread_scope_device>;

    CUTE_DEVICE
    void try_until_signal(){
        while(doorbell.load() == 0 && !stop.load()){}
    }

    CUTE_DEVICE
    void communicator_batch_send(n_bytes_repr* heap_iter, // heap iterator
                                 unsigned long long int* sync_grid,
                                 unsigned char* symmetric_heap_byte,
                                 const size_t n_bytes_index,
                                 const size_t blocks_to_peers, // r
                                 const uint peer_stripe_len, // l
                                 const uint intra_peer_index,
                                 const uint n_experts){
        // TODO avoid copy for self-send?
        size_t n_b = 0; // TODO move to shared?
        // TODO move i init outside
        for(int i = static_cast<int>(aristos::bid() / blocks_to_peers); i < peer_stripe_len; ++i){
            if(n_b = atomicAdd((heap_iter + (n_bytes_index * i)), 0); n_b > 0){ // atomicAdd == load
                // TODO __syncwarp() somewhere here?
                auto chunk_size = cuda::ceil_div(n_b, blocks_to_peers);
                auto data_ptr = symmetric_heap_byte + (intra_peer_index*chunk_size);
                if(intra_peer_index == (blocks_to_peers - 1) && n_b % blocks_to_peers != 0){
                    // residual chunk
                    nvshmemx_putmem_nbi_block(data_ptr,
                                              (data_ptr + (n_b % chunk_size)),
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
        if(block_tid() == 0 && n_b > 0){
            nvshmem_quiet();
            //last block signals doorbell, sync and send signal.z, zero n_bytes
            for(int i = static_cast<int>(aristos::bid() / blocks_to_peers); i < peer_stripe_len; ++i){

                atomicAdd((sync_grid + ((n_experts*i) + 3)), 1);
            }
        }
        __syncthreads();
        // decrement doorbell?
    }

    CUTE_DEVICE
    void communicator_start(void* symmetric_heap,
                            unsigned long long int* const sync_grid,
                            const uint n_peers,
                            const uint cap,
                            const uint k,
                            const uint embed_dim,
                            const uint embed_p,
                            const uint n_experts){
        // TODO annotated ptr @ startup instead
        cuda::access_property accessProperty(cuda::access_property::persisting{});
        cuda::associate_access_property(symmetric_heap, accessProperty);

        const auto t_per_block = (blockDim.x * blockDim.y * blockDim.z);
        // requires gridDim.z > 1,
        // <= 2 is reasonable
        const auto n_comm_blocks = (gridDim.z - 1) * (gridDim.x * gridDim.y);
        auto blocks_to_peers = size_t(max(n_comm_blocks / n_peers, 1)); // floor division by default
        auto peer_stripe_len = cuda::ceil_div(1UL, blocks_to_peers);
        auto heap_iter = static_cast<n_bytes_repr*>(symmetric_heap);
        auto symmetric_heap_byte = static_cast<unsigned char*>(symmetric_heap);
        auto n_bytes_index = symmetric_heap_peer_offset(cap, k, embed_dim, embed_p) / sizeof(size_t);
        auto intra_peer_index = aristos::bid() - ((aristos::bid() / blocks_to_peers) * blocks_to_peers);

        __shared__ uint checkpoint;
        __shared__ bool done_first_broadcast;
        extern __shared__ DeviceAtomicRef atomic_refs [];
        __syncthreads();

        // broadcast()
        // Most likely == 1
        while(!stop.load()){
            try_until_signal();
            while(!stop.load() && doorbell.load() > 0){
                communicator_batch_send(heap_iter,
                           sync_grid,
                           symmetric_heap_byte,
                           n_bytes_index,
                           blocks_to_peers,
                           peer_stripe_len,
                           intra_peer_index,
                           n_experts);
            }
        }
    }

    CUTE_DEVICE
    void communicator_batch_enqueue(int n_requests){
        if(last.fetch_add(1) == n_requests){
            doorbell++;
            last.store(1);
        }
    }

}
#endif //ARISTOS_COMMUNICATOR_CUH
