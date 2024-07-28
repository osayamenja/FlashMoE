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

    template<bool isPrimingStage = false>
    CUTE_DEVICE
    void communicator_batch_send(unsigned int* heap_iter, // heap iterator
                                 unsigned long long int* sync_grid,
                                 unsigned char* symmetric_heap_byte,
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
        // TODO move i init outside
        for(int i = first_peer; i < peer_stripe_len; ++i){
            if(auto n_k = atomicAdd((heap_iter + (peer_offset * i)), 0); n_k > 0){ // atomicAdd == load
                // TODO __syncwarp() somewhere here?
                auto chunk_size = cuda::ceil_div(payload_bytes, blocks_to_peers);
                auto data_ptr = symmetric_heap_byte + (intra_peer_index*chunk_size);
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
            //last block signals doorbell, sync and send signal.z, zero n_bytes
            unsigned int j = first_peer + peer_stripe_len;
            for(int i = first_peer; i < peer_stripe_len; ++i, ++j){
                if(checkpoints[j] > 0){
                    auto expert_i = packet_trailer_index(send_cell(1),
                                                         checkpoints[j],
                                                         cap,
                                                         embed_bytes,
                                                         k);
                    auto old = atomicAdd((sync_grid + ((n_experts*i) + heap_iter[(peer_offset * i) + expert_i])), 1);
                    if(old == blocks_to_peers){
                        // I am last
                        //send nvshmem signal
                        doorbell--;
                    }
                }
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

        // requires gridDim.z > 1,
        // <= 2 is reasonable
        __shared__ unsigned long stripe_len, blocksToPeers; //TODO force register use instead of shared?
        extern __shared__ unsigned int checkpoints[];
        if(aristos::block_tid() == 0){
            stripe_len = cuda::ceil_div(1UL, blocksToPeers);
            blocksToPeers = size_t(max((gridDim.z - 1) * (gridDim.x * gridDim.y) / n_peers, 1));
            for(unsigned int i = 0; i < 2*stripe_len; ++i){
                checkpoints[i] = 0;
            }
        }
        __threadfence_block();
        __syncthreads();

        // broadcast()
        // Most likely == 1
        while(!stop.load()){
            try_until_signal();
            while(!stop.load() && doorbell.load() > 0){
                //TODO please cache the arguments outside the loop
                communicator_batch_send(
                        static_cast<unsigned int*>(symmetric_heap),
                        sync_grid,
                        static_cast<unsigned char*>(symmetric_heap),
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
    void communicator_batch_enqueue(int n_requests){
        if(last.fetch_add(1) == n_requests){
            doorbell++;
            last.store(1);
        }
    }

}
#endif //ARISTOS_COMMUNICATOR_CUH
