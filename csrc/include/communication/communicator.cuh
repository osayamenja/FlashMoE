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
    __device__ cuda::atomic<medium_int, cuda::thread_scope_device> doorbell{0};
    __device__ cuda::atomic<unsigned int, cuda::thread_scope_device> last{1};

    using DeviceAtomicRef = cuda::atomic_ref<n_bytes_repr, cuda::thread_scope_device>;
    class Communicator{
    public:
        Communicator() = default;
        static CUTE_DEVICE
        void start(void* symmetric_heap, medium_int rank, uint n_peers,
                   uint cap, uint k, uint embed_dim, uint embed_p){
            // TODO annotated ptr @ startup instead
            cuda::access_property accessProperty(cuda::access_property::persisting{});
            cuda::associate_access_property(symmetric_heap, accessProperty);
            // Most likely == 1
            while(!stop.load()){
                __shared__ uint checkpoint;
                check_doorbell_until_signal();
                while(!stop.load() && doorbell.load() > 0){
                    batch_send(symmetric_heap, rank, n_peers, cap, k, embed_dim, embed_p);
                }
            }
        }
        static CUTE_DEVICE
        void batch_enqueue(int n_requests){
            if(last.fetch_add(1) == n_requests){
                doorbell++;
                last.store(1);
            }
        }
    private:
        static CUTE_DEVICE
        void check_doorbell_until_signal(){
            while(doorbell.load() == 0 && !stop.load()){}
        }
        static CUTE_DEVICE
        void batch_send(void* symmetric_heap,
                        const medium_int rank,
                        const uint n_peers,
                        const uint cap,
                        const uint k,
                        const uint embed_dim,
                        const uint embed_precision){
            // TODO grid_constant
            const auto t_per_block = (blockDim.x * blockDim.y * blockDim.z);
            // gridDim.z > 1, 2 is okay
            const auto n_comm_blocks = (gridDim.z - 1) * (gridDim.x * gridDim.y);
            auto blocks_to_peers = n_comm_blocks / n_peers; // floor division by default
            auto peer_stripe_len = cuda::ceil_div(1U, blocks_to_peers);
            auto heap_iter = static_cast<n_bytes_repr*>(symmetric_heap);
            auto symmetric_heap_byte = static_cast<char*>(symmetric_heap);
            auto n_bytes_index = symmetric_heap_index(cap, k, embed_dim, embed_precision) / sizeof(size_t);
            auto intra_peer_index = aristos::bid() - ((aristos::bid() / blocks_to_peers) * blocks_to_peers);
            for(uint i = (aristos::bid() / blocks_to_peers); i < peer_stripe_len; ++i){
                if(size_t n_b = DeviceAtomicRef(heap_iter[(n_bytes_index*i)]).exchange(0) > 0){
                    // TODO __syncwarp() somewhere here?
                    auto chunk_size = cuda::ceil_div(n_b, size_t(blocks_to_peers));
                    auto data_ptr = symmetric_heap_byte + (intra_peer_index*chunk_size);
                    if(intra_peer_index == (blocks_to_peers - 1) && n_b % blocks_to_peers != 0){
                        // residual chunk
                        nvshmemx_putmem_nbi_block(data_ptr,
                                                  (data_ptr + (n_b % chunk_size)),
                                                  chunk_size,
                                                  static_cast<int>(i));
                    }
                    else{
                        // send complete chunk
                        nvshmemx_putmem_nbi_block(data_ptr,
                                                  (data_ptr + chunk_size),
                                                  chunk_size,
                                                  static_cast<int>(i));
                    }
                }
            }
            // decrement doorbell?

        }
    };

    __device__ __constant__ Communicator communicator{};
}
#endif //ARISTOS_COMMUNICATOR_CUH
