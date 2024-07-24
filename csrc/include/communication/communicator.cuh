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
    __device__ cuda::atomic<medium_int, cuda::thread_scope_device> queue_index{0};
    __device__ cuda::atomic<unsigned int, cuda::thread_scope_device> comm_barrier{1};

    class Communicator{
    public:
        Communicator() = default;
        static CUTE_DEVICE
        void start(void* message_queue, medium_int rank, uint n_peers,
                   uint cap, uint k, uint embed_dim, uint embed_p){
            cuda::access_property accessProperty(cuda::access_property::persisting{});
            cuda::associate_access_property(message_queue, accessProperty);

            // Most likely == 1
            while(!stop.load()){
                check_doorbell_until_signal();
                while(!stop.load() && doorbell.load() > 0){
                    batch_dequeue(message_queue, rank, n_peers, cap, k, embed_dim, embed_p);
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
        void batch_dequeue(void* message_queue, medium_int rank, const uint n_peers,
                           uint cap, uint k, uint embed_dim, uint embed_precision){
            const auto t_per_block = (blockDim.x * blockDim.y * blockDim.z);
            // gridDim.z > 1, 2 is okay
            const auto n_comm_blocks = (gridDim.z - 1) * (gridDim.x * gridDim.y);
            const auto total_communicator_threads = t_per_block * n_comm_blocks;
            uint thread_stripe_len = cuda::ceil_div(n_peers, total_communicator_threads);
            using IndexPair = cuda::std::pair<uint, size_t>;
            extern __shared__ IndexPair communication_queue[];
            using value_type = size_t;
            using DeviceAtomicRef = cuda::atomic_ref<value_type, cuda::thread_scope_device>;
            auto tid = get_tid() % n_comm_blocks;
            auto heap_iter = static_cast<value_type*>(message_queue);
            auto heap_index = symmetric_heap_index(cap, k, embed_dim, embed_precision) / sizeof(size_t);
            CUTE_UNROLL
            for(uint i = 0; i < thread_stripe_len && (i * tid) < n_peers; ++i){
                if(size_t n_b = DeviceAtomicRef(heap_iter[(heap_index*i)]).exchange(0) != 0){
                    auto j = queue_index.fetch_add(1);
                    communication_queue[j] = IndexPair{(i * tid), n_b};
                }
            }
            __syncthreads();

            // decrement doorbell?
        }

        static CUTE_DEVICE
        void sync(const CUTE_GRID_CONSTANT auto n_participants){
            // arrive
            if(auto token = comm_barrier.fetch_add(1); token == n_participants){

            }
            // You have nothing to do in the interim, so kindly
            // wait
//            while(token != comm_barrier.load()){}

        }
    };

    __device__ __constant__ Communicator communicator{};
}
#endif //ARISTOS_COMMUNICATOR_CUH
