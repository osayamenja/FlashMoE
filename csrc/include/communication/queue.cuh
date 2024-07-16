//
// Created by Jonathan on 7/4/24.
//

#ifndef ARISTOS_QUEUE_CUH
#define ARISTOS_QUEUE_CUH

#include "packet.cuh"

namespace aristos{
    class task_queue{
        using ratio = cuda::std::micro;
        using clk = cuda::std::chrono::high_resolution_clock;
        cuda::std::chrono::time_point<clk> snapshot_now{};
        cuda::std::chrono::duration<float, ratio> timeout_duration;
        unsigned int optimism_factor;
        uint64_t* flags;
        size_t* indices;
        unsigned int n_pes;
        const unsigned int default_val;
        int* status;
        bool activated_countdown = false;

    public:
        CUTE_DEVICE
        task_queue(cuda::std::chrono::duration<float, ratio> _timeout_duration,
                   const unsigned int _var_epsilon,
                   uint64_t* _flags,
                   size_t* _indices,
                   unsigned int _n_pes,
                   int* _status) :
                timeout_duration(_timeout_duration), optimism_factor(_var_epsilon), flags(_flags), indices(_indices),
                n_pes(_n_pes), default_val(_flags[0]), status(_status)
                {
                    if(optimism_factor == 0){
                        snapshot_now = clk::now();
                    }
                };

        template<TensorValueType V>
        CUTE_DEVICE
        void dequeue(){
            size_t n_received_pes;
            do{
                n_received_pes = nvshmem_uint64_test_some(flags, n_pes, indices, status,
                                                             NVSHMEM_CMP_LT, default_val);
            }while(n_received_pes == 0 && (optimism_factor > 0 || (clk::now() - snapshot_now) < timeout_duration));

            for(int i = 0; i < n_received_pes; ++i){
                auto index = indices[i];
                if(flags[index] == NOOP || flags[index] == processed){
                    status[index] = 1; // Exclude this PE from subsequent tests
                }
                optimism_factor = (optimism_factor > 0 && flags[index] == processed)?
                        (optimism_factor - 1) : optimism_factor;
                if(optimism_factor == 0 && !activated_countdown){
                    snapshot_now = clk::now();
                    activated_countdown = true;
                }
            }
            // TODO Bundle data according to shouldProcess or processed
            // TODO Return array of tensors?

        }
    };
}
#endif //ARISTOS_QUEUE_CUH
