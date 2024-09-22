//
// Created by Jonathan on 7/4/24.
//

#ifndef ARISTOS_QUEUE_CUH
#define ARISTOS_QUEUE_CUH

#include "../definition/types.cuh"
#include "../definition/tensor.cuh"

namespace aristos::subscriber{
    using ratio = cuda::std::micro;
    using clk = cuda::std::chrono::high_resolution_clock;
    cuda::std::chrono::time_point<clk> snapshot_now{};
    cuda::std::chrono::duration<float, ratio> timeout_duration;
    unsigned int optimism_factor;
    uint64_t* flags;
    size_t* indices;
    unsigned int n_pes;
    const unsigned int default_val = 3;
    int* status;
    bool activated_countdown = false;

    CUTE_DEVICE
    void start(){

    }

    template<TensorValueType V>
    CUTE_DEVICE
    void dequeue(){
        size_t n_received_pes;
        do{
            n_received_pes = nvshmem_uint64_test_some(flags, n_pes, indices, status,
                                                      NVSHMEM_CMP_LT, default_val);
        }while(n_received_pes == 0 && (optimism_factor > 0 || (clk::now() - snapshot_now) < timeout_duration));

        for(uint i = 0; i < n_received_pes; ++i){
            auto index = indices[i];
            if(flags[index] == NOOP){
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
}
#endif //ARISTOS_QUEUE_CUH
