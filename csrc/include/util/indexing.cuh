//
// Created by osayamen on 7/13/24.
//

#ifndef ARISTOS_INDEXING_CUH
#define ARISTOS_INDEXING_CUH

#include <cute/config.hpp>
namespace aristos{
    int
    CUTE_DEVICE
    get_tid() {
    #if defined(__CUDA_ARCH__)
        auto blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
        return blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    #else
        return 0;
    #endif
    }

    int
    CUTE_DEVICE
    get_bid() {
    #if defined(__CUDA_ARCH__)
            return blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    #else
            return 0;
    #endif
    }

    bool
    CUTE_DEVICE
    within_block_range(int threshold){
    #if defined(__CUDA_ARCH__)
        return blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y <= threshold;
    #else
        return true;
    #endif
    }

    bool
    CUTE_DEVICE
    outside_block_range(int threshold){
    #if defined(__CUDA_ARCH__)
        return blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y > threshold;
    #else
        return true;
    #endif
    }

    bool
    CUTE_DEVICE
    thread_within_range(int threshold){
    #if defined(__CUDA_ARCH__)
        return get_tid() <= threshold;
    #else
        return true;
    #endif
    }

    bool
    CUTE_DEVICE
    thread_strictly_within_range(uint threshold){
    #if defined(__CUDA_ARCH__)
            return get_tid() < threshold;
    #else
            return true;
    #endif
    }
}

#endif //ARISTOS_INDEXING_CUH
