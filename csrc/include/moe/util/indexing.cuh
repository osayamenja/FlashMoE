//
// Created by Jonathan on 7/13/24.
//

#ifndef ARISTOS_INDEXING_CUH
#define ARISTOS_INDEXING_CUH

#include "../../../../../../../../../../global/homes/o/osayamen/.local/cutlass/include/cute/config.hpp"
#include "../../../cmake-build-debug/_deps/cccl-src/libcudacxx/include/cuda/cmath"

namespace aristos{
    /// Block-scoped thread id
    decltype(auto)
    CUTE_HOST_DEVICE
    block_tid() {
    #if defined(__CUDA_ARCH__)
            return threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    #else
            return 0;
    #endif
    }

    /// Grid-scoped thread id
    decltype(auto)
    CUTE_HOST_DEVICE
    grid_tid() {
    #if defined(__CUDA_ARCH__)
        auto blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
        return blockId * (blockDim.x * blockDim.y * blockDim.z)
                        + (threadIdx.z * (blockDim.x * blockDim.y))
                        + (threadIdx.y * blockDim.x)
                        + threadIdx.x;
    #else
        return 0;
    #endif
    }

    decltype(auto)
    CUTE_HOST_DEVICE
    bid() {
    #if defined(__CUDA_ARCH__)
            return blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    #else
            return 0;
    #endif
    }
}

#endif //ARISTOS_INDEXING_CUH
