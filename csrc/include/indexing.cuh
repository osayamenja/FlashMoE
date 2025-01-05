//
// Created by Jonathan on 7/13/24.
//

#ifndef ARISTOS_INDEXING_CUH
#define ARISTOS_INDEXING_CUH

namespace aristos{
    namespace block{
        /// Block-scoped thread id
        decltype(auto)
        __device__ __forceinline__
        threadID() {
            return threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
        }

        /// Block-scoped warp id
        decltype(auto)
        __device__ __forceinline__
        warpID() {
            return (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) >> 5;
        }
    }
    namespace warp{
        /// Block-scoped and warp-scoped thread id
        decltype(auto)
        __device__ __forceinline__
        laneID() {
            return (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) % 32;
        }
    }
    namespace grid{
        /// Grid-scoped thread id
        decltype(auto)
        __device__ __forceinline__
        threadID() {
            return (blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z)
                        * (blockDim.x * blockDim.y * blockDim.z)
                        + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
        }

        decltype(auto)
        __device__ __forceinline__
        blockID() {
            return blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
        }
    }
}

#endif //ARISTOS_INDEXING_CUH
