//
// Created by oja7 on 1/1/25.
//

#ifndef OS_CUH
#define OS_CUH

#include <cuda/std/cstddef>

#include "packet.cuh"
#include "scheduler.cuh"
#include "subscriber.cuh"
#include "observer.cuh"
#include "processor/processor.cuh"

namespace aristos::os {
    template<unsigned int blocks>
    __device__ __forceinline__
    void start(cuda::std::byte* __restrict__ const& workspace) {
        if (!threadIdx.x) {
            *CAST_TO(unsigned int, workspace) = moeConfig.nTiles;
        }
        __syncthreads();
        if (!threadIdx.x) {
            // last thread
            scheduler::start<blocks>(CAST_TO(unsigned int, workspace));
        }
        else {
            // first n threads
            subscriber::start(CAST_TO(unsigned int, workspace));
        }
    }
}
#endif //OS_CUH
