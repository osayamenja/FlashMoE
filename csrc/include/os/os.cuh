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
        constexpr auto subscriberCount = THREADS - 2;
        __shared__ Config mC;
        __shared__ SchedulerConfig sC;

        // build arguments for scheduler, subscriber and observer
    }
}
#endif //OS_CUH
