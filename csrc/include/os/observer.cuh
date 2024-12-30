//
// Created by oja7 on 12/26/24.
//

#ifndef OBSERVER_CUH
#define OBSERVER_CUH

#include "../definition/types.cuh"

namespace aristos::workQ {
    /// workQ Observer
    __device__ __forceinline__
    void observe(unsigned int* __restrict__ wQHead,
        unsigned int* __restrict__ wQTail,
        uint2* __restrict__ wQ) {
        auto* __restrict__ globalTQ = schedulerState.taskQ;
    }
}

#endif //OBSERVER_CUH
