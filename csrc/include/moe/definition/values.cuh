//
// Created by osayamen on 8/20/24.
//

#ifndef CSRC_VALUES_CUH
#define CSRC_VALUES_CUH

#include "types.cuh"

namespace aristos{
    __constant__ Config moeConfig{};
    __device__ AtomicBoolType stillExecuting = 1;
    Config hostMoEConfig;
}
#endif //CSRC_VALUES_CUH
