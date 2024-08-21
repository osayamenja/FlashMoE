//
// Created by osayamen on 8/20/24.
//

#ifndef CSRC_VALUES_CUH
#define CSRC_VALUES_CUH

#include "packet.cuh"
#include "types.cuh"

namespace aristos{
    __constant__ Config moeConfig{};
    /// len = |E|
    __device__ specType* expertParallelSpec;
    /// len <= |D|
    __device__ specType* peerTranslation;

    __device__ AtomicBoolType stillExecuting = 1;
    __device__ unsigned long sequenceNumber = aristos::header::begin + 1;
}
#endif //CSRC_VALUES_CUH
