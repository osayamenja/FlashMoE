/******************************************************************************
 * Copyright (c) 2024, Osayamen Jonathan Aimuyo.
 ******************************************************************************/

#ifndef ARISTOS_MOE_CUH
#define ARISTOS_MOE_CUH

#include "gate.cuh"

namespace aristos::moe{
    template<Matrix M, Tensor T>
    __global__ __maxnreg__(128) void forward(
        M const& activations,
        T const& expertsWeights,
        M const& gateWeights,
        M gateOutput) {
    }

    template<Matrix M, Tensor T>
    __global__ __maxnreg__(128) void backward(){

    }
}
#endif //ARISTOS_MOE_CUH
