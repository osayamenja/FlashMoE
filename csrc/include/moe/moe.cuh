/******************************************************************************
 * Copyright (c) 2024, Osayamen Jonathan Aimuyo.
 ******************************************************************************/

#ifndef ARISTOS_MOE_CUH
#define ARISTOS_MOE_CUH

#include "../definition/tensor.cuh"
#include "../algorithm/algorithm.cuh"

namespace aristos{
    template<Matrix M>
    CUTE_DEVICE
    void gate(M const& activations, M const& weights, M routing){
        // 1. TODO Compute Fused GEMM and Softmax output routing
        //2. Apply top_idx and return result
        topKMask(activations);
    }

    template<Matrix M, Tensor T>
    __global__ __maxnreg__(128) void forward(
        M const& activations,
        T const& expertsWeights,
        M const& gateWeights,
        M gateOutput) {
        gate(activations, gateWeights, gateOutput);
    }

    template<Matrix M, Tensor T>
    __global__ __maxnreg__(128) void backward(){

    }
}
#endif //ARISTOS_MOE_CUH
