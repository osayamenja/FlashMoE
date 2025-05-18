//
// Created by oja7 on 5/17/25.
//

#ifndef CORRECTNESS_CUH
#define CORRECTNESS_CUH
#include <cuda/std/array>
#include <matx.h>
#include "include/types.cuh"

namespace aristos {
    // reference expert
    template<
        unsigned int S,
        unsigned int H,
        unsigned int P,
        typename Element
    >
    __host__ __forceinline__
    void rExpert(Element* __restrict__ const& act,
        Element* __restrict__ const& gateWeights,
        Element* __restrict__ const& expertWeights,
        Element* __restrict__ const& bias,
        Element* __restrict__ const& gateOutput,
        Element* __restrict__ const& moeOutput,
        const unsigned int& nLx) {
        using T = typename ToCDx<Element>::T;
        auto activation = matx::make_tensor<T>(act, {S, H});
    }
}
#endif //CORRECTNESS_CUH
