/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Kleos Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 */

//
// Created by oja7 on 5/17/25.
//

#ifndef CORRECTNESS_CUH
#define CORRECTNESS_CUH
#include <matx.h>
#include "include/kleos/types.cuh"

namespace kleos {
    // reference expert
    template<
        unsigned int S,
        unsigned int H,
        unsigned int P,
        unsigned int E,
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
        auto a = matx::make_tensor<Element>(act, {S, H});
        auto gW = matx::make_tensor<Element>(gateWeights, {H, E});
        auto gO = matx::make_tensor<Element>(gateOutput, {S, E});
        auto t0 = matx::make_tensor<Element>({});
        auto t0i = matx::make_tensor<matx::index_t>({});
        matx::cudaExecutor exec{kleosStream};
        // do Gate
        // 1) GEMM + Softmax
        (gO = matx::softmax(matx::matmul(a, gW), {1})).run(exec);
        (matx::mtie(t0, t0i) = matx::argmax(gO, {1})).run(exec);
    }
}
#endif //CORRECTNESS_CUH
