//
// Created by Jonathan on 7/13/24.
//

#ifndef ARISTOS_GATE_CUH
#define ARISTOS_GATE_CUH

#include "../util/tensor.cuh"
#include "../algorithm/algorithm.cuh"

namespace aristos {
    template<Matrix T>
    CUTE_DEVICE
    void gate(T activations, T weights, T routing){
        // 1. TODO Compute Fused GEMM and Softmax output routing
        //2. Apply top_idx and return result
        in_place_fused_top_k_mask(activations);
    }
}
#endif //ARISTOS_GATE_CUH
