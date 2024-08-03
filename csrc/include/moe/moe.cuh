//
// Created by Jonathan on 7/5/24.
//

#ifndef ARISTOS_MOE_CUH
#define ARISTOS_MOE_CUH

#include "../definition/tensor.cuh"
#include "../util/indexing.cuh"
#include "../definition/types.cuh"
#include <cuda/atomic>
#include <cuda/cmath>

__constant__ aristos::Config moeConfig{};

namespace aristos{
    __device__ cuda::atomic<bool, cuda::thread_scope_device> stop{};
    //TODO gate, expert fusion and control plane
    // TODO Figure out spec translation, but for now: spendthrift memory allocation
    template<Matrix M>
    CUTE_DEVICE
    void gate(M activations, M weights, M routing){
        // 1. TODO Compute Fused GEMM and Softmax output routing
        //2. Apply top_idx and return result
        in_place_fused_top_k_mask(activations);
    }

    template<Matrix M, Tensor T>
    CUTE_DEVICE
    void forward(M activations, T w_experts, M w_gate) {
        auto token_dim = cute::size<0>(activations.layout());
        extern __shared__ float workspace[];
        auto routing_tensor = cute::make_tensor(cute::make_smem_ptr(workspace),
                                                cute::make_shape(token_dim, moeConfig.numExperts));
        gate(activations, w_gate, routing_tensor);
        extern __shared__ uint scratch[];
        auto mapping_tensor = cute::make_tensor(cute::make_smem_ptr(scratch),
                                                cute::make_shape(token_dim, moeConfig.worldSize));
        auto shard_spec = cute::make_tensor(cute::make_gmem_ptr(moeConfig.shardSpec),
                                            cute::make_shape(moeConfig.worldSize, moeConfig.numExperts));
        get_token_to_peer_mapping(routing_tensor, shard_spec, mapping_tensor);
        //workspace is free for reuse
        // All blocks send slices to workers

        if (gridDim.z > 0) {
            // Posters and Receivers
        }
    }
}
#endif //ARISTOS_MOE_CUH
