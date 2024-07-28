//
// Created by Jonathan on 7/5/24.
//

#ifndef ARISTOS_MOE_CUH
#define ARISTOS_MOE_CUH

#include "../definition/tensor.cuh"
#include "../util/indexing.cuh"
#include "../definition/types.cuh"
#include <cuda/cmath>

namespace aristos{
    __device__ cuda::atomic<bool, cuda::thread_scope_device> stop{false};
    //TODO gate, expert fusion and control plane
    /// MoE + Gate
    template<Matrix M, Tensor T>
    class FusedMoELayer{
        void* symmetric_heap;
        M shard_spec;
        M spec_translation;
        medium_int n_peers;
        medium_int capacity_factor;
        medium_int k;

    public:
        // TODO Figure out spec translation, but for now: spendthrift memory allocation
        CUTE_DEVICE
        FusedMoELayer(void* _symmetric_heap, M _shard_spec, M _spec_translation, medium_int _capacity_factor, medium_int _k):
        symmetric_heap(_symmetric_heap), shard_spec(_shard_spec),
        spec_translation(_spec_translation), capacity_factor(_capacity_factor), k(_k)
        {
            n_peers = cute::size<0>(spec_translation.layout());
        }

        uint_fast16_t capacity(uint token_dim, uint n_experts){
            return cuda::ceil_div(token_dim, n_experts) * capacity_factor * k;
        }

        void gate(T activations, T weights, T routing){
            // 1. TODO Compute Fused GEMM and Softmax output routing
            //2. Apply top_idx and return result
            in_place_fused_top_k_mask(activations);
        }

        CUTE_DEVICE
        void forward(M activations, T w_experts, M w_gate){
            auto token_dim = cute::size<0>(activations.layout());
            auto n_experts = cute::size<0>(shard_spec.layout());
            extern __shared__ float workspace[];
            auto routing_tensor = cute::make_tensor(cute::make_smem_ptr(workspace), cute::make_shape(token_dim, n_experts));
            gate(activations, w_gate, routing_tensor);
            extern __shared__ uint scratch[];
            auto mapping_tensor = cute::make_tensor(cute::make_smem_ptr(scratch), cute::make_shape(token_dim, n_peers));
            get_token_to_peer_mapping(routing_tensor, shard_spec, mapping_tensor);
            //workspace is free for reuse
            // All blocks send slices to workers

            if(within_block_range(0)){

            }

        }
    };
}
#endif //ARISTOS_MOE_CUH
