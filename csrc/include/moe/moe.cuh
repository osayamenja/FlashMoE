//
// Created by Jonathan on 7/5/24.
//

#ifndef ARISTOS_MOE_CUH
#define ARISTOS_MOE_CUH

#include "../util/tensor.cuh"
#include "gate.cuh"

namespace aristos{
    //TODO gate, expert fusion and control plane
    template<Matrix M, Tensor T>
    CUTE_DEVICE
    void FusedMoE(M activations, T w_experts, M w_gate, M shard_spec, uint_fast16_t n_peers){
        auto token_dim = cute::size<0>(activations.layout());
        auto n_experts = cute::size<0>(shard_spec.layout());
        extern __shared__ float routing[];
        auto routing_tensor = cute::make_tensor(cute::make_smem_ptr(routing), cute::make_shape(token_dim, n_experts));
        gate(activations, w_gate, routing_tensor);
        extern __shared__ uint_fast16_t mapping[]; // assumes n_peers in an ep_group <= UINT16_MAX = (64k - 1)
        auto mapping_tensor = cute::make_tensor(cute::make_smem_ptr(mapping), cute::make_shape(token_dim, n_peers));
        get_token_to_peer_mapping(routing_tensor, shard_spec, mapping_tensor);

        // send slices to workers
        
    }
}
#endif //ARISTOS_MOE_CUH
