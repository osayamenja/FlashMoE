//
// Created by Jonathan on 7/13/24.
//

#ifndef ARISTOS_ALGORITHM_CUH
#define ARISTOS_ALGORITHM_CUH

// Heap things
#include <cuda/std/__algorithm/make_heap.h>
#include <cuda/std/__algorithm/pop_heap.h>
#include <cuda/std/concepts>
#include "../definition/tensor.cuh"
#include "../util/indexing.cuh"

//TODO dimensions for all params
namespace aristos {
    /// Complexity: O(2k + (n-k)*log(k))
    template<Matrix T, unsigned int k=1>
    CUTE_DEVICE
    void in_place_fused_top_k_mask(T input){
        if(grid_tid() < cute::size<0>(input.layout())){
            using MatrixType = typename decltype(input)::value_type;
            cute::array<cuda::std::pair<MatrixType, uint>, k> window{};
            auto my_slice = input(grid_tid(), cute::_);
            CUTE_UNROLL
            for(uint i = 0; i < k; ++i){
                window[i] = {my_slice(i), i};
            }
            cuda::std::make_heap(window.begin(), window.end(), cuda::std::greater<>{});
            cuda::std::pop_heap(window.begin(), window.end(), cuda::std::greater<>{});
            // min element now at the end of the array
            CUTE_UNROLL
            for(uint i = k; i < cute::size(my_slice); ++i){
                auto min_elem = window.back();
                if(cuda::std::pair<MatrixType, uint>{my_slice(i), i} > min_elem){
                    my_slice(min_elem.second) = MatrixType(0);
                    window[k - 1] = cuda::std::pair<MatrixType, uint>{my_slice(i), i};
                    cuda::std::push_heap(window.begin(), window.end(), cuda::std::greater<>{});
                    cuda::std::pop_heap(window.begin(), window.end(), cuda::std::greater<>{});
                }
                else{
                    my_slice(i) = MatrixType(0);
                }
            }
        }
    }
    /// Akin to scatter
    template<Matrix T>
    CUTE_DEVICE
    void get_token_to_peer_mapping(T gate_routing, T shard_spec, T mapping){
        auto index = grid_tid();
        if(index < cute::size<0>(gate_routing.layout())){
            using MatrixType = typename decltype(gate_routing)::value_type;
            auto in_slice = gate_routing(index, cute::_);
            auto out_slice = mapping(index, cute::_);
            auto zero_val = MatrixType(0);
            const medium_int n_peers = cute::size<0>(mapping.layout());
            cute::array<medium_int, n_peers> k_indices;
            k_indices.clear(); // may be redundant
            CUTE_UNROLL
            for(uint i = 0; i < cute::size<0>(in_slice.layout()); ++i){
                if(in_slice(i) > zero_val){
                    auto peer = shard_spec(i);
                    out_slice(peer, k_indices[i]++) = i; // The value i denotes the expert index.
                }
            }
        }
    }
}

#endif //ARISTOS_ALGORITHM_CUH
