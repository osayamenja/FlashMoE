//
// Created by Jonathan on 7/13/24.
//

#ifndef ARISTOS_TENSOR_CUH
#define ARISTOS_TENSOR_CUH

#include "../../../../../../../../../../opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/targets/x86_64-linux/include/cuda/std/concepts"
#include "../../../../../../../../../../global/homes/o/osayamen/.local/cutlass/include/cute/tensor.hpp"

namespace aristos{
    template<typename V>
    concept TensorValueType = requires(V v){
        {V(0)} -> std::same_as<V>;
    };

    template<typename T>
    concept Tensor = requires(T t){
        cute::is_tensor<T>::value == true;
        requires TensorValueType<typename decltype(t)::value_type>;
    };

    template<typename M>
    concept Matrix = requires(M m){
        requires Tensor<M>;
        cute::rank(m.layout()) == 2;
    };
}

#endif //ARISTOS_TENSOR_CUH
