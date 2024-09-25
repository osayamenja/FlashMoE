//
// Created by Jonathan on 7/13/24.
//

#ifndef ARISTOS_TENSOR_CUH
#define ARISTOS_TENSOR_CUH

#include <cute/tensor.hpp>

namespace aristos{
    template<typename V>
    concept TensorValueType = requires(V v){
        {V(0)} -> cuda::std::same_as<V>;
    };

    template<typename T>
    concept Tensor = requires(T t){
        requires cute::is_tensor<T>::value == true;
        requires TensorValueType<typename decltype(t)::value_type>;
    };

    template<typename M>
    concept Matrix = requires(M m){
        requires Tensor<M> && cute::rank<0>(m) == 1 && cute::rank<1>(m) == 1;
    };
}

#endif //ARISTOS_TENSOR_CUH
