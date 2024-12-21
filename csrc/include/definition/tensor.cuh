//
// Created by Jonathan on 7/13/24.
//

#ifndef ARISTOS_TENSOR_CUH
#define ARISTOS_TENSOR_CUH

#include <cute/tensor.hpp>
#include <cuda/std/type_traits>
namespace aristos{
    template<typename V>
    concept TensorValueType = cuda::std::is_same_v<V, cute::half_t> ||
        cuda::std::is_same_v<V, cute::bfloat16_t> ||
        cuda::std::is_same_v<V, cute::tfloat32_t> ||
        cuda::std::is_same_v<V, float> ||
        cuda::std::is_same_v<V, cute::float_e4m3_t> ||
        cuda::std::is_same_v<V, cute::float_e5m2_t>;

    template<typename T>
    concept Tensor = cute::is_tensor<T>::value && TensorValueType<typename T::value_type>;

    template<typename M>
    concept Matrix = requires(M m){
        requires Tensor<M> && rank(m) == 2;
    };
}

#endif //ARISTOS_TENSOR_CUH
