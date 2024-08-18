//
// Created by osayamen on 8/17/24.
//

#ifndef CSRC_ATOMICS_CUH
#define CSRC_ATOMICS_CUH

#include <cuda/std/concepts>
namespace aristos{
    template<typename B>
    concept AtomicBitWiseType = requires(B b){
        requires cuda::std::same_as<B, int> ||
                cuda::std::same_as<B, unsigned int> || cuda::std::same_as<B, unsigned long long int>;
    };

    template<typename F>
    concept AtomicFloatType = requires(F f){
        requires cuda::std::same_as<F, float> || cuda::std::same_as<F, double> ||
                cuda::std::same_as<F, __half> || cuda::std::same_as<F, nv_bfloat16>;
    };

    template<AtomicBitWiseType B>
    CUTE_DEVICE
    B atomicLoad(B* addr){
        return atomicOr(addr, B(0));
    }

    template<AtomicFloatType F>
    CUTE_DEVICE
    F atomicFloatLoad(F* addr){
        return atomicAdd(addr, F(0.0));
    }
}
#endif //CSRC_ATOMICS_CUH
