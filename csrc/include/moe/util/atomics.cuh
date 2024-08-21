//
// Created by osayamen on 8/17/24.
//

#ifndef CSRC_ATOMICS_CUH
#define CSRC_ATOMICS_CUH

#include <cuda/std/concepts>
namespace aristos{
    template<typename B>
    concept AtomicType = requires(B b){
        requires cuda::std::same_as<B, int> ||
                cuda::std::same_as<B, unsigned int> || cuda::std::same_as<B, unsigned long long int> ||
                cuda::std::same_as<B, unsigned short int>;
    };

    template<AtomicType T>
    CUTE_DEVICE
    T atomicLoad(T* addr){
        return atomicCAS(addr, 0U, 0U);
    }
}
#endif //CSRC_ATOMICS_CUH
