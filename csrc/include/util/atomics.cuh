//
// Created by osayamen on 8/17/24.
//

#ifndef CSRC_ATOMICS_CUH
#define CSRC_ATOMICS_CUH

#include <cuda/annotated_ptr>
#define USE_BARRIER 0

namespace aristos{
    template<typename B>
    concept AtomicType = cuda::std::same_as<B, int> || cuda::std::same_as<B, unsigned int>
    || cuda::std::same_as<B, unsigned long long int>;

    template<typename T> requires AtomicType<T>
    CUTE_DEVICE
    T atomicLoad(T* addr){
        return atomicOr(addr, 0U);
    }
}
#endif //CSRC_ATOMICS_CUH
