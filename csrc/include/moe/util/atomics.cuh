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
                cuda::std::same_as<B, unsigned int> || cuda::std::same_as<B, unsigned long long int>;
    };

    /// Empirical benchmarks shows atomicOr is the fastest
    template<AtomicType T>
    CUTE_DEVICE
    T atomicLoad(T* addr){
        return atomicOr(addr, 0U);
    }
}
namespace aristos::barrier{

}
#endif //CSRC_ATOMICS_CUH
