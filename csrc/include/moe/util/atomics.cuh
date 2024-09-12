//
// Created by osayamen on 8/17/24.
//

#ifndef CSRC_ATOMICS_CUH
#define CSRC_ATOMICS_CUH

#include <cuda/std/concepts>
#include <cuda/annotated_ptr>

namespace aristos{
    template<typename B>
    concept AtomicType = requires(B b){
        requires cuda::std::same_as<B, int> ||
                cuda::std::same_as<B, unsigned int> || cuda::std::same_as<B, unsigned long long int>;
    };

    /// Empirical benchmarks shows atomicCAS is the fastest at scale
    template<AtomicType T>
    CUTE_DEVICE
    T atomicLoad(T* addr){
        return atomicCAS(addr, 0U, 0U);
    }
}

namespace aristos::barrier{
    __device__ unsigned int barrierStages = 0;
    __device__ unsigned int counter = 0;
    __device__ unsigned int numProcesses;

    CUTE_DEVICE void init(const unsigned int& n, bool shouldPersist = true){
        if(shouldPersist){
            cuda::associate_access_property(&barrierStages, cuda::access_property::persisting{});
            cuda::associate_access_property(&counter, cuda::access_property::persisting{});
            cuda::associate_access_property(&numProcesses, cuda::access_property::persisting{});
        }
        atomicExch(&numProcesses, n);
    }

    CUTE_DEVICE unsigned int arrive(){
        return aristos::atomicLoad(&barrierStages) + 1;
    }

    CUTE_DEVICE void wait(const unsigned int& token){
        if((atomicAdd(&counter, 1U) + 1) == numProcesses * token){
            atomicAdd(&barrierStages, 1U);
        }
        else{
            while(aristos::atomicLoad(&barrierStages) != token){
                __nanosleep(2);
            }
        }
    }
}
#endif //CSRC_ATOMICS_CUH
