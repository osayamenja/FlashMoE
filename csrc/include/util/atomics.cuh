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

    template<cuda::thread_scope scope = cuda::thread_scope_device, typename T> requires AtomicType<T>
    CUTE_DEVICE
    T atomicLoad(T* addr){
        if constexpr (scope == cuda::thread_scope_block || scope == cuda::thread_scope_thread) {
            return atomicOr_block(addr, 0U);
        }
        if constexpr (scope == cuda::thread_scope_system) {
            return atomicOr_system(addr, 0U);
        }
        return atomicOr(addr, 0U);
    }

    namespace ring {
        //TODO use partially specialized struct for this
        // Ring-based polling.
        template<cuda::thread_scope scope = cuda::thread_scope_device, typename T>
        __device__ __forceinline__
        void awaitTurn(T* addr, const T& baton) {
            while (atomicLoad<scope>(addr) != baton){}
        }

        template<cuda::thread_scope scope = cuda::thread_scope_device, typename T>
        __device__ __forceinline__
        void signal(T* addr) {
            if constexpr (scope == cuda::thread_scope_block || scope == cuda::thread_scope_thread) {
                return atomicAdd_block(addr, 0U);
            }
            if constexpr (scope == cuda::thread_scope_system) {
                return atomicAdd_system(addr, 0U);
            }
            return atomicAdd(addr, 0U);
        }
    }

}
#endif //CSRC_ATOMICS_CUH
