//
// Created by osayamen on 8/17/24.
//

#ifndef CSRC_ATOMICS_CUH
#define CSRC_ATOMICS_CUH

#include <cuda/annotated_ptr>
#include <thrust/system/cuda/error.h>
#define USE_BARRIER 0

namespace aristos{
    template<typename B>
    concept AtomicType = cuda::std::same_as<B, int> || cuda::std::same_as<B, unsigned int>
    || cuda::std::same_as<B, unsigned long long int>;

    template<typename B>
    concept AtomicCASType = cuda::std::same_as<B, int> || cuda::std::same_as<B, unsigned int>
    || cuda::std::same_as<B, unsigned long long int> || cuda::std::same_as<B, unsigned short int>;

    template<cuda::thread_scope scope>
    concept AtomicScope = scope == cuda::thread_scope_thread ||
        scope == cuda::thread_scope_block || scope == cuda::thread_scope_device || scope == cuda::thread_scope_system;

    template<cuda::thread_scope scope = cuda::thread_scope_device, typename T>
    requires AtomicType<T> && AtomicScope<scope>
    __device__ __forceinline__
    T atomicLoad(T* const& addr){
        if constexpr (scope == cuda::thread_scope_block || scope == cuda::thread_scope_thread) {
            return atomicOr_block(addr, 0U);
        }
        if constexpr (scope == cuda::thread_scope_system) {
            return atomicOr_system(addr, 0U);
        }
        return atomicOr(addr, 0U);
    }

    template<cuda::thread_scope scope = cuda::thread_scope_device,
    unsigned int bound = cuda::std::numeric_limits<unsigned int>::max()>
    requires(AtomicScope<scope> && bound <= cuda::std::numeric_limits<unsigned int>::max())
    __device__ __forceinline__
    unsigned int atomicIncrement(unsigned int* const& addr) {
        if constexpr (scope == cuda::thread_scope_block || scope == cuda::thread_scope_thread) {
            return atomicInc_block(addr, bound);
        }
        if constexpr (scope == cuda::thread_scope_system) {
            return atomicInc_system(addr, bound);
        }
        return atomicInc(addr, bound);
    }

    // Atomic Test and set
    template<cuda::thread_scope scope = cuda::thread_scope_device, typename T>
    requires AtomicCASType<T> && AtomicScope<scope> &&
        (!cuda::std::is_same_v<T, unsigned short int> || scope == cuda::thread_scope_device)
    __device__ __forceinline__
    T atomicTAS(T* const& addr) {
        if constexpr (scope == cuda::thread_scope_block || scope == cuda::thread_scope_thread) {
            return atomicCAS_block(addr, 0U, 1U);
        }
        if constexpr (scope == cuda::thread_scope_system) {
            return atomicCAS_system(addr, 0U, 1U);
        }
        return atomicCAS(addr, 0U, 1U);
    }


    namespace ring {
        //TODO use partially specialized struct for this
        // Ring-based polling.
        template<cuda::thread_scope scope = cuda::thread_scope_device, typename T>
        __device__ __forceinline__
        void awaitTurn(T* const& addr, const T& baton = static_cast<T>(1U)) {
            while (atomicLoad<scope>(addr) != baton){}
        }

        // non-blocking await routine
        template<cuda::thread_scope scope = cuda::thread_scope_device, typename T>
        __device__ __forceinline__
        bool tryAwait(T* const& addr, const T& baton = static_cast<T>(1U)) {
            return atomicLoad<scope>(addr) == baton;
        }

        template<cuda::thread_scope scope = cuda::thread_scope_device, typename T>
        requires AtomicScope<scope> && AtomicType<T>
        __device__ __forceinline__
        void signal(T* const& addr) {
            if constexpr (scope == cuda::thread_scope_block || scope == cuda::thread_scope_thread) {
                atomicAdd_block(addr, 1U);
            }
            if constexpr (scope == cuda::thread_scope_system) {
                atomicAdd_system(addr, 1U);
            }
            atomicAdd(addr, 1U);
        }
    }

}
#endif //CSRC_ATOMICS_CUH
