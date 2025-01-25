//
// Created by osayamen on 8/17/24.
//

#ifndef CSRC_ATOMICS_CUH
#define CSRC_ATOMICS_CUH

#include "types.cuh"
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

    template<cuda::thread_scope scope = cuda::thread_scope_device>
    __device__ __forceinline__
    void fence() {
        if constexpr (scope == cuda::thread_scope_block) {
            __threadfence_block();
        }
        else if constexpr (scope == cuda::thread_scope_device) {
            __threadfence();
        }
        __threadfence_system();
    }


    // no fences are needed because the payload and signal comprise the atomically received data word
    template<cuda::thread_scope scope = cuda::thread_scope_device, typename Payload>
    requires(sizeof(Payload) == sizeof(unsigned long long int) && alignof(Payload) == alignof(unsigned long long int))
    __device__ __forceinline__
    void awaitPayload(unsigned long long int* const& addr, Payload* const& dest, const uint16_t& baton = 1U) {
        auto mail = atomicLoad<scope>(addr);
        auto* payload = CAST_TO(Payload, &mail);
        while (payload->signal != baton) {
            mail = atomicLoad<scope>(addr);
            payload = CAST_TO(Payload, &mail);
        }
        *dest = *payload;
    }

    template<
        cuda::thread_scope scope = cuda::thread_scope_device,
        typename Payload
    >
    requires(AtomicScope<scope>
        && sizeof(unsigned long long int) == sizeof(Payload) && alignof(Payload) == alignof(unsigned long long int))
    __device__ __forceinline__
    void signal(Payload* const& addr, Payload const& payload) {
        if constexpr (scope == cuda::thread_scope_block || scope == cuda::thread_scope_thread) {
            atomicExch_block(CAST_TO(unsigned long long int, addr), *CAST_TO(unsigned long long int, payload));
        }
        if constexpr (scope == cuda::thread_scope_system) {
            atomicExch_system(CAST_TO(unsigned long long int, addr), *CAST_TO(unsigned long long int, payload));
        }
        atomicExch(CAST_TO(unsigned long long int, addr), *CAST_TO(unsigned long long int, payload));
    }
}
#endif //CSRC_ATOMICS_CUH
