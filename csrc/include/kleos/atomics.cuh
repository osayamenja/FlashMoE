/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Kleos Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 */

//
// Created by osayamen on 8/17/24.
//

#ifndef CSRC_ATOMICS_CUH
#define CSRC_ATOMICS_CUH

#include "types.cuh"

namespace kleos{
    template<typename B>
    concept AtomicType = cuda::std::same_as<B, int> || cuda::std::same_as<B, unsigned int>
    || cuda::std::same_as<B, ull_t>;

    template<typename B>
    concept AtomicCASType = cuda::std::same_as<B, int> || cuda::std::same_as<B, unsigned int>
    || cuda::std::same_as<B, ull_t> || cuda::std::same_as<B, unsigned short int>;

    template<cuda::thread_scope scope>
    concept AtomicScope = scope == cuda::thread_scope_thread ||
        scope == cuda::thread_scope_block || scope == cuda::thread_scope_device || scope == cuda::thread_scope_system;

    template<cuda::thread_scope scope = cuda::thread_scope_device, typename T>
    requires AtomicType<T> && AtomicScope<scope>
    __device__ __forceinline__
    auto atomicLoad(T* __restrict__ const& addr){
        if constexpr (scope == cuda::thread_scope_block || scope == cuda::thread_scope_thread) {
            return atomicOr_block(addr, 0U);
        }
        else if constexpr (scope == cuda::thread_scope_system) {
            return atomicOr_system(addr, 0U);
        }
        else {
            return atomicOr(addr, 0U);
        }
    }

    template<cuda::thread_scope scope = cuda::thread_scope_device,
    unsigned int bound = cuda::std::numeric_limits<unsigned int>::max()>
    requires(AtomicScope<scope> && bound <= cuda::std::numeric_limits<unsigned int>::max())
    __device__ __forceinline__
    unsigned int atomicIncrement(unsigned int* __restrict__ const& addr) {
        if constexpr (scope == cuda::thread_scope_block || scope == cuda::thread_scope_thread) {
            return atomicInc_block(addr, bound);
        }
        else if constexpr (scope == cuda::thread_scope_system) {
            return atomicInc_system(addr, bound);
        }
        else {
            return atomicInc(addr, bound);
        }
    }

    // Atomic Test and set
    template<cuda::thread_scope scope = cuda::thread_scope_device, typename T>
    requires AtomicCASType<T> && AtomicScope<scope> &&
        (!cuda::std::is_same_v<T, unsigned short int> || scope == cuda::thread_scope_device)
    __device__ __forceinline__
    T atomicTAS(T* __restrict__ const& addr) {
        if constexpr (scope == cuda::thread_scope_block || scope == cuda::thread_scope_thread) {
            return atomicCAS_block(addr, 0U, 1U);
        }
        else if constexpr (scope == cuda::thread_scope_system) {
            return atomicCAS_system(addr, 0U, 1U);
        }
        else {
            return atomicCAS(addr, 0U, 1U);
        }
    }

    template<cuda::thread_scope scope = cuda::thread_scope_device>
    requires(AtomicScope<scope>)
    __device__ __forceinline__
    void fence() {
        if constexpr (scope == cuda::thread_scope_block) {
            __threadfence_block();
        }
        else if constexpr (scope == cuda::thread_scope_device) {
            __threadfence();
        }
        else {
            __threadfence_system();
        }
    }


    /// Expected Signal is known
    template<cuda::thread_scope scope = cuda::thread_scope_device,typename Payload, typename T>
    requires(sizeof(Payload) == sizeof(ull_t) && alignof(Payload) == alignof(ull_t))
    __device__ __forceinline__
    void awaitPayload(Payload* __restrict__ const& addr, Payload* __restrict__ const& dest, const T& expected = 1U) {
        static_assert(cuda::std::is_same_v<decltype(dest->signal), T>, "Signal types should be the same!");
        auto mail = atomicLoad<scope>(CAST_TO(ull_t, addr));
        auto* __restrict__ payload = CAST_TO(Payload, &mail);
        while (payload->signal != expected) {
            mail = atomicLoad<scope>(CAST_TO(ull_t, addr));
            payload = CAST_TO(Payload, &mail);
        }
        *dest = *payload;
    }

    template<cuda::thread_scope scope = cuda::thread_scope_device, typename Notification, typename T>
    requires(sizeof(Notification) == sizeof(ull_t) && alignof(Notification) == alignof(ull_t))
    __device__ __forceinline__
    void awaitNotification(Notification* __restrict__ const& addr, Notification* __restrict__ const& dest,
        const T& previous) {
        static_assert(cuda::std::is_same_v<decltype(dest->signal), T>, "Signal types should be the same!");
        auto mail = atomicLoad<scope>(CAST_TO(ull_t, addr));
        auto* __restrict__ payload = CAST_TO(Notification, &mail);
        while (!payload->interrupt && payload->signal == previous) {
            mail = atomicLoad<scope>(CAST_TO(ull_t, addr));
            payload = CAST_TO(Notification, &mail);
        }
        *dest = *payload;
    }

    template<cuda::thread_scope scope = cuda::thread_scope_device, typename Notification, typename T>
    requires(sizeof(Notification) == sizeof(ull_t) && alignof(Notification) == alignof(ull_t))
    __device__ __forceinline__
    void awaitBarrier(Notification* __restrict__ const& addr, Notification* __restrict__ const& dest,
        const T& token) {
        static_assert(cuda::std::is_same_v<decltype(addr->signal), T>, "Signal types should be the same!");
        auto mail = atomicLoad<scope>(CAST_TO(ull_t, addr));
        auto* __restrict__ payload = CAST_TO(Notification, &mail);
        while (payload->signal < token) {
            mail = atomicLoad<scope>(CAST_TO(ull_t, addr));
            payload = CAST_TO(Notification, &mail);
        }
        *dest = *payload;
    }

    template<
        cuda::thread_scope scope = cuda::thread_scope_device,
        typename Payload
    >
    requires(AtomicScope<scope>
        && sizeof(ull_t) == sizeof(Payload) && alignof(Payload) == alignof(ull_t))
    __device__ __forceinline__
    void signalPayload(Payload* __restrict__ const& addr, const Payload* __restrict__ const& payload) {
        if constexpr (scope == cuda::thread_scope_block || scope == cuda::thread_scope_thread) {
            atomicExch_block(CAST_TO(ull_t, addr), *CONST_CAST_TO(ull_t, payload));
        }
        else if constexpr (scope == cuda::thread_scope_system) {
            atomicExch_system(CAST_TO(ull_t, addr), *CONST_CAST_TO(ull_t, payload));
        }
        else {
            atomicExch(CAST_TO(ull_t, addr), *CONST_CAST_TO(ull_t, payload));
        }
    }

    __device__ __forceinline__
    void gridBarrier() {
        __syncthreads();
        if (!threadIdx.x) {
            __threadfence();
            bookkeeping.dB()->arrive_and_wait();
        }
        __syncthreads();
    }
}
#endif //CSRC_ATOMICS_CUH
