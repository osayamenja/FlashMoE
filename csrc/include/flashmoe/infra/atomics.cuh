/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Flashmoe Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 */

//
// Created by osayamen on 8/17/24.
//

#ifndef CSRC_ATOMICS_CUH
#define CSRC_ATOMICS_CUH
#include <cuda/atomic>
#include <cuda/barrier>
namespace flashmoe{
    using ull_t = unsigned long long int;
    template<typename B>
    concept AtomicType = cuda::std::same_as<B, int> || cuda::std::same_as<B, unsigned int>
    || cuda::std::same_as<B, ull_t>;

    template<typename B>
    concept AtomicCASType = cuda::std::same_as<B, int> || cuda::std::same_as<B, unsigned int>
    || cuda::std::same_as<B, ull_t> || cuda::std::same_as<B, unsigned short int>;

    template<cuda::thread_scope scope>
    concept AtomicScope = scope == cuda::thread_scope_thread ||
        scope == cuda::thread_scope_block || scope == cuda::thread_scope_device || scope == cuda::thread_scope_system;


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

    enum InitState: int {
        stale = -1,
        initializing = 0,
        initialized = 1
    };
    // Set every byte to 0xFF → each int becomes 0xFFFFFFFF → -1
    // needed for cudaMemSet
    constexpr auto STALE_AS_BYTE= 0xFF;
    static_assert(cuda::std::is_same_v<cuda::std::underlying_type_t<InitState>, int>);
    static_assert(initialized == 1 && stale != initializing && initialized != initializing);

    // Enables in-kernel initialization of global accumulators
    __device__ __forceinline__
    auto guardedAtomicAdd(int* __restrict__ const& guard, int* __restrict__ const& vals, const int& val,
        const int& participants) {
        const cuda::atomic_ref<int, cuda::thread_scope_device> g{*guard};
        int expected = stale;
        if (g.compare_exchange_strong(expected, initializing, cuda::memory_order_acquire)) {
            // initialize with my value
            *vals = val;
            g.store(initialized, cuda::memory_order_release);
            return 0;
        }
        while (expected == initializing) {
            expected = g.load(cuda::memory_order_acquire);
        }
        if (g.fetch_add(1, cuda::memory_order_relaxed) + 1 == participants) {
            // everyone has arrived at this point, but I am the last.
            g.store(stale, cuda::memory_order_relaxed);
        }
        return atomicAdd(vals, val);
    }
}
#endif //CSRC_ATOMICS_CUH
