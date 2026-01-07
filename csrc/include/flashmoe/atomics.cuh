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
        stale = 0,
        initializing = 1,
        initialized = 2
    };

    __device__ __forceinline__
    void tryGuardInit(int* __restrict__ const& guards, int* __restrict__ const& vals) {
        cuda::atomic_ref<int, cuda::thread_scope_device> g{*guards};
        if (int expected = stale;
            g.compare_exchange_strong(expected, initializing, cuda::memory_order_relaxed)) {
            *vals = 0;
            g.store(initialized, cuda::memory_order_release);
        }
    }
    // Enables in-kernel initialization of global accumulators
    __device__ __forceinline__
    auto guardedAtomicAdd(int* __restrict__ const& guard, int* __restrict__ const& vals, const int& val) {
        const cuda::atomic_ref<int, cuda::thread_scope_device> g{*guard};
        int expected = stale;
        if (g.compare_exchange_strong(expected, initializing, cuda::memory_order_acquire)) {
            // initialize with my value
            *vals = val;
            g.store(initialized, cuda::memory_order_release);
            return 0;
        }
        while (expected != initialized) {
            expected = g.load(cuda::memory_order_acquire);
        }
        return atomicAdd(vals, val);
    }

    template<int threads>
    __device__ __forceinline__
    void clearGuardsCoop(int* __restrict__ guards, const int& guardsLength, int* __restrict__ const& checkpoint,
        const int& blocks, int* __restrict__ const& isLastBlock) {
        __syncthreads();
        if (!threadIdx.x) {
            const cuda::atomic_ref<int, cuda::thread_scope_device> c{*checkpoint};
            *isLastBlock = c.fetch_add(1, cuda::memory_order_acq_rel) + 1 == blocks;
        }
        __syncthreads();
        if (*isLastBlock) {
            for (int i = static_cast<int>(threadIdx.x); i < guardsLength; i += threads) {
                guards[i] = 0;
            }
        }
    }
}
#endif //CSRC_ATOMICS_CUH
