//
// Created by oja7 on 3/3/25.
//

#ifndef SYNC_CUH
#define SYNC_CUH
#include <nvshmem.h>

#include "../atomics.cuh"
#include "../types.cuh"
namespace aristos {
    // kernel to sync among a subset of the global group
    //
    template<
        unsigned int threads = ACC::PeakHardware::OS::threads::value
    >
    __global__ void syncAll(PLI* __restrict__ const __grid_constant__ peers,
        const __grid_constant__ uint rank,
        const __grid_constant__ uint world) {
        // Use 𝛼 synchronization algorithm
        // Notify arrival
        static_assert(sizeof(decltype(bookkeeping.syncCount)) == sizeof(ull_t) &&
            alignof(decltype(bookkeeping.syncCount)) == alignof(ull_t));
        const auto sC = __ldg(bookkeeping.syncCount);
        for (uint i = threadIdx.x; i < world; i += threads) {
            // scramble the peer id to potentially mitigate congestion
            const auto pLI = peers[(i + rank) % world];
            nvshmem_uint64_atomic_inc(bookkeeping.syncArray + rank, pLI.pe);
        }
        // Await notification
        auto* __restrict__ sA = CAST_TO(ull_t, bookkeeping.syncArray);
        for (uint i = threadIdx.x; i < world; i += threads) {
            while (atomicLoad(sA + i) < sC) {}
        }
        __syncthreads();
        if (!threadIdx.x) {
            *bookkeeping.syncCount = sC + 1;
        }
    }
}
#endif //SYNC_CUH
