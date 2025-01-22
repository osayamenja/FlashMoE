//
// Created by oja7 on 1/1/25.
//

#ifndef OS_CUH
#define OS_CUH

#include <cuda/std/cstddef>
#include "../types.cuh"

#include "packet.cuh"
#include "scheduler.cuh"
#include "subscriber.cuh"
#include "processor/processor.cuh"

namespace aristos::os {
    template<unsigned int processors>
    __device__ __forceinline__
    void start(cuda::std::byte* __restrict__ const& workspace) {
        constexpr auto subscriberCount = THREADS - 32;
        __shared__ Bookkeeping bk;
        if (!threadIdx.x) {
            bk = bookkeeping;
        }
        auto* __restrict__ tQHeads = CAST_TO(uint, workspace);
        auto* __restrict__ rQ = tQHeads + subscriberCount;
        auto* __restrict__ interrupt = rQ + processors;
        __syncthreads();
        // build arguments for scheduler and subscriber
        if (threadIdx.x / WARP_SIZE == 0) {
            // scheduler
            const auto gtQCl = bk.tM;
            const auto tQRl = cute::ceil_div(gtQCl * Config::tiles<BLOCK_N>(bk.pd),
                subscriberCount);
            auto* __restrict__ gtQHeads = bk.tQH();
            auto* __restrict__ sQ = bk.tQS();

        }
        else {
            // subscriber

        }
    }
}
#endif //OS_CUH
