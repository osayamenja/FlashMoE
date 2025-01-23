//
// Created by oja7 on 1/1/25.
//

#ifndef OS_CUH
#define OS_CUH

#include <cuda/std/cstddef>
#include "../types.cuh"

#include "scheduler.cuh"
#include "subscriber.cuh"

namespace aristos::os {
    template<
        unsigned int processors,
        DropTokens d = DropTokens::yes,
        typename ExpertsTensor,
        typename BiasTensor,
        typename Activations
    >
    __device__ __forceinline__
    void start(cuda::std::byte* __restrict__ const& workspace,
        Activations const& activations,
        ExpertsTensor const& experts,
        BiasTensor const& biasT) {
        // each subscriber thread gets 64 bytes of workspace
        constexpr auto wSet = 16U; // working set size
        constexpr auto subscriberCount = THREADS - 32;
        auto* __restrict__ scratch = workspace + THREADS * wSet * sizeof(uint);
        __shared__ Bookkeeping bk;
        if (!threadIdx.x) {
            bk = bookkeeping;
        }
        const auto nx = bk.nx;
        const auto* __restrict__ eC = bk.eC();
        const auto* __restrict__ gPT = bk.pT();
        const auto* __restrict__ eD = bk.eD();
        const auto nRx = *bk.nRx();
        const auto tNx = Config::tiles<BLOCK_N>(bk.pd);
        const auto tN = bk.tN;
        const auto eCap = bk.eCap;
        const auto world = bk.world;
        // shared memory arrays
        // Upper bound for expectant tasks
        auto*  __restrict__ taskBound = CAST_TO(uint, scratch);
        const auto* __restrict__ eCs = taskBound + 1;
        for (uint i = threadIdx.x; i < nx; i += THREADS) {
            (taskBound + 1)[i] = __ldg(eC + i);
        }
        __syncthreads();
        // compute taskBound
        for (uint i = threadIdx.x; i < nx; i += THREADS) {
            const auto eCt = Config::tiles<BLOCK_M>(d == DropTokens::yes ? cute::min(eCs[i], eCap)
                : eCs[i]);
            atomicAdd_block(taskBound, eCt * tN);
            #pragma unroll 4
            for (uint j = 0; j < world; ++j) {
                atomicAdd_block(taskBound, eCt * tNx);
            }
        }
        __syncthreads();
        auto* __restrict__ tQHeads = CAST_TO(uint, taskBound + 1);
        auto* __restrict__ rQ = tQHeads + subscriberCount;
        #pragma unroll
        for (uint i = threadIdx.x; i < processors; i += THREADS) {
            rQ[i] = i; // initially, all processors are ready
        }
        auto* __restrict__ interrupt = rQ + processors;
        #pragma unroll
        for (uint i = threadIdx.x; i < SUBSCRIBERS; i += THREADS) {
            interrupt[i] = 0U;
        }
        auto* __restrict__ status = interrupt + SUBSCRIBERS;
        auto* __restrict__ pT = status + world;
        for (uint i = threadIdx.x; i < world; i += THREADS) {
            pT[i] = __ldg(gPT + i);
            status[i] = 0U;
        }
        auto* __restrict__ xD = static_cast<cuda::std::tuple<uint, uint, uint>*>(static_cast<void*>(pT + world));
        for (uint i = threadIdx.x; i < nx; i += THREADS) {
            // copy from above
            xD[i] = eD[i];
        }
        __syncthreads();
        // build arguments for scheduler and subscriber
        if (threadIdx.x / WARP_SIZE == 0) {
            // scheduler
            const auto gtQCl = bk.tM;
            const auto tQRl = cute::ceil_div(gtQCl * Config::tiles<BLOCK_N>(bk.pd),
                subscriberCount);
            auto* __restrict__ gtQHeads = bk.tQH();
            auto* __restrict__ sQ = bk.tQS();
            auto* __restrict__ pDB = bk.pDB();
            scheduler::start<processors>(CAST_TO(cuda::std::byte, xD + nx),
                tQRl, gtQCl, interrupt, tQHeads, gtQHeads, taskBound, rQ, sQ, pDB);
        }
        else {
            // subscriber
            subscriber::start(workspace, interrupt, pT, xD, xD + nRx, nRx, status,
                taskBound, bk, activations, experts, biasT);
        }
    }
}
#endif //OS_CUH
