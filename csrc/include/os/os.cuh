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
        typename Output,
        typename ExpertsUp,
        typename ExpertsDown,
        typename BiasUp,
        typename BiasDown
    >
    __device__ __forceinline__
    void start(cuda::std::byte* __restrict__ const& workspace,
        Output const& moeOutput,
        ExpertsUp const& expertsUp,
        ExpertsDown const& expertsDown,
        BiasUp const& biasUp,
        BiasDown const& biasDown,
        const uint16_t& lSeqBit) {
        const auto nRx = __ldg(bookkeeping.nRx());
        const auto* __restrict__ eC = bookkeeping.eC();
        const auto world = bookkeeping.world;

        // each subscriber thread gets wSet * sizeof(uint) bytes of workspace
        constexpr auto wSet = 16U; // working set size
        constexpr auto subscriberCount = THREADS - 32;
        constexpr auto E = ACC::E::value;
        constexpr auto TNx = ACC::TNx::value;
        constexpr auto TN = ACC::TNx::value;
        constexpr auto EC = ACC::TNx::value;

        // subscriber shared memory allocation
        auto* __restrict__ eL = CAST_TO(ELI, workspace);
        auto* __restrict__ pL = CAST_TO(PLI, eL + E);
        const auto* __restrict__ geL = bookkeeping.eL();
        const auto* __restrict__ gpL = bookkeeping.pLI();
        const auto z = sizeof(ELI) * E + sizeof(PLI) * world;
        // Below is to minimize bank conflicts for the subscriber threads
        auto* __restrict__ sWorkspace = workspace + cute::ceil_div(z, 128U) * 128U;
        #pragma unroll
        for (uint i = threadIdx.x; i < E; ++i) {
            eL[i] = geL[i];
            pL[i] = gpL[i];
        }
        auto* __restrict__ scratch = CAST_TO(uint, sWorkspace + SUBSCRIBERS * wSet * sizeof(uint));
        // shared memory arrays
        // Upper bound for expectant tasks
        auto*  taskBound = scratch;
        const auto* __restrict__ eCs = taskBound + 1;
        scratch += 1;
        #pragma unroll
        for (uint i = threadIdx.x; i < E; i += THREADS) {
            scratch[i] = __ldg(eC + i);
        }
        __syncthreads();
        // compute taskBound
        #pragma unroll
        for (uint i = threadIdx.x; i < E; i += THREADS) {
            const auto eCt = Bookkeeping::tiles<BLOCK_M>(d == DropTokens::yes ? cute::min(eCs[i], EC)
                : eCs[i]);
            atomicAdd_block(taskBound, eCt * TN);
            #pragma unroll 4
            for (uint j = 0; j < world; ++j) {
                atomicAdd_block(taskBound, eCt * TNx);
            }
        }
        __syncthreads();
        auto* __restrict__ tQHeads = taskBound + 1;
        auto* __restrict__ rQ = tQHeads + subscriberCount;
        #pragma unroll
        for (uint i = threadIdx.x; i < processors; i += THREADS) {
            rQ[i] = i; // initially, all processors are ready
        }
        auto* interrupt = rQ + processors;
        #pragma unroll
        for (uint i = threadIdx.x; i < SUBSCRIBERS; i += THREADS) {
            tQHeads[i] = 0U;
            interrupt[i] = 0U;
        }
        auto* __restrict__ status = interrupt + SUBSCRIBERS;
        auto* schedulerScratch = status + world;
        __syncthreads();
        // build arguments for scheduler and subscriber
        if (threadIdx.x / WARP_SIZE == 0) {
            // scheduler
            const auto gtQCl = bookkeeping.gtQCl;
            const auto tQRl = cute::ceil_div(gtQCl * ACC::TN::value, subscriberCount);
            auto* __restrict__ gtQHeads = bookkeeping.tQH();
            auto* __restrict__ sQ = bookkeeping.tQS();
            auto* __restrict__ pDB = bookkeeping.pDB();
            scheduler::start<processors>(schedulerScratch,tQRl, gtQCl, interrupt, tQHeads,
                gtQHeads, taskBound, rQ, sQ, pDB);
        }
        else {
            // subscriber
            subscriber::start<wSet>(sWorkspace, interrupt, pL, eL, nRx,
                status, taskBound, moeOutput, expertsUp, expertsDown, biasUp, biasDown, lSeqBit);
        }
    }
}
#endif //OS_CUH
