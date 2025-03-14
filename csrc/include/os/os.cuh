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
        typename BiasDown,
        unsigned int threads = ACC::PeakHardware::OS::threads::value
    >
    __device__ __forceinline__
    void start(cuda::std::byte* __restrict__ const& workspace,
        Output const& moeOutput,
        ExpertsUp const& expertsUp,
        ExpertsDown const& expertsDown,
        BiasUp const& biasUp,
        BiasDown const& biasDown,
        const uint16_t& lSeqBit) {
        const auto ssfC = __ldg(bookkeeping.ssFc());
        const auto* __restrict__ eC = bookkeeping.eC();
        const auto world = bookkeeping.world;
        const auto nLx = bookkeeping.nLx;
        constexpr auto subscriberCount = threads - WARP_SIZE;

        // each subscriber thread gets wSet * sizeof(uint) bytes of workspace
        constexpr auto wSet = 16U; // working set size
        constexpr auto bitSetSizePs = cute::ceil_div(wSet, sizeof(uint) * 8U);
        const auto bSSI = nSI<subscriberCount>(nLx * world) + nSI<subscriberCount>(ssfC);
        constexpr auto E = ACC::E::value;
        constexpr auto TNx = ACC::TNx::value;
        constexpr auto EC = ACC::TNx::value;

        // subscriber shared memory allocation
        auto* __restrict__ eL = CAST_TO(ELI, workspace);
        static_assert(alignof(ELI) % alignof(PLI) == 0);
        auto* __restrict__ pL = CAST_TO(PLI, eL + E);
        static_assert(alignof(PLI) % alignof(uint) == 0);
        auto* __restrict__ lX = CAST_TO(LXI, pL + world);
        const auto dZ = roundToCacheLine(sizeof(ELI) * E + sizeof(PLI) * world + sizeof(LXI) * nLx);
        auto* __restrict__ bitSet = CAST_TO(BitSet, workspace + dZ);
        const auto cbSSI = roundToCacheLine(bSSI * sizeof(uint));
        static_assert(alignof(BitSet) % alignof(uint) == 0);
        auto* __restrict__ subscriberScratch = CAST_TO(uint, workspace + dZ + cbSSI);
        auto* __restrict__ taskBound = subscriberScratch + (SUBSCRIBERS * wSet);
        const auto* __restrict__ geL = bookkeeping.eL();
        const auto* __restrict__ gpL = bookkeeping.pL();
        const auto* __restrict__ gLx = bookkeeping.lX();
        const auto z = dZ + cbSSI + (SUBSCRIBERS * wSet + 1) * sizeof(uint);
        for (uint i = threadIdx.x; i < bSSI; i += threads) {
            bitSet[i] = BitSet{0U};
        }
        #pragma unroll
        for (uint i = threadIdx.x; i < E; i += threads) {
            eL[i] = geL[i];
            pL[i] = gpL[i];
        }
        for (uint i = threadIdx.x; i < world; i += threads) {
            lX[i] = gLx[i];
        }
        auto* __restrict__ scratch = CAST_TO(uint, workspace + roundToCacheLine(z));
        auto* __restrict__ tQHeads = scratch;
        auto* __restrict__ interrupt = tQHeads + subscriberCount;
        auto* __restrict__ rQ = interrupt + subscriberCount;
        auto* __restrict__ status = rQ + processors;
        auto* __restrict__ schedulerScratch = CAST_TO(cuda::std::byte, status + world);
        // shared memory arrays
        // Upper bound for expectant tasks
        const auto* __restrict__ eCs = scratch;
        scratch += 1;
        if (!threadIdx.x) {
            // Expert computation expectant tasks
            // unknown a priori
            *taskBound = bookkeeping.nLx * bookkeeping.world *
                ACC::TCM::value * (ACC::TN::value + ACC::TNx::value);
        }
        #pragma unroll
        for (uint i = threadIdx.x; i < E; i += threads) {
            scratch[i] = __ldg(eC + i);
        }
        __syncthreads();
        // Combine tasks
        // known a priori
        #pragma unroll
        for (uint i = threadIdx.x; i < E; i += threads) {
            const auto eCt = Bookkeeping::tiles<BLOCK_M>(d == DropTokens::yes ?
                cute::min(eCs[i], EC)
                : eCs[i]);
            atomicAdd_block(taskBound, eCt * TNx);
        }
        __syncthreads();
        #pragma unroll
        for (uint i = threadIdx.x; i < processors; i += threads) {
            rQ[i] = i; // initially, all processors are ready
        }
        #pragma unroll
        for (uint i = threadIdx.x; i < SUBSCRIBERS; i += threads) {
            tQHeads[i] = 0U;
            interrupt[i] = 0U;
        }
        __syncthreads();
        // build arguments for scheduler and subscriber
        if (threadIdx.x / WARP_SIZE == 0) {
            // scheduler
            const auto gtQCl = bookkeeping.gtQCl;
            const auto sO = bookkeeping.sT;
            auto* __restrict__ gtQHeads = bookkeeping.tQH();
            auto* __restrict__ sQ = bookkeeping.tSA();
            auto* __restrict__ pDB = bookkeeping.pDB();
            scheduler::start<processors>(schedulerScratch, sO, gtQCl, interrupt, tQHeads,
                gtQHeads, taskBound, rQ, sQ, pDB);
        }
        else {
            const auto tIdx = threadIdx.x - WARP_SIZE;
            // subscriber
            subscriber::start<bitSetSizePs, wSet>(bitSet, CAST_TO(cuda::std::byte, bitSet + bSSI), interrupt,
                tQHeads + tIdx, pL, lX, eL, ssfC, status, taskBound,
                moeOutput, expertsUp, expertsDown, biasUp, biasDown, lSeqBit, tIdx);
        }
    }
}
#endif //OS_CUH
