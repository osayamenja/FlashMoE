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
        constexpr auto EC = ACC::EC::value;

        // subscriber shared memory allocation
        auto* __restrict__ pL = CAST_TO(PLI, workspace);
        static_assert(alignof(PLI) % alignof(ELI) == 0);
        auto* __restrict__ eL = CAST_TO(ELI, pL + world);
        static_assert(alignof(ELI) % alignof(uint) == 0);
        auto* __restrict__ lX = CAST_TO(LXI, eL + E);
        const auto dZ = roundToCacheLine<LXI>(sizeof(ELI) * E + sizeof(PLI) * world + sizeof(LXI) * nLx);
        auto* __restrict__ bitSet = CAST_TO(BitSet, workspace + dZ);
        const auto bSSIz = bSSI * sizeof(uint);
        static_assert(alignof(BitSet) % alignof(uint) == 0);
        auto* __restrict__ subscriberScratch = CAST_TO(uint, workspace + dZ + bSSIz);
        auto* __restrict__ taskBound = subscriberScratch + (SUBSCRIBERS * wSet);
        const auto* __restrict__ geL = bookkeeping.eL();
        const auto* __restrict__ gpL = bookkeeping.pL();
        const auto* __restrict__ gLx = bookkeeping.lX();
        const auto z = dZ + bSSIz + (SUBSCRIBERS * wSet + 1) * sizeof(uint);
        for (uint i = threadIdx.x; i < bSSI; i += threads) {
            bitSet[i] = BitSet{0U};
        }
        for (uint i = threadIdx.x; i < world; i += threads) {
            pL[i] = gpL[i];
        }
        #pragma unroll
        for (uint i = threadIdx.x; i < E; i += threads) {
            eL[i] = geL[i];
        }
        for (uint i = threadIdx.x; i < nLx; i += threads) {
            lX[i] = gLx[i];
        }
        auto* __restrict__ scratch = CAST_TO(uint, workspace + roundToCacheLine<uint>(z));
        auto* __restrict__ tQHeads = scratch;
        auto* __restrict__ interrupt = tQHeads + subscriberCount;
        auto* __restrict__ rQ = interrupt + subscriberCount;
        auto* __restrict__ status = rQ + roundToCacheLine<uint>(processors);
        static_assert(alignof(uint) % alignof(BitSet) == 0);
        auto* __restrict__ schedulerScratch = status + world;
        auto* __restrict__ eCs = scratch;
        if (!threadIdx.x) {
            // Expert computation expectant tasks
            // unknown a priori
            *taskBound = bookkeeping.nLx * bookkeeping.world *
                ACC::TCM::value * (ACC::TN::value + ACC::TNx::value);
        }
        #pragma unroll
        for (uint i = threadIdx.x; i < E; i += threads) {
            eCs[i] = __ldg(eC + i);
        }
        __syncthreads();
        // Combine tasks
        // known a priori
        #pragma unroll
        for (uint i = threadIdx.x; i < E; i += threads) {
            const auto eCt = Bookkeeping::tiles<BLOCK_M>(d == DropTokens::yes ?
                cute::min(eCs[i], EC) : eCs[i]);
            atomicAdd_block(taskBound, eCt * TNx);
        }
        __syncthreads();
        #pragma unroll
        for (uint i = threadIdx.x; i < processors; i += threads) {
            rQ[i] = i; // initially, all processors are ready
        }
        const auto gtQCl = bookkeeping.gtQCl;
        #pragma unroll
        for (uint i = threadIdx.x; i < processors; i += threads) {
            schedulerScratch[i] = 1U; // pre-fill the scheduler's bitmask
        }
        #pragma unroll
        for (uint i = threadIdx.x; i < SUBSCRIBERS; i += threads) {
            tQHeads[i] = 0U;
            interrupt[i] = 0U;
        }
        for (uint i = threadIdx.x; i < world; i += threads) {
            status[i] = 0U;
        }
        __syncthreads();
        // build arguments for scheduler and subscriber
        if (threadIdx.x / WARP_SIZE == 0) {
            // scheduler
            const auto sO = bookkeeping.sT;
            auto* __restrict__ gtQHeads = bookkeeping.tQH();
            auto* __restrict__ sQ = bookkeeping.sQ();
            auto* __restrict__ pDB = bookkeeping.pDB();
            scheduler::start<processors>(schedulerScratch, sO, gtQCl, interrupt, tQHeads,
                gtQHeads, taskBound, rQ, sQ, pDB);
        }
        else {
            __shared__ uint16_t sSeqBit[SUBSCRIBERS];
            const auto tIdx = threadIdx.x - WARP_SIZE;
            // Operand for a NOOP instruction
            sSeqBit[tIdx] = lSeqBit;
            // subscriber
            subscriber::start<bitSetSizePs, wSet>(bitSet, subscriberScratch, sSeqBit + tIdx,
                interrupt, tQHeads + tIdx, pL, lX, eL, ssfC, status, taskBound,
                moeOutput, expertsUp, expertsDown, biasUp, biasDown, lSeqBit, tIdx);
        }
    }
}
#endif //OS_CUH
