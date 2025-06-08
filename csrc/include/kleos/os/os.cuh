/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Kleos Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 */

//
// Created by oja7 on 1/1/25.
//

#ifndef OS_CUH
#define OS_CUH

#include <cuda/std/cstddef>
#include "../types.cuh"

#include "scheduler.cuh"
#include "subscriber.cuh"

namespace kleos::os {
    template<
        unsigned int processors,
        DropTokens d = DropTokens::yes,
        typename ExpertsUp,
        typename ExpertsDown,
        typename BiasUp,
        typename BiasDown
    >
    __device__ __forceinline__
    void start(cuda::std::byte* __restrict__ const& workspace,
        ExpertsUp const& expertsUp,
        ExpertsDown const& expertsDown,
        BiasUp const& biasUp,
        BiasDown const& biasDown,
        const uint16_t& lSeqBit) {
        const auto ssfC = __ldg(bookkeeping.ssFc());
        const auto* __restrict__ eC = bookkeeping.eC();
        const auto world = bookkeeping.world;
        const auto nLx = bookkeeping.nLx;
        constexpr auto threads = ACC::PeakHardware::OS::threads::value;
        constexpr auto subscriberCount = threads - WARP_SIZE;
        constexpr auto sNW = subscriberCount / WARP_SIZE;
        // each subscriber thread gets wSet * sizeof(uint) bytes of workspace
        constexpr auto uSfC = ACC::TCM::value * ACC::TNx::value / subscriberCount;
        constexpr auto wSet = uSfC >= 32 ? 32U : 16U;
        const auto bSSI = nSI<sNW>(nLx * world) +
            nSI<subscriberCount>(ssfC);
        constexpr auto E = ACC::E::value;
        constexpr auto TNx = ACC::TNx::value;
        constexpr auto EC = ACC::EC::value;

        // subscriber shared memory allocation
        auto* __restrict__ pL = CAST_TO(PLI, workspace);
        static_assert(alignof(PLI) % alignof(ELI) == 0);
        auto* __restrict__ eL = CAST_TO(ELI, pL + world);
        static_assert(alignof(ELI) % alignof(uint) == 0);
        auto* __restrict__ lX = CAST_TO(LXI, eL + E);
        const auto dZ = rTCL<LXI>(sizeof(ELI) * E +
            sizeof(PLI) * world +
            sizeof(LXI) * nLx);
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
        // Scheduler shared memory allocation
        const auto sBz = nSI<WARP_SIZE>(bookkeeping.gtQCl);
        auto* __restrict__ scratch = CAST_TO(uint, workspace + rTCL<uint>(z));
        auto* __restrict__ tQHeads = scratch;
        auto* __restrict__ interrupt = tQHeads + subscriberCount;
        auto* __restrict__ rQ = interrupt + subscriberCount;
        static_assert(alignof(uint) % alignof(BitSet) == 0);
        auto* __restrict__ schedulerBitSet = CAST_TO(BitSet, rQ + rTCL<uint>(processors));
        static_assert(alignof(BitSet) % alignof(uint) == 0);
        auto* __restrict__ interruptScratch = CAST_TO(uint, schedulerBitSet + rTCL<BitSet>(sBz));
        auto* __restrict__ status = interruptScratch + rTCL<uint>(processors);
        using WarpScan = cub::WarpScan<uint>;
        static_assert(alignof(uint) % alignof(WarpScan::TempStorage) == 0);
        auto* __restrict__ wSt = CAST_TO(WarpScan::TempStorage, status + rTCL<uint>(world));

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
        if (threadIdx.x / WARP_SIZE == 0) {
            uint clearEC = 0U;
            if (!threadIdx.x) {
                constexpr auto expected = ACC::DBZ::value + 1;
                __threadfence();
                clearEC = atomicIncrement(bookkeeping.eCSync()) + 1 == expected;
            }
            __syncwarp();
            clearEC = __shfl_sync(0xffffffff, clearEC, 0);
            if (clearEC) {
                auto* __restrict__ bEC = bookkeeping.eC();
                constexpr auto tL = ACC::E::value / WARP_SIZE;
                for (uint i = 0; i < tL; ++i) {
                    bEC[threadIdx.x + i * WARP_SIZE] = 0U;
                }
                if constexpr (constexpr auto residue = ACC::E::value % WARP_SIZE; residue != 0) {
                    if (threadIdx.x < residue) {
                        bEC[threadIdx.x + tL * WARP_SIZE] = 0U;
                    }
                }
            }
        }
        // Combine tasks
        // known a priori
        #pragma unroll
        for (uint i = threadIdx.x; i < E; i += threads) {
            const auto eCt = Bookkeeping::tiles<BLOCK_M>(d == DropTokens::yes ?
                cute::min(eCs[i], EC) : eCs[i]);
            atomicAdd_block(taskBound, eCt * TNx);
        }
        __syncthreads();
        // Pre-populate rQ under the assumption that all processors are initially ready.
        // However, some processors are currently specialized for packet dispatch, while others are idle.
        // To maximize utilization, we time-shift idle brethren to earlier slots in the rQ.
        constexpr auto fL  = processors - ACC::DBZ::value;
        constexpr auto sL = fL / threads;
        constexpr auto rL = fL % threads;
        if constexpr (sL > 0) {
            #pragma unroll
            for (uint i = 0; i < sL; ++i) {
                const auto idx = i * threads + threadIdx.x;
                rQ[idx] = ACC::DBZ::value + idx;
            }
        }
        if constexpr (fL % threads != 0) {
            if (threadIdx.x < rL) {
                const auto idx = sL * threads + threadIdx.x;
                rQ[idx] = ACC::DBZ::value + idx;
            }
        }
        constexpr auto psL = ACC::DBZ::value / threads;
        constexpr auto prL = ACC::DBZ::value % threads;
        if constexpr (psL > 0) {
            #pragma unroll
            for (uint i = 0; i < psL; ++i) {
                const auto idx = i * threads + threadIdx.x;
                rQ[fL + idx] = idx;
            }
        }
        if constexpr (prL % threads != 0) {
            if (threadIdx.x < prL) {
                const auto idx = psL * threads + threadIdx.x;
                rQ[fL + idx] = idx;
            }
        }

        const auto gtQCl = bookkeeping.gtQCl;
        #pragma unroll
        for (uint i = threadIdx.x; i < processors; i += threads) {
            interruptScratch[i] = 1U; // pre-fill the scheduler's bitmask
        }
        #pragma unroll
        for (uint i = threadIdx.x; i < SUBSCRIBERS; i += threads) {
            tQHeads[i] = 0U;
            interrupt[i] = 0U;
        }
        for (uint i = threadIdx.x; i < world; i += threads) {
            status[i] = 0U;
        }
        for (uint i = threadIdx.x; i < sBz; i += threads) {
            schedulerBitSet[i] = BitSet{0U};
        }
        __syncthreads();
        // build arguments for scheduler and subscriber
        if (threadIdx.x / WARP_SIZE == 0) {
            // scheduler
            const auto sO = bookkeeping.sT;
            auto* __restrict__ gtQHeads = bookkeeping.tQH();
            auto* __restrict__ sQ = bookkeeping.sQ();
            auto* __restrict__ pDB = bookkeeping.pDB();
            scheduler::start<processors>(wSt, interruptScratch, schedulerBitSet,
                sO, gtQCl, interrupt, tQHeads,
                gtQHeads, taskBound, rQ, sQ, pDB);
        }
        else {
            __shared__ uint16_t sSeqBit[SUBSCRIBERS];
            const auto tIdx = threadIdx.x - WARP_SIZE;
            // Operand for a NOOP instruction
            sSeqBit[tIdx] = lSeqBit;
            // subscriber
            subscriber::start<wSet>(bitSet, subscriberScratch, sSeqBit + tIdx,
                interrupt, tQHeads + tIdx, pL, lX, eL, ssfC, status, taskBound,
                expertsUp, expertsDown, biasUp, biasDown, lSeqBit, tIdx);
        }
    }
}
#endif //OS_CUH
