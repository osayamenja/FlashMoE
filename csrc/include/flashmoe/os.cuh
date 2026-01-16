/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Flashmoe Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 */

//
// Created by oja7 on 1/1/25.
//

#ifndef OS_CUH
#define OS_CUH

#include <cuda/std/cstddef>

#include "infra/heap.cuh"
#include "scheduler.cuh"
#include "subscriber.cuh"

namespace flashmoe::os {
    // Also applies to shared memory banks
    template<typename Element>
    requires(128 % sizeof(Element) == 0)
    __device__ __forceinline__
    constexpr auto rTCL(uint const& zb) {
        return cute::ceil_div(zb, 128U / sizeof(Element)) * (128U / sizeof(Element));
    }
    template<
        Topology topo,
        int subscriberCount,
        int threads,
        int bM,
        DropTokens d,
        typename ElementC
    >
    __device__ __forceinline__
    void start(cuda::std::byte* __restrict__ const& workspace,
        const Heap& symHeap, const Context& ctx, const int& EC, // not-padded EC
        const int& dispatchBlocks,
        const int& E, const uint& processors) {
        // no funny business
        static_assert(scheduler::WARP_SIZE == subscriber::WARP_SIZE);
        const auto ssfC = ctx.ssFc;
        const auto* __restrict__ eC = ctx.expertCounts;
        const auto world = ctx.world;
        const auto nLx = bookkeeping.nLx;
        constexpr auto sNW = subscriberCount / subscriber::WARP_SIZE;
        const auto bSSI = nSI<sNW>(nLx * world) +
            nSI<subscriberCount>(ssfC);

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
        auto* __restrict__ taskBound = reinterpret_cast<uint*>(workspace + dZ + bSSIz);
        const auto* __restrict__ geL = bookkeeping.eL();
        const auto* __restrict__ gpL = bookkeeping.pL();
        const auto* __restrict__ gLx = bookkeeping.lX();
        const auto z = dZ + bSSIz + sizeof(uint);
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
        const auto sBz = nSI<scheduler::WARP_SIZE>(bookkeeping.gtQCl);
        auto* __restrict__ scratch = CAST_TO(uint, workspace + rTCL<uint>(z));
        auto* __restrict__ tQHeads = scratch;
        auto* __restrict__ interrupt = tQHeads + subscriberCount;
        auto* __restrict__ rQ = interrupt + subscriberCount;
        static_assert(alignof(uint) % alignof(BitSet) == 0);
        auto* __restrict__ schedulerBitSet = CAST_TO(BitSet, rQ + rTCL<uint>(processors));
        static_assert(alignof(BitSet) % alignof(uint) == 0);
        auto* __restrict__ interruptScratch = CAST_TO(uint, schedulerBitSet + rTCL<BitSet>(sBz));
        auto* __restrict__ status = interruptScratch + rTCL<uint>(processors);

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
            const auto eCt = Bookkeeping::tiles<bM>(d == DropTokens::yes ?
                cute::min(eCs[i], EC) : eCs[i]);
            atomicAdd_block(taskBound, eCt * TNx);
        }
        __syncthreads();
        // Pre-populate rQ under the assumption that all processors are initially ready.
        // However, some processors are currently specialized for packet dispatch, while others are idle.
        // To maximize utilization, we time-shift idle brethren to earlier slots in the rQ.
        const auto fL  = processors - dispatchBlocks;
        constexpr auto sL = fL / threads;
        constexpr auto rL = fL % threads;
        if (sL > 0) {
            #pragma unroll
            for (int i = 0; i < sL; ++i) {
                const auto idx = i * threads + threadIdx.x;
                rQ[idx] = dispatchBlocks + idx;
            }
        }
        if (fL % threads != 0) {
            if (threadIdx.x < rL) {
                const auto idx = sL * threads + threadIdx.x;
                rQ[idx] = dispatchBlocks + idx;
            }
        }
        const auto psL = dispatchBlocks / threads;
        const auto prL = dispatchBlocks % threads;
        if constexpr (psL > 0) {
            #pragma unroll
            for (int i = 0; i < psL; ++i) {
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
        for (int i = threadIdx.x; i < processors; i += threads) {
            interruptScratch[i] = 1U; // pre-fill the scheduler's bitmask
        }
        #pragma unroll
        for (int i = threadIdx.x; i < subscriberCount; i += threads) {
            tQHeads[i] = 0U;
            interrupt[i] = 0U;
        }
        for (int i = threadIdx.x; i < world; i += threads) {
            status[i] = 0U;
        }
        for (int i = threadIdx.x; i < sBz; i += threads) {
            schedulerBitSet[i] = BitSet{0U};
        }
        __syncthreads();
        // build arguments for scheduler and subscriber
        if (threadIdx.x / scheduler::WARP_SIZE == 0) {
            // scheduler
            const auto sO = bookkeeping.sT;
            auto* __restrict__ gtQHeads = bookkeeping.tQH();
            auto* __restrict__ sQ = bookkeeping.sQ();
            auto* __restrict__ pDB = bookkeeping.pDB();
            scheduler::start<processors>(interruptScratch, schedulerBitSet,
                sO, gtQCl, interrupt, tQHeads,
                gtQHeads, taskBound, rQ, sQ, pDB);
        }
        else {
            // build subscriber::Args args{};
            // subscriber
            subscriber::start<topo, subscriberCount, bM, ElementC>(symHeap, {});
        }
    }
}
#endif //OS_CUH
