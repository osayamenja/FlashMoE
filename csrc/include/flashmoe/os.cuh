/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Flashmoe Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 */

//
// Created by Osayamen on 1/1/25.
//

#ifndef OS_CUH
#define OS_CUH

#include <cuda/std/cstddef>

#include "context.cuh"
#include "infra/heap.cuh"
#include "scheduler.cuh"
#include "subscriber.cuh"

namespace flashmoe::os {
    // Also applies to shared memory banks
    template<typename Element>
    __device__ __forceinline__
    constexpr auto rTCL(uint const& len) {
        constexpr auto lineBytes = 128;
        return cute::ceil_div(len * sizeof(Element), lineBytes) * lineBytes;
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
        const Heap& symHeap,
        const Context& ctx, const int& EC, // not-padded EC
        const int& tilesN0, const int& tilesN1,
        const int& dispatchBlocks,
        const int& E, const int& I,
        const uint& processors) {
        // no funny business
        static_assert(scheduler::WARP_SIZE == subscriber::WARP_SIZE);
        const auto ecTilesM = cute::ceil_div(EC, bM);
        const auto ecSignalCount = ecTilesM * tilesN1;
        const auto ssfC = E * ecSignalCount;
        const auto* __restrict__ eC = ctx.expertCounts;
        const auto world = ctx.world;
        const auto nLx = ctx.nLx;
        constexpr auto sNW = subscriberCount / subscriber::WARP_SIZE;
        static_assert(subscriberCount % subscriber::WARP_SIZE == 0 && sNW >= 1);

        const auto gtQCl = ctx.world * ctx.nLx * ecTilesM;
        const auto sBz = nSI<scheduler::WARP_SIZE>(gtQCl);

        // subscriber shared memory allocation
        size_t offset = 0;
        auto* __restrict__ pL = reinterpret_cast<PLI*>(workspace);
        static_assert(alignof(PLI) % alignof(ELI) == 0);
        offset += rTCL<PLI>(world);
        auto* __restrict__ eL = reinterpret_cast<ELI*>(workspace + offset);
        static_assert(alignof(ELI) % alignof(uint) == 0);
        offset += rTCL<ELI>(E);
        auto* __restrict__ lX = reinterpret_cast<LXI*>(workspace + offset);
        static_assert(alignof(LXI) % alignof(BitSet) == 0);
        offset += rTCL<LXI>(nLx);
        auto* __restrict__ subVisitedSet = reinterpret_cast<BitSet*>(workspace + offset);
        const auto fSbSL = nSI<sNW>(nLx * world);
        const auto bScL = nSI<subscriberCount>(ssfC);
        const auto bSSI = fSbSL + bScL;
        offset += rTCL<BitSet>(bSSI);
        auto* __restrict__ senseBitset = reinterpret_cast<BitSet*>(workspace + offset);
        offset += rTCL<BitSet>(bScL);

        // scheduler shared memory allocation
        auto* __restrict__ schedulerBitSet = reinterpret_cast<BitSet*>(workspace + offset);
        offset += rTCL<BitSet>(sBz);
        static_assert(alignof(BitSet) % alignof(uint) == 0);
        auto* __restrict__ tQHeads = reinterpret_cast<uint*>(workspace + offset);
        offset += sizeof(uint) * subscriberCount;
        auto* __restrict__ interrupt = reinterpret_cast<uint*>(workspace + offset);
        offset += sizeof(uint) * subscriberCount;
        auto* __restrict__ rQ = reinterpret_cast<uint*>(workspace + offset);
        offset += rTCL<uint>(processors);
        auto* __restrict__ interruptScratch = reinterpret_cast<uint*>(workspace + offset);
        offset += rTCL<uint>(processors);
        auto* __restrict__ status = reinterpret_cast<uint*>(workspace + offset);
        offset += rTCL<uint>(world);
        auto* __restrict__ taskBound = reinterpret_cast<uint*>(workspace + offset);
        offset += sizeof(uint);
        auto* __restrict__ scratch = reinterpret_cast<uint*>(workspace + offset); // [E]

        const auto* __restrict__ geL = ctx.eli;
        const auto* __restrict__ gpL = ctx.pli;
        const auto* __restrict__ gLx = ctx.lxi;
        for (uint i = threadIdx.x; i < fSbSL; i += threads) {
            subVisitedSet[i] = BitSet{0U};
        }
        for (uint i = threadIdx.x; i < world; i += threads) {
            pL[i] = gpL[i];
        }
        for (uint i = threadIdx.x; i < E; i += threads) {
            eL[i] = geL[i];
        }
        for (uint i = threadIdx.x; i < nLx; i += threads) {
            lX[i] = gLx[i];
        }
        for (uint i = threadIdx.x; i < bScL; i += threads) {
            senseBitset[i] = ctx.consumerCombineBitMap[i];
        }
        // Scheduler shared memory allocation
        auto* __restrict__ eCs = scratch;
        if (!threadIdx.x) {
            // Expert computation expectant tasks
            // unknown a priori
            *taskBound = ctx.nLx * ctx.world * ecTilesM * (tilesN0 + tilesN1);
        }
        for (uint i = threadIdx.x; i < E; i += threads) {
            eCs[i] = eC[i];
        }
        __syncthreads();
        // Combine tasks
        // known a priori
        for (uint i = threadIdx.x; i < E; i += threads) {
            const auto eCount = d == DropTokens::yes ? cute::min(eCs[i], EC) : eCs[i];
            const auto eCt = cute::ceil_div(eCount, bM);
            atomicAdd_block(taskBound, eCt * tilesN1);
        }
        // initialize combine visited set
        if (threadIdx.x < subscriberCount) {
            auto* __restrict__ vs = subVisitedSet + fSbSL;
            for (uint i = threadIdx.x; i < bScL; i += subscriberCount) {
                const auto slot = i / subscriberCount;
                const auto prefix = slot * subscriberCount * sizeof(BitSet) * 8;
                auto bitset = vs[i];
                for (int j = 0; j < sizeof(BitSet) * 8; ++j) {
                    const auto tileIdx = (prefix + threadIdx.x) + (j * subscriberCount);
                    if (tileIdx >= ecSignalCount * E) {
                        break;
                    }
                    const auto expertIdx = tileIdx / ecSignalCount;
                    const auto intraTileIdx = tileIdx % ecSignalCount;
                    const auto expertTileCount = cute::ceil_div(eCs[expertIdx], bM) * tilesN1;
                    if (intraTileIdx < expertTileCount) {
                        bitset.set(j);
                    }
                    else {
                        bitset.clear(j);
                    }
                }
                vs[i] = bitset;
            }
        }
        __syncthreads();
        // Pre-populate rQ under the assumption that all processors are initially ready at kernel start-up
        // However, some processors are currently specialized for packet dispatch, while others are idle.
        // To maximize utilization, we time-shift idle brethren to earlier slots in the rQ.
        const auto fL  = processors - dispatchBlocks;
        const auto sL = fL / threads;
        const auto rL = fL % threads;
        if (sL > 0) {
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
            for (int i = 0; i < psL; ++i) {
                const auto idx = i * threads + threadIdx.x;
                rQ[fL + idx] = idx;
            }
        }
        if (prL % threads != 0) {
            if (threadIdx.x < prL) {
                const auto idx = psL * threads + threadIdx.x;
                rQ[fL + idx] = idx;
            }
        }

        for (uint i = threadIdx.x; i < processors; i += threads) {
            interruptScratch[i] = 1U; // pre-fill the scheduler's bitmask
        }
        for (uint i = threadIdx.x; i < subscriberCount; i += threads) {
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
        if (threadIdx.x / scheduler::WARP_SIZE == 0) {
            // scheduler
            const auto sO = subscriberTQLength<subscriberCount, subscriber::WARP_SIZE>(ctx.world,
                ctx.nLx, ecTilesM, E, tilesN0, tilesN1);
            auto* __restrict__ gtQHeads = ctx.gTqHeads();
            auto* __restrict__ sQ = ctx.statusQueue();
            auto* __restrict__ pDB = ctx.tqs();
            scheduler::start<subscriberCount>(interruptScratch, schedulerBitSet, processors, tilesN1,
                sO, gtQCl, interrupt, tQHeads,
                gtQHeads, taskBound, rQ, sQ, pDB);
        }
        else {
            subscriber::Args args{
                ctx.signals, ctx.tQ, ctx.GEMM0Staging, senseBitset, subVisitedSet, interrupt, tQHeads,
                pL, lX, eL, status, taskBound, ctx.world, ctx.nLx, ctx.firstStageFlagCount, ctx.epRank, ecTilesM * bM, E, I,
                threadIdx.x - scheduler::WARP_SIZE, tilesN0, tilesN1, ecTilesM, ctx.stateNumber
            };
            subscriber::start<topo, subscriberCount, bM, ElementC>(symHeap, args);
        }
        __syncthreads();
        for (uint i = threadIdx.x; i < bScL; i += threads) {
            ctx.consumerCombineBitMap[i] = senseBitset[i];
        }
    }
}
#endif //OS_CUH
