/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Flashmoe Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 */

//
// Created by oja7 on 11/17/24.
//

#ifndef SCHEDULER_CUH
#define SCHEDULER_CUH

#include <cub/cub.cuh>
#include <cuda/atomic>
#include <cute/numeric/integral_constant.hpp>
#include <cutlass/array.h>

#include "infra/constants.cuh"
#include "infra/bitset.cuh"
#include "infra/checks.cuh"
#include "infra/dq.cuh"
#include "infra/tq.cuh"

namespace flashmoe::scheduler {
    using WarpScan = cub::WarpScan<uint>;
    constexpr int SCHEDULER_COUNT = WARP_SIZE; // warp_size
    // register state sizes
    constexpr int PROCESSOR_STATE_SIZE = 16; // 8 is recommended
    static_assert(PROCESSOR_STATE_SIZE <= 64);
    constexpr int MAX_PROCESSORS = WARP_SIZE * PROCESSOR_STATE_SIZE; // can be relaxed but with slower perf
    constexpr int WORK_SET_SIZE = 4;
    constexpr int QUEUE_STATE_SIZE = 2;
    template<
        DQType dqt = DQType::stride,
        int nQ = 0
    >
    __device__ __forceinline__
    void schedule(const uint& processors, const uint& cSetB,
    const uint& canSchedule, const uint& qIdx, uint& lRQIdx,
    const uint& gRQIdx, uint* __restrict__ const& rQ,
    TQSignal* __restrict__ const& pDB) {
        cutlass::Array<uint, WORK_SET_SIZE> wSet{};
        static_assert(sizeof(TQSignal) == sizeof(uint64_t) && alignof(TQSignal) == alignof(uint64_t));
        auto sig = TQSignal{0U, 0U};
        for (uint k = 0; k < cSetB; ++k) {
            #pragma unroll
            for (uint l = 0; l < wSet.size(); ++l) {
                wSet[l] = rQ[(gRQIdx + lRQIdx++) % processors];
            }
            #pragma unroll
            for (uint l = 0; l < wSet.size(); ++l) {
                // signal processor
                auto* __restrict__ pdbAddr = reinterpret_cast<uint64_t*>(pDB + wSet[l]);
                cuda::atomic_ref<uint64_t, cuda::thread_scope_device> pdb{*pdbAddr};
                sig.encodeSig(DQ::next<dqt, nQ>(qIdx, k * wSet.size() + l));
                pdb.store(cuda::std::bit_cast<uint64_t>(sig), cuda::memory_order_release);
            }
        }
        // Residual scheduling
        const auto residue = canSchedule - cSetB * wSet.size();
        #pragma unroll
        for (uint l = 0; l < wSet.size(); ++l) {
            if (l < residue) {
                wSet[l] = rQ[(gRQIdx + lRQIdx++) % processors];
            }
        }
        #pragma unroll
        for (uint l = 0; l < wSet.size(); ++l) {
            if (l < residue) {
                auto* __restrict__ pdbAddr = reinterpret_cast<uint64_t*>(pDB + wSet[l]);
                cuda::atomic_ref<uint64_t, cuda::thread_scope_device> pdb{*pdbAddr};
                sig.encodeSig(DQ::next<dqt, nQ>(qIdx, cSetB * wSet.size() + l));
                pdb.store(cuda::std::bit_cast<uint64_t>(sig), cuda::memory_order_release);
            }
        }
    }

    template<
        int subscriberCount,
        int sL,
        int schedulerCount,
        typename TQState
    >
    requires (schedulerCount == 32 && isRegisterV<TQState>)
    __device__ __forceinline__
    void schedulerLoop(TQState& tqState,
        const uint& processors,
        const unsigned int& tilesN1,
        const unsigned int& tQOffset,
        const unsigned int& gTbO,
        uint& lTt, uint& processorTally,
        uint& gRQIdx, uint& scheduled,
        unsigned int* __restrict__ const& sQ,
        uint* __restrict__ const& rQ,
        TQSignal* __restrict__ const& pDB,
        const bool& isMedley = false) {
        __shared__ WarpScan::TempStorage wSt[2];
        uint queueSlot;
        uint taskTally;
        // things are about to get warped :)
        // Aggregate tally across the warp
        __syncwarp();
        WarpScan(wSt[0]).InclusiveSum(lTt, queueSlot, taskTally);
        queueSlot -= lTt;
        auto prefixTaskSum = 0U;
        while (taskTally) {
            // Find processors if we are not currently aware of any
            while (!processorTally) {
                cutlass::Array<uint, PROCESSOR_STATE_SIZE> sQState{};
                // sweep sQ to identify ready processes
                uint lPt = 0U; // local processor tally
                #pragma unroll
                for (int j = 0; j < sQState.size(); ++j) {
                    const auto idx = j * schedulerCount + threadIdx.x;
                    if (idx < processors) {
                        const auto readiness = atomicExch(sQ + (j * schedulerCount + threadIdx.x),
                        observed) == ready;
                        lPt += readiness;
                        sQState[j] = readiness;
                    }
                }
                uint startIdx;
                // Aggregate tally across the warp
                WarpScan(wSt[1]).InclusiveSum(lPt, startIdx, processorTally);
                startIdx -= lPt;
                // write to rQ
                const auto qSIdx = gRQIdx + prefixTaskSum;
                #pragma unroll
                for (uint j = 0; j < sQState.size(); ++j) {
                    const auto idx = j * schedulerCount + threadIdx.x;
                    if (idx < processors && sQState[j]) {
                        // write ready process pid to rQ
                        rQ[(qSIdx + startIdx++) % processors] = idx;
                    }
                }
                if (processorTally) {
                    // Below ensures writes to rQ in shared memory are visible warp-wide before consumption
                    __syncwarp();
                }
            }
            // schedule tasks
            const auto tasks = cute::min(processorTally, taskTally);
            prefixTaskSum += tasks;
            scheduled += tasks;
            processorTally -= tasks;
            taskTally -= tasks;
            // these will get scheduled now
            if (lTt > 0 && queueSlot < prefixTaskSum) {
                auto tasksToSchedule = umin(lTt, prefixTaskSum - queueSlot);
                lTt -= tasksToSchedule;
                if (isMedley) {
                    if constexpr (sL > 0) {
                        #pragma unroll
                        for (uint j = 0; j < sL; ++j) {
                            if (tqState[j].tasks > 0 && tasksToSchedule) {
                                const auto canSchedule = cute::min(tasksToSchedule, tqState[j].tasks);
                                const auto qIdx = DQ::next<DQType::stride, subscriberCount>(j * schedulerCount + threadIdx.x,
                                    tqState[j].tQTail);
                                tasksToSchedule -= canSchedule;
                                tqState[j].tasks -= canSchedule;
                                // have to increment tails as we will revisit this queue later on
                                tqState[j].tQTail += canSchedule;
                                const auto cSetB = canSchedule / WORK_SET_SIZE;
                                schedule<DQType::stride, subscriberCount>(processors, cSetB, canSchedule, qIdx,
                                    queueSlot, gRQIdx, rQ, pDB);
                            }
                        }
                    }
                }
                #pragma unroll
                for (uint j = sL; j < tqState.size(); ++j) {
                    if (tqState[j].tasks && tasksToSchedule) {
                        const auto canSchedule = cute::min(tasksToSchedule, tqState[j].tasks);
                        const auto qHead = (schedulerCount * (gTbO + (j - sL)) + threadIdx.x) * tilesN1 + tqState[j].tQTail;
                        const auto qIdx = tQOffset + qHead;
                        tasksToSchedule -= canSchedule;
                        tqState[j].tasks -= canSchedule;
                        // checkpoint state in case of partial scheduling
                        tqState[j].tQTail += canSchedule;
                        const auto cSetB = canSchedule / WORK_SET_SIZE;
                        schedule<DQType::block>(processors, cSetB, canSchedule,
                            qIdx, queueSlot, gRQIdx, rQ, pDB);
                    }
                }
            }
        }
        // clear checkpoints
        #pragma unroll
        for (uint j = sL; j < tqState.size(); ++j) {
            tqState[j].tQTail = 0;
        }
        // Advance global rQ index
        gRQIdx = (gRQIdx + prefixTaskSum) % processors;
    }

    template<unsigned int schedulerCount>
    /// Schedule Processor interrupts
    __device__ __forceinline__
    void sPI(unsigned int* __restrict__ const& rQ,
        uint* __restrict__ const& sQ,
        TQSignal* __restrict__ const& pDB,
        uint& gRQIdx,
        uint* __restrict__ const& scratch, // pre-filled with 1
        const uint& processors,
        const uint& processorTally) {
        __shared__ WarpScan::TempStorage wSt;
        cutlass::Array<uint, PROCESSOR_STATE_SIZE> sQState{};
        /// read through the ready queue first
        constexpr auto sig = TQSignal{0U, 1U}; // set interrupt to 1
        // Below must be <= ceil(processors / wS) == sQsL, so we can repurpose sQState registers as temporary storage
        const auto tS = processorTally / schedulerCount + (threadIdx.x < processorTally % schedulerCount);
        const auto gRO = gRQIdx + (threadIdx.x * (processorTally / schedulerCount) +
            cute::min(threadIdx.x, processorTally % schedulerCount));
        // index can only wrap around once
        gRQIdx = gRO % processors;
        #pragma unroll
        for (uint i = 0; i < sQState.size(); ++i) {
            if (i < tS) {
                // shared -> registers
                sQState[i] = rQ[(gRQIdx + i) % processors];
            }
        }
        #pragma unroll
        for (uint i = 0; i < sQState.size(); ++i) {
            if (i < tS) {
                // notify interrupts
                const auto pid = sQState[i];
                auto* __restrict__ db_p = reinterpret_cast<uint64_t*>(pDB + pid);
                cuda::atomic_ref<uint64_t, cuda::thread_scope_device> pdb{*db_p};
                pdb.store(cuda::std::bit_cast<uint64_t>(sig), cuda::memory_order_release);
                scratch[pid] = 0U;
            }
        }
        __syncwarp();
        // Consolidate findings and populate the ready queue
        uint uI = 0U;
        // shared -> registers
        #pragma unroll
        for (uint i = 0; i < sQState.size(); ++i) {
            const auto idx = i * schedulerCount + threadIdx.x;
            if (idx < processors) {
                sQState[i] = scratch[idx];
            }
        }

        #pragma unroll
        for (uint i = 0; i < sQState.size(); ++i) {
            if ((i * schedulerCount + threadIdx.x) < processors) {
                uI += sQState[i];
            }
        }

        uint startIdx;
        uint pending;
        WarpScan(wSt).InclusiveSum(uI, startIdx, pending);
        startIdx -= uI;
        // enqueue all pending processes we discovered into the rQ
        #pragma unroll
        for (uint i = 0; i < sQState.size(); ++i) {
            const auto idx = i * schedulerCount + threadIdx.x;
            if (idx < processors && sQState[i]) {
                rQ[startIdx++] = idx;
            }
        }
        __syncwarp();
        auto remaining = pending / schedulerCount + (threadIdx.x < pending % schedulerCount);
        cuda::std::array<uint, sQState.size()> pids{};
        // read from rQ to registers
        #pragma unroll
        for (uint i = 0; i < sQState.size(); ++i) {
            const auto idx = i * schedulerCount + threadIdx.x;
            if (idx < pending) {
                sQState[i] = 1;
                pids[i] = rQ[idx];
            }
            else {
                sQState[i] = 0;
            }
        }

        while (remaining) {
            #pragma unroll
            for (uint j = 0; j < sQState.size(); ++j) {
                if (sQState[j]) {
                    const auto pid = pids[j];
                    const auto isReady = atomicExch(sQ + pid, observed) == ready;
                    sQState[j] = !isReady;
                    if (isReady) {
                        // interrupt processor
                        remaining -= 1;
                        auto* __restrict__ db_p = reinterpret_cast<uint64_t*>(pDB + pid);
                        cuda::atomic_ref<uint64_t, cuda::thread_scope_device> pdb{*db_p};
                        pdb.store(cuda::std::bit_cast<uint64_t>(sig), cuda::memory_order_release);
                    }
                }
            }
        }
    }

    template<
        unsigned int subscribers
    >
    __device__ __forceinline__
    void start(uint* __restrict__ const& interruptScratch,
        BitSet* __restrict__ const& bitSet,
        const uint& processors,
        const unsigned int& tilesN1,
        const unsigned int& sO,
        const unsigned int& gtQCL,
        unsigned int* __restrict__ const& sInterrupts,
        unsigned int* __restrict__ const& tQHeads, // shared
        unsigned int* __restrict__ const& gtQHeads, // global
        unsigned int* __restrict__ const& taskBound, // shared
        unsigned int* __restrict__ const& rQ, // shared
        unsigned int* __restrict__ const& sQ, // global
        TQSignal* __restrict__ const& pDB) { //  global
        static_assert(sizeof(TQSignal) == sizeof(uint64_t) && alignof(TQSignal) == alignof(uint64_t));
        // Assumptions are below, will assert them host-side
        // RMEM_STATE_PER_THREAD * schedulerCount >= processors
        uint scheduled = 0U;
        constexpr auto schedulerCount = SCHEDULER_COUNT; // number of scheduler threads
        static_assert(subscribers % schedulerCount == 0);
        constexpr auto sL = subscribers / schedulerCount;
        // initialize register buffers
        constexpr auto dQL = QUEUE_STATE_SIZE;
        constexpr auto bSw = sizeof(uint) * 8U;
        static_assert(dQL <= bSw);
        cutlass::Array<TQState, dQL + sL> tqState{};
        tqState.fill({0U,0U});
        const uint dT = gtQCL / (schedulerCount * dQL);

        uint gRQIdx = 0U;
        uint processorTally = processors; // initially, all processors are available, ensure that rQ has all pids
        cuda::atomic_ref<unsigned int, cuda::thread_scope_block> tb{*taskBound};
        auto tTB = tb.load(cuda::memory_order_relaxed);
        while (scheduled < tTB) {
            // statically sweep tQ for tasks
            uint lTt = 0U; // local task tally
            #pragma unroll
            for (uint i = 0; i < sL; ++i) {
                auto* __restrict__ tqHp = tQHeads + (i * schedulerCount + threadIdx.x);
                cuda::atomic_ref<unsigned int, cuda::thread_scope_block> tqh{*tqHp};
                const auto tasks = tqh.load(cuda::memory_order_acquire) - tqState[i].tQTail;
                tqState[i].tasks = tasks;
                lTt += tasks;
            }

            // Abstract queues as a 3-D tensor (B, Q, T),
            // where B is the batch dimension or total queue / (Q * T);
            // Q is the number of queues a thread observes in one-pass;
            // and T is the number of threads in a warp
            if (dT > 0) {
                auto sBS = bitSet[threadIdx.x];
                // One-shot scheduling, so tails are irrelevant.
                // Special case, where i == 0
                #pragma unroll
                for (uint j = sL; j < tqState.size(); ++j) {
                    const auto pJ = j - sL;
                    if (const auto isVisited = sBS.get(pJ % bSw); !isVisited) {
                        const auto qIdx = schedulerCount * pJ + threadIdx.x;
                        cuda::atomic_ref<unsigned int, cuda::thread_scope_device> gtqH{*(gtQHeads + qIdx)};
                        const auto tasks = gtqH.load(cuda::memory_order_acquire);
                        if (tasks) {
                            // one and done
                            sBS.set(pJ % bSw);
                            gtqH.store(tQHeadGroundState, cuda::memory_order_relaxed);
                        }
                        tqState[j].tasks = tasks;
                        lTt += tasks;
                    }
                }
                bitSet[threadIdx.x] = sBS;
                // schedule observed tasks
                schedulerLoop<subscribers, sL, schedulerCount>(tqState, processors, tilesN1, sO, 0, lTt,
                    processorTally, gRQIdx, scheduled, sQ, rQ, pDB, true);

                for (uint i = 1; i < dT; ++i) {
                    const uint sBIdx = threadIdx.x + (i * dQL / bSw) * schedulerCount;
                    sBS = bitSet[sBIdx];
                    // Needed to enforce register storage
                    #pragma unroll
                    for (uint j = sL; j < tqState.size(); ++j) {
                        const auto pJ = j - sL;
                        const uint bIdx = (i * dQL + pJ) % bSw;
                        if (const auto isVisited = sBS.get(bIdx); !isVisited) {
                            const auto qIdx = schedulerCount * (dQL * i + pJ) + threadIdx.x;
                            cuda::atomic_ref<unsigned int, cuda::thread_scope_device> gtqH{*(gtQHeads + qIdx)};
                            const auto tasks = gtqH.load(cuda::memory_order_acquire);
                            if (tasks) {
                                sBS.set(bIdx);
                                gtqH.store(tQHeadGroundState, cuda::memory_order_relaxed);
                            }
                            tqState[j].tasks = tasks;
                            lTt += tasks;
                        }
                    }
                    bitSet[sBIdx] = sBS;
                    // schedule observed tasks
                    schedulerLoop<subscribers, sL, schedulerCount>(tqState,processors, tilesN1, sO, i * dQL,
                        lTt, processorTally, gRQIdx, scheduled, sQ, rQ, pDB);
                }
            }
            if (threadIdx.x < gtQCL - dT * dQL * schedulerCount) {
                const uint sBIdx = threadIdx.x + (dT * dQL / bSw) * schedulerCount;
                auto sBS = bitSet[sBIdx];
                // residue
                #pragma unroll
                for (uint j = sL; j < tqState.size(); ++j) {
                    const auto pJ = j - sL;
                    if (const auto qIdx = schedulerCount * (dQL * dT + pJ) + threadIdx.x; qIdx < gtQCL) {
                        const uint bIdx = (dT * dQL + pJ) % bSw;
                        if (const auto isVisited = sBS.get(bIdx); !isVisited) {
                            cuda::atomic_ref<unsigned int, cuda::thread_scope_device> gtqH{*(gtQHeads + qIdx)};
                            const auto tasks = gtqH.load(cuda::memory_order_acquire);
                            if (tasks) {
                                sBS.set(bIdx);
                                gtqH.store(tQHeadGroundState, cuda::memory_order_relaxed);
                            }
                            tqState[j].tasks = tasks;
                            lTt += tasks;
                        }
                    }
                }
                bitSet[sBIdx] = sBS;
            }
            // schedule observed tasks
            schedulerLoop<subscribers, sL, schedulerCount>(tqState, processors, tilesN1, sO, dQL * dT,
                lTt, processorTally, gRQIdx, scheduled, sQ, rQ, pDB, dT == 0);

            if (!threadIdx.x) {
                tTB = tb.load(cuda::memory_order_relaxed);
            }
            __syncwarp();
            tTB = __shfl_sync(0xffffffff, tTB, 0);
        }
        // interrupt subscribers
        #pragma unroll
        for (uint sid = threadIdx.x; sid < subscribers; sid += SCHEDULER_COUNT) {
            cuda::atomic_ref<uint, cuda::thread_scope_block> inr{*(sInterrupts + sid)};
            inr.store(1U, cuda::memory_order_release);
        }
        // interrupt processors
        sPI<schedulerCount>(rQ, sQ, pDB, gRQIdx, interruptScratch, processors, processorTally);
    }
}
#endif //SCHEDULER_CUH
