//
// Created by oja7 on 11/17/24.
//

#ifndef SCHEDULER_CUH
#define SCHEDULER_CUH

#include <cub/cub.cuh>
#include <cutlass/array.h>

#include "../atomics.cuh"
#include "../types.cuh"

namespace aristos::scheduler {
    template<
        unsigned int processors,
        DQType dqt = DQType::stride,
        typename WSet
    >
    requires(processors > 0 && isRegisterV<WSet>)
    __device__ __forceinline__
    void schedule(WSet& wSet, const uint& cSetB,
    const uint& canSchedule, const uint& qIdx, uint& lRQIdx,
    const uint& gRQIdx, uint* __restrict__ const& rQ,
    TQSignal* __restrict__ const& pDB) {
        auto sig = TQSignal{0U};
        for (uint k = 0; k < cSetB; ++k) {
            #pragma unroll
            for (uint l = 0; l < WSet::kElements; ++l) {
                wSet[l] = rQ[(gRQIdx + lRQIdx++) % processors];
            }
            #pragma unroll
            for (uint l = 0; l < WSet::kElements; ++l) {
                // signal processor
                sig.encodeSig(DQ::next<dqt>(qIdx, k * WSet::kElements + l));
                signalPayload(pDB + wSet[l], &sig);
            }
        }
        // Residual scheduling
        const auto residue = canSchedule - cSetB * WSet::kElements;
        #pragma unroll
        for (uint l = 0; l < WSet::kElements; ++l) {
            if (l < residue) {
                wSet[l] = rQ[(gRQIdx + lRQIdx++) % processors];
            }
        }
        #pragma unroll
        for (uint l = 0; l < WSet::kElements; ++l) {
            if (l < residue) {
                sig.encodeSig(DQ::next<dqt>(qIdx, cSetB * WSet::kElements + l));
                signalPayload(pDB + wSet[l], &sig);
            }
        }
    }

    template<
        unsigned int processors,
        unsigned int sL = (THREADS - WARP_SIZE) / WARP_SIZE,
        unsigned int wS = WARP_SIZE,
        unsigned int blockQStride = ACC::TNx::value,
        typename WarpScan = cub::WarpScan<uint>,
        typename SQState,
        typename TQState,
        typename WSet
    >
    requires (processors > 0 && wS == 32 &&
        isRegisterV<SQState> && isRegisterV<TQState> && isRegisterV<WSet>)
    __device__ __forceinline__
    void schedulerLoop(SQState& sQState, TQState& tqState, WSet& wSet,
        const unsigned int& tQOffset,
        uint& lTt, uint& processorTally,
        uint& gRQIdx, uint& scheduled,
        typename WarpScan::TempStorage* __restrict__ const& wSt,
        unsigned int* __restrict__ const& sQ,
        uint* __restrict__ const& rQ,
        TQSignal* __restrict__ const& pDB,
        const bool& isMedley = false) {
        uint lRQIdx;
        uint taskTally;
        // things are about to get warped :)
        // Aggregate tally across the warp
        WarpScan(wSt[0]).InclusiveSum(lTt, lRQIdx, taskTally);
        lRQIdx -= lTt;
        auto prefixTaskSum = 0U;
        while (taskTally) {
            // Find processors if we are not currently aware of any
            while (!processorTally) {
                // sweep sQ to identify ready processes
                uint lPt = 0U; // local processor tally
                constexpr auto pL = processors / wS;
                #pragma unroll
                for (uint j = 0; j < pL; ++j) {
                    const auto readiness = atomicExch(sQ + (j * wS + threadIdx.x),
                        observed) == ready;
                    lPt += readiness;
                    sQState[j] = readiness;
                }
                if (threadIdx.x < processors - pL * wS) {
                    const auto readiness = atomicExch(sQ + (pL * wS + threadIdx.x),
                        observed) == ready;
                    lPt += readiness;
                    sQState[pL] = readiness;
                }
                uint startIdx;
                // Aggregate tally across the warp
                WarpScan(wSt[1]).InclusiveSum(lPt, startIdx, processorTally);
                startIdx -= lPt;
                // write to rQ
                #pragma unroll
                for (uint j = 0; j < SQState::kElements; ++j) {
                    if (sQState[j]) {
                        // write ready process pid to rQ
                        rQ[(gRQIdx + startIdx++) % processors] = j * wS + threadIdx.x;
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
            if (lTt > 0 && lRQIdx < prefixTaskSum) {
                auto tasksToSchedule = lRQIdx + lTt > tasks ? tasks - lRQIdx : lTt;
                lTt -= tasksToSchedule;
                if (isMedley) {
                    if constexpr (sL > 0) {
                        #pragma unroll
                        for (uint j = 0; j < sL; ++j) {
                            if (tqState[j].tasks > 0 && tasksToSchedule) {
                                const auto canSchedule = cute::min(tasksToSchedule, tqState[j].tasks);
                                const auto qIdx = DQ::next(j * wS + threadIdx.x, tqState[j].tQTail);
                                tasksToSchedule -= canSchedule;
                                tqState[j].tasks -= canSchedule;
                                // have to increment tails as we will revisit this queue later on
                                tqState[j].tQTail += canSchedule;
                                const auto cSetB = canSchedule / WSet::kElements;
                                schedule<processors>(wSet, cSetB, canSchedule, qIdx,
                                    lRQIdx, gRQIdx, rQ, pDB);
                            }
                        }
                    }
                }
                #pragma unroll
                for (uint j = sL; j < TQState::kElements; ++j) {
                    if (tqState[j].tasks && tasksToSchedule) {
                        const auto canSchedule = cute::min(tasksToSchedule, tqState[j].tasks);
                        const auto qHead = ((j - sL) * wS + threadIdx.x) * blockQStride;
                        const auto qIdx = tQOffset + qHead;
                        tasksToSchedule -= canSchedule;
                        tqState[j].tasks -= canSchedule;
                        const auto cSetB = canSchedule / WSet::kElements;
                        schedule<processors, DQType::block>(wSet, cSetB, canSchedule,
                            qIdx, lRQIdx, gRQIdx, rQ, pDB);
                    }
                }
            }
            gRQIdx = (gRQIdx + tasks) % processors;
        }
    }

    template<
        unsigned int processors,
        unsigned int wS = WARP_SIZE,
        typename WarpScan = cub::WarpScan<uint>,
        typename SQState
    >
    /// Schedule Processor interrupts
    __device__ __forceinline__
    void sPI(SQState& sQState,
        unsigned int* __restrict__ const& rQ,
        uint* __restrict__ const& sQ,
        TQSignal* __restrict__ const& pDB,
        uint& gRQIdx,
        typename WarpScan::TempStorage* __restrict__ const& wSt,
        uint* __restrict__ scratch,
        const uint& processorTally) {
        static_assert(cuda::std::is_same_v<typename SQState::value_type, uint>);
        #pragma unroll
        for (uint i = threadIdx.x; i < processors; i += wS) {
            scratch[i] = 1U;
        }
        __syncwarp();
        /// read through the ready queue first
        constexpr auto sig = TQSignal{1U};
        // Below must be <= ceil(processors / wS) == sQsL, so we can repurpose sQState registers as temporary storage
        const auto tS = processorTally / wS + (threadIdx.x < processorTally % wS);
        const auto gRO = gRQIdx + (threadIdx.x * (processorTally / wS) +
            cute::min(threadIdx.x, processorTally % wS));
        // index can only wrap around once
        auto unguarded = gRO >= processors;
        gRQIdx = gRO % processors;
        unguarded = unguarded || gRQIdx + tS - 1 < processors;
        if (unguarded) {
            #pragma unroll
            for (uint i = 0; i < SQState::kElements; ++i) {
                if (i < tS) {
                    // shared -> registers
                    sQState[i] = rQ[gRQIdx + i];
                }
            }
        }
        else {
            #pragma unroll
            for (uint i = 0; i < SQState::kElements; ++i) {
                if (i < tS) {
                    // shared -> registers
                    sQState[i] = rQ[(gRQIdx + i) % processors];
                }
            }
        }
        #pragma unroll
        for (uint i = 0; i < SQState::kElements; ++i) {
            if (i < tS) {
                // notify interrupts
                const auto pid = sQState[i];
                signalPayload(pDB + pid, &sig);
                scratch[pid] = 0U;
            }
        }
        __syncwarp();
        constexpr auto pL = processors / wS;
        // Consolidate findings and populate the ready queue
        auto uI = 0U;
        // shared -> registers
        #pragma unroll
        for (uint i = 0; i < pL; ++i) {
            sQState[i] = scratch[i * wS + threadIdx.x];
        }
        if (threadIdx.x < processors % wS) {
            sQState[pL] = scratch[pL * wS + threadIdx.x];
        }

        #pragma unroll
        for (uint i = 0; i < pL; ++i) {
            uI += sQState[i];
        }
        if (threadIdx.x < processors % wS) {
            uI += sQState[pL];
        }

        uint startIdx;
        uint pending;
        WarpScan(*wSt).InclusiveSum(uI, startIdx, pending);
        startIdx -= uI;
        // enqueue all pending processes we discovered into the rQ
        #pragma unroll
        for (uint i = 0; i < pL; ++i) {
            if (sQState[i]) {
                rQ[startIdx++] = i * wS + threadIdx.x;
            }
        }
        if (threadIdx.x < processors % wS && sQState[pL]) {
            rQ[startIdx++] = pL * wS + threadIdx.x;
        }
        __syncwarp();
        auto remaining = pending / wS + (threadIdx.x < pending % wS);
        cuda::std::array<uint, SQState::kElements> pids{};
        // read from rQ to registers
        #pragma unroll
        for (uint i = 0; i < SQState::kElements; ++i) {
            const auto idx = i * wS + threadIdx.x;
            sQState[i] = idx < pending;
            if (sQState[i]) {
                pids[i] = rQ[idx];
            }
        }

        while (remaining) {
            #pragma unroll
            for (uint j = 0; j < SQState::kElements; ++j) {
                if (sQState[j]) {
                    const auto pid = pids[j];
                    const auto isReady = atomicExch(sQ + pid, observed) == ready;
                    sQState[j] = !isReady;
                    if (isReady) {
                        // interrupt processor
                        remaining -= 1;
                        signalPayload(pDB + pid, &sig);
                    }
                }
            }
        }
    }

    /// Making processorCount a compile-time constant is not a functional requirement but rather strictly
    /// for globally optimizing the modulo operation, which is incredibly expensive.
    /// Benchmarks confirm an order of magnitude performance improvement for that operation.
    template<
        unsigned int processors,
        unsigned int subscribers = SUBSCRIBERS
    >
    requires(processors > 0)
    __device__ __forceinline__
    void start(uint* __restrict__ const& workspace,
        const unsigned int& sO,
        const unsigned int& gtQCL,
        unsigned int* __restrict__ const& sInterrupts,
        unsigned int* __restrict__ const& tQHeads, // shared
        unsigned int* __restrict__ const& gtQHeads, // global
        unsigned int* __restrict__ const& taskBound, // shared
        unsigned int* __restrict__ const& rQ, // shared
        unsigned int* __restrict__ const& sQ, // global
        TQSignal* __restrict__ const& pDB) { //  global
        uint scheduled = 0U;
        constexpr auto wS = 32U;
        constexpr auto sQsL = cute::ceil_div(processors, wS);
        static_assert(sQsL <= 32);

        static_assert(subscribers % wS == 0);
        constexpr auto sL = subscribers / wS;
        // initialize register buffers
        constexpr auto gTQWSet = 8;
        cutlass::Array<TQState, gTQWSet + sL> tqState{};
        cutlass::Array<uint, sQsL> sQState{};
        constexpr auto wSz = 8U;
        cutlass::Array<uint, wSz> wSet{};
        tqState.fill({0U,0U});
        sQState.fill(0U);

        constexpr auto dQL = decltype(tqState)::kElements - sL;
        const uint dT = gtQCL / (wS * dQL);

        // cub stuff
        using WarpScan = cub::WarpScan<uint>;
        static_assert(alignof(uint) % alignof(WarpScan::TempStorage) == 0);
        auto* __restrict__ wSt = CAST_TO(WarpScan::TempStorage, workspace + processors);
        uint gRQIdx = 0U;
        uint processorTally = processors; // initially, all processors are available, ensure that rQ has all pids
        auto tTB = atomicLoad<cuda::thread_scope_block>(taskBound);
        while (scheduled < tTB) {
            // statically sweep tQ for tasks
            uint lTt = 0U; // local task tally
            #pragma unroll
            for (uint i = 0; i < sL; ++i) {
                const auto tasks = atomicLoad<cuda::thread_scope_block>(tQHeads +
                    (i * wS + threadIdx.x)) - tqState[i].tQTail;
                tqState[i].tasks = tasks;
                lTt += tasks;
            }

            // Abstract queues as a 3-D tensor (B, Q, T),
            // where B is the batch dimension or total queue / (Q * T);
            // Q is the number of queues a thread observes in one-pass;
            // and T is the number of threads in a warp
            // TODO guard against atomic checks using bitset
            if (dT > 0) {
                // One-shot scheduling, so tails are irrelevant.
                // Special case, where i == 0
                #pragma unroll
                for (uint j = sL; j < decltype(tqState)::kElements; ++j) {
                    const auto qIdx = wS * (j - sL) + threadIdx.x;
                    const auto tasks = atomicExch(gtQHeads + qIdx, tQHeadGroundState);
                    tqState[j].tasks = tasks;
                    lTt += tasks;
                }
                // schedule observed tasks
                schedulerLoop<processors>(sQState, tqState, wSet, sO, lTt,
                    processorTally, gRQIdx, scheduled,
                    wSt, sQ, rQ, pDB, true);

                for (uint i = 1; i < dT; ++i) {
                    // Needed to enforce register storage
                    #pragma unroll
                    for (uint j = sL; j < decltype(tqState)::kElements; ++j) {
                        const auto qIdx = wS * (dQL * i + (j - sL)) + threadIdx.x;
                        const auto tasks = atomicExch(gtQHeads + qIdx, tQHeadGroundState);
                        tqState[j].tasks = tasks;
                        lTt += tasks;
                    }
                    // schedule observed tasks
                    schedulerLoop<processors>(sQState, tqState, wSet, sO, lTt,
                        processorTally, gRQIdx, scheduled,
                        wSt, sQ, rQ, pDB);
                }
            }

            // residue
            #pragma unroll
            for (uint j = sL; j < decltype(tqState)::kElements; ++j) {
                if (const auto qIdx = wS * (dQL * dT + (j - sL)) + threadIdx.x; qIdx < gtQCL) {
                    const auto tasks = atomicExch(gtQHeads + qIdx, tQHeadGroundState);
                    tqState[j].tasks = tasks;
                    lTt += tasks;
                }
            }
            // schedule observed tasks
            schedulerLoop<processors>(sQState, tqState, wSet, sO, lTt,
                processorTally, gRQIdx, scheduled,
                wSt, sQ, rQ, pDB, dT == 0);
            if (!threadIdx.x) {
                tTB = atomicLoad<cuda::thread_scope_block>(taskBound);
            }
            tTB = __shfl_sync(0xffffffff, tTB, 0);
        }
        if (!threadIdx.x) {
            printf("About to interrupt\n");
        }
        // interrupt subscribers
        static_assert(subscribers % wS == 0);
        #pragma unroll
        for (uint i = 0; i < sL; ++i) {
            const auto sid = i * wS + threadIdx.x;
            atomicExch_block(sInterrupts + sid, 1U);
        }
        // interrupt processors
        sPI<processors, wS>(sQState, rQ, sQ, pDB, gRQIdx, wSt, workspace, processorTally);
    }
}
#endif //SCHEDULER_CUH
