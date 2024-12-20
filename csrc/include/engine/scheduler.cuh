//
// Created by oja7 on 11/17/24.
//

#ifndef SCHEDULER_CUH
#define SCHEDULER_CUH

#include "../definition/types.cuh"

namespace aristos::scheduler {
    // fast, batch scheduling with register and shared memory pipelining
    // Gives about 3.5x speedup over vanilla scheduling,
    // see https://github.com/osayamenja/cuPlayground/blob/master/examples/pipelining.cuh
    template<
        unsigned int scratchSize,
        unsigned int regSize,
        unsigned int processorCount,
        typename RegisterScratch
    >
    requires(processorCount > 0 && scratchSize > 0 && regSize > 0 &&
        scratchSize >= regSize && scratchSize % regSize == 0)
    __device__ __forceinline__
    void fastSchedule(RegisterScratch& registerScratch,
        unsigned int& tasksToSchedule,
        unsigned int& scheduled,
        unsigned int& rQTail,
        unsigned int* __restrict__ fastScratch,
        unsigned int* __restrict__ rQ,
        unsigned int* __restrict__ pDB) {
        constexpr auto vectorLength = sizeof(uint4) / sizeof(unsigned int);
        static_assert(regSize % vectorLength == 0, "regSize must be a multiple of vectorLength");
        constexpr auto vectorSize = scratchSize / vectorLength;
        // Batch reads from global memory
        const auto trips = tasksToSchedule / scratchSize;
        // Quickly prefetch pids to fast shared memory
        if (trips) {
            #pragma unroll
            for (uint i = 0; i < vectorSize; i++) {
                CAST_TO(uint4, fastScratch)[i] = CAST_TO(uint4, rQ)[rQTail % processorCount];
                rQTail += vectorLength;
            }
        }

        for (uint i = 0; i < trips; i++) {
            constexpr auto regTrips = scratchSize / registerScratch.size();
            #pragma unroll
            for (uint j = 0; j < regTrips; j++) {
                #pragma unroll
                for (uint k = 0; k < regSize; k++) {
                    registerScratch[k] = fastScratch[k + j * regSize];
                }

                // Eagerly prefetch next batch to smem
                if (i + 1 < trips) {
                    constexpr auto rVs = regSize / vectorLength;
                    #pragma unroll
                    for (uint k = 0; k < rVs; k++) {
                        CAST_TO(uint4, fastScratch + j * regSize)[k] = CAST_TO(uint4, rQ)[rQTail % processorCount];
                        rQTail += vectorLength;
                    }
                }
                #pragma unroll
                for (uint k = 0; k < regSize; k++) {
                    const auto pid = registerScratch[k];
                    --tasksToSchedule;
                    // Inform this process of a single task
                    atomicExch(pDB + pid, ++scheduled);
                }
            }
        }

        // Handle residue
        const auto residue = tasksToSchedule - trips * scratchSize;
        const auto vR = residue / vectorLength;
        const auto rVr = residue - vR * vectorLength;
        for (uint i = 0; i < vR; i++) {
            CAST_TO(uint4, fastScratch)[i] = CAST_TO(uint4, rQ)[rQTail % processorCount];
            rQTail += vectorLength;
        }
        // Residual shared trip
        for (uint i = 0; i < rVr; i++) {
            // Stage in shared memory
            fastScratch[i] = rQ[rQTail++ % processorCount];
        }

        const auto regTrips = residue / registerScratch.size();
        for (uint i = 0; i < regTrips; i++) {
            #pragma unroll
            for (uint j = 0; j < regSize; j++) {
                registerScratch[j] = fastScratch[j + i * regSize];
            }

            #pragma unroll
            for (uint j = 0; j < regSize; j++) {
                const auto pid = registerScratch[j];
                --tasksToSchedule;
                // Inform this process of a single task
                atomicExch(pDB + pid, ++scheduled);
            }
        }

        // Residual register trip
        const auto rRt = residue - regTrips * regSize;
        for (uint j = 0; j < rRt; j++) {
            registerScratch[j] = fastScratch[j + regTrips * regSize];
        }

        for (uint j = 0; j < rRt; j++) {
            const auto pid = registerScratch[j];
            --tasksToSchedule;
            // Inform this process of a single task
            atomicExch(pDB + pid, ++scheduled);
        }
    }

    /// Making processorCount a compile-time constant is not a functional requirement but rather strictly
    /// for optimizing the modulo operation, which is incredibly expensive.
    /// Micro-benchmarks show 10 vs. 100 ns performance difference.
    template<unsigned int processorCount, unsigned int scratchSize = 64U>
    requires(processorCount > 0 && scratchSize >= 64 && scratchSize % 64 == 0 && cutlass::ispow2(scratchSize))
    __device__ __forceinline__
    void start(unsigned int* __restrict__ taskBound, unsigned int* __restrict__ fastScratch) {
        constexpr auto schedulerRegSize = 32U;
        static_assert(scratchSize >= schedulerRegSize && scratchSize % schedulerRegSize == 0,
            "scratchSize must be a multiple of register scratch size");
        // Register allocations
        auto* rQ = schedulerState.readyQ;
        auto* tQ = schedulerState.taskQ;
        auto* pDB = schedulerState.taskSignal;
        auto* rQHead = schedulerState.readyQHead;
        auto* doorbell = schedulerState.taskQSignals;
        auto* tQHead = schedulerState.taskQSignals + 1;
        unsigned int scheduled = 0U;
        unsigned int rQTail = 0U;
        cutlass::AlignedArray<unsigned int, schedulerRegSize> registerScratch{};

        while (scheduled < atomicLoad<cuda::thread_scope_block>(taskBound)) {
            // Batch process doorbell
            auto tasks = atomicLoad(doorbell) - scheduled;
            while (tasks) {
                auto readyProcesses = atomicLoad(rQHead) - rQTail;
                // The below if is not strictly needed as no operation will execute if tasksToSchedule == 0
                // However, that scenario will involve many more conditionals than just a single if check like below
                if (auto tasksToSchedule = cute::min(tasks, readyProcesses); tasksToSchedule) {
                    tasks -= tasksToSchedule;
                    fastSchedule<scratchSize, schedulerRegSize, processorCount>(
                        registerScratch,
                        tasksToSchedule,
                        scheduled,
                        rQTail,
                        fastScratch,
                        rQ,
                        pDB);
                }
            }
        }

        #pragma unroll
        for (unsigned int i = 0; i < processorCount; ++i) {
            tQ[atomicAdd(tQHead, 1U)] = Task{TaskType::Interrupt};
        }
        __threadfence();

        unsigned int interrupted = 0U;
        auto tasks = processorCount;
        while (tasks) {
            auto readyProcesses = atomicLoad(rQHead) - rQTail;
            if (auto tasksToSchedule = cute::min(tasks, readyProcesses); tasksToSchedule) {
                tasks -= tasksToSchedule;
                fastSchedule<scratchSize, schedulerRegSize, processorCount>(
                    registerScratch,
                    tasksToSchedule,
                    interrupted,
                    rQTail,
                    fastScratch,
                    rQ,
                    pDB);
            }
        }
    }
}
#endif //SCHEDULER_CUH
