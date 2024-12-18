//
// Created by oja7 on 11/17/24.
//

#ifndef SCHEDULER_CUH
#define SCHEDULER_CUH

#include "../definition/types.cuh"

namespace aristos::scheduler {
    /// Making processorCount a compile-time constant is not a functional requirement but rather strictly
    /// for optimizing the modulo operation, which is incredibly expensive.
    /// Micro-benchmarks show 10 vs. 100 ns performance difference.
    template<unsigned int processorCount> requires(processorCount > 0)
    __device__ __forceinline__
    void start(unsigned int* taskBound) {
        // Register allocations
        const auto* rQ = schedulerState.readyQ;
        auto* tQ = schedulerState.taskQ;
        auto* pDB = schedulerState.taskSignal;
        auto* rQHead = schedulerState.readyQHead;
        auto* doorbell = schedulerState.taskQSignals;
        auto* tQHead = schedulerState.taskQSignals + 1;
        unsigned int scheduled = 0U;
        unsigned int rQTail = 0U;

        while (scheduled < atomicLoad<cuda::thread_scope_block>(taskBound)) {
            auto tasks = atomicLoad(doorbell) - scheduled;
            // Batch read to global memory
            while (tasks > 0) {
                auto readyProcesses = atomicLoad(rQHead) - rQTail;
                while (readyProcesses > 0) {
                    // Gets a ready processor
                    const auto pid = rQ[rQTail++ % processorCount];
                    // Update state
                    --tasks;
                    --readyProcesses;
                    // Inform them of a single task
                    atomicExch(pDB + pid, ++scheduled);
                }
            }
        }

        #pragma unroll
        for (unsigned int i = 0; i < processorCount; ++i) {
            tQ[atomicAdd(tQHead, 1U)] = Task{TaskType::Interrupt};
        }
        __threadfence();

        unsigned int interrupted = 0U;
        while (interrupted < processorCount) {
            auto readyProcesses = atomicLoad(rQHead) - rQTail;
            while (readyProcesses > 0) {
                // Gets a ready processor
                const auto pid = rQ[rQTail++ % processorCount];
                // Update state
                --readyProcesses;
                ++interrupted;
                // Inform them of a single task
                atomicExch(pDB + pid, ++scheduled);
            }
        }
    }
}
#endif //SCHEDULER_CUH
