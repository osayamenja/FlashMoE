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
    void start() {
        auto* rQHead = schedulerState.readyQSignals;
        auto* rQTail = schedulerState.readyQSignals + 1;
        auto* doorbell = schedulerState.taskQSignals;
        auto* tQHead = schedulerState.taskQSignals + 1;
        unsigned long long int scheduled = 0U;
        while (scheduled < schedulerState.taskBound) {
            while (atomicLoad(doorbell) > scheduled && atomicLoad(rQHead) > atomicLoad(rQTail)) {
                scheduled++;
                // Gets a ready processor
                const auto pid = schedulerState.readyQ[atomicAdd(rQTail, 1U) % processorCount];
                // Inform them of a single task
                atomicExch(schedulerState.taskSignal + pid, scheduled);
            }
        }

        #pragma unroll
        for (unsigned int i = 0; i < processorCount; ++i) {
            schedulerState.taskQ[atomicAdd(tQHead, 1U)] = Task{TaskType::Interrupt};
        }
        __threadfence();

        unsigned int interrupted = 0U;
        while (interrupted < processorCount) {
            while (atomicLoad(rQHead) > atomicLoad(rQTail)) {
                scheduled++;
                const auto pid = schedulerState.readyQ[atomicAdd(rQTail, 1U) % processorCount];
                atomicExch(schedulerState.taskSignal + pid, scheduled);
                interrupted++;
            }
        }
    }
}
#endif //SCHEDULER_CUH
