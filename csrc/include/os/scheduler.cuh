//
// Created by oja7 on 11/17/24.
//

#ifndef SCHEDULER_CUH
#define SCHEDULER_CUH

#include "../definition/types.cuh"

namespace aristos::scheduler {
    template<unsigned int processorCount, typename Registers, typename RScratch>
    requires(processorCount > 0 && isRegisterV<Registers> && isRegisterV<RScratch>)
    __device__ __forceinline__
    void scheduleLoop(unsigned int& tasks,
        unsigned int& scheduled,
        const unsigned int& wSIdx,
        Registers& rtQTails,
        RScratch& rQSet,
        unsigned int* __restrict__ const& rQHead,
        unsigned int& rQTail,
        const unsigned int* __restrict__ const& rQ,
        const unsigned int& tQRow,
        const unsigned int& tQRl,
        unsigned int* __restrict__ const& pDB) {
        while (tasks) {
            const auto readyProcesses = atomicLoad<cuda::thread_scope_block>(rQHead) - rQTail;
            const auto tasksToSchedule = cute::min(readyProcesses, tasks);
            tasks -= tasksToSchedule;
            scheduled += tasksToSchedule;
            if (tasksToSchedule) {
                // ensures reads to ready Q happen after signal reception
                __threadfence_block();
            }
            constexpr auto batchSize = RScratch::kElements;
            const auto tSb = tasksToSchedule / batchSize; // tasks to schedule batches
            for (uint i = 0; i < tSb; ++i) {
                #pragma unroll
                for (uint j = 0; j < batchSize; ++j) {
                    rQSet[j] = rQ[rQTail++ % processorCount];
                }

                #pragma unroll
                for (uint j = 0; j < batchSize; ++j) {
                    const auto pid = rQSet[j];
                    ++rtQTails[wSIdx];
                    atomicExch(pDB + pid, tQRow * tQRl + rtQTails[wSIdx]);
                }
            }
            const auto residue = tasksToSchedule - tSb * batchSize;
            for (uint i = 0; i < residue; ++i) {
                rQSet[i] = rQ[rQTail++ % processorCount];
            }

            for (uint i = 0; i < residue; ++i) {
                const auto pid = rQSet[i];
                ++rtQTails[wSIdx];
                // Fence is not needed prior to signal given task producers already issue one.
                atomicExch(pDB + pid, tQRow * tQRl + rtQTails[wSIdx]);
            }
        }
    }

    // all loops are unrolled
    template<unsigned int processorCount, unsigned int producerCount, unsigned int wSetSize,
    typename Registers, typename RScratch>
    requires(processorCount > 0 && producerCount > 0 && producerCount < 128 && wSetSize > 1 && wSetSize % 4 == 0
        && isRegisterV<Registers> && Registers::kElements == wSetSize
        && isRegisterV<RScratch>)
    __device__ __forceinline__
    void staticSchedule(unsigned int& scheduled,
        Registers& rtQTails,
        RScratch& rQSet,
        const unsigned int& tQRl,
        unsigned int* __restrict__ const& tQTails,
        unsigned int* __restrict__ const& tQHeads,
        unsigned int* __restrict__ const& rQHead,
        unsigned int& rQTail,
        const unsigned int* __restrict__ const& rQ,
        unsigned int* __restrict__ const& pDB) {
        constexpr auto producerBatches = producerCount / wSetSize;

        #pragma unroll
        for (uint i = 0; i < producerBatches; ++i) {
            // load wQ tails to registers
            #pragma unroll
            for (uint j = 0; j < wSetSize; ++j) {
                rtQTails[j] = tQTails[j + i * wSetSize];
            }
            #pragma unroll
            for (uint j = 0; j < wSetSize; ++j) {
                const auto tQRow = j + i * wSetSize;
                auto tasks = atomicLoad<cuda::thread_scope_block>(tQHeads + tQRow) - rtQTails[j];
                // Let's schedule these tasks
                scheduleLoop<processorCount>(tasks, scheduled, j, rtQTails, rQSet, rQHead,
                    rQTail, rQ, tQRow, tQRl, pDB);
            }

            // persist wQ tails
            #pragma unroll
            for (uint j = 0; j < wSetSize; ++j) {
                tQTails[j + i * wSetSize] = rtQTails[j];
            }
        }
        constexpr auto residue = producerCount - producerBatches * wSetSize;
        #pragma unroll
        for (uint j = 0; j < residue; ++j) {
            // prefetch to registers
            rtQTails[j] = tQTails[j + producerBatches * wSetSize];
        }

        // complete static scheduling
        #pragma unroll
        for (uint j = 0; j < residue; ++j) {
            const auto tQRow = j + producerBatches * wSetSize;
            auto tasks = atomicLoad<cuda::thread_scope_block>(tQHeads + tQRow) - rtQTails[j];
            scheduleLoop<processorCount>(tasks, scheduled, j, rtQTails, rQSet, rQHead,
                    rQTail, rQ, tQRow, tQRl, pDB);
        }

        #pragma unroll
        for (uint j = 0; j < residue; ++j) {
            // persist
            tQTails[j + producerBatches * wSetSize] = rtQTails[j];
        }
    }

    // dynamic trip, used for external task queue whose heads are in global memory not shared.
    template<unsigned int processorCount, unsigned int wSetSize, typename Registers, typename RScratch>
    requires(processorCount > 0 && wSetSize > 1 && wSetSize % 4 == 0
        && isRegisterV<Registers> && Registers::kElements == wSetSize
        && isRegisterV<RScratch>)
    __device__ __forceinline__
    void dynamicSchedule(unsigned int& scheduled,
        Registers& rtQTails,
        RScratch& rQSet,
        const unsigned int& tQRl,
        unsigned int* __restrict__ const& tQTails,
        unsigned int* __restrict__ const& tQHeads,
        unsigned int* __restrict__ const& rQHead,
        unsigned int& rQTail,
        const unsigned int* __restrict__ const& rQ,
        unsigned int* __restrict__ const& pDB,
        const unsigned int& taskMailboxes) {
        const auto mailboxBatches = taskMailboxes / wSetSize;

        for (uint i = 0; i < mailboxBatches; ++i) {
            // load wQ tails to registers
            #pragma unroll
            for (uint j = 0; j < wSetSize; ++j) {
                rtQTails[j] = tQTails[j + i * wSetSize];
            }
            #pragma unroll
            for (uint j = 0; j < wSetSize; ++j) {
                const auto tQRow = j + i * wSetSize;
                auto tasks = atomicLoad(tQHeads + tQRow) - rtQTails[j];
                // Let's schedule these tasks
                scheduleLoop<processorCount>(tasks, scheduled, j, rtQTails, rQSet, rQHead,
                    rQTail, rQ, tQRow, tQRl, pDB);
            }

            // persist wQ tails
            #pragma unroll
            for (uint j = 0; j < wSetSize; ++j) {
                tQTails[j + i * wSetSize] = rtQTails[j];
            }
        }
        const auto residue = taskMailboxes - mailboxBatches * wSetSize;
        for (uint j = 0; j < residue; ++j) {
            // prefetch to registers
            rtQTails[j] = tQTails[j + mailboxBatches * wSetSize];
        }

        for (uint j = 0; j < residue; ++j) {
            const auto tQRow = j + mailboxBatches * wSetSize;
            auto tasks = atomicLoad(tQHeads + tQRow) - rtQTails[j];
            scheduleLoop<processorCount>(tasks, scheduled, j, rtQTails, rQSet, rQHead,
                    rQTail, rQ, tQRow, tQRl, pDB);
        }

        for (uint j = 0; j < residue; ++j) {
            // persist
            tQTails[j + mailboxBatches * wSetSize] = rtQTails[j];
        }
    }
    /// Making processorCount a compile-time constant is not a functional requirement but rather strictly
    /// for optimizing the modulo operation, which is incredibly expensive.
    /// Benchmarks show an order of magnitude performance difference for runtime vs. compile-time evaluation
    template<unsigned int processorCount, unsigned int producerCount>
    requires(processorCount > 0 && producerCount > 0 && producerCount < THREADS)
    __device__ __forceinline__
    void start(const unsigned int& tQRl,
        const unsigned int& gtQCL,
        const unsigned int& gtQRl,
        unsigned int* __restrict__ const& tQHeads,
        unsigned int* __restrict__ const& gtQHeads,
        unsigned int* __restrict__ const& gtQTails,
        unsigned int* __restrict__ const& tQTails,
        unsigned int* __restrict__ const& taskBound,
        unsigned int* __restrict__ const& rQHead,
        const unsigned int* __restrict__ const& rQ,
        unsigned int* __restrict__ const& pDB) {
        // assert(__isShared(all arguments above) and alignment is 16)
        unsigned int scheduled = 0U;
        unsigned int rQTail = 0U;
        constexpr auto wSetSize = 16;
        constexpr auto rQSetSize = 16;
        cutlass::AlignedArray<unsigned int, wSetSize> rtQTails{};
        cutlass::AlignedArray<unsigned int, rQSetSize> rQSet{};
        rtQTails.fill(0U);

        while (scheduled < atomicLoad<cuda::thread_scope_block>(taskBound)) {
            // sweep all static wQHeads and schedule pending tasks
            staticSchedule<processorCount, producerCount, wSetSize>(scheduled, rtQTails, rQSet, tQRl, tQTails,
                tQHeads, rQHead, rQTail, rQ, pDB);
            // Now do dynamic scheduling
            dynamicSchedule<processorCount, wSetSize>(scheduled, rtQTails, rQSet, gtQRl, gtQTails,
                gtQHeads, rQHead, rQTail, rQ, pDB, gtQCL);
        }
    }
}
#endif //SCHEDULER_CUH
