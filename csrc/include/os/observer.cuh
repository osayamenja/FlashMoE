//
// Created by oja7 on 12/26/24.
//

#ifndef OBSERVER_CUH
#define OBSERVER_CUH

#include <cutlass/array.h>

#include "../definition/types.cuh"
#include "../util/atomics.cuh"
namespace aristos::ReadyQ {
    /// readyQ Observer
    /// only actor that writes to the rQ
    template<unsigned int processorCount, unsigned int setSize>
    requires(processorCount > 0 && setSize > 1 && cutlass::ispow2(setSize))
    __device__ __forceinline__
    void observe(unsigned int* __restrict__ const& rQHead,
        unsigned int* __restrict__ const& rQ,
        unsigned int* __restrict__ const& interrupt) {
        //assert(__isShared(all arguments))
        auto* __restrict__ sQ = schedulerState.statusQ;
        unsigned int lRQHead = *rQHead; // assert(*rQHead == 0)
        cutlass::AlignedArray<unsigned int, setSize> cacheSet{};
        while (!atomicLoad<cuda::thread_scope_block>(interrupt)) {
            // linearly search sQ for "ready" processes
            unsigned int readySeen = 0U;
            #pragma unroll
            for (uint i = 0; i < processorCount; ++i) {
                if (atomicExch(sQ + i, observed) == ready) {
                    cacheSet[readySeen++] = i;
                    if (readySeen == setSize) {
                        // batch insert pids to the readyQ
                        #pragma unroll
                        for (uint j = 0; j < setSize; ++j) {
                            // register to shared memory transfer
                            rQ[lRQHead++ % processorCount] = cacheSet[j];
                        }
                        __threadfence_block();
                        // notify scheduler of inserts
                        atomicAdd_block(rQHead, setSize);
                    }
                }
            }
            // Address the unfortunate case where the loop terminates and readySeen < setSize
            for (uint j = 0; j < readySeen; ++j) {
                rQ[lRQHead++ % processorCount] = cacheSet[j];
            }
            __threadfence_block();
            atomicAdd_block(rQHead, readySeen);
        }
        // Adios!
    }
}

#endif //OBSERVER_CUH
