//
// Created by Jonathan on 7/4/24.
//

#ifndef ARISTOS_QUEUE_CUH
#define ARISTOS_QUEUE_CUH

#include <nvshmemx.h>
#include <nvshmem.h>
#include <host/nvshmemx_api.h>

namespace aristos::subscriber{
    ///Receive and decode packets deposited
    template<unsigned int subscriberCount = THREADS - 2>
    __device__ __forceinline__
    void start(unsigned int* __restrict__ const& interrupt){
        auto* __restrict__ flags = moeConfig.flags;
        const auto fSfC = moeConfig.worldSize * moeConfig.numExperts; // first stage flag count
        const auto fSl = fSfC / subscriberCount + (threadIdx.x < fSfC % subscriberCount);
        while (!atomicLoad<cuda::thread_scope_block>(interrupt)) {
            // sweep through flags by stages
            // TODO if necessary precondition this loop to execute only when necessary
            // doing so would consume already scarce shared memory
            for (uint i = 0; i < fSl; ++i) {
                const auto flagIdx = threadIdx.x + i * subscriberCount;

            }
        }
    }
}
#endif //ARISTOS_QUEUE_CUH
