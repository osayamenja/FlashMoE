//
// Created by Jonathan on 7/21/24.
//

#ifndef ARISTOS_MEMORY_LAYOUT_CUH
#define ARISTOS_MEMORY_LAYOUT_CUH

#include "types.cuh"
namespace aristos::heap{
    template<unsigned int stage = 0, unsigned cell = 0>
    requires (stage < STAGES && cell < CELLS)
    __device__ __forceinline__
    auto* advance(unsigned int const& peer, unsigned int const& expert) {
        return moeConfig.sHeap + moeConfig.capacity * moeConfig.embedDim * (CELLS * (peer * STAGES + stage) + cell);
    }
}
#endif //ARISTOS_MEMORY_LAYOUT_CUH
