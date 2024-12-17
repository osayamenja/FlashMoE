//
// Created by Jonathan on 7/21/24.
//

#ifndef ARISTOS_MEMORY_LAYOUT_CUH
#define ARISTOS_MEMORY_LAYOUT_CUH

#include "types.cuh"
namespace aristos::heap{
    template<HeapType h = HeapType::remote, unsigned int stage = 0, unsigned cell = 0>
    requires (stage < STAGES && cell < CELLS)
    __device__ __forceinline__
    auto* advance(unsigned int const& peer) {
        if constexpr (h == HeapType::remote) {
            return moeConfig.sHeap + moeConfig.capacity * moeConfig.embedDim * (CELLS * (peer * STAGES + stage) + cell);
        }
        return moeConfig.p2pSHeap + moeConfig.capacity * moeConfig.embedDim * (peer * STAGES + stage);
    }
}
#endif //ARISTOS_MEMORY_LAYOUT_CUH
