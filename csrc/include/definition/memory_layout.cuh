//
// Created by Jonathan on 7/21/24.
//

#ifndef ARISTOS_MEMORY_LAYOUT_CUH
#define ARISTOS_MEMORY_LAYOUT_CUH

#include "types.cuh"
namespace aristos::heap{
    template<unsigned int stage> requires (stage < STAGES)
    __device__ __forceinline__
    auto* advanceP2P(unsigned int const& peer) {
        return moeConfig.sHeap + moeConfig.capacity * moeConfig.embedDim * (peer * STAGES + stage);
    }

    template<unsigned int stage, unsigned cell> requires (stage < STAGES && cell < CELLS)
    __device__ __forceinline__
    auto* advanceRemote(unsigned int const& peer){
        // Note the following,
        // tokenStride = moeConfig.embedDim;
        // cellStride = moeConfig.capacity
        // stageStride = CELLS * cellStride
        // peerStride = STAGES * stageStride
        // Thus, indexing is a prefix product of the above quantities, namely
        // ptr = moeConfig.sHeap + (peer * peerStride) + (stage * stageStride) + (cell * cellStride)
        return moeConfig.sHeap + moeConfig.capacity * moeConfig.embedDim * (CELLS * (peer * STAGES + stage) + cell);
    }
}
#endif //ARISTOS_MEMORY_LAYOUT_CUH
