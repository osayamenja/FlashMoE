//
// Created by Jonathan on 7/21/24.
//

#ifndef ARISTOS_MEMORY_LAYOUT_CUH
#define ARISTOS_MEMORY_LAYOUT_CUH

#include "types.cuh"
namespace aristos{
    CUTE_DEVICE
    cuda::std::byte* getTokenPointer(unsigned int const& peer, unsigned int const& stage, unsigned int const& cell, unsigned int const& token){
        // Note the following,
        // tokenStride = (moeConfig.embedDim + moeConfig.k + 2);
        // cellStride = moeConfig.capacity * tokenStride
        // stageStride = CELLS * cellStride
        // peerStride = STAGES * stageStride
        // Thus, indexing is a prefix product of the above quantities, namely
        // ptr = moeConfig.sHeap + ((peer * peerStride) + (stage * stageStride) + (cell * cellStride) + (token * tokenStride))
        // However, the above would require storing those variables, which we would rather avoid.
        // Computing them on-the-fly requires 18 arithmetic operations and 11 reads,
        // whereas the below demands 9 operations and 3 reads.
        return moeConfig.sHeap + ((moeConfig.embedDim + moeConfig.k + 2) * ((moeConfig.capacity * (CELLS * ((STAGES * peer) + stage)) + cell) + token));
    }

}
#endif //ARISTOS_MEMORY_LAYOUT_CUH
