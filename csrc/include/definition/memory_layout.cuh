//
// Created by Jonathan on 7/21/24.
//

#ifndef ARISTOS_MEMORY_LAYOUT_CUH
#define ARISTOS_MEMORY_LAYOUT_CUH

/// The symmetric heap is a 4-D tensor (P, S, C, T)
/// where P, S, C, and T denote dimensions for peers, communication stages,
/// cells and tokens.
/// Number of communication stages S
#define STAGES 2

/// Per stage, there is one cell for send and another for receive
#define CELLS 2
#define SEND_CELL 0
#define RECEIVE_CELL 1

#define HEAP_ALIGNMENT 16

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
