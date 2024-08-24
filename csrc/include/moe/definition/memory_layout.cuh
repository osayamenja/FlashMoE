//
// Created by Jonathan on 7/21/24.
//

#ifndef ARISTOS_MEMORY_LAYOUT_CUH
#define ARISTOS_MEMORY_LAYOUT_CUH

namespace aristos{
    /// The symmetric heap is a 4-D tensor (P, S, C, T)
    /// where P, S, C, and T denote dimensions for peers, communication stages,
    /// cells and tokens.
    /// Number of communication stages S
    __constant__ constexpr unsigned int stages = 2;

    /// Per stage, there is one cell for send and another for receive
    __constant__ constexpr unsigned int numCells = 2;

    __constant__ constexpr unsigned int sendCell = 0;
    __constant__ constexpr unsigned int receiveCell = 1;
    
    CUTE_DEVICE
    cuda::std::byte* getTokenPointer(unsigned int const& peer, unsigned int const& stage, unsigned int const& cell, unsigned int const& token){
        return moeConfig.sHeap + ((peer * moeConfig.peerStride) + (stage * moeConfig.stageStride) + (cell * moeConfig.cellStride) + (token * moeConfig.tokenStride));
    }

}
#endif //ARISTOS_MEMORY_LAYOUT_CUH
