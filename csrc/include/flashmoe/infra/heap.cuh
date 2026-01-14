//
// Created by osay on 1/5/26.
//

#ifndef FLASHMOE_HEAP_CUH
#define FLASHMOE_HEAP_CUH

/// Number of communication stages
#define STAGES 2U
/// Per stage, there is one cell for sending and another for reception
#define CELLS 2U
#define SEND_CELL 0U
#define RECEIVE_CELL 1U

namespace flashmoe {
    /// The symmetric tensor from the FlashMoE paper
    struct Heap {
        cuda::std::byte* const sHeap;
        const int expertSlots;
        const int EC; // round to multiple of bM
        const int tokenDim;
        const int elementBytes;
        Heap(cuda::std::byte* const& _sHeap,
            const int& _expertSlots, const int& _EC,
            const int& _tokenDim, const int& _elementBytes) :
        sHeap(_sHeap), expertSlots(_expertSlots), EC(_EC),
        tokenDim(_tokenDim), elementBytes(_elementBytes){}
        template<
            unsigned long int stage,
            unsigned long int cell
        >
        requires (stage < STAGES && cell < CELLS)
        __device__ __forceinline__
        constexpr long int advanceOffset(const int& peer, const int& expert, const int& token = 0) const {
            return static_cast<long int>(elementBytes) * tokenDim * (EC * (CELLS * (STAGES * (peer * expertSlots +
                static_cast<long int>(expert)) + stage) + cell) + token);
        }
        template<
            unsigned long int stage,
            unsigned long int cell
        >
        requires (stage < STAGES && cell < CELLS)
        __device__ __forceinline__
        cuda::std::byte* advance(const int& peer, const int& expert, const int& token = 0) const {
                return sHeap + advanceOffset<stage, cell>(peer, expert, token);
        }
    };
}
#endif //FLASHMOE_HEAP_CUH