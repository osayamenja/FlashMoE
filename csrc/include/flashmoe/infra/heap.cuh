//
// Created by osay on 1/5/26.
//

#ifndef FLASHMOE_HEAP_CUH
#define FLASHMOE_HEAP_CUH

namespace flashmoe {
    /// Number of communication stages
    constexpr int HEAP_STAGES = 2;
    /// Per stage, there is one cell for sending and another for reception
    constexpr int HEAP_CELLS = 2;
    constexpr int HEAP_SEND_CELL = 0;
    constexpr int HEAP_RECEIVE_CELL = 1;
    /// The symmetric tensor from the FlashMoE paper
    struct Heap {
        cuda::std::byte* const sHeap;
        __device__ __forceinline__
        Heap(cuda::std::byte* const& _sHeap,
            const uint& _expertSlots, const uint& _EC,
            const uint& _tokenDim, const uint& _elementBytes) :
        sHeap(_sHeap), expertSlots(_expertSlots), EC(_EC),
        tokenDim(_tokenDim), elementBytes(_elementBytes){}
        template<
            int stage,
            int cell
        >
        requires (stage < HEAP_STAGES && cell < HEAP_CELLS)
        __device__ __forceinline__
        constexpr auto advanceOffset(const size_t& peer, const size_t& expert, const size_t& token = 0) const {
            return static_cast<size_t>(elementBytes)
                    * static_cast<size_t>(tokenDim)
                    * (static_cast<size_t>(EC)
                        * (static_cast<size_t>(HEAP_CELLS)
                            * (static_cast<size_t>(HEAP_STAGES)
                                * (static_cast<size_t>(peer) * expertSlots + static_cast<size_t>(expert))
                                + static_cast<size_t>(stage))
                            + static_cast<size_t>(cell))
                        + static_cast<size_t>(token));
        }
        template<
            int stage,
            int cell
        >
        requires (stage < HEAP_STAGES && cell < HEAP_CELLS)
        __device__ __forceinline__
        cuda::std::byte* advance(const size_t& peer, const size_t& expert, const size_t& token = 0) const {
                return sHeap + advanceOffset<stage, cell>(peer, expert, token);
        }
    private:
        const uint expertSlots;
        const uint EC; // round to multiple of bM
        const uint tokenDim;
        const uint elementBytes;
    };
}
#endif //FLASHMOE_HEAP_CUH