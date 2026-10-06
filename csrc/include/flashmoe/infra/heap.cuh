//
// Created by osay on 1/5/26.
//

#ifndef FLASHMOE_HEAP_CUH
#define FLASHMOE_HEAP_CUH
#include <cuda/std/cstddef>

#include "mx.cuh"
namespace flashmoe {
  /// Number of communication stages
  constexpr int HEAP_STAGES = 2;
  /// Per stage, there is one cell for sending and another for reception
  constexpr int HEAP_CELLS = 2;
  constexpr int HEAP_SEND_CELL = 0;
  constexpr int HEAP_RECEIVE_CELL = 1;

  /// Bytes of the token region of the symmetric heap: [peer, expert, stage, cell, roundEC, H]
  __host__ __device__ __forceinline__
  constexpr size_t heapDataBytes(const size_t& slots, const size_t& roundEC, const size_t& tokenDim,
                                 const size_t& elementBytes) {
    return elementBytes * HEAP_STAGES * HEAP_CELLS * slots * roundEC * tokenDim;
  }

  /// Block-scaled recipes only. Bytes of the scale factors of one first-stage cell: the canonical layout
  /// (mx::sfOffset) of the quantized [roundEC, H] matrix, i.e., one scale per sfVec columns (quantSFVecSize:
  /// 32 for mxfp8 and mxfp4, 16 for nvfp4).
  /// roundEC and H are multiples of 128, see initialize().
  __host__ __device__ __forceinline__
  constexpr size_t heapSFCellBytes(const size_t& roundEC, const size_t& tokenDim, const size_t& sfVec = mx::BLOCK_SIZE) {
    return mx::sfBytes(roundEC, tokenDim / sfVec);
  }

  /// Block-scaled recipes only. Bytes of the scale factor region that is appended to the token region of the
  /// symmetric heap: [peer, expert, cell, canonical scale factors of [roundEC, H]]. Only the first stage (dispatch)
  /// is quantized, the second stage (combine) is not.
  __host__ __device__ __forceinline__
  constexpr size_t heapSFBytes(const size_t& slots, const size_t& roundEC, const size_t& tokenDim,
                               const size_t& sfVec = mx::BLOCK_SIZE) {
    return HEAP_CELLS * slots * heapSFCellBytes(roundEC, tokenDim, sfVec);
  }

  /// The symmetric tensor from the FlashMoE paper
  struct Heap {
    cuda::std::byte *const sHeap;
    /// sfOffset is the byte offset of the scale factor region (block-scaled recipes only), see heapSFBytes.
    /// It is a symmetric offset, so a peer's scale factors are at remoteSHeap + advanceSFOffset(...).
    /// sfVec is the number of K elements per scale factor, i.e., quantSFVecSize(recipe).
    __device__ __forceinline__
    Heap(cuda::std::byte *const&_sHeap,
         const uint &_expertSlots, const uint &_EC,
         const uint &_tokenDim, const uint &_elementBytes, const size_t &_sfOffset = 0,
         const uint &_sfVec = mx::BLOCK_SIZE) : sHeap(_sHeap),
                                                expertSlots(_expertSlots), EC(_EC),
                                                tokenDim(_tokenDim), elementBytes(_elementBytes),
                                                sfOffset(_sfOffset), sfVec(_sfVec) {
    }

    template<
      int stage,
      int cell
    >
      requires (stage < HEAP_STAGES && cell < HEAP_CELLS)
    __device__ __forceinline__
    constexpr auto advanceOffset(const size_t &peer, const size_t &expert, const size_t &token = 0) const {
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
    cuda::std::byte *advance(const size_t &peer, const size_t &expert, const size_t &token = 0) const {
      return sHeap + advanceOffset<stage, cell>(peer, expert, token);
    }

    /// Block-scaled recipes only. Offset of the scale factors of the packet at advance<stage, cell>(peer, expert):
    /// the canonical layout of its quantized [EC, H] rows, see mx::sfOffset. Only the first stage is quantized.
    template<
      int stage,
      int cell
    >
      requires (stage == 0 && cell < HEAP_CELLS)
    __device__ __forceinline__
    constexpr auto advanceSFOffset(const size_t &peer, const size_t &expert) const {
      return sfOffset + heapSFCellBytes(EC, tokenDim, sfVec) *
             (HEAP_CELLS * (static_cast<size_t>(peer) * expertSlots + static_cast<size_t>(expert)) +
              static_cast<size_t>(cell));
    }

    template<
      int stage,
      int cell
    >
      requires (stage == 0 && cell < HEAP_CELLS)
    __device__ __forceinline__
    cuda::std::byte *advanceSF(const size_t &peer, const size_t &expert) const {
      return sHeap + advanceSFOffset<stage, cell>(peer, expert);
    }

  private:
    const uint expertSlots;
    const uint EC; // round to multiple of bM
    const uint tokenDim;
    const uint elementBytes;
    const size_t sfOffset;
    const uint sfVec;
  };
}
#endif //FLASHMOE_HEAP_CUH
