//
// Created by osayamen on 1/15/26.
//

#ifndef FLASHMOE_CONTEXT_CUH
#define FLASHMOE_CONTEXT_CUH

#include "infra/bitset.cuh"
#include "infra/packed.cuh"
#include "infra/signal.cuh"
#include "infra/structures.cuh"
#include "infra/task.cuh"
#include "infra/tq.cuh"

namespace flashmoe {
  struct Context {
    cuda::std::byte *const symHeap = nullptr;
    uint64_t *const signals = nullptr; // [[world, num_local_experts], [E, tiles(roundEC), tiles(H)]]
    Task *const tQ = nullptr; // [subscriberTQLength]
    Task *const pTq = nullptr; //[secondaryTQLength]
    // [world, num_local_experts, roundEC, I] ~= [S, I]
    cuda::std::byte *const GEMM0Staging = nullptr;
    BitSet *const consumerCombineBitMap = nullptr; // nSI<subscriberCount>(tiles(S) * tiles(H))
    uint8_t *const producerCombineBitMap = nullptr; // [world, nLx, ecTilesM, N1] ~= tiles(S) * tiles(H)
    PEL *const pel = nullptr; // [E]
    PLI *const pli = nullptr; // [world]
    ELI *const eli = nullptr; // [E]
    LXI *const lxi = nullptr; // [num_local_experts]
    TQSignal *const tqs = nullptr; // [processors]
    uint *const dispatchSync = nullptr; // [E]
    uint *const gTqHeads = nullptr; // [world, num_local_experts, ecTilesM] = tiles(S)
    uint *const tileSync = nullptr; // [world, num_local_experts, ecTilesM] = tiles(S)
    uint *const statusQueue = nullptr; // [processors]
    TPS *const tokenIndices = nullptr; // [E, roundEC]
    uint8_t* const stateNumbers = nullptr; // [processorCTAs]
    const cuda::fast_mod_div<uint> processors_v;
    const uint blocks = 0;
    const uint smemSize = 0;
    const uint S = 0; //  max number of tokens for this rank
    const uint H = 0; // max hidden dimension or model dim
    const uint I = 0; //  max FFN intermediate size
    const uint EC = 0; // max EC
    const uint16_t bM = 0;
    const uint16_t bN0 = 0;
    const uint16_t bN1 = 0;
    const uint16_t nLx = 0;
    const uint16_t E = 0;
    const uint16_t world = 0;
    const uint16_t epRank = 0;
    const uint16_t myPE = 0;
    const Topology topo = Topology::MIXED;
  };

  struct GateContext {
    TPS *const staging = nullptr; // [E, S], one record per input row
    SoftmaxStatePacked *const ssp = nullptr; // [S, tiles(E)]
    RingTopKPayload *const rtp = nullptr; // [2, S, tiles(E)]
    uint blocks = 0;
  };
}
#endif //FLASHMOE_CONTEXT_CUH
