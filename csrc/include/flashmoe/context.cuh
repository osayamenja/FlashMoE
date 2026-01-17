//
// Created by osayamen on 1/15/26.
//

#ifndef FLASHMOE_CONTEXT_CUH
#define FLASHMOE_CONTEXT_CUH
#include <cuda/barrier>
#include <cute/int_tuple.hpp>

#include "infra/bitset.cuh"
#include "infra/packed.cuh"
#include "infra/signal.cuh"
#include "infra/structures.cuh"
#include "infra/task.cuh"
#include "infra/tq.cuh"

namespace flashmoe
{
    template<int subscriberCount, int subscriberWarpSize>
    __host__ __device__ __forceinline__
    auto subscriberTQLength(const int& world, const int& numLocalExperts, const uint& ecTilesM,
        const uint& E, const uint& tilesN0, const uint& tilesN1) {
        static_assert(subscriberCount % subscriberWarpSize == 0 && subscriberWarpSize == 32);
        return cute::ceil_div(world * numLocalExperts, subscriberCount / subscriberWarpSize) *
                    cute::ceil_div(ecTilesM * tilesN0, subscriberWarpSize) +
                    cute::ceil_div(ecTilesM * E, subscriberCount) * tilesN1;
    }
    struct __align__(16) Context {
        uint64_t* const signals; // [[world, num_local_experts], [E, tiles(roundEC), tiles(H)]]
        Task* const tQ;// []
        Task* const pTq; // []
        // [world, num_local_experts, roundEC, I] ~= [S, I]
        cuda::std::byte* const GEMM0Staging;
        BitSet* const consumerCombineBitMap; // nSI<subscriberCount>(tiles(S) * tiles(H))
        uint8_t* const producerCombineBitMap; // [world, nLx, ecTilesM, N1] ~= tiles(S) * tiles(H)
        PEL* const pel; // [E]
        PLI* const pli; // [world]
        ELI* const eli; // [E]
        LXI* const lxi; // [num_local_experts]
        TPS* const tokenIndices; // [E, roundEC]
        TQSignal* const tqs; // [processors]
        uint* const dispatchSync; // [E]
        int* const expertCounts; //[E]
        int* const ecGuards; // [E]
        uint* const gTqHeads; // [world, num_local_experts, ecTilesM] = tiles(S)
        uint* const tileSync; // [world, num_local_experts, ecTilesM] = = tiles(S)
        uint* const statusQueue; // [processors]
        SoftmaxStatePacked* const ssp = nullptr; // [S, tiles(E)]
        RingTopKPayload* const rtp = nullptr; // [2, S, tiles(E)]
        cuda::barrier<cuda::thread_scope_device>* const db = nullptr; // [1]
        const uint S; //  max number of tokens for this rank
        const uint H; // max hidden dimension or model dim
        const uint I; //  max FFN intermediate size
        const uint EC; // max EC
        const uint ecTilesM;
        const uint gtqHeadLength; // world*num_local_experts*ecTilesM
        const uint16_t nLx;
        const uint16_t E;
        const uint16_t world;
        const uint16_t epRank;
        const uint8_t stateNumber;
        const Topology topology = Topology::NVLINK_ONLY;

        // __host__
        // Context(const uint& S, const uint& H, const uint& I, const uint& E,
        //     const int& bM, const int& bNGate, const int& bN1) {
        //
        // }
        void compatible(const uint& S, const uint& H, const uint& I, const uint& E,
            const int& bM, const int& bNGate, const int& bN1) {
            // check compatibility
            // refil
        }
        void finalize() {

        }
    };
}
#endif //FLASHMOE_CONTEXT_CUH