//
// Created by osayamen on 1/15/26.
//

#ifndef FLASHMOE_CONTEXT_CUH
#define FLASHMOE_CONTEXT_CUH
#include <cuda/barrier>

#include "infra/packed.cuh"
#include "infra/signal.cuh"
#include "infra/structures.cuh"
#include "infra/task.cuh"
#include "infra/tq.cuh"

namespace flashmoe
{
    struct __align__(8) Context {
        uint64_t* const signals; // [[world, num_local_experts], [E, tiles(roundEC), tiles(H)]]
        Task* const tQ;// []
        Task* const pTq; // []
        // [world, num_local_experts, roundEC, I]
        cuda::std::byte* const GEMM0Staging;
        PEL* const pel; // [E]
        PLI* const pli; // [world]
        ELI* const eli; // // [E]
        LXI* const lxi; // [num_local_experts]
        TPS* const tokenIndices; // [E, roundEC]
        TQSignal* const tqs; // [processors]
        uint* const dispatchSync; // [E]
        int* const expertCounts; //[E]
        int* const ecGuards; // [E]
        uint* const gTqHeads; // [world, num_local_experts, ecTilesM]
        uint* const tileSync; // [world, num_local_experts, ecTilesM]
        uint* const statusQueue; // [processors]
        SoftmaxStatePacked* const ssp = nullptr; // [S, tiles(E)]
        RingTopKPayload* const rtp = nullptr; // [2, S, tiles(E)]
        cuda::barrier<cuda::thread_scope_device>* const db = nullptr; // [1]
        uint* const tileIndices = nullptr; // [E, ecTilesM, tiles(H)]
        const uint S; //  max number of tokens for this rank
        const uint H; // max hidden dimension or model dim
        const uint I; //  max FFN intermediate size
        const uint EC; // max EC
        const uint ecTilesM;
        const uint ssFc; // // second stage flag count
        const uint gtqHeadLength; // world*num_local_experts*ecTilesM
        const uint16_t nLx;
        const uint16_t E;
        const uint16_t world;
        const uint16_t epRank;
        const uint16_t seqNumber;
        const Topology topology = Topology::NVLINK_ONLY;

        __host__
        Context(const uint& S, const uint& H, const uint& I, const uint& E,
            const int& bM, const int& bNGate, const int& bN1) {

        }
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