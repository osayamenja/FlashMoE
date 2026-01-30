//
// Created by osayamen on 1/15/26.
//

#ifndef FLASHMOE_CONTEXT_CUH
#define FLASHMOE_CONTEXT_CUH

#include <cuda/barrier>
#include <cute/int_tuple.hpp>

#include "cuda/memory"
#include "infra/bitset.cuh"
#include "infra/packed.cuh"
#include "infra/signal.cuh"
#include "infra/structures.cuh"
#include "infra/task.cuh"
#include "infra/tq.cuh"

namespace flashmoe
{
    __host__ __forceinline__ __device__
    auto checkAlignment(const void* const& p, const bool supports32 = false) {
        const auto alignment = supports32 ? 32 : 16;
        if (!cuda::is_aligned(p, alignment)) {
            printf("Pointer is not %d-byte aligned:\n", alignment);
            cuda::std::terminate();
        }
    }
    template<int subscriberWarpSize>
    __host__ __forceinline__
    constexpr auto subscriberTQLength(const int& world, const uint& numLocalExperts, const uint& ecTilesM,
        const uint& E, const uint& tilesN0, const uint& tilesN1, const uint& subscriberCount) {
        const auto dispatchTaskQL = cute::ceil_div(world * numLocalExperts,subscriberCount / subscriberWarpSize) *
            (cute::ceil_div(ecTilesM * tilesN0, subscriberWarpSize) + cute::ceil_div(tilesN0, subscriberWarpSize));
        const auto combineTaskQL = cute::ceil_div(ecTilesM * E * tilesN1, subscriberCount);
        return (dispatchTaskQL + combineTaskQL) * subscriberCount;
    }

    template<int subscriberCount, int subscriberWarpSize>
    __device__ __forceinline__
    constexpr auto subscriberTQLength(const int& world, const int& numLocalExperts, const uint& ecTilesM,
        const uint& E, const uint& tilesN0, const uint& tilesN1) {
        static_assert(subscriberCount % subscriberWarpSize == 0);
        return subscriberTQLength<subscriberWarpSize>(world, numLocalExperts, ecTilesM, E, tilesN0, tilesN1, subscriberCount);
    }
    __host__ __device__ __forceinline__
    auto secondaryTQLength(const int& world, const int& numLocalExperts, const uint& ecTilesM, const uint& tilesN1) {
        return world * numLocalExperts * ecTilesM * tilesN1;
    }

    struct Context {
        cuda::std::byte* const symHeap = nullptr;
        uint64_t* const signals = nullptr; // [[world, num_local_experts], [E, tiles(roundEC), tiles(H)]]
        Task* const tQ = nullptr;// [subscriberTQLength]
        Task* const pTq = nullptr; //[secondaryTQLength]
        // [world, num_local_experts, roundEC, I] ~= [S, I]
        cuda::std::byte* const GEMM0Staging = nullptr;
        BitSet* const consumerCombineBitMap = nullptr; // nSI<subscriberCount>(tiles(S) * tiles(H))
        uint8_t* const producerCombineBitMap = nullptr; // [world, nLx, ecTilesM, N1] ~= tiles(S) * tiles(H)
        PEL* const pel = nullptr; // [E]
        PLI* const pli = nullptr; // [world]
        ELI* const eli = nullptr; // [E]
        LXI* const lxi = nullptr; // [num_local_experts]
        TQSignal* const tqs = nullptr; // [processors]
        uint* const dispatchSync = nullptr; // [E]
        uint* const gTqHeads = nullptr; // [world, num_local_experts, ecTilesM] = tiles(S)
        uint* const tileSync = nullptr; // [world, num_local_experts, ecTilesM] = = tiles(S)
        uint* const statusQueue = nullptr; // [processors]
        TPS* const tokenIndices = nullptr; // [E, roundEC]
        const cuda::fast_mod_div<uint> processors_v;
        const uint blocks = 0;
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
        const bool initialized = false;
        const Topology topo = Topology::MIXED;
        uint8_t stateNumber = 0;
        static_assert(alignof(cuda::fast_mod_div<uint>) <= 8);
    };

    struct __align__(8) GateContext{
        int* const ecGuards = nullptr; // [E]
        SoftmaxStatePacked* const ssp = nullptr; // [S, tiles(E)]
        RingTopKPayload* const rtp = nullptr; // [2, S, tiles(E)]
        //cuda::barrier<cuda::thread_scope_device>* const db = nullptr; // [1]
    };
}
#endif //FLASHMOE_CONTEXT_CUH