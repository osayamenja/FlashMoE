/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Kleos Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 */

//
// Created by oja7 on 3/3/25.
//

#ifndef FFFN_CUH
#define FFFN_CUH

#include <cub/cub.cuh>
#include <cuda/std/array>
#include <cute/tensor.hpp>

#include "../types.cuh"
#include "../os/processor/gemm.cuh"
#include "../os/processor/processor.cuh"
namespace kleos {
    /// fused ffn with on-demand tile scheduling
    template<
        unsigned int M = ACC::S::value,
        unsigned int N = ACC::P::value,
        unsigned int K = ACC::H::value,
        typename Element = ACC::Element
    >
    __global__ __maxnreg__(ACC::PeakHardware::registers::value) void fffn(
        const void* __restrict__ iP /* A, B, D*/, void* __restrict__ oP /*C*/) {
        constexpr unsigned int blocks = ACC::PeakHardware::OS::processorBlocks::value;
        constexpr unsigned int sharedSize = ACC::sharedSize::value;
        __shared__ __align__(16) Element workspace[sharedSize / sizeof(Element)];
        __shared__ __align__(16) uint tQ[ACC::TMU::value];
        using Operation = BlockMM<ACC::ActivationOp, Element>;
        using OperationX = BlockMM<ACC::ActivationOpX, Element>;

        // we require M, N, K to be evenly divisible by corresponding block tiling dimensions
        constexpr auto tilesM = M / cute::get<0>(typename Operation::TilerOut{});
        constexpr auto tilesN = N / cute::get<1>(typename Operation::TilerOut{});
        constexpr auto tilesK = K / cute::get<1>(typename Operation::TilerOut{});
        constexpr auto tiles = tilesM * tilesN;
        constexpr auto tiles2 = tilesM * tilesK;
        constexpr auto threads = Operation::Threads::value;

        const auto* __restrict__ pA = CONST_CAST_TO(Element, iP);
        const auto* __restrict__ pB1 = pA + M * K;
        const auto* __restrict__ pB2 = pB1 + N * K;
        const auto* __restrict__ pD1 = pB2 + N * K;
        const auto* __restrict__ pD2 = pD1 + K;
        auto* __restrict__ pC1 = bookkeeping.xM();
        auto* __restrict__ pC2 = CAST_TO(Element, oP);

        constexpr auto tSCl = cute::min(tilesK, blocks);
        const auto tSync = make_tensor(cute::make_gmem_ptr(bookkeeping.tSA()),
            cute::Layout<cute::Shape<cute::Int<tSCl>, cute::Int<tilesM>>,
                cute::Stride<cute::Int<tilesM>, cute::_1>>{});

        constexpr auto bL = tiles / blocks;
        for (uint i = 0; i < bL; ++i) {
            const auto tileIdx = blockIdx.x + i * blocks;
            processor::sfGET<Operation, M, N, K>(workspace, pA, pB1, pC1, pD1, tileIdx);
            __syncthreads();
            if (constexpr auto wS = 32; threadIdx.x / wS == 0) {
                const auto tM = tileIdx / tilesN;
                uint propagate = 0U;
                if (!threadIdx.x) {
                    __threadfence();
                    // Notify this tile's completion
                    propagate = atomicIncrement(&tSync(0, tM)) == tilesN - 1;
                }
                // Broadcast from t0 to everyone else in the warp
                propagate = __shfl_sync(0xffffffff, propagate, 0);
                if (propagate) {
                    // We were the last, let's propagate this information down
                    #pragma unroll
                    for (uint j = threadIdx.x + 1; j < tSCl; j += wS) {
                        atomicExch(&tSync(j, tM), tilesN);
                    }
                }
            }
        }
        if (blockIdx.x < tiles % blocks) {
            const auto tileIdx = blockIdx.x + bL * blocks;
            processor::sfGET<Operation, M, N, K>(workspace, pA, pB1, pC1, pD1, tileIdx);
            __syncthreads();
            if (constexpr auto wS = 32; threadIdx.x / wS == 0) {
                const auto tM = tileIdx / tilesN;
                uint propagate = 0U;
                if (!threadIdx.x) {
                    __threadfence();
                    // Notify this tile's completion
                    propagate = atomicIncrement(&tSync(0, tM)) == tilesN - 1;
                }
                // Broadcast from t0 to everyone else in the warp
                propagate = __shfl_sync(0xffffffff, propagate, 0);
                if (propagate) {
                    // We were the last, let's propagate this information down
                    #pragma unroll
                    for (uint j = threadIdx.x + 1; j < tSCl; j += wS) {
                        atomicExch(&tSync(j, tM), tilesN);
                    }
                }
            }
        }

        using BlockScan = cub::BlockScan<uint, threads>;
        constexpr auto tSlice = ACC::TMU::value / threads;
        // Register allocations
        uint predicates[tSlice];
        FlagState flagState[tSlice];
        #pragma unroll
        for (uint i = 0; i < tSlice; ++i) {
            predicates[i] = 0U;
            flagState[i] = FlagState::unidentified;
        }
        auto processed = 0U;
        // Below presumes a row-major layout of blocks over tiles
        const auto nT = tiles2 / blocks + (blockIdx.x < (tiles2 % blocks));
        static_assert(sizeof(BlockScan::TempStorage) <= sharedSize);
        auto* __restrict__ bTs = CAST_TO(typename BlockScan::TempStorage, workspace);
        constexpr auto underSubscribed = tilesK > blocks;
        constexpr auto fStride = underSubscribed ? blocks * (tilesK / blocks) : blocks;
        while (processed < nT) {
            // concurrently sweep pending flags
            #pragma unroll
            for (uint i = 0; i < tSlice; ++i) {
                const auto tileIdx = blockIdx.x + fStride * (i * threads + threadIdx.x);
                const auto flagColIdx = tileIdx / tilesK;
                const auto isCollision = underSubscribed && (i > 0 || threadIdx.x > 0) &&
                    (tileIdx - fStride) / tilesK == flagColIdx;
                const auto flagRowIdx = underSubscribed ? blockIdx.x : tileIdx % tilesK;
                if (tileIdx < tiles2 && flagState[i] != FlagState::completed && !isCollision) {
                    predicates[i] = atomicLoad(&tSync(flagRowIdx, flagColIdx)) == tilesN;
                    flagState[i] = predicates[i] ? FlagState::identified : FlagState::unidentified;
                }
            }
            uint identifiedFlags = 0U;
            // Perform block-wide Aggregation
            BlockScan(*bTs).InclusiveSum(predicates, predicates, identifiedFlags);
            // Populate task queue with identified flag indices
            #pragma unroll
            for (uint i = 0; i < tSlice; ++i) {
                const auto tileIdx = blockIdx.x + fStride * (i * threads + threadIdx.x);
                if (tileIdx < tiles2 && flagState[i] == FlagState::identified) {
                    flagState[i] = FlagState::completed;
                    tQ[predicates[i] - 1] = tileIdx / tilesK;
                }
                predicates[i] = 0U;
            }
            // needed for global visibility of tQ updates
            __syncthreads();
            for (uint i = 0; i < identifiedFlags; ++i) {
                const auto rowIdx = tQ[i];
                // Compute prefix index for my first tile in this row
                const auto startIdx = (rowIdx * tilesK) / blocks + (blockIdx.x < rowIdx * tilesK % blocks);
                const auto endTile = (rowIdx + 1) * tilesK;
                // do gemm
                for (uint j = blockIdx.x + startIdx * blocks; j < endTile; j += blocks) {
                    processor::sfGET<OperationX, M, K, N>(workspace, pC1, pB2, pC2, pD2, j);
                    processed++;
                }
            }
        }
    }
}
#endif //FFFN_CUH
