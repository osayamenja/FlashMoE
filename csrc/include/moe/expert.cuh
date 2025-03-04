//
// Created by oja7 on 1/28/25.
//

#ifndef EXPERT_CUH
#define EXPERT_CUH

#include <cub/cub.cuh>
#include <cuda/std/array>
#include <cute/tensor.hpp>

#include "../arch.cuh"
#include "../atomics.cuh"
#include "../types.cuh"
#include "../os/processor/gemm.cuh"
#include "../os/processor/processor.cuh"
#define TIME_EXPERT 1
namespace aristos {
    template<
        typename BlockGEMM,
        unsigned int N,
        unsigned int K,
        CombineMode c,
        unsigned int elems,
        typename Activations,
        typename Weights,
        typename Output,
        typename Bias,
        unsigned int threads = BlockGEMM::GEMM::block_dim.x
    >
    __forceinline__ __device__
    void fGST(typename BlockGEMM::MatrixDType* const& workspace,
        const Activations& mA,
        const Weights& mB,
        const Bias& mD,
        const Output& mC,
        const typename BlockGEMM::MatrixDType* __restrict__ const& scaleWeights,
        const typename BlockGEMM::MatrixDType* __restrict__ const& combineWeights,
        const unsigned int& tileIdx,
        const unsigned int& M) {
        auto accumulator = cute::partition_fragment_C(typename BlockGEMM::MMA{}, typename BlockGEMM::TilerOut{});
        static_assert(cute::size(accumulator) % elems == 0);
        cuda::std::array<typename BlockGEMM::MatrixDType, elems> rScratch{};
        constexpr auto bM = cute::get<0>(typename BlockGEMM::BlockTiler{});
        constexpr auto bN = cute::get<1>(typename BlockGEMM::BlockTiler{});
        static_assert(size(accumulator) % rScratch.size() == 0);
        // Instantiate mainloop
        typename BlockGEMM::CollectiveMainloop mainLoop{};
        cute::clear(accumulator);
        const auto tilesM = M / cute::get<0>(typename BlockGEMM::BlockTiler{});
        constexpr auto tilesN = N / cute::get<1>(typename BlockGEMM::BlockTiler{});
        constexpr auto tilesK = K / cute::get<2>(typename BlockGEMM::BlockTiler{});
        const auto tileCoord = idx2crd(tileIdx, cute::Shape(tilesM, tilesN), cute::Stride(tilesN ,1));
        const auto ctaCoord = make_coord(cute::get<0>(tileCoord), cute::get<1>(tileCoord), cute::_);
        using BlockTiler = cute::Shape<cute::Int<bM>, cute::Int<bN>>;
        constexpr BlockTiler tiler{};
        const auto gA = cute::local_tile(mA, typename BlockGEMM::BlockTiler{}, ctaCoord, cute::Step<cute::_1, cute::X,cute::_1>{});
        const auto gB = cute::local_tile(mB, typename BlockGEMM::BlockTiler{}, ctaCoord, cute::Step< cute::X,cute::_1,cute::_1>{});
        const auto gC = cute::local_tile(mC, typename BlockGEMM::BlockTiler{}, ctaCoord, cute::Step<cute::_1,cute::_1, cute::X>{});
        const auto gD = cute::local_tile(mD, typename BlockGEMM::BlockTiler{}, ctaCoord, cute::Step<cute::_1,cute::_1, cute::X>{});
        // get combine and scale weights
        const auto rowIdx = (tileIdx / tilesN) * bM;
        auto sW = scaleWeights[rowIdx + threadIdx.x];
        auto cW = combineWeights[rowIdx + threadIdx.x];

        auto k_tile_iter = cute::make_coord_iterator(tilesK);

        using ElementD = typename BlockGEMM::MatrixDType;
        mainLoop(
            accumulator,
            gA,
            gB,
            accumulator,
            k_tile_iter, tilesK,
            cute::Underscore{},
            threadIdx.x,
            CAST_TO(char, workspace));
        /// There is a block-wide barrier at the end of the above ^

        // Epilogue
        constexpr typename BlockGEMM::MMA tiledMMA{};
        const auto tDgD = tiledMMA.get_slice(threadIdx.x).partition_C(gD);

        constexpr auto gCStoreOp = cutlass::NumericConverter<typename decltype(gC)::value_type,
                                                    typename decltype(accumulator)::value_type>{};
        constexpr auto gDLoadOp = cutlass::NumericConverter<typename decltype(accumulator)::value_type,
                                                    ElementD>{};

        // Assume elementwise operator
        constexpr typename BlockGEMM::FusedEpilogue epilogueOp{};
        constexpr auto trips = size(accumulator) / rScratch.size();
        const auto nW = gDLoadOp(sW / cW);

        // Prefetch bias from global to shared memory
        #pragma unroll
        for (int j = 0; j < elems; ++j) {
            workspace[threadIdx.x + j * threads] = tDgD(j);
        }

        #pragma unroll
        for (unsigned int i = 0; i < trips; ++i) {
            #pragma unroll
            for (unsigned int j = 0; j < elems; ++j) {
                rScratch[j] = workspace[threadIdx.x + j * threads];
                if (i + 1 < trips) {
                    // Eagerly start loads for the next batch, if needed
                    workspace[threadIdx.x + j * threads] = tDgD(j + (i + 1) * elems);
                }
            }

            #pragma unroll
            for (int j = 0; j < elems; ++j) {
                accumulator(j + i * elems) = nW * epilogueOp(accumulator(j + i * elems), gDLoadOp(rScratch[j]));
            }
        }

        __syncthreads();
        constexpr auto sCLay = cute::make_layout(cute::Shape<cute::Int<bM>, cute::Int<elems>>{},
            cute::LayoutRight{});
        const auto sC = cute::make_tensor(cute::make_smem_ptr(workspace), sCLay);
        const auto tCsC = tiledMMA.get_slice(threadIdx.x).partition_C(sC);
        const auto rIdx = threadIdx.x / elems * elems;
        const auto cIdx = threadIdx.x % elems;
        using CDxT = typename ToCDx<ElementD>::T;
        constexpr auto cTCx = cutlass::NumericConverter<CDxT, ElementD>{};
        // Transpose data
        #pragma unroll
        for (unsigned int i = 0; i < trips; ++i) {
            #pragma unroll
            for (unsigned int j = 0; j < elems; ++j) {
                tCsC(j) = gCStoreOp(accumulator(j + i * elems));
            }
            __syncthreads();
            // Load striped slice into register fragment
            #pragma unroll
            for (unsigned int j = 0; j < elems; ++j) {
                rScratch[j] = sC(rIdx + j, cIdx);
            }
            #pragma unroll
            for (unsigned int j = 0; j < elems; ++j) {
                if constexpr (c == CombineMode::single) {
                    gC(rIdx + j, cIdx + i * elems) = rScratch[j];
                }
                else {
                    atomicAdd(CAST_TO(CDxT, &gC(rIdx + j, cIdx + i * elems)), cTCx(rScratch[j]));
                }
            }
        }
        __syncthreads();
    }
    // Fused FFN
    template<
        UseBarrier u = UseBarrier::no,
        unsigned int N,
        unsigned int K,
        typename Element
    >
    /// D = A * B1 + C1
    /// A = D * B2 + C2
    __global__ __maxnreg__(ACC::PeakHardware::registers::value) void expert(
        const __grid_constant__ unsigned int M,
        cuda::barrier<cuda::thread_scope_device>* dB,
        float* deviceThroughput, uint* tileSync,
        const Element* __restrict__ iP /* A, B, D, S, W*/,
        Element* __restrict__ oP /*C*/,
        const bool skip = true) {
        constexpr auto c = ACC::CM::value;
        constexpr unsigned int blocks = ACC::PeakHardware::OS::processorBlocks::value;
        constexpr unsigned int sharedSize = ACC::PeakHardware::sharedMemory::value;
        constexpr unsigned int elems = ACC::PeakHardware::rScratch::value;

        __shared__ __align__(16) Element workspace[sharedSize / sizeof(Element)];
        __shared__ __align__(16) uint16_t tQ[ACC::TMU::value];
        using Operation = BlockMM<ACC::ActivationOp, Element>;
        using OperationX = BlockMM<ACC::ActivationOpX, Element>;
        constexpr auto threads = Operation::GEMM::block_dim.x;
        // we require M, N, K to be evenly divisible by corresponding block tiling dimensions
        const auto tilesM = M / cute::get<0>(typename Operation::TilerOut{});
        constexpr auto tilesN = N / cute::get<1>(typename Operation::TilerOut{});
        constexpr auto tilesK = K / cute::get<1>(typename Operation::TilerOut{});
        const auto tiles = tilesM * tilesN;
        const auto tiles2 = tilesM * tilesK;

        const auto* __restrict__ pA = iP;
        const auto* __restrict__ pB1 = pA + M * K;
        const auto* __restrict__ pB2 = pB1 + N * K;
        const auto* __restrict__ pD1 = pB2 + N * K;
        const auto* __restrict__ pD2 = pD1 + K;
        const auto* __restrict__ pS = pD2 + N;
        const auto* __restrict__ pCw = pS + M;
        auto* __restrict__ pC1 = oP;
        auto* __restrict__ pC2 = pC1 + M * N;

        constexpr auto tSCl = cute::min(tilesK, blocks);
        // transposed layout to enable coalescing during polling,
        // tradeoffs by disabling coalescing in one-shot propagation.

        const auto tSync = make_tensor(cute::make_gmem_ptr(tileSync),
            make_layout(cute::make_shape(tSCl, tilesM), cute::LayoutRight{}));
#if TIME_EXPERT
        uint64_t start = 0, end = 0;
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
#endif
        for (uint i = blockIdx.x; i < tiles; i += blocks) {
            processor::fGET<Operation>(workspace, pA, pB1, pC1, pD1, M, i);
            // Pick warp 0
            if constexpr (u == UseBarrier::no) {
                __syncthreads();
                if (constexpr auto wS = 32; threadIdx.x / wS == 0) {
                    const auto tM = i / tilesN;
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
        }
        // Make tensors for below
        // Row-major
        const auto mA = make_tensor(cute::make_gmem_ptr(pC1),
            make_layout(cute::make_shape(M, N), cute::LayoutRight{}));
        // Row-major, transposed
        const auto mB = make_tensor(cute::make_gmem_ptr(pB2),
            make_layout(cute::make_shape(K, N), cute::LayoutRight{}));
        // Row-major
        const auto mC = make_tensor(cute::make_gmem_ptr(pC2),
            make_layout(cute::make_shape(M, K), cute::LayoutRight{}));
        const auto mD = make_tensor(cute::make_gmem_ptr(pD2),
            make_layout(cute::make_shape(M, K), cute::make_stride(0, 1)));

        if constexpr (u == UseBarrier::yes) {
            __syncthreads();
            if (!threadIdx.x) {
                __threadfence();
                dB->arrive_and_wait();
            }
            __syncthreads();
            for (uint i = blockIdx.x; i < tiles2; i += blocks) {
                fGST<OperationX, c, elems>(workspace, mA, mB, mD, mC, pS, pCw, i, M);
            }
        }
        else {
            // Below, is a more sophisticated method for scheduling the next GEMM task.
            // Not necessarily more performant than the barrier.
            using BlockScan = cub::BlockScan<uint16_t, threads>;
            constexpr auto tSlice = ACC::TMU::value / threads;
            // Register allocations
            uint16_t predicates[tSlice];
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
            const auto underSubscribed = tilesK > blocks;
            const auto fStride = underSubscribed ? blocks * (tilesK / blocks) : blocks;
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
                uint16_t identifiedFlags = 0U;
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
                        fGST<OperationX, N, K, c, elems>(workspace, mA, mB, mD, mC, pS, pCw, j, M);
                        processed++;
                    }
                }
            }
        }
#if TIME_EXPERT
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
        if (!skip) {
            auto tDt = (end - start) / 1e6f; // convert nano to milliseconds
            // Intra-block reduction to get maximum latency
            using BlockReduce = cub::BlockReduce<float, threads>;
            auto* __restrict__ rTs = CAST_TO(typename BlockReduce::TempStorage, workspace);
            // Compute the block-wide max for thread0
            auto bT = BlockReduce(*rTs).Reduce(tDt, cub::Max());
            // Inter-block max reduction
            if (!threadIdx.x) {
                cuda::atomic_ref{*deviceThroughput}.fetch_max(bT);
            }
        }
#endif
    }
}
#endif //EXPERT_CUH
