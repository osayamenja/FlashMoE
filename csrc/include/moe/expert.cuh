//
// Created by oja7 on 1/28/25.
//

#ifndef EXPERT_CUH
#define EXPERT_CUH

#include <cub/cub.cuh>
#include <cuda/std/array>
#include <cuda/std/cstddef>
#include <cute/tensor.hpp>

#include "../arch.cuh"
#include "../atomics.cuh"
#include "../types.cuh"
#include "../os/processor/gemm.cuh"
#include "../os/processor/processor.cuh"

namespace aristos {
    template<
        typename BlockGEMM,
        typename Activations,
        typename Weights,
        typename Output,
        typename Bias
    >
    __forceinline__ __device__
    void fGST(typename BlockGEMM::MatrixDType* const& workspace,
        const Activations& mA,
        const Weights& mB,
        const Bias& mD,
        const Output& mC,
        const typename BlockGEMM::MatrixDType* __restrict__ const& scaleWeights,
        const typename BlockGEMM::MatrixDType* __restrict__ const& combineWeights,
        const unsigned int& M,
        const unsigned int& N,
        const unsigned int& tileIdx) {
        auto accumulator = cute::partition_fragment_C(typename BlockGEMM::MMA{}, typename BlockGEMM::TilerOut{});
        constexpr auto elems = SHARED_SIZE / (THREADS * sizeof(typename BlockGEMM::MatrixDType));
        static_assert(cute::size(accumulator) % elems == 0);
        cuda::std::array<typename BlockGEMM::MatrixDType, elems> rScratch{};
        constexpr auto bM = cute::get<0>(typename BlockGEMM::BlockTiler{});
        constexpr auto bN = cute::get<1>(typename BlockGEMM::BlockTiler{});
        static_assert(size(accumulator) % rScratch.size() == 0);
        // Instantiate mainloop
        typename BlockGEMM::CollectiveMainloop mainLoop{};
        cute::clear(accumulator);
        const auto tilesM = M / cute::get<0>(typename BlockGEMM::BlockTiler{});
        const auto tilesN = N / cute::get<1>(typename BlockGEMM::BlockTiler{});
        const auto tileCoord = idx2crd(tileIdx, cute::Shape(tilesM, tilesN), cute::Stride(tilesN ,1));
        const auto ctaCoord = make_coord(cute::get<0>(tileCoord), cute::get<1>(tileCoord), cute::_);
        using BlockTiler = cute::Shape<cute::Int<bM>, cute::Int<bN>>;
        constexpr BlockTiler tiler{};
        const auto gA = cute::local_tile(mA, typename BlockGEMM::BlockTiler{}, ctaCoord, cute::Step<cute::_1, cute::X,cute::_1>{});
        const auto gB = cute::local_tile(mB, typename BlockGEMM::BlockTiler{}, ctaCoord, cute::Step< cute::X,cute::_1,cute::_1>{});
        const auto gC = cute::local_tile(mC, typename BlockGEMM::BlockTiler{}, ctaCoord, cute::Step<cute::_1,cute::_1, cute::X>{});
        const auto gD = cute::local_tile(mD, typename BlockGEMM::BlockTiler{}, ctaCoord, cute::Step<cute::_1,cute::_1, cute::X>{});
        // get combine and scale weights
        auto sW = scaleWeights[tileIdx * bM + threadIdx.x];
        auto cW = combineWeights[tileIdx * bM + threadIdx.x];

        auto k_tile_iter = cute::make_coord_iterator(size<2>(gA));
        int k_tile_count = size<2>(gA);

        using ElementD = typename BlockGEMM::MatrixDType;
        mainLoop(
            accumulator,
            gA,
            gB,
            accumulator,
            k_tile_iter, k_tile_count,
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
        constexpr auto scaleOp = cutlass::epilogue::thread::Scale<typename decltype(accumulator)::value_type>{};

        // Assume elementwise operator
        constexpr typename BlockGEMM::FusedEpilogue epilogueOp{};
        constexpr auto trips = size(accumulator) / rScratch.size();
        const auto nW = gDLoadOp(sW / cW);

        // Prefetch bias from global to shared memory
        #pragma unroll
        for (int j = 0; j < elems; ++j) {
            workspace[threadIdx.x + j * THREADS] = tDgD(j);
        }

        #pragma unroll
        for (unsigned int i = 0; i < trips; ++i) {
            #pragma unroll
            for (unsigned int j = 0; j < elems; ++j) {
                rScratch[j] = workspace[threadIdx.x + j * THREADS];
                if (i + 1 < trips) {
                    // Eagerly start loads for the next batch, if needed
                    workspace[threadIdx.x + j * THREADS] = tDgD(j + (i + 1) * elems);
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
                atomicAdd(CAST_TO(CDxT, &gC(rIdx + j, cIdx + i * elems)), cTCx(rScratch[j]));
            }
        }
        __syncthreads();
    }

    __device__
    enum class TileState : uint8_t {
        unidentified,
        identified,
        completed
    };
    // Stylized expert for evaluation only
    // For simplicity, assume single element types for all matrices,
    // This is typical of PyTorch workloads
    template<
        unsigned int Arch,
        typename ActivationOp,
        typename ElementC,
        typename Element,
        typename ProblemShape_MNK
    >
    requires(cute::is_tuple_v<ProblemShape_MNK> && rank(ProblemShape_MNK{}) == 3
    && aristos::TensorValueType<Element> &&
    !(cuda::std::is_same_v<Element, cute::float_e4m3_t> ||
            cuda::std::is_same_v<Element, cute::float_e5m2_t> ||
                cuda::std::is_same_v<Element, cute::tfloat32_t>))
    /// D = A * B1 + C1
    /// A = D * B2 + C2
    __global__ __maxnreg__(REGINALD) void FFN(ProblemShape_MNK pShape,
        uint* deviceThroughput,
        uint* tileSync,
        const Element* __restrict__ iP /* A, B, D, S, W*/,
        Element* __restrict__ oP /*C*/, const bool skip = true) {
        uint64_t start = 0, end = 0;
        constexpr auto blocks = Hardware<Arch>::blocks::value - 1U;
        constexpr auto tUpB = 1024; // max supported number of tiles per block per gemm is 1536
        __shared__ __align__(16) Element workspace[SHARED_SIZE / sizeof(Element)];
        __shared__ __align__(16) uint16_t tQ[tUpB];
        const auto [M, N, K] = pShape;
        using Operation = BlockMM<Arch, Element, Element, ElementC, ActivationOp>;
        using OperationX = BlockMM<Arch, Element, Element, ElementC>;
        constexpr auto preGEMM = processor::FGT<TaskType::preGEMM, Operation>{};
        const auto tilesM = Bookkeeping::tiles<BLOCK_M>(M);
        const auto tilesN = Bookkeeping::tiles<BLOCK_N>(N);
        const auto tilesK = Bookkeeping::tiles<BLOCK_N>(K);
        const auto tilesX = tilesM * tilesK;
        const auto tiles = tilesM * tilesN;

        const auto* __restrict__ pA = iP;
        const auto* __restrict__ pB1 = pA + M * K;
        const auto* __restrict__ pB2 = pB1 + N * K;
        const auto* __restrict__ pD1 = pB2 + N * K;
        const auto* __restrict__ pD2 = pD1 + K;
        const auto* __restrict__ pS = pD2 + N;
        const auto* __restrict__ pCw = pS + M;
        auto* __restrict__ pC1 = oP;
        auto* __restrict__ pC2 = pC1 + M * N; // write results back to input memory

        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
        #pragma unroll 4
        for (uint i = blockIdx.x; i < tiles; i += blocks) {
            const auto tM = i / tilesN;
            preGEMM(workspace, pA, pB1, pC1, pD1, M, N, K, i);
            // notify this tile's completion
            if (!threadIdx.x) {
                __threadfence();
                atomicIncrement(tileSync + tM);
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

        // Below, is a more sophisticated method for scheduling the next GEMM task.
        using BlockScan = cub::BlockScan<uint16_t, THREADS>;
        constexpr auto tSlice = tUpB / THREADS;
        // Register allocations
        uint16_t predicates[tSlice];
        TileState tileStates[tSlice];
        #pragma unroll
        for (uint i = 0; i < tSlice; ++i) {
            predicates[i] = 0U;
            tileStates[i] = TileState::unidentified;
        }
        const auto nT = tiles / blocks + (blockIdx.x < blocks - tiles / blocks * blocks);
        const auto tNt = nT / THREADS + (threadIdx.x < THREADS - nT / THREADS * THREADS);
        auto processed = 0U;
        static_assert(sizeof(BlockScan::TempStorage) <= SHARED_SIZE);
        auto* __restrict__ bTs = CAST_TO(BlockScan::TempStorage, workspace);
        while (processed < nT) {
            // concurrently sweep pending tiles
            #pragma unroll
            for (uint i = 0; i < tSlice; ++i) {
                if (i < tNt) {
                    const auto idx = i * THREADS + threadIdx.x;
                    if (tileStates[i] != TileState::completed) {
                        predicates[i] = atomicLoad(tileSync + idx) == tilesK;
                        tileStates[i] = predicates[i] ? TileState::identified : TileState::unidentified;
                    }
                }
            }
            uint16_t completedTiles = 0U;
            // Perform block-wide Aggregation
            BlockScan(*bTs).InclusiveSum(predicates, predicates, completedTiles);
            // Populate task queue with tiles
            #pragma unroll
            for (uint i = 0; i < tSlice; ++i) {
                if (i < tNt) {
                    const uint16_t idx = i * THREADS + threadIdx.x;
                    const uint16_t lIdx = i + threadIdx.x * tSlice;
                    if (tileStates[i] == TileState::identified) {
                        tileStates[i] = TileState::completed;
                        tQ[predicates[i] - lIdx] = idx * blocks + blockIdx.x;
                    }
                }
            }
            // needed for global visibility of tQ updates
            __syncthreads();
            #pragma unroll 4
            for (uint i = 0; i < completedTiles; ++i) {
                const auto tileIdx = tQ[i];
                // do gemm
                fGST<OperationX>(workspace, mA, mB, mD, mC, pS, pCw, M, N, tileIdx);
            }
            processed += completedTiles;
        }
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
        if (!skip) {
            auto tDt = static_cast<float>(end - start) / 1e6f; // convert nano to milliseconds
            // Intra-block reduction to get maximum latency
            using BlockReduce = cub::BlockReduce<float, THREADS>;
            auto* __restrict__ rTs = CAST_TO(BlockReduce::TempStorage, workspace);
            // Compute the block-wide max for thread0
            auto bT = BlockReduce(*rTs).Reduce(tDt, cub::Max());
            // Inter-block max reduction
            if (!threadIdx.x) {
                atomicMin(deviceThroughput, __float2uint_rn(__fdividef(1, bT))); // experts per ms
            }
        }
    }
}
#endif //EXPERT_CUH
