//
// Created by oja7 on 1/28/25.
//

#ifndef EXPERT_CUH
#define EXPERT_CUH

#include <cub/cub.cuh>
#include <cuda/std/cstddef>
#include <cute/tensor.hpp>

#include "../arch.cuh"
#include "../atomics.cuh"
#include "../types.cuh"
#include "../os/processor/gemm.cuh"
#include "../os/processor/processor.cuh"
namespace aristos {
    template<typename BlockGEMM>
    __forceinline__ __device__
    void fGST(typename BlockGEMM::MatrixDType* __restrict__& workspace,
    const typename BlockGEMM::MatrixAType* __restrict__& inputs,
    const typename BlockGEMM::MatrixBType* __restrict__& weights,
    typename BlockGEMM::MatrixDType* __restrict__& output,
    const typename BlockGEMM::MatrixDType* __restrict__& bias,
    const typename BlockGEMM::MatrixDType* __restrict__& scaleWeights,
    const unsigned int& M,
    const unsigned int& N,
    const unsigned int& K,
    const unsigned int& tileIdx) {
        auto accumulator = cute::partition_fragment_C(typename BlockGEMM::MMA{}, typename BlockGEMM::TilerOut{});
        constexpr auto elems = SHARED_SIZE / (THREADS * sizeof(typename BlockGEMM::MatrixDType));
        static_assert(cute::size(accumulator) % elems == 0);
        cutlass::AlignedArray<typename BlockGEMM::MatrixDType, elems> rScratch{};
        constexpr auto bM = cute::get<0>(typename BlockGEMM::BlockTiler{});
        static_assert(size(accumulator) % rScratch.size() == 0);
        // Instantiate mainloop
        typename BlockGEMM::CollectiveMainloop mainLoop{};
        cute::clear(accumulator);

        // Row-major
        const auto mA = make_tensor(cute::make_gmem_ptr(inputs),
            make_layout(cute::make_shape(M, K), cute::make_stride(K, 1)));
        // Row-major, transposed
        const auto mB = make_tensor(cute::make_gmem_ptr(weights),
            make_layout(cute::make_shape(N, K), cute::make_stride(K, 1)));
        // Row-major
        const auto mC = make_tensor(cute::make_gmem_ptr(output,
            make_layout(cute::make_shape(M, N), cute::make_stride(N, 1))));
        const auto mD = make_tensor(cute::make_gmem_ptr(bias),
            make_layout(cute::make_shape(M, N), cute::make_stride(0, 1)));
        const auto mS = make_tensor(cute::make_gmem_ptr(scaleWeights),
               make_layout(cute::make_shape(M, N), cute::make_stride(1, 0)));

        const auto tilesM = M / cute::get<0>(typename BlockGEMM::BlockTiler{});
        const auto tilesN = N / cute::get<1>(typename BlockGEMM::BlockTiler{});

        const auto tileCoord = idx2crd(tileIdx, cute::Shape(tilesM, tilesN), cute::Stride(tilesN ,1));
        const auto ctaCoord = make_coord(cute::get<0>(tileCoord), cute::get<1>(tileCoord), cute::_);
        const auto gA = cute::local_tile(mA, typename BlockGEMM::BlockTiler{}, ctaCoord, cute::Step<cute::_1, cute::X,cute::_1>{});
        const auto gB = cute::local_tile(mB, typename BlockGEMM::BlockTiler{}, ctaCoord, cute::Step< cute::X,cute::_1,cute::_1>{});
        const auto gC = cute::local_tile(mC, typename BlockGEMM::BlockTiler{}, ctaCoord, cute::Step<cute::_1,cute::_1, cute::X>{});
        const auto gD = cute::local_tile(mD, typename BlockGEMM::BlockTiler{}, ctaCoord, cute::Step<cute::_1,cute::_1, cute::X>{});
        const auto gS = cute::local_tile(mS, typename BlockGEMM::BlockTiler{}, ctaCoord, cute::Step<cute::_1,cute::_1, cute::X>{});

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
        const auto tSgS = tiledMMA.get_slice(threadIdx.x).partition_C(gS);

        constexpr auto gCStoreOp = cutlass::NumericConverter<typename decltype(gC)::value_type,
                                                    typename decltype(accumulator)::value_type>{};
        constexpr auto gDLoadOp = cutlass::NumericConverter<typename decltype(accumulator)::value_type,
                                                    ElementD>{};
        constexpr auto scaleOp = cutlass::epilogue::thread::Scale<typename decltype(accumulator)::value_type>{};

        // Assume elementwise operator
        constexpr typename BlockGEMM::FusedEpilogue epilogueOp{};
        constexpr auto trips = size(accumulator) / rScratch.size();

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
                accumulator(j + i * elems) = epilogueOp(accumulator(j + i * elems), gDLoadOp(rScratch[j]));
            }
        }

        // Do Scale on register fragment
        #pragma unroll
        for (int j = 0; j < elems; ++j) {
            workspace[threadIdx.x + j * THREADS] = tSgS(j);
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
                accumulator(j + i * elems) = scaleOp(accumulator(j + i * elems), gDLoadOp(rScratch[j]));
            }
        }

        __syncthreads();
        constexpr auto sCLay = cute::make_layout(cute::Shape<cute::Int<bM>, cute::Int<elems>>{},
            cute::LayoutRight{});
        const auto sC = cute::make_tensor(cute::make_smem_ptr(workspace), sCLay);
        const auto tCsC = tiledMMA.get_slice(threadIdx.x).partition_C(sC);
        const auto rIdx = threadIdx.x / elems * elems;
        const auto cIdx = threadIdx.x % elems;
        #pragma unroll
        for (unsigned int i = 0; i < trips; ++i) {
            #pragma unroll
            for (unsigned int j = 0; j < elems; ++j) {
                tCsC(j) = gCStoreOp(accumulator(j + i * elems));
            }
            __syncthreads();
            #pragma unroll
            for (unsigned int j = 0; j < elems; ++j) {
                gC(rIdx + j, cIdx + i * elems) = sC(rIdx + j, cIdx);
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
    // single shot expert used for evaluation only
    // For simplicity, assume single element types for all matrices,
    // This is typical of PyTorch workloads
    template<
        unsigned int Arch,
        typename ActivationOp,
        unsigned int batch = GEMMs,
        typename Element,
        typename ProblemShape_MNK
    >
    requires(cute::is_tuple_v<ProblemShape_MNK> && rank(ProblemShape_MNK{}) == 4
        && aristos::TensorValueType<Element>)
    /// D = A * B1 + C1
    /// A = D * B2 + C2
    __global__ void expert(ProblemShape_MNK const pS,
        uint* __restrict__ tileSync, Element* __restrict__ pA,
        const cuda::std::array<Element* __restrict__, batch> pB,
        const cuda::std::array<Element* __restrict__, batch> pC /*bias*/,
        const cuda::std::array<Element* __restrict__, batch> pSw /*scale weights*/,
        Element* __restrict__ pD, uint* __restrict__ deviceThroughput) {
        uint64_t start = 0, end = 0;
        constexpr auto blocks = Hardware<Arch>::blocks::value - 1U;
        constexpr auto tUpB = 1536U; // max supported number of tiles per block per gemm
        __shared__ __align__(16) cuda::std::byte workspace[SHARED_SIZE];
        __shared__ __align__(16) uint16_t tQ[tUpB];
        const auto M = cute::get<0>(pS);
        const auto N = cute::get<1>(pS);
        const auto Nx = cute::get<2>(pS);
        const auto K = cute::get<3>(pS);
        using Operation = BlockMM<Arch, Element, ActivationOp>;
        using OperationX = BlockMM<Arch, Element>;
        constexpr auto preGEMM = processor::FGT<TaskType::preGEMM, Operation>{};
        const auto tilesM = Config::tiles<BLOCK_M>(M);
        const auto tilesN = Config::tiles<BLOCK_N>(N);
        const auto tilesNx = Config::tiles<BLOCK_N>(Nx);
        const auto tilesX = tilesM * tilesNx;
        const auto tiles = tilesM * tilesN;

        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
        #pragma unroll 4
        for (uint i = blockIdx.x; i < tiles; i += blocks) {
            const auto tM = i / tilesNx;
            preGEMM(CAST_TO(Element, workspace), pA, pB[0], pD, pC[0], M, Nx, K, i);
            // notify this tile's completion
            if (!threadIdx.x) {
                __threadfence();
                atomicIncrement(tileSync + tM);
            }
        }

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
        auto* __restrict__ bTs = CAST_TO(BlockScan::TempStorage, workspace);
        while (processed < nT) {
            // concurrently sweep pending tiles
            #pragma unroll
            for (uint i = 0; i < tSlice; ++i) {
                if (i < tNt) {
                    const auto idx = i * THREADS + threadIdx.x;
                    if (tileStates[i] != TileState::completed) {
                        predicates[i] = atomicLoad(tileSync + idx) == tilesNx;
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
                fGST<OperationX>(workspace, pD, pB[1], pA, pC[1], pSw, M, N, K,
                    tileIdx);
            }
            processed += completedTiles;
        }
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
        auto tDt = static_cast<float>(end - start) / 1e6f;
        // Intra-block reduction to get maximum latency
        using BlockReduce = cub::BlockReduce<float, THREADS>;
        auto* __restrict__ rTs = CAST_TO(BlockReduce::TempStorage, workspace);
        // Compute the block-wide max for thread0
        auto bT = BlockReduce(*rTs).Reduce(tDt, cub::Max());
        // Inter-block max reduction
        if (!threadIdx.x) {
            atomicMax(deviceThroughput, __float2uint_rn(bT));
        }
    }
}
#endif //EXPERT_CUH
