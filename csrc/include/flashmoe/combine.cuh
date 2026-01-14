//
// Created by osayamen on 1/11/26.
//

#ifndef FLASHMOE_COMBINE_CUH
#define FLASHMOE_COMBINE_CUH
#include "infra/packed.cuh"
#include "infra/task.cuh"
#include "infra/heap.cuh"
#include "tile.cuh"
#include "infra/atomics.cuh"
#include "infra/rvt.cuh"
#include "infra/vt.cuh"
namespace flashmoe
{
    enum class CombineMode {
        single, // top k = 1
        plural // top k > 1
    };

    // MoE combine at tile granularity
    template<
        typename TileGEMM1,
        int threads,
        CombineMode c,
        typename Element,
        typename TileCoord
    >
    void combine(const Heap& symHeap,
        const int& S, const int& E, const int& H, const int& k,
        void* __restrict__ const& workspace,
        const TPS* __restrict__ const& tokenIndices, // [E, EC], where EC is padded to a multiple of bM
        int* __restrict__ const& tokenGuards, // [S, vH], where vH = H / RVD::VectorWidth::value, only needed k > 1
        Element* __restrict__ const& moeOutput, // [S, H]
        const Element* __restrict__ const& tokens, // [bM, H]
        const Task& task, const TileCoord& tileCoord) {
        using BLAS = TileGEMM1::BLAS;
        const uint tbs = task.tokenBatchStart();
        constexpr auto bM = cublasdx::size_of<BLAS>::m;
        constexpr auto bN = cublasdx::size_of<BLAS>::n;
        __shared__ TPS stIds[bM];
        const auto tIds = cute::make_tensor(cute::make_gmem_ptr(tokenIndices),
            cute::make_layout(cute::make_shape(E, symHeap.EC), cute::LayoutRight{}));
        auto sC = cublasdx::make_tensor(static_cast<Element*>(workspace), BLAS::get_layout_smem_c());
        const auto gC = tile::getC<bM, bN, cublasdx::arrangement_of_v_c<BLAS>>(tokens, bM, H, tileCoord);
        static_assert(cute::is_compatible<decltype(gC.layout()), decltype(sC.layout())>::value);

        #pragma unroll
        for (int i = threadIdx.x; i < bM; i += threads) {
            stIds[i] = tIds(task.expertIdx(), tbs + i);
        }
        // copy processed tile from gmem -> smem
        cublasdx::copy<BLAS, TileGEMM1::CAlign::value>(gC, sC);
        cublasdx::copy_wait();
        __syncthreads();
        if constexpr (c == CombineMode::single) {
            using VTD = VectorTypeDescriptor<Element, ElementAlignment<Element, bN>>;
            using VT = VTD::VectorType;
            constexpr auto vw = VTD::VectorWidth::value;
            constexpr auto vbN = bN / vw;
            constexpr auto nElems = vbN * bM;
            const auto actualElems = task.tileSize() * vbN;
            static_assert(nElems % threads == 0);
            constexpr auto elemsPerThread = nElems / threads;
            uint cache[cute::ceil_div(nElems, threads)]; // use ceil_div to avoid 0
            // row major output
            const auto vH = H / vw;
            auto mC = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<VT*>(moeOutput)),
                cute::make_layout(cute::make_shape(S, vH), cute::LayoutRight{}));
            // we enforce H % bN == 0 cpu-side
            auto tC = cute::local_tile(mC, cute::make_shape(S, cute::Int<vbN>{}), tileCoord);
            // row major layout
            using VSL = cute::Layout<cute::Shape<cute::Int<bM>, cute::Int<vbN>>,
                    cute::Stride<cute::Int<vbN>, cute::_1>>;
            static_assert(cute::is_compatible<VSL, decltype(sC.layout())>::value);
            const auto vsC = cute::make_tensor(cute::make_smem_ptr(static_cast<const VT*>(workspace)),
                VSL{});
            // vector copy results to gmem and call it a day
            #pragma unroll
            for (int i = 0; i < elemsPerThread; ++i) {
                const auto idx = threadIdx.x + i * threads;
                const auto rowIdx = idx / vbN;
                const auto colIdx = idx % vbN;
                cache[i] = stIds[rowIdx].tokenIdx;
            }
            if (task.tileSize() == bM) {
                #pragma unroll
                for (int i = 0; i < elemsPerThread; ++i) {
                    const auto idx = threadIdx.x + i * threads;
                    const auto rowIdx = idx / vbN;
                    const auto colIdx = idx % vbN;
                    const auto tokenIdx = cache[i];
                    // smem -> gmem.
                    tC(tokenIdx, colIdx) = vsC(rowIdx, colIdx);
                }
            }
            else {
                #pragma unroll
                for (int i = 0; i < elemsPerThread; ++i) {
                    const auto idx = threadIdx.x + i * threads;
                    const auto rowIdx = idx / vbN;
                    const auto colIdx = idx % vbN;
                    const auto tokenIdx = cache[i];
                    // smem -> gmem.
                    if (idx < actualElems) {
                        tC(tokenIdx, colIdx) = vsC(rowIdx, colIdx);
                    }
                }
            }
            constexpr auto residue = nElems - (elemsPerThread * threads);
            if constexpr (residue && threadIdx.x < residue) {
                const auto idx = threadIdx.x + elemsPerThread * threads;
                const auto rowIdx = idx / vbN;
                const auto colIdx = idx % vbN;
                const auto tokenIdx = stIds[rowIdx].tokenIdx;
                // smem -> gmem.
                if (idx < actualElems) {
                    tC(tokenIdx, colIdx) = vsC(rowIdx, colIdx);
                }
            }
        }
        else {
            // we need to atomically reduce to the output buffer here
            constexpr auto arch = cublasdx::sm_of_v<BLAS>;
            // promotes to fp16x2 or bf16x2, if it can.
            using RAD = RedAddType<Element, ElementAlignment<Element, bN>>;
            using RAT = RAD::Type;
            constexpr int bNp = bN / RAD::Width::value;
            static_assert(RAD::Width::value == 1 || RAD::Width::value == 2);
            using RedAddOp = RedAdd<RedArch<arch>, RAT, ElementAlignment<RAT, bNp>>;
            using RVD = VectorTypeDescriptor<RAT, RAD::Width * sizeof(RAT)>;
            using RV = RVD::VectorType;
            constexpr auto rbN = bN / RVD::VectorWidth::value;
            using RSL = cute::Layout<cute::Shape<cute::Int<bM>, cute::Int<rbN>>,
                    cute::Stride<cute::Int<rbN>, cute::_1>>;
            const auto vsC = cute::make_tensor(cute::make_smem_ptr(static_cast<const RV*>(workspace)), RSL{});
            // row major output
            const auto vH = H / RVD::VectorWidth::value;
            const auto vHo = H / RAD::Width::value;
            auto mC = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<RAT*>(moeOutput)),
                cute::make_layout(cute::make_shape(S, vHo), cute::LayoutRight{}));
            auto mGuards = cute::make_tensor(cute::make_gmem_ptr(tokenGuards),
                cute::make_layout(cute::make_shape(S, vH), cute::LayoutRight{}));
            auto tC = cute::local_tile(mC, cute::make_shape(S, cute::Int<rbN>{}), tileCoord);
            auto tGuards = cute::local_tile(mGuards, cute::make_shape(S, cute::Int<rbN>{}), tileCoord);
            constexpr auto totalElems = bM * rbN; // cublasdx::cosize(RSL{})
            const auto actualElems = task.tileSize() * rbN;
            constexpr auto redElemsPerThread = totalElems / threads;
            TPS cache[cute::ceil_div(totalElems, threads)]; // use ceil_div to avoid zero
            #pragma unroll
            for (int i = 0; i < redElemsPerThread; ++i) {
                const auto idx = threadIdx.x + i * threads;
                const auto rowIdx = idx / rbN;
                const auto colIdx = idx % rbN;
                cache[i] = stIds[rowIdx];
            }
            constexpr auto elemsPerPack = RVD::VectorWidth::value;
            if (task.tileSize() == bM) {
                #pragma unroll
                for (int i = 0; i < redElemsPerThread; ++i) {
                    const auto idx = threadIdx.x + i * threads;
                    const auto rowIdx = idx / rbN;
                    const auto colIdx = idx % rbN;
                    const auto indexAndScale = cache[i];
                    auto tokenValue = vsC(rowIdx, colIdx);
                    const auto tokIdx = indexAndScale.tokenIdx;
                    if constexpr (RAD::Width::value == 2) {
                        constexpr Converter<float2, RAT> loadOp{};
                        constexpr Converter<RAT, float2> storeOp{};
                        // fp16x2 or bf16x2
                        const auto scale2 = float2{indexAndScale.probability, indexAndScale.probability};
                        #pragma unroll
                        for (int j = 0; j < RVD::VectorWidth::value; ++j) {
                            // convert to float2 -> multiply -> convert back
                            tokenValue[j] = storeOp(__fmul2_rn(loadOp(tokenValue[j]), scale2));
                        }
                    }
                    else {
                        constexpr Converter<float, RAT> loadOp{};
                        constexpr Converter<RAT, float> storeOp{};
                        #pragma unroll
                        for (int j = 0; j < elemsPerPack; ++j) {
                            // convert to float -> multiply -> convert back
                            tokenValue[j] = storeOp(loadOp(tokenValue[j]) * indexAndScale.probability);
                        }
                    }
                    // account for the fact that the type of the below tile is either Element or RAT.
                    // Since we read 'elemsPerPack' per iteration we need to advance the colIx by that much.
                    // The row index is preserved because we only apply this vectorization across columns
                    // which are contiguous in memory.
                    auto* __restrict__ tCp = (&tC(tokIdx, colIdx * elemsPerPack));
                    guardedRedAdd<RedAddOp>(&tGuards(tokIdx, colIdx), tCp, tokenValue, k);
                }
            }
            else {
                #pragma unroll
                for (int i = 0; i < redElemsPerThread; ++i) {
                    const auto idx = threadIdx.x + i * threads;
                    if (idx < actualElems) {
                        const auto rowIdx = idx / rbN;
                        const auto colIdx = idx % rbN;
                        const auto indexAndScale = cache[i];
                        auto tokenValue = vsC(rowIdx, colIdx);
                        const auto tokIdx = indexAndScale.tokenIdx;
                        if constexpr (RAD::Width::value == 2) {
                            constexpr Converter<float2, RAT> loadOp{};
                            constexpr Converter<RAT, float2> storeOp{};
                            // fp16x2 or bf16x2
                            const auto scale2 = float2{indexAndScale.probability, indexAndScale.probability};
                            #pragma unroll
                            for (int j = 0; j < RVD::VectorWidth::value; ++j) {
                                // convert to float2 -> multiply -> convert back
                                tokenValue[j] = storeOp(__fmul2_rn(loadOp(tokenValue[j]), scale2));
                            }
                        }
                        else {
                            constexpr Converter<float, RAT> loadOp{};
                            constexpr Converter<RAT, float> storeOp{};
                            #pragma unroll
                            for (int j = 0; j < RVD::VectorWidth::value; ++j) {
                                // convert to float -> multiply -> convert back
                                tokenValue[j] = storeOp(loadOp(tokenValue[j]) * indexAndScale.probability);
                            }
                        }
                        auto* __restrict__ tCp = &tC(tokIdx, colIdx * elemsPerPack);
                        guardedRedAdd<RedAddOp>(&tGuards(tokIdx, colIdx), tCp, tokenValue, k);
                    }
                }
            }
            constexpr auto residue = totalElems - (redElemsPerThread * threads);
            if constexpr (residue && threadIdx.x < residue) {
                const auto idx = threadIdx.x + redElemsPerThread * threads;
                const auto rowIdx = idx / rbN;
                const auto colIdx = idx % rbN;
                const auto indexAndScale = stIds[rowIdx];
                auto tokenValue = vsC(rowIdx, colIdx);
                const auto tokIdx = indexAndScale.tokenIdx;
                // smem -> gmem.
                if (idx < actualElems) {
                    if constexpr (RAD::Width::value == 2) {
                        constexpr Converter<float2, RAT> loadOp{};
                        constexpr Converter<RAT, float2> storeOp{};
                        // fp16x2 or bf16x2
                        const auto scale2 = float2{indexAndScale.probability, indexAndScale.probability};
                        #pragma unroll
                        for (int j = 0; j < RVD::VectorWidth::value; ++j) {
                            // convert to float2 -> multiply -> convert back
                            tokenValue[j] = storeOp(__fmul2_rn(loadOp(tokenValue[j]), scale2));
                        }
                    }
                    else {
                        constexpr Converter<float, RAT> loadOp{};
                        constexpr Converter<RAT, float> storeOp{};
                        #pragma unroll
                        for (int j = 0; j < RVD::VectorWidth::value; ++j) {
                            // convert to float -> multiply -> convert back
                            tokenValue[j] = storeOp(loadOp(tokenValue[j]) * indexAndScale.probability);
                        }
                    }
                    auto* __restrict__ tCp = &tC(tokIdx, colIdx * elemsPerPack);
                    guardedRedAdd<RedAddOp>(&tGuards(tokIdx, colIdx), tCp, tokenValue, k);
                }
            }
        }
    }
}
#endif //FLASHMOE_COMBINE_CUH