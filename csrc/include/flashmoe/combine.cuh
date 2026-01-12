//
// Created by osayamen on 1/11/26.
//

#ifndef FLASHMOE_COMBINE_CUH
#define FLASHMOE_COMBINE_CUH
#include "packed.cuh"
#include "task.cuh"
#include "tile.cuh"
#include "rvt.cuh"
#include "vt.cuh"
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
        typename Element
    >
    void combine(const Heap& symHeap,
            void* __restrict__ const& workspace,
            const TPS* __restrict__ const& tokenIndices,
            int* __restrict__ const& tokenGuards,
            const int& S,
            const int& E, const int& H,
            Element* __restrict__ const& moeOutput,
            const Task& task) {
        using BLAS = TileGEMM1::BLAS;
        const uint tileIdx = task.combineTileIdx();
        const uint tbs = task.tokenBatchStart();
        constexpr auto bM = cublasdx::size_of<BLAS>::m;
        constexpr auto bN = cublasdx::size_of<BLAS>::m;
        __shared__ TPS stIds[bM];
        const auto tileCoord = cute::make_coord(cute::_0{}, tileIdx);
        const auto tIds = cute::make_tensor(cute::make_gmem_ptr(tokenIndices),
            cute::make_layout(cute::make_shape(E, symHeap.EC), cute::LayoutRight{}));
        #pragma unroll
        for (int i = threadIdx.x; i < bM; i += threads) {
            stIds[i] = tIds(task.expertIdx(), tbs + i);
        }
        const auto* __restrict__ tokens = reinterpret_cast<Element*>(symHeap.advance<1,1>(task.epRank(),
            task.localExpertIdx(), tbs));
        auto sC = cublasdx::make_tensor(static_cast<Element*>(workspace), BLAS::get_layout_smem_c());
        const auto gC = tile::getC<bM, bN, cublasdx::arrangement_of_v_c<BLAS>>(tokens, bM, H, tileCoord);
        static_assert(cute::is_compatible<decltype(gC.layout()), decltype(sC.layout())>::value);
        // copy processed tile from gmem -> smem
        cublasdx::copy<BLAS, TileGEMM1::CAlign::value>(gC, sC);
        cublasdx::copy_wait();
        if constexpr (c == CombineMode::single) {
            // vector copy results to gmem and call it a day
            using VTD = VectorTypeDescriptor<Element, ElementAlignment<Element, bN>>;
            using VT = VTD::VectorType;
            constexpr auto vw = VTD::VectorWidth::value;
            constexpr auto vbN = bN / vw;
            // row major output
            const auto mC = cublasdx::make_tensor(cute::make_gmem_ptr(reinterpret_cast<VTD::VectorType*>(moeOutput)),
                cute::make_layout(cute::make_shape(S, H), cute::LayoutRight{}));
            // we enforce H % bN == 0 cpu-side
            const auto tC = cute::local_tile(mC, cute::make_shape(S, cute::Int<vbN>{}), tileCoord);
            // row major layout
            using VSL = cute::Layout<cute::Shape<cute::Int<bM>, cute::Int<vbN>>,
                    cute::Stride<cute::Int<vbN>, cute::_1>>;
            static_assert(cute::is_compatible<VSL, decltype(sC.layout())>::value);
            const auto vsC = cute::make_tensor(cute::make_smem_ptr(static_cast<const VTD::VectorType*>(workspace)),
                VSL{});
            constexpr auto nElems = vbN * bM;
            const auto actualElems = task.tileSize() * vbN;
            static_assert(nElems % threads == 0);
            constexpr auto elemsPerThread = nElems / threads;
            uint cache[elemsPerThread];
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
            constexpr auto redArch = arch < 800 ? 700 : (arch < 900 ? 800 : 900);
        }
    }
}
#endif //FLASHMOE_COMBINE_CUH