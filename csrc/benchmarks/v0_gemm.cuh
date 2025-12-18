//
// Created by osayamen on 12/18/25.
//

#ifndef FLASHMOE_V0_GEMM_CUH
#define FLASHMOE_V0_GEMM_CUH

#include "../include/flashmoe/os/processor/gemm.cuh"
namespace v0
{
    template<typename BlockGEMM, int threads, typename MMA_C, typename ElementC, typename Element>
    __device__ __forceinline__
    void gemmMainloopV1(void* __restrict__ const& workspace,
        const Element* __restrict__ const& a,
        const Element* __restrict__ const& b,
        void* __restrict__ const& c,
        const void* __restrict__ const& bias,
        const int& M, const int& N, const int& K, const int& tileIdx) {
        constexpr auto bM = cute::get<0>(typename BlockGEMM::BlockTiler{});
        constexpr auto bN = cute::get<1>(typename BlockGEMM::BlockTiler{});
        constexpr auto bK = cute::get<2>(typename BlockGEMM::BlockTiler{});
        auto accumulator = cute::partition_fragment_C(typename BlockGEMM::MMA{}, typename BlockGEMM::TilerOut{});
        static_assert(cute::size(accumulator) == bN);
        // Instantiate mainloop
        typename BlockGEMM::CollectiveMainloop mainLoop{};
        cute::clear(accumulator);
        // Row-major
        const auto mA = cute::make_tensor(cute::make_gmem_ptr(a),
            make_layout(cute::make_shape(M, K), cute::LayoutRight{}));
        // Row-major, transposed
        const auto mB = make_tensor(cute::make_gmem_ptr(b),
            make_layout(cute::make_shape(N, K), cute::LayoutRight{}));
        const auto* __restrict__ vBias = static_cast<const float4*>(bias);
        constexpr int vbN = bN / 4;
        const int tilesM = M / bM;
        const int tilesN = N / bN;
        const int tilesK = K / bK;
        const auto tileCoord = idx2crd(tileIdx, cute::Shape(tilesM, tilesN), cute::Stride(tilesN, 1));
        const auto ctaCoord = make_coord(cute::get<0>(tileCoord), cute::get<1>(tileCoord), cute::_);
        const auto gA = cute::local_tile(mA, typename BlockGEMM::BlockTiler{}, ctaCoord,
            cute::Step<cute::_1, cute::X,cute::_1>{});
        const auto gB = cute::local_tile(mB, typename BlockGEMM::BlockTiler{}, ctaCoord,
            cute::Step< cute::X,cute::_1,cute::_1>{});
        const auto k_tile_iter = cute::make_coord_iterator(tilesK);
        // prefetch bias
        constexpr auto trips = cute::ceil_div(vbN, threads);
        constexpr auto elems = cute::min(vbN, threads);
        float4 biasCache[trips];
        const int biasOffset = (tileIdx % tilesN) * vbN;
        const auto* __restrict__ bP = vBias + biasOffset;
        if constexpr (threads >= vbN) {
            static_assert(threads % vbN == 0);
            biasCache[0] = bP[threadIdx.x % vbN];
        }
        else {
            // below is not strictly necessary, but it makes my life easier :)
            static_assert(vbN % threads == 0);
            #pragma unroll
            for (int i = 0; i < trips; ++i) {
                biasCache[i] = bP[threadIdx.x + i * vbN];
            }
        }
        mainLoop(
            accumulator,
            gA,
            gB,
            accumulator,
            k_tile_iter, tilesK,
            cute::Underscore{},
            threadIdx.x,
            CAST_TO(char, workspace));
        __syncthreads();

        typename BlockGEMM::MMA tiledMMA{};
        constexpr auto gCStoreOp = cutlass::NumericConverter<ElementC, MMA_C>{};
        // Assume elementwise operator
        typename BlockGEMM::FusedEpilogue epilogueOp{};
        constexpr auto psCLay = cute::make_layout(cute::Shape<cute::Int<bM>, cute::Int<bN>>{},
            cute::LayoutRight{});
        const auto psC = cute::make_tensor(cute::make_smem_ptr(CAST_TO(MMA_C, workspace)), psCLay);
        const auto tCsC = tiledMMA.get_slice(threadIdx.x).partition_C(psC);
        #pragma unroll
        for (unsigned int j = 0; j < bN; ++j) {
            tCsC(j) = accumulator(j);
        }
        __syncthreads();
        const auto rIdx = threadIdx.x / vbN * vbN;
        const auto cIdx = threadIdx.x % vbN;
        constexpr auto sCLay = cute::make_layout(cute::Shape<cute::Int<bM>, cute::Int<vbN>>{},
            cute::LayoutRight{});
        const auto sC = cute::make_tensor(cute::make_smem_ptr(CAST_TO(float4, workspace)), sCLay);
        // vectorize here
        float4 vrC[vbN];
        #pragma unroll
        for (int i = 0; i < trips; ++i) {
            #pragma unroll
            for (unsigned int j = 0; j < elems; ++j) {
                vrC[j + i * elems] = sC(rIdx + j, cIdx + i * elems);
            }
        }
        // apply epilogue -> Activation(fragment_c + bias)
        #pragma unroll
        for (uint i = 0; i < trips; ++i) {
            #pragma unroll
            for (int j = 0; j < elems; ++j) {
                const auto v = vrC[j + i * elems];
                const auto bc = biasCache[i];
                vrC[j + i * elems] = float4{
                    epilogueOp(v.x, bc.x),
                    epilogueOp(v.y, bc.y),
                    epilogueOp(v.z, bc.z),
                    epilogueOp(v.w, bc.w)};
            }
        }
        // Coalesced, vectorized copy from registers to global memory
        auto* __restrict__ vC = static_cast<float4*>(c);
        const int vN = N / 4;
        // Row Major
        const auto mC = cute::make_tensor(cute::make_gmem_ptr(vC),
                make_layout(cute::make_shape(M, vN), cute::LayoutRight{}));
        const auto gC = cute::local_tile(mC,
            cute::Shape<cute::Int<bM>, cute::Int<vbN>, cute::Int<bK>>{}, ctaCoord,
            cute::Step<cute::_1,cute::_1, cute::X>{});
        #pragma unroll
        for (uint i = 0; i < trips; ++i) {
            #pragma unroll
            for (unsigned int j = 0; j < elems; ++j) {
                gC(rIdx + j, cIdx + i * elems) = vrC[j + i * elems];
            }
        }
    }

    #define SC(T, v) static_cast<T>(v)

    template<int bM, int bN, int bK, int pipeStages, int threads,
    typename ActivationFunction, typename MMA_C,
    typename Element, typename ElementC>
    __global__ void gk(const Element* __restrict__ a, const Element* __restrict__ b,
        ElementC* __restrict__ c, const ElementC* __restrict__ bias,
        const __grid_constant__ int M, const __grid_constant__ int N, const int __grid_constant__ K) {
        static_assert(threads == bM);
        static_assert(bM % bN == 0);
        const int nTiles = (M / bM) * (N / bN);
        constexpr auto sharedSize = cute::max(bK * pipeStages * (bM + bN) * sizeof(ElementC),
            bM * bN * sizeof(MMA_C));
        __shared__ __align__(16) cuda::std::byte workspace[sharedSize];
            using V1G = flashmoe::BlockMM<ActivationFunction, Element, Element, MMA_C, bM, bN, bK, pipeStages, threads>;
        for (int tileIdx = SC(int, blockIdx.x); tileIdx < nTiles; tileIdx += SC(int, gridDim.x)) {
            gemmMainloopV1<V1G, threads, MMA_C, ElementC>(workspace, a, b, c, bias, M, N, K, tileIdx);
        }
    }
}
#endif //FLASHMOE_V0_GEMM_CUH