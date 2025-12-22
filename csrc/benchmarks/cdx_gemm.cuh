//
// Created by osayamen on 12/18/25.
//

#ifndef FLASHMOE_CDX_GEMM_CUH
#define FLASHMOE_CDX_GEMM_CUH

#include <cublasdx.hpp>
#include <cuda/utility>
#include <cutlass/numeric_conversion.h>
namespace v1
{
    template<int bM, int bN>
    __device__ __forceinline__
    constexpr auto tileIdx2Crd(const int& M, const int& N, const int& tileIdx) {
        const int tilesM = M / bM;
        const int tilesN = N / bN;
        const auto tileCoord = cute::idx2crd(tileIdx, cute::make_shape(tilesM, tilesN),
            cute::make_stride(tilesN, cute::_1{}));
        return cute::make_coord(cute::get<0>(tileCoord), cute::get<1>(tileCoord), cute::_);
    }
    template<int bM, int bK, cublasdx::arrangement ar, typename Element, typename TileCoord>
    __device__ __forceinline__
    constexpr auto getTileA(const Element* __restrict__ const& a, const int& M, const int& K, const TileCoord& tileCoord) {
        const auto strideA = cute::conditional_return<ar == cublasdx::row_major>
        (cute::make_stride(K, cute::_1{}), cute::make_stride(cute::_1{}, M));
        const auto mA = cute::make_tensor(cute::make_gmem_ptr(a),
            cute::make_layout(cute::make_shape(M, K), strideA));
        return cute::local_tile(mA, cute::Shape<cute::Int<bM>, cute::Int<bK>>{}, cute::select<0, 2>(tileCoord));
    }
    template<int bK, int bN, cublasdx::arrangement br, typename Element, typename TileCoord>
    __device__ __forceinline__
    constexpr auto getTileB(const Element* __restrict__ const& b, const int& K, const int& N, const TileCoord& tileCoord) {
        const auto strideB = cute::conditional_return<br == cublasdx::row_major>
        (cute::make_stride(N, cute::_1{}), cute::make_stride(cute::_1{}, K));
        const auto mB = cute::make_tensor(cute::make_gmem_ptr(b),
            cute::make_layout(cute::make_shape(K, N), strideB));
        return cute::local_tile(mB, cute::Shape<cute::Int<bK>, cute::Int<bN>>{}, cute::select<2, 1>(tileCoord));
    }
    template<int bM, int bN, cublasdx::arrangement cr, typename Element, typename TileCoord>
    __device__ __forceinline__
    constexpr auto getTileC(Element* __restrict__ const& c, const int& M, const int& N, const TileCoord& tileCoord) {
        const auto strideC = cute::conditional_return<cr == cublasdx::row_major>
        (cute::make_stride(N, cute::_1{}), cute::make_stride(cute::_1{}, M));
        const auto mC = cute::make_tensor(cute::make_gmem_ptr(c),
            cute::make_layout(cute::make_shape(M, N), strideC));
        return cute::local_tile(mC, cute::Shape<cute::Int<bM>, cute::Int<bN>>{}, cute::select<0, 1>(tileCoord));
    }
    template<int bM, int bN, typename Element, typename TileCoord>
    __device__ __forceinline__
    constexpr auto getTileBias(const Element* __restrict__ const& bias, const int& M, const int& N, const TileCoord& tileCoord) {
        // broadcast from {1, N} -> {M, N}
        const auto mD = cute::make_tensor(cute::make_gmem_ptr(bias),
            cute::make_layout(cute::make_shape(M, N), cute::Stride<cute::_0, cute::_1>{}));
        return cute::local_tile(mD, cute::Shape<cute::Int<bM>, cute::Int<bN>>{}, cute::select<0, 1>(tileCoord));
    }
    template<int N>
    __device__ __forceinline__
    void cpWait() {
        cute::cp_async_wait<N>();
        __syncthreads();
    }

    template<int offset, typename Tensor, typename Element>
    __device__ __forceinline__
    void update_buffer(Tensor& tensor, Element* __restrict__ const& base_ptr, const int& stage) {
        tensor.data() = base_ptr + (stage * offset);
    }
    template<int stage, int offset, typename Tensor, typename Element>
    __device__ __forceinline__
    void update_buffer(Tensor& tensor, Element* __restrict__ const& base_ptr) {
        tensor.data() = base_ptr + (stage * offset);
    }
    template<typename BLAS, int pipeStages, typename Element, typename Accumulator, typename TileCoord>
    requires(cublasdx::is_blas_execution_v<BLAS>)
    __device__ __forceinline__
    void tileMainLoop(void* __restrict__ const& workspace,
        const Element* __restrict__ const& a,
        const Element* __restrict__ const& b,
        Accumulator& accumulator,
        const int& M, const int& N, const int& K, const TileCoord& tileCoord) {
        accumulator.clear();
        using BK = cute::Int<cublasdx::size_of<BLAS>::k>;
        const int tilesK = K / BK{};

        const auto gA = getTileA<cublasdx::size_of<BLAS>::m, cublasdx::size_of<BLAS>::k,
            cublasdx::arrangement_of_v_a<BLAS>>(a, M, K, tileCoord);
        const auto gB = getTileB<cublasdx::size_of<BLAS>::k, cublasdx::size_of<BLAS>::n,
            cublasdx::arrangement_of_v_b<BLAS>>(b, K, N, tileCoord);

        /*shared layouts
        we could simplify with the below hierarchical layout, but it would not dispatch to LDSM.
        constexpr auto sALay = cute::tile_to_shape(BLAS::suggest_layout_smem_a().layout,
        cute::Shape<BM, BK, cute::Int<PIPE_STAGES>>{});
        constexpr auto sBLay = cute::tile_to_shape(BLAS::suggest_layout_smem_b().layout,
            cute::Shape<BK, BN, cute::Int<PIPE_STAGES>>{});
        const auto [sA, sB] = cublasdx::shared_memory::slice<Element, Element>(
            workspace, cublasdx::alignment_of_v_a<BLAS>, sALay, cublasdx::alignment_of_v_b<BLAS>, sBLay);
        */
        constexpr auto sASS = cublasdx::cosize(BLAS::suggest_layout_smem_a());
        constexpr auto sBSS = cublasdx::cosize(BLAS::suggest_layout_smem_b());
        auto* __restrict__ sAP = static_cast<Element*>(workspace);
        auto* __restrict__ sBP = static_cast<Element*>(workspace) + (sASS * pipeStages);
        auto sA = cublasdx::make_tensor(sAP, BLAS::suggest_layout_smem_a());
        auto sB = cublasdx::make_tensor(sBP, BLAS::suggest_layout_smem_b());
        cuda::static_for<pipeStages>([&](auto stage){
            update_buffer<stage, sASS>(sA, sAP);
            update_buffer<stage, sBSS>(sB, sBP);
            cublasdx::copy<BLAS, cublasdx::alignment_of_v_a<BLAS>>(gA(cute::_, cute::_, stage), sA);
            cublasdx::copy<BLAS, cublasdx::alignment_of_v_b<BLAS>>(gB(cute::_, cute::_, stage), sB);
            cute::cp_async_fence();
        });
        // mainloop
        for (int kStage = pipeStages; kStage < tilesK; ++kStage) {
            const int ps = kStage % pipeStages;
            update_buffer<sASS>(sA, sAP, ps);
            update_buffer<sBSS>(sB, sBP, ps);
            cpWait<pipeStages - 1>();
            BLAS().execute(sA, sB, accumulator);
            __syncthreads();
            cublasdx::copy<BLAS, cublasdx::alignment_of_v_a<BLAS>>(gA(cute::_, cute::_, kStage), sA);
            cublasdx::copy<BLAS, cublasdx::alignment_of_v_b<BLAS>>(gB(cute::_, cute::_, kStage), sB);
            cute::cp_async_fence();
        }
        // tail
        cuda::static_for<pipeStages>([&](auto stage) {
            const int ps = (tilesK + stage) % pipeStages;
            update_buffer<sASS>(sA, sAP, ps);
            update_buffer<sBSS>(sB, sBP, ps);
            cpWait<(pipeStages - 1) - stage>();
            BLAS().execute(sA, sB, accumulator);
        });
    }

    template<typename BLAS, typename Activation, int pipeStages, typename ElementC, typename Element>
    __device__ __forceinline__
    void gemmMainloop(void* __restrict__ const& workspace,
        const Element* __restrict__ const& a,
        const Element* __restrict__ const& b,
        ElementC* __restrict__ const& c,
        const ElementC* __restrict__ const& bias,
        const int& M, const int& N, const int& K, const int& tileIdx) {
        auto accumulator = BLAS::suggest_accumulator();
        using BM = cute::Int<cublasdx::size_of<BLAS>::m>;
        using BN = cute::Int<cublasdx::size_of<BLAS>::n>;
        const auto tileCoord = tileIdx2Crd<BM{}, BN{}>(M, N, tileIdx);
        // gmem -> rmem: prefetch bias
        const auto gD = getTileBias<BM{}, BN{}>(bias, M, N, tileCoord);
        auto d_frag = cublasdx::make_fragment_like<Element>(accumulator.get_results());
        cublasdx::copy_fragment<cublasdx::alignment_of<BLAS>::c>(gD, d_frag, accumulator);
        // compute Tile
        tileMainLoop<BLAS, pipeStages>(workspace, a, b, accumulator, M, N, K, tileCoord);
        // Epilogue
        constexpr Activation act{};
        constexpr cutlass::NumericConverter<ElementC, typename decltype(accumulator)::value_type> conv{};
        const auto c_frag = accumulator.get_results();
        constexpr int accum_size = cublasdx::size(c_frag);
        cuda::static_for<accum_size>([&c_frag, &d_frag](auto i) {
            // TODO: vectorize the addition?
            d_frag(i) = act(conv(c_frag(i)) + d_frag(i));
        });
        auto gC = getTileC<BM{}, BN{}, cublasdx::arrangement_of_v_c<BLAS>>(c, M, N, tileCoord);
        // rmem -> gmem
        cublasdx::copy_fragment<cublasdx::alignment_of<BLAS>::c>(d_frag, gC, accumulator);
    }

    #define SC(T, v) static_cast<T>(v)
    template<typename BLAS, int pipeStages, typename Activation,
    typename Element, typename ElementC>
    requires(cublasdx::is_blas_execution_v<BLAS> && pipeStages >= 0)
    __global__ void gk(const Element* __restrict__ a, const Element* __restrict__ b,
        ElementC* __restrict__ c, const ElementC* __restrict__ bias,
        const __grid_constant__ int M, const __grid_constant__ int N, const int __grid_constant__ K) {
        constexpr int bM = cublasdx::size_of<BLAS>::m;
        constexpr int bN = cublasdx::size_of<BLAS>::n;
        constexpr int bK = cublasdx::size_of<BLAS>::k;
        static_assert(bM % bN == 0);
        const int nTiles = (M / bM) * (N / bN);
        __shared__ __align__(16) cuda::std::byte workspace[bK * pipeStages * (bM + bN) * sizeof(Element)];
        for (int tileIdx = SC(int, blockIdx.x); tileIdx < nTiles; tileIdx += SC(int, gridDim.x)) {
            gemmMainloop<BLAS, Activation, pipeStages, ElementC>(workspace, a, b, c, bias, M, N, K, tileIdx);
        }
    }
}
#endif //FLASHMOE_CDX_GEMM_CUH