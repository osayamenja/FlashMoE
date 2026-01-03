//
// Created by osayamen on 12/25/25.
//

#ifndef FLASHMOE_TILE_CUH
#define FLASHMOE_TILE_CUH
#include <cublasdx.hpp>

#define MAX_ALIGNMENT 16 // blackwell is 32?
namespace flashmoe
{
    template<typename T, typename S>
    struct Converter {
        __device__ auto operator()(const S& x) const {
            return static_cast<T>(x);
        }
    };
    template<>
    struct Converter<__half, float> {
        __device__ auto operator()(const float& x) const {
            return  __float2half(x);
        }
    };
    template<>
    struct Converter<__nv_bfloat16, float> {
        __device__ auto operator()(const float& x) const {
            return  __float2bfloat16(x);
        }
    };

    template<typename T, int Alignment = MAX_ALIGNMENT>
    struct VectorTypeDescriptor {
        using VectorWidth = cute::C<Alignment / sizeof(T)>;
        using VectorType = cutlass::AlignedArray<T, VectorWidth::value, Alignment>;
    };

}
namespace flashmoe::tile
{
    template<int M, int N, int K, typename Element>
    struct Heuristics {
        using BM = cute::Int<cute::min(M, 128)>;
        using BN = cuda::std::conditional_t<N <= 64, cute::Int<N>, cute::Int<256 / sizeof(Element)>>;
        using BK = cute::Int<cute::min(K, 32)>;
    };
    template<int N>
    __device__ __forceinline__
    void cpWait() {
        cute::cp_async_wait<N>();
        __syncthreads();
    }
    __device__ __forceinline__
    void cpFence() {
        cute::cp_async_fence();
    }
    __device__ __forceinline__
    constexpr auto idx2Coord(const int& tilesM, const int& tilesN, const int& tileIdx) {
        const auto tileCoord = cute::idx2crd(tileIdx, cute::make_shape(tilesM, tilesN),
            cute::make_stride(tilesN, cute::_1{}));
        return cute::make_coord(cute::get<0>(tileCoord), cute::get<1>(tileCoord), cute::_);
    }
    template<int tRow, int tCol, cublasdx::arrangement ar, typename Element, typename TileCoord>
    __device__ __forceinline__
    constexpr auto get(const Element* __restrict__ const& p, const int& nRow, const int& nCol, const TileCoord& tileCoord) {
        const auto stride = cute::conditional_return<ar == cublasdx::row_major>
        (cute::make_stride(nCol, cute::_1{}), cute::make_stride(cute::_1{}, nRow));
        const auto m = cute::make_tensor(cute::make_gmem_ptr(p),
            cute::make_layout(cute::make_shape(nRow, nCol), stride));
        return cute::local_tile(m, cute::Shape<cute::Int<tRow>, cute::Int<tCol>>{}, tileCoord);
    }
    template<int tRow, int tCol, cublasdx::arrangement ar, typename Element, typename TileCoord>
    __device__ __forceinline__
    constexpr auto getC(Element* __restrict__ const& p, const int& nRow, const int& nCol, const TileCoord& tileCoord) {
        const auto stride = cute::conditional_return<ar == cublasdx::row_major>
        (cute::make_stride(nCol, cute::_1{}), cute::make_stride(cute::_1{}, nRow));
        auto m = cute::make_tensor(cute::make_gmem_ptr(p),
            cute::make_layout(cute::make_shape(nRow, nCol), stride));
        return cute::local_tile(m, cute::Shape<cute::Int<tRow>, cute::Int<tCol>>{}, tileCoord);
    }
    template<int bM, int bN, typename Element, typename TileCoord>
    __device__ __forceinline__
    constexpr auto getBias(const Element* __restrict__ const& bias, const int& M, const int& N, const TileCoord& tileCoord) {
        // broadcast from {1, N} -> {M, N}
        const auto mD = cute::make_tensor(cute::make_gmem_ptr(bias),
            cute::make_layout(cute::make_shape(M, N), cute::Stride<cute::_0, cute::_1>{}));
        return cute::local_tile(mD, cute::Shape<cute::Int<bM>, cute::Int<bN>>{}, cute::select<0, 1>(tileCoord));
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

    template<typename Element, int dim>
    constexpr int ElementWidth = cute::min(dim, MAX_ALIGNMENT / sizeof(Element));
    template<typename Element, int dim>
    constexpr int ElementAlignment = (cutlass::is_pow2<ElementWidth<Element, dim>>::value ?
        ElementWidth<Element, dim> : 1) * sizeof(Element);
    template<cublasdx::arrangement ar, int bM, int bK>
    constexpr int ldA = ar == cublasdx::row_major ? bK : bM;
    template<cublasdx::arrangement br, int bK, int bN>
    constexpr int ldB = br == cublasdx::col_major ? bK : bN;
    template<cublasdx::arrangement cr, int bM, int bN>
    constexpr int ldC = cr == cublasdx::row_major ? bN : bM;
    template<
        int bM, int bN, int bK, // tile shape
        int Arch, // compute capability
        typename Element, // type for A and B
        typename MMA_C,  // compute type
        cublasdx::arrangement ar = cublasdx::row_major,
        cublasdx::arrangement br = cublasdx::col_major,
        cublasdx::arrangement cr = cublasdx::row_major,
        int aAlignment = MAX_ALIGNMENT,
        int bAlignment = MAX_ALIGNMENT,
        int cAlignment = MAX_ALIGNMENT
    >
    constexpr int suggest_thread_count() {
        using GhostBLAS = decltype(
            cublasdx::Size<bM, bN, bK>() +
            cublasdx::Precision<Element, Element, MMA_C>() +
            cublasdx::Type<cublasdx::type::real>() +
            cublasdx::Function<cublasdx::function::MM>() +
            cublasdx::Arrangement<ar, br, cr>() +
            cublasdx::Block() +
            cublasdx::Alignment<aAlignment, bAlignment, cAlignment>() +
            cublasdx::StaticBlockDim() +
            cublasdx::EnableInputStreaming() +
            cublasdx::SM<Arch, Arch >= 900 ? cublasdx::sm_modifier::arch_specific : cublasdx::sm_modifier::generic>());
        return GhostBLAS::max_threads_per_block;
    }
    template<
        int bM, int bN, int bK, // tile shape
        int Arch, // compute capability
        typename Element, // type for A and B
        typename MMA_C,  // compute type
        int threads,
        int pipeStages = 1, // pipeline stages
        cublasdx::arrangement ar = cublasdx::row_major,
        cublasdx::arrangement br = cublasdx::col_major,
        cublasdx::arrangement cr = cublasdx::row_major,
        int aAlignment = MAX_ALIGNMENT,
        int bAlignment = MAX_ALIGNMENT,
        int cAlignment = MAX_ALIGNMENT
    >
    requires(pipeStages > 0 && Arch >= 700)
    struct CollectiveMainloop {
        using BLAS = decltype(
            cublasdx::Size<bM, bN, bK>() +
            cublasdx::Precision<Element, Element, MMA_C>() +
            cublasdx::Type<cublasdx::type::real>() +
            cublasdx::Function<cublasdx::function::MM>() +
            cublasdx::Arrangement<ar, br, cr>() +
            cublasdx::Block() +
            cublasdx::Alignment<aAlignment, bAlignment, cAlignment>() +
            cublasdx::BlockDim<threads>() +
            cublasdx::StaticBlockDim() +
            cublasdx::EnableInputStreaming() +
            cublasdx::SM<Arch, Arch >= 900 ? cublasdx::sm_modifier::arch_specific : cublasdx::sm_modifier::generic>());
        using Threads = cute::C<threads>;
        using TileShape = cute::Shape<cute::Int<bM>, cute::Int<bN>, cute::Int<bK>>;
        using SharedSize = cute::Int<bK * pipeStages * (bM + bN) * sizeof(Element)>;
        using GeneralAlignment = cute::Int<MAX_ALIGNMENT>;
        using CArr = cute::C<cr>;
        using AAlign = cute::C<aAlignment>;
        using BAlign = cute::C<bAlignment>;
        using CAlign = cute::C<cAlignment>;
        using AccumType = MMA_C;
        using PipeStages = cute::C<pipeStages>;

        template<typename Accumulator, typename TileCoord>
        requires(cute::rank_v<TileCoord> == 3)
        __device__ __forceinline__
        void operator()(void* __restrict__ const& workspace,
        const Element* __restrict__ const& a,
        const Element* __restrict__ const& b,
        Accumulator& accumulator,
        const int& M, const int& N, const int& K, const TileCoord& tileCoord) const {
            accumulator.clear();
            const int tilesK = K / bK;
            const auto gA = tile::get<bM, bK, ar>(a, M, K, cute::select<0, 2>(tileCoord)); //  M, K
            const auto gB = tile::get<bK, bN, br>(b, K, N, cute::select<2, 1>(tileCoord)); // K, N
            constexpr auto sASS = cublasdx::cosize(BLAS::suggest_layout_smem_a());
            constexpr auto sBSS = cublasdx::cosize(BLAS::suggest_layout_smem_b());
            auto* __restrict__ sAP = static_cast<Element*>(workspace);
            auto* __restrict__ sBP = sAP + (sASS * pipeStages);
            auto sA = cublasdx::make_tensor(sAP, BLAS::suggest_layout_smem_a());
            auto sB = cublasdx::make_tensor(sBP, BLAS::suggest_layout_smem_b());
            // prime pipeline
            cute::for_each(cute::make_int_sequence<pipeStages>{}, [&](auto stage){
                update_buffer<stage, sASS>(sA, sAP);
                update_buffer<stage, sBSS>(sB, sBP);
                cublasdx::copy<BLAS, aAlignment>(gA(cute::_, cute::_, stage), sA);
                cublasdx::copy<BLAS, bAlignment>(gB(cute::_, cute::_, stage), sB);
                cpFence();
            });
            // mainloop
            for (int kStage = pipeStages; kStage < tilesK; ++kStage) {
                const int ps = kStage % pipeStages;
                update_buffer<sASS>(sA, sAP, ps);
                update_buffer<sBSS>(sB, sBP, ps);
                cpWait<pipeStages - 1>();
                BLAS().execute(sA, sB, accumulator);
                __syncthreads();
                cublasdx::copy<BLAS, aAlignment>(gA(cute::_, cute::_, kStage), sA);
                cublasdx::copy<BLAS, bAlignment>(gB(cute::_, cute::_, kStage), sB);
                cpFence();
            }
            // tail
            cute::for_each(cute::make_int_sequence<pipeStages>{}, [&](auto stage) {
                const int ps = (tilesK + stage) % pipeStages;
                update_buffer<sASS>(sA, sAP, ps);
                update_buffer<sBSS>(sB, sBP, ps);
                cpWait<(pipeStages - 1) - stage>();
                BLAS().execute(sA, sB, accumulator);
            });
        }
    };
}
#endif //FLASHMOE_TILE_CUH