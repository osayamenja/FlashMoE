//
// Created by osayamen on 12/25/25.
//

#ifndef FLASHMOE_TILE_CUH
#define FLASHMOE_TILE_CUH
#include <cublasdx.hpp>

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
    struct Converter<float, __half> {
        __device__ auto operator()(const __half& x) const {
            return  __half2float(x);
        }
    };
    template<>
    struct Converter<__nv_bfloat16, float> {
        __device__ auto operator()(const float& x) const {
            return  __float2bfloat16(x);
        }
    };
    template<>
    struct Converter<float, __nv_bfloat16> {
        __device__ auto operator()(const __nv_bfloat16& x) const {
            return  __bfloat162float(x);
        }
    };
    template<>
    struct Converter<float2, __half2> {
        __device__ auto operator()(const __half2& x) const {
            return  __half22float2(x);
        }
    };
    template<>
    struct Converter<__half2, float2> {
        __device__ auto operator()(const float2& x) const {
            return  __float22half2_rn(x);
        }
    };
    template<>
    struct Converter<float2, __nv_bfloat162> {
        __device__ auto operator()(const __nv_bfloat162& x) const {
            return  __bfloat1622float2(x);
        }
    };
    template<>
    struct Converter<__nv_bfloat162, float2> {
        __device__ auto operator()(const float2& x) const {
            return  __float22bfloat162_rn(x);
        }
    };

    template<typename V>
        concept TensorValueType = cuda::std::is_same_v<V, __half> ||
            cuda::std::is_same_v<V, __nv_bfloat16> ||
            cuda::std::is_same_v<V, cublasdx::tfloat32_t> ||
            cuda::std::is_same_v<V, float> ||
            cuda::std::is_same_v<V, cublasdx::float_e4m3_t> ||
            cuda::std::is_same_v<V, cublasdx::float_e5m2_t>;

    template <class T>
    struct isTensor : cuda::std::false_type {};
    template <class Engine, class Layout>
    requires(TensorValueType<typename Engine::value_type>)
    struct isTensor<cute::Tensor<Engine,Layout>> : cuda::std::true_type {};
    template <class Engine, class Layout>
    requires(TensorValueType<typename Engine::value_type>)
    struct isTensor<const cute::Tensor<Engine,Layout>> : cuda::std::true_type {};
}
namespace flashmoe::tile
{
    constexpr int MAX_ALIGN = 16;
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
        int aAlignment = MAX_ALIGN,
        int bAlignment = MAX_ALIGN,
        int cAlignment = MAX_ALIGN
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
        int aAlignment = MAX_ALIGN,
        int bAlignment = MAX_ALIGN,
        int cAlignment = MAX_ALIGN
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
        using GeneralAlignment = cute::Int<MAX_ALIGN>;
        using CArr = cute::C<cr>;
        using AAlign = cute::C<aAlignment>;
        using BAlign = cute::C<bAlignment>;
        using CAlign = cute::C<cAlignment>;
        using AccumType = MMA_C;
        using PipeStages = cute::C<pipeStages>;

        template<typename Accumulator, typename TileCoord>
        requires(cute::rank_v<TileCoord> == 3 && cublasdx::is_blas_execution_v<BLAS>)
        __device__ __forceinline__
        void operator()(void* __restrict__ const& workspace,
        const Element* __restrict__ const& a,
        const Element* __restrict__ const& b,
        Accumulator& accumulator,
        const int& M, const int& N, const int& K, const TileCoord& tileCoord) const {
            // assert(__isShared(workspace));
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