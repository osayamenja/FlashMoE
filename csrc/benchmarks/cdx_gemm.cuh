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
    template<int N>
    __device__ __forceinline__
    void cpWait() {
        cute::cp_async_wait<N>();
        __syncthreads();
    }
    template<typename BLAS, int pipeStages, typename Element, typename Accumulator>
    requires(cublasdx::is_blas_execution_v<BLAS>)
    __device__ __forceinline__
    void mainLoop(void* __restrict__ const& workspace,
        const Element* __restrict__ const& a,
        const Element* __restrict__ const& b,
        Accumulator& accumulator,
        const int& M, const int& N, const int& K, const int& tileIdx) {
        cublasdx::clear(accumulator);
        using BM = cute::Int<cublasdx::size_of<BLAS>::m>;
        using BN = cute::Int<cublasdx::size_of<BLAS>::n>;
        using BK = cute::Int<cublasdx::size_of<BLAS>::k>;

        const int tilesM = M / BM{};
        const int tilesN = N / BN{};
        const int tilesK = K / BK{};

        const auto tileCoord = cute::idx2crd(tileIdx, cute::make_shape(tilesM, tilesN),
            cute::make_stride(tilesN, cute::_1{}));
        const auto ctaCoord = cute::make_coord(cute::get<0>(tileCoord),
            cute::get<1>(tileCoord), cute::_);
        const auto strideA = cute::conditional_return<cublasdx::arrangement_of_v_a<BLAS> == cublasdx::row_major>
        (cute::make_stride(K, cute::_1{}), cute::make_stride(cute::_1{}, M));
        const auto mA = cute::make_tensor(cute::make_gmem_ptr(a),
            cute::make_layout(cute::make_shape(M, K), strideA));
        const auto strideB = cute::conditional_return<cublasdx::arrangement_of_v_b<BLAS> == cublasdx::row_major>
        (cute::make_stride(N, cute::_1{}), cute::make_stride(cute::_1{}, K));
        const auto mB = cute::make_tensor(cute::make_gmem_ptr(b),
            cute::make_layout(cute::make_shape(K, N), strideB));

        const auto gA = cute::local_tile(mA, cute::Shape<BM, BK>{}, cute::select<0, 2>(ctaCoord));
        const auto gB = cute::local_tile(mB, cute::Shape<BK, BN>{}, cute::select<2, 1>(ctaCoord));
        // shared layouts
        constexpr auto sALay = cute::tile_to_shape(BLAS::suggest_layout_smem_a().layout,
            cute::Shape<BM, BK, cute::Int<pipeStages>>{});
        constexpr auto sBLay = cute::tile_to_shape(BLAS::suggest_layout_smem_b().layout,
            cute::Shape<BK, BN, cute::Int<pipeStages>>{});
        const auto [sA, sB] = cublasdx::shared_memory::slice<Element, Element>(
            workspace, cublasdx::alignment_of_v_a<BLAS>, sALay, cublasdx::alignment_of_v_b<BLAS>, sBLay);
        cuda::static_for<pipeStages>([&](auto stage){
            cublasdx::copy<BLAS, cublasdx::alignment_of_v_a<BLAS>>(gA(cute::_, cute::_, stage), sA(cute::_, cute::_, stage));
            cublasdx::copy<BLAS, cublasdx::alignment_of_v_b<BLAS>>(gB(cute::_, cute::_, stage), sB(cute::_, cute::_, stage));
            cute::cp_async_fence();
        });
        #pragma unroll 1
        for (int kStage = pipeStages; kStage < tilesK; ++kStage) {
            const int ps = kStage % pipeStages;
            cpWait<pipeStages - 1>();
            BLAS().execute(sA(cute::_, cute::_, ps), sB(cute::_, cute::_, ps), accumulator);
            __syncthreads();
            cublasdx::copy<BLAS, cublasdx::alignment_of_v_a<BLAS>>(gA(cute::_, cute::_, kStage), sA(cute::_, cute::_, ps));
            cublasdx::copy<BLAS, cublasdx::alignment_of_v_b<BLAS>>(gB(cute::_, cute::_, kStage), sB(cute::_, cute::_, ps));
            cute::cp_async_fence();
        }
        cuda::static_for<pipeStages>([&](auto stage) {
            const int ps = (tilesK + stage) % pipeStages;
            cpWait<pipeStages - 1 - stage>();
            BLAS().execute(sA(cute::_, cute::_, ps), sB(cute::_, cute::_, ps), accumulator);
        });
    }

    template <typename T>
    struct VectorType {
    };

    struct __align__(16) fp32x4 {
        float2 x;
        float2 y;
        template<typename Activation>
        requires(cuda::std::is_invocable_r_v<float, Activation, float>)
        __device__ __forceinline__
        void faa(const fp32x4& other, const Activation& actOp) {
            const auto vx = __fadd2_rn(x, other.x);
            x = float2{actOp(vx.x), actOp(vx.y)};
            const auto vy = __fadd2_rn(y, other.y);
            y = float2{actOp(vy.x), actOp(vy.y)};
        }
    };
    template <>
    struct VectorType<float> {
        using Type = fp32x4;
    };

    struct __align__(16) tf32x4 {
        cublasdx::tfloat32_t x;
        cublasdx::tfloat32_t y;
        cublasdx::tfloat32_t z;
        cublasdx::tfloat32_t w;
        template<typename Activation>
        requires(cuda::std::is_invocable_r_v<cublasdx::tfloat32_t, Activation, cublasdx::tfloat32_t>)
        __device__ __forceinline__
        void faa(const tf32x4& other, const Activation& actOp) {
            x = actOp(x + other.x);
            y = actOp(y + other.y);
            z = actOp(z + other.z);
            w = actOp(w + other.w);
        }
    };
    template <>
    struct VectorType<cublasdx::tfloat32_t> {
        using Type = float4;
    };
    struct __align__(16) halfx8 {
        __half2 x;
        __half2 y;
        __half2 z;
        __half2 w;
        template<typename Activation>
        requires(cuda::std::is_invocable_r_v<__half, Activation, __half>)
        __device__ __forceinline__
        void faa(const halfx8& other, const Activation& actOp) {
            const auto vx = __hadd2(x, other.x);
            x = __half2{actOp(vx.x), actOp(vx.y)};
            const auto vy = __hadd2(y, other.y);
            y = __half2{actOp(vy.x), actOp(vy.y)};
            const auto vz = __hadd2(z, other.z);
            z = __half2{actOp(vz.x), actOp(vz.y)};
            const auto vw = __hadd2(w, other.w);
            w = __half2{actOp(vw.x), actOp(vw.y)};
        }
    };
    template <>
    struct VectorType<__half> {
        using Type = halfx8;
    };
    struct __align__(16) bfx8 {
        __nv_bfloat162 x;
        __nv_bfloat162 y;
        __nv_bfloat162 z;
        __nv_bfloat162 w;
        template<typename Activation>
        requires(cuda::std::is_invocable_r_v<__nv_bfloat16, Activation, __nv_bfloat16>)
        __device__ __forceinline__
        void faa(const bfx8& other, const Activation& actOp) {
            const auto vx = __hadd2(x, other.x);
            x = __half2{actOp(vx.x), actOp(vx.y)};
            const auto vy = __hadd2(y, other.y);
            y = __half2{actOp(vy.x), actOp(vy.y)};
            const auto vz = __hadd2(z, other.z);
            z = __half2{actOp(vz.x), actOp(vz.y)};
            const auto vw = __hadd2(w, other.w);
            w = __half2{actOp(vw.x), actOp(vw.y)};
        }
    };
    template <>
    struct VectorType<__nv_bfloat16> {
        using Type = bfx8;
    };

    template<typename BLAS, typename Activation, int pipeStages, int threads, typename ElementC, typename Element>
    requires(cublasdx::is_blas_execution_v<BLAS>)
    __device__ __forceinline__
    void gemmMainloop(void* __restrict__ const& workspace,
        const Element* __restrict__ const& a,
        const Element* __restrict__ const& b,
        void* __restrict__ const& c,
        const void* __restrict__ const& bias,
        const int& M, const int& N, const int& K, const int& tileIdx) {
        // MMA Accumulation Type
        using MAT = cublasdx::precision_of_c_t<BLAS>;
        const auto partitioner = BLAS::suggest_partitioner();
        auto accumulator = partitioner.make_accumulator_fragment();
        // prefetch bias
        using VectorElement = VectorType<ElementC>;
        const auto* __restrict__ vBias = static_cast<const VectorElement*>(bias);
        constexpr int vF = sizeof(VectorElement) / sizeof(ElementC);
        using BM = cute::Int<cublasdx::size_of<BLAS>::m>;
        using BN = cute::Int<cublasdx::size_of<BLAS>::n>;
        const int tilesM = M / BM{};
        constexpr int bN = cublasdx::size_of<BLAS>::n;
        const int tilesN = N / bN;
        constexpr int vbN = bN / vF;
        using VBN = cute::Int<vbN>;
        constexpr auto trips = cute::ceil_div(vbN, threads);
        constexpr auto elems = cute::min(vbN, threads);
        VectorElement biasCache[trips];
        const int biasOffset = (tileIdx % tilesN) * vbN;
        const auto* __restrict__ bP = vBias + biasOffset;
        if constexpr (threads >= vbN) {
            static_assert(threads % vbN == 0);
            biasCache[0] = bP[threadIdx.x % vbN];
        }
        else {
            static_assert(vbN % threads == 0);
            #pragma unroll
            for (int i = 0; i < trips; ++i) {
                biasCache[i] = bP[threadIdx.x + i * vbN];
            }
        }
        mainLoop<BLAS, pipeStages, threads>(workspace, a, b, accumulator, M, N, K, tileIdx);
        // in-place conversion from MMA accumulator type (typically float) to final value type
        static_assert(cublasdx::size(accumulator) == bN);
        constexpr auto gCStoreOp = cutlass::NumericConverter<ElementC, MAT>{};
        MAT rC[bN];
        cuda::static_for<bN>([&](auto i) {
            rC[i] = gCStoreOp(accumulator(i));
        });
        auto sC = cute::make_tensor(cute::make_smem_ptr(static_cast<ElementC*>(workspace)),
                cute::Layout<cute::Shape<BM, BN>,
                cuda::std::conditional_t<cublasdx::arrangement_of_v_c<BLAS> == cublasdx::row_major,
                    cute::Stride<BN, cute::_1>, cute::Stride<cute::_1, BM>>>{});
        auto vsC = cute::make_tensor(cute::make_smem_ptr(static_cast<ElementC*>(workspace)),
                cute::Layout<cute::Shape<BM, VBN>,
                cuda::std::conditional_t<cublasdx::arrangement_of_v_c<BLAS> == cublasdx::row_major,
                    cute::Stride<cute::Int<VBN>, cute::_1>, cute::Stride<cute::_1, BM>>>{});
        // rmem -> smem
        cublasdx::copy_fragment<cublasdx::alignment_of_v_c<BLAS>>(rC, sC, partitioner);
        __syncthreads();
        const auto rIdx = threadIdx.x / vbN * vbN;
        const auto cIdx = threadIdx.x % vbN;
        // vectorized smem -> rmem
        // striped layout
        VectorElement vrC[vbN];
        cuda::static_for<trips>([&](auto i) {
            cuda::static_for<elems>([&](auto j) {
                vrC[j + i * elems] = vsC(rIdx + j, cIdx + i * elems);
            });
        });
        // apply epilogue -> Activation(fragment_c + bias)
        cuda::static_for<trips>([&](auto i) {
           cuda::static_for<elems>([&](auto j) {
               // fused add and activation
               vrC[j + i * elems].faa(biasCache[i], Activation{});
           });
        });

        // vectorized rmem -> gmem
        const int vN = N / vF;
        const auto strideC = cute::conditional_return<cublasdx::arrangement_of_v_c<BLAS> == cublasdx::row_major>
        (cute::make_stride(vN, cute::_1{}), cute::make_stride(cute::_1{}, M));
        const auto mC = cute::make_tensor(cute::make_gmem_ptr(c),
            cute::make_layout(cute::make_shape(M, vN), strideC));
        const auto tileCoord = cute::idx2crd(tileIdx, cute::Shape(tilesM, tilesN),
                    cute::Stride(tilesM, tilesN));
        const auto ctaCoord = cute::make_coord(cute::get<0>(tileCoord), cute::get<1>(tileCoord), cute::_);
        const auto gC = cute::local_tile(mC, cute::Shape<BM, VBN>{}, cute::select<0, 1>(ctaCoord));
        cute::for_each(cute::make_int_sequence<trips>{}, [&](auto i) {
            cute::for_each(cute::make_int_sequence<elems>{}, [&](auto j) {
                gC(rIdx + j, cIdx + i * elems) = vrC[j + i * elems];
            });
        });
    }

    #define SC(T, v) static_cast<T>(v)
    template<int bM, int bN, int bK, int pipeStages, int threads,
    typename Activation, typename MMA_C,
    typename Element, typename ElementC>
    __global__ void gk(const Element* __restrict__ a, const Element* __restrict__ b,
        ElementC* __restrict__ c, const ElementC* __restrict__ bias,
        const __grid_constant__ int M, const __grid_constant__ int N, const int __grid_constant__ K) {
        static_assert(threads == bM);
        static_assert(bM % bN == 0);
        const int nTiles = (M / bM) * (N / bN);
        constexpr auto sharedSize = cute::max(bK * pipeStages * (bM + bN) * sizeof(ElementC),
            bM * bN * sizeof(ElementC));
        __shared__ __align__(16) cuda::std::byte workspace[sharedSize];
        using BLAS = decltype(
            cublasdx::Size<128, 128, 32>() +
            cublasdx::Precision<Element, Element, MMA_C>() +
            cublasdx::Type<cublasdx::type::real>() +
            cublasdx::Function<cublasdx::function::MM>() +
            cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major, cublasdx::row_major>() +
            cublasdx::Block() +
            cublasdx::BlockDim<threads>() +
            cublasdx::MaxAlignment() +
            cublasdx::experimental::StaticBlockDim() +
            cublasdx::SM<FLASHMOE_ARCH>());
        for (int tileIdx = SC(int, blockIdx.x); tileIdx < nTiles; tileIdx += SC(int, gridDim.x)) {
            gemmMainloop<BLAS, Activation, pipeStages, threads, ElementC>(workspace, a, b, c, bias, M, N, K, tileIdx);
        }
    }
}
#endif //FLASHMOE_CDX_GEMM_CUH