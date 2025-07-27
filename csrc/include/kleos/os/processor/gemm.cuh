/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Kleos Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 */
//
// Created by oja7 on 11/25/24.
//

#ifndef GEMM_CUH
#define GEMM_CUH

#include <cublasdx.hpp>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/gemm/collective/collective_mma.hpp>

#include "mmaConfig.cuh"
#include "../../arch.cuh"

namespace kleos {
    /// Fused, Add, Activate
    template <typename Element, typename ActivationFunction>
    requires(kleos::TensorValueType<Element> && cuda::std::is_invocable_r_v<Element, ActivationFunction, Element>)
    struct FAA {
        __forceinline__ __device__
        Element operator()(const Element& accumulator, const Element& term) const {
            constexpr ActivationFunction op{};
            return op(accumulator + term);
        }
    };

    // specialization for half-precision and relu
    template<>
    struct FAA<cute::half_t, cutlass::epilogue::thread::ReLU<cute::half_t>> {
        __forceinline__ __device__
        cute::half_t operator()(const cute::half_t& accumulator, const cute::half_t& term) const {
            return cute::half_t(__hfma_relu(__half(1.0f),accumulator.to_half(), term.to_half()));
        }
    };

    // specialization for bfloat16 and relu
    template<>
    struct FAA<cute::bfloat16_t, cutlass::epilogue::thread::ReLU<cute::bfloat16_t>> {
        __forceinline__ __device__
        cute::bfloat16_t operator()(const cute::bfloat16_t& accumulator, const cute::bfloat16_t& term) const {
            return cute::bfloat16_t(__hfma_relu(__nv_bfloat16(1.0f),
                accumulator.to_nv_bfloat16(), term.to_nv_bfloat16()));
        }
    };

    template<typename F>
    struct isFAA : cuda::std::false_type {};

    template<typename Element, typename ActivationFunction>
    struct isFAA<FAA<Element, ActivationFunction>> : cuda::std::true_type {};

    template<int N>
    __device__ __forceinline__
    void cpWait() {
        cute::cp_async_wait<N>();
        __syncthreads();
    }
    // GEMM Mainloop like that of CUTLASS
    template<
        typename BLAS,
        int N,
        int K,
        int pipeStages,
        typename Accumulator
    >
    requires(cutlass::is_pow2<pipeStages>::value)
    struct GM {
        __forceinline__ __device__
        void operator()(void* __restrict__ const& workspace,
            Accumulator& accumulator,
            const typename BLAS::a_value_type* __restrict__ const& a,
            const typename BLAS::b_value_type* __restrict__ const& b,
            typename BLAS::a_value_type* __restrict__ const& c,
            const int& M, const int& tileIdx) const {
            using BM = cute::Int<cublasdx::size_of<BLAS>::m>;
            using BN = cute::Int<cublasdx::size_of<BLAS>::n>;
            using BK = cute::Int<cublasdx::size_of<BLAS>::k>;
            using BT = cute::Shape<BM, BN, BK>;

            const auto tilesM = M / BM{};
            using KT = cute::Int<K>;
            using NT = cute::Int<N>;
            using tilesN = cute::Int<(N / BN{})>;
            using tilesK = cute::Int<(K / BK{})>;

            const auto tileCoord = cute::idx2crd(tileIdx, cute::Shape(tilesM, tilesN{}),
                cute::Stride<cute::Int<tilesN{}>, cute::_1>{});
            const auto ctaCoord = cute::make_coord(cute::get<0>(tileCoord), cute::get<1>(tileCoord), cute::_);
            const auto strideA = cute::conditional_return<cublasdx::arrangement_of_v_a<BLAS> == cublasdx::row_major>
            (cute::Stride<KT, cute::_1>{}, cute::make_stride(cute::_1{}, M));
            const auto mA = cute::make_tensor(cute::make_gmem_ptr(a),
                cute::make_layout(cute::make_shape(M, KT{}), strideA));
            using StrideB = cuda::std::conditional_t<cublasdx::arrangement_of_v_b<BLAS> == cublasdx::row_major,
            cute::Stride<NT, cute::_1>, cute::Stride<cute::_1, KT>>;
            using ShapeB = cute::Shape<KT, NT>;
            const auto mB = cute::make_tensor(cute::make_gmem_ptr(a),
                cute::Layout<ShapeB, StrideB>{});
            const auto strideC = cute::conditional_return<cublasdx::arrangement_of_v_c<BLAS> == cublasdx::row_major>
            (cute::Stride<NT, cute::_1>{}, cute::make_stride(cute::_1{}, M));
            const auto mC = make_tensor(cute::make_gmem_ptr(c),
            make_layout(cute::make_shape(M, N), strideC));

            const auto gA = cute::local_tile(mA, BT{}, ctaCoord, cute::Step<cute::_1, cute::X,cute::_1>{});
            const auto gB = cute::local_tile(mB, BT{}, ctaCoord, cute::Step<cute::_1, cute::X,cute::_1>{});
            const auto gC = cute::local_tile(mC, BT{}, ctaCoord, cute::Step<cute::_1, cute::X,cute::_1>{});

            // shared layouts
            constexpr auto sALay = cute::tile_to_shape(BLAS::suggest_layout_smem_a().layout,
                cute::Shape<BM, BK, cute::Int<pipeStages>>{});
            constexpr auto sBLay = cute::tile_to_shape(BLAS::suggest_layout_smem_b().layout,
                cute::Shape<BK, BN, cute::Int<pipeStages>>{});
            const auto [sA, sB] = cublasdx::shared_memory::slice<typename BLAS::a_value_type, typename BLAS::b_value_type>(
                workspace, cublasdx::alignment_of_v_a<BLAS>, sALay, cublasdx::alignment_of_v_b<BLAS>, sBLay);
            cute::for_each(cute::make_int_sequence<pipeStages>{}, [&](auto stage) {
                cublasdx::copy<BLAS, cublasdx::alignment_of_v_a<BLAS>>(gA(cute::_, cute::_, stage), sA(cute::_, cute::_, stage));
                cublasdx::copy<BLAS, cublasdx::alignment_of_v_b<BLAS>>(gB(cute::_, cute::_, stage), sB(cute::_, cute::_, stage));
                cute::cp_async_fence();
            });

            cute::for_each(cute::make_int_range<pipeStages, tilesK{}>{}, [&](auto kStage) {
                constexpr auto ps = cute::Int<kStage % pipeStages>{};
                cpWait<pipeStages - 1>();
                BLAS().execute(sA(cute::_, cute::_, ps), sB(cute::_, cute::_, ps), accumulator);
                __syncthreads();
                cublasdx::copy<BLAS, cublasdx::alignment_of_v_a<BLAS>>(gA(cute::_, cute::_, kStage), sA(cute::_, cute::_, ps));
                cublasdx::copy<BLAS, cublasdx::alignment_of_v_b<BLAS>>(gB(cute::_, cute::_, kStage), sB(cute::_, cute::_, ps));
                cute::cp_async_fence();
            });

            cute::for_each(cute::make_int_rsequence<pipeStages>{}, [&](auto rStage) {
                constexpr auto ps = cute::Int<((tilesK{} - (rStage + 1)) % pipeStages)>{};
                cpWait<rStage>();
                BLAS().execute(sA(cute::_, cute::_, ps), sB(cute::_, cute::_, ps), accumulator);
            });
        }
    };
    template<
        typename ActivationOp,
        typename ElementA,
        typename ElementB = ElementA,
        typename ElementC = ACC::ElementC,
        unsigned int sizeK = ACC::PeakHardware::bKBase::value,
        unsigned int Arch = cute::min(ACC::PeakHardware::arch::value,800), // clamp at 800 for now
        unsigned int threads = ACC::PeakHardware::OS::threads::value,
        unsigned int pipeStages = ACC::PeakHardware::pipeStages::value
    >
    requires(cuda::std::is_same_v<ElementC, ACC::ElementC> ||
        (cuda::std::is_same_v<ElementC, cute::half_t> &&
            cuda::std::is_same_v<ElementA, cute::half_t> &&
            cuda::std::is_same_v<ElementB, cute::half_t>))
    struct BlockMM {
        // will clamp at Ampere for now, until we implement Hopper specific GEMM
        static_assert(BLOCK_M == THREADS && BLOCK_M == threads);
        static_assert(BLOCK_M == 128);
        static_assert(BLOCK_N == 64, "64 is a very good value for N, change it back!");
        using Threads = cute::C<threads>;
        using MatrixAType = ElementA;
        using MatrixBType = ElementB;
        using MatrixCType = ElementC;
        using MatrixDType = ElementA;
        using BlockTiler = cute::Shape<cute::Int<BLOCK_M>, cute::Int<BLOCK_N>, cute::Int<sizeK>>;
        using TilerOut = cute::Shape<cute::Int<BLOCK_M>, cute::Int<BLOCK_N>>;
        using Parameters = CollectiveMMAConfig<BLOCK_M, BLOCK_N, sizeK, Arch, ElementA, ElementB, ElementC,
            LayoutOptimization::UseSwizzle>;
        using MMA = typename Parameters::mma_t;
        using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
            cuda::std::conditional_t<Arch < 800,
                    cutlass::gemm::MainloopSm70TwoStageUnpredicated,
                        cutlass::gemm::MainloopSm80CpAsyncUnpredicated<pipeStages>>,
            BlockTiler,
            ElementA,
            cute::Underscore,
            ElementB,
            cute::Underscore,
            typename Parameters::mma_t,
            typename Parameters::gCopyA,
            typename Parameters::sLayA,
            typename Parameters::sCopyA,
            cute::identity,
            typename Parameters::gCopyB,
            typename Parameters::sLayB,
            typename Parameters::sCopyB,
            cute::identity
        >;
        using FusedEpilogue = FAA<ElementC, ActivationOp>;
    };
}
#endif //GEMM_CUH
