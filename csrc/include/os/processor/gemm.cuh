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

namespace aristos {
    /// Fused, Add, Activate
    template <typename Element, typename ActivationFunction>
    requires(aristos::TensorValueType<Element> && cuda::std::is_invocable_r_v<Element, ActivationFunction, Element>)
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

    template<
        typename ActivationOp,
        typename ElementA,
        typename ElementB = ElementA,
        typename ElementC = ACC::ElementC,
        unsigned int sizeK = ACC::PeakHardware::bKBase::value,
        unsigned int Arch = ACC::PeakHardware::arch::value,
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
        // TODO deprecate the below
        using GEMM = decltype(cublasdx::Size<BLOCK_M, BLOCK_N, sizeK>()
                              + cublasdx::Precision<typename ToCDx<ElementA>::T, typename ToCDx<ElementB>::T, typename ToCDx<ElementC>::T>()
                              + cublasdx::Type<cublasdx::type::real>()
                              + cublasdx::Arrangement<cublasdx::row_major, cublasdx::row_major, cublasdx::row_major>()
                              + cublasdx::Function<cublasdx::function::MM>()
                              + cublasdx::SM<cute::min(Arch, 800)>()
                              + cublasdx::Block()
                              + cublasdx::BlockDim<threads>());
        using MatrixAType = ElementA;
        using MatrixBType = ElementB;
        using MatrixCType = ElementC;
        using MatrixDType = ElementA;
        using BlockTiler = cute::Shape<cute::Int<cublasdx::size_of<GEMM>::m>,
                                        cute::Int<cublasdx::size_of<GEMM>::n>,
                                        cute::Int<cublasdx::size_of<GEMM>::k>>;
        using TilerOut = cute::Shape<cute::Int<cublasdx::size_of<GEMM>::m>, cute::Int<cublasdx::size_of<GEMM>::n>>;
        using Parameters = CollectiveMMAConfig<GEMM, ElementA, ElementB, ElementC, LayoutOptimization::UseSwizzle>;
        using MMA = typename Parameters::mma_t;
        using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
            cuda::std::conditional_t<cublasdx::sm_of<GEMM>::value < 800,
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
