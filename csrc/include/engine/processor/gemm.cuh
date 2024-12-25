//
// Created by oja7 on 11/25/24.
//

#ifndef GEMM_CUH
#define GEMM_CUH

#include <cublasdx.hpp>
#include <cute/tensor.hpp>
#include <cutlass/array.h>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/gemm/collective/collective_mma.hpp>
#include "mmaConfig.cuh"

namespace aristos {
    /// A more apropos name would be "static storage" rather than registers.
    template<class T>
    struct isRegister : cuda::std::false_type {};

    template<class T, int N, int Alignment>
    struct isRegister<cutlass::AlignedArray<T, N, Alignment>> : cuda::std::true_type {};

    template<class T, int N, bool RegisterSized>
    struct isRegister<cutlass::Array<T, N, RegisterSized>> : cuda::std::true_type {};

    template<class Engine, class Layout>
    struct isRegister<cute::Tensor<Engine, Layout>> :
    cuda::std::conditional_t<cute::is_rmem_v<cute::Tensor<Engine, Layout>>,
    cuda::std::true_type, cuda::std::false_type> {};

    template <class T>
    constexpr bool isRegisterV = isRegister<T>::value;

    template<typename Element>
    requires aristos::TensorValueType<Element>
    struct Scale {
        __forceinline__ __device__
        Element operator()(const Element& accumulator, const Element& scale) const {
            return scale * accumulator;
        }
    };

    /// Vector atomic add
    template<unsigned int Arch, typename Element = float>
    requires SupportedArch<Arch> && TensorValueType<Element>
    struct VAA {
        template<class Registers>
        requires isRegisterV<Registers> &&
            cuda::std::is_same_v<typename Registers::value_type, Element>
        __device__ __forceinline__
        void operator()(Element* __restrict__ const& gS, Registers registers) const {
            // Float is the "safe accumulator type"
            // We acknowledge this by converting registers to float before accumulating.
            auto regLoadOp = cutlass::NumericConverter<float, typename Registers::value_type>{};
            #pragma unroll
            for (uint i = 0; i < registers.size(); ++i) {
                atomicAdd(gS + i, regLoadOp(registers(i)));
            }
        }
    };

    // specialization for half-precision
    template<unsigned int Arch>
    struct VAA<Arch, cute::half_t> {
        template<class Registers>
        requires isRegisterV<Registers> &&
            cuda::std::is_same_v<typename Registers::value_type, cute::half_t>
        __device__ __forceinline__
        void operator()(cute::half_t* __restrict__ const& gS, Registers registers) const {
            using vType = cuda::std::conditional_t<registers.size() % 2 == 0, __half2, __half>;
            constexpr auto len = registers.size() / (sizeof(vType) / sizeof(__half));
            auto* __restrict__ gSv = CAST_TO(vType, gS);
            const auto* __restrict__ vRegs = CAST_TO(vType, registers.data());
            #pragma unroll
            for (uint i = 0; i < len; ++i) {
                atomicAdd(gSv + i, vRegs[i]);
            }
        }
    };

    // specialization for bfloat16
    template<unsigned int Arch> requires(Arch >= 800)
    struct VAA<Arch, cute::bfloat16_t> {
        template<class Registers>
        requires isRegisterV<Registers> &&
            cuda::std::is_same_v<typename Registers::value_type, cute::bfloat16_t>
        __device__ __forceinline__
        void operator()(cute::bfloat16_t* __restrict__ const& gS, Registers registers) const {
            using vType = cuda::std::conditional_t<registers.size() % 2 == 0, __nv_bfloat162, __nv_bfloat16>;
            constexpr auto len = registers.size() / (sizeof(vType) / sizeof(__half));
            auto* __restrict__ gSv = CAST_TO(vType, gS);
            const auto* __restrict__ vRegs = CAST_TO(vType, registers.data());
            #pragma unroll
            for (uint i = 0; i < len; ++i) {
                atomicAdd(gSv + i, vRegs[i]);
            }
        }
    };

    // specialization for float on Hopper
    template<>
    struct VAA<900, float> {
        template<class Registers>
        requires isRegisterV<Registers> &&
            cuda::std::is_same_v<typename Registers::value_type, float>
        __device__ __forceinline__
        void operator()(float* __restrict__ const& gS, Registers registers) const {
            static_assert(registers.size() % 2 == 0, "Register tensor does not vectorize");
            using vType = cuda::std::conditional_t<registers.size() % 4 == 0, float4,
                cuda::std::conditional_t<registers.size() % 2 == 0, float2, float>>;
            constexpr auto len = registers.size() / (sizeof(vType) / sizeof(float));
            auto* __restrict__ gSv = CAST_TO(vType, gS);
            const auto* __restrict__ vRegs = CAST_TO(vType, registers.data());
            #pragma unroll
            for (uint i = 0; i < len; ++i) {
                atomicAdd(gSv + i, vRegs[i]);
            }
        }
    };

    /// Fused, Add, Activate
    template <typename Element, typename ActivationFunction>
    requires(aristos::TensorValueType<Element> && cuda::std::is_invocable_r_v<Element, ActivationFunction, Element>)
    struct FAA {
        // fp8
        static_assert(sizeof(Element) == 2);
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
        unsigned int Arch,
        typename ElementA,
        typename ElementB,
        typename ElementC = float,
        typename ActivationOp = cute::identity>
    struct BlockMM {
        static_assert(BLOCK_M == THREADS);
        static_assert(BLOCK_M == 128);
        static_assert(BLOCK_N == 64, "64 is a very good value for N, change it back!");
        using GEMM = decltype(cublasdx::Size<BLOCK_M, BLOCK_N, BLOCK_K_FULL>()
                              + cublasdx::Precision<toCDX<ElementA>, toCDX<ElementB>, toCDX<ElementC>>()
                              + cublasdx::Type<cublasdx::type::real>()
                              + cublasdx::Arrangement<cublasdx::row_major, cublasdx::row_major, cublasdx::row_major>()
                              + cublasdx::Function<cublasdx::function::MM>()
                              + cublasdx::SM<Arch>()
                              + cublasdx::Block()
                              + cublasdx::BlockDim<THREADS>());
        using MatrixAType = ElementA;
        using MatrixBType = ElementB;
        using MatrixCType = ElementC;
        using MatrixDType = ElementA;
        using BlockTiler = cute::Shape<cute::Int<cublasdx::size_of<GEMM>::m>,
                                        cute::Int<cublasdx::size_of<GEMM>::n>,
                                        cute::Int<cublasdx::size_of<GEMM>::k>>;
        using TilerOut = cute::Shape<cute::Int<cublasdx::size_of<GEMM>::m>, cute::Int<cublasdx::size_of<GEMM>::n>>;
        using Parameters = CollectiveMMAConfig<GEMM, LayoutOptimization::UseSwizzle>;
        using MMA = typename Parameters::mma_t;
        using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
            typename Parameters::dispatch,
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
        // TODO CollectiveMMA support for Hopper
    };
}
#endif //GEMM_CUH
