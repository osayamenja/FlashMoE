//
// Created by oja7 on 11/25/24.
//

#ifndef GEMM_CUH
#define GEMM_CUH

#include <cublasdx.hpp>
#include <cutlass/gemm/collective/collective_mma.hpp>
#include "mmaConfig.cuh"

namespace aristos {
    template<
        unsigned int Arch,
        typename ElementA,
        typename ElementB,
        typename ElementC = float,
        typename ActivationOp = cute::identity>
    struct BlockMM {
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

        using EpilogueOp = ActivationOp;
        // TODO CollectiveMMA support for Hopper
    };
}
#endif //GEMM_CUH
