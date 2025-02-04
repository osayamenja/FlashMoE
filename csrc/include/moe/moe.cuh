/******************************************************************************
 * Copyright (c) 2024, Osayamen Jonathan Aimuyo.
 ******************************************************************************/

#ifndef ARISTOS_MOE_CUH
#define ARISTOS_MOE_CUH

#include "gate.cuh"
#include "../os/os.cuh"

namespace aristos::moe{
    template<
        unsigned int Arch,
        GateReductionLevel g = GateReductionLevel::singleBlock,
        DropTokens d = DropTokens::yes,
        CombineMode c = CombineMode::single,
        typename ActivationOp = cute::identity,
        typename ActivationOpX = cute::identity,
        typename ElementC = float,
        typename ElementA = cute::half_t,
        typename ElementB = ElementA,
        typename ElementD = ElementA,
        typename Activations,
        typename MoEOut,
        typename ExpertsUp,
        typename ExpertsDown,
        typename BiasUp,
        typename BiasDown,
        typename Gates,
        typename GateOut>
    requires(aristos::SupportedArch<Arch> && aristos::Matrix<Activations> &&
        aristos::Matrix<MoEOut> &&
        aristos::Tensor<ExpertsUp> && aristos::Tensor<ExpertsDown> &&
        aristos::Tensor<BiasUp> && aristos::Tensor<BiasDown> &&
        aristos::Matrix<Gates> && aristos::Matrix<GateOut>)
    __global__ __maxnreg__(128) void forward(
        Activations const __grid_constant__ activations,
        ExpertsUp const __grid_constant__ expertsUp,
        ExpertsDown const __grid_constant__ expertsDown,
        BiasUp const __grid_constant__ biasUp,
        BiasDown const __grid_constant__ biasDown,
        Gates const __grid_constant__ gateWeights,
        GateOut const __grid_constant__ gateOutput) {
        constexpr auto blocks = Hardware<Arch>::blocks;
        constexpr auto processors = blocks - 1;
        __shared__ __align__(16) cuda::std::byte workspace[SHARED_SIZE];
        // wipe gTQHeads here and read the sequence bit, before the grid-wide barrier
        const auto gtQCl = bookkeeping.gtQCl;
        const auto sb = seqBit;
        auto* __restrict__ gtQHeads = bookkeeping.tQH();
        for (uint i = THREADS * blockIdx.x + threadIdx.x; i < gtQCl; i += blocks * THREADS) {
            gtQHeads[i] = 0U;
        }

        gate::forward<Arch, blocks, g, ElementC>(activations,
            gateWeights, gateOutput, CAST_TO(ElementC, workspace));
        if (blockIdx.x + 1 < blocks) {
            packet::encode<processors, d>(activations, gateOutput, workspace);
            processor::start<
                processors,
                Arch,
                c,
                ElementA,
                ElementB,
                ElementC,
                ElementD,
                ActivationOp,
                ActivationOpX>(CAST_TO(ElementD, workspace, rSb));
        }
        else {
            os::start<processors, d>();
        }
    }

    template<
        unsigned int Arch
    >
    requires(aristos::SupportedArch<Arch>)
    __global__ __maxnreg__(128) void backward(){

    }
}
#endif //ARISTOS_MOE_CUH
