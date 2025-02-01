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
        unsigned int blocks,
        GateReductionLevel g = GateReductionLevel::singleBlock,
        typename ActivationOp = cute::identity,
        typename ActivationOpX = cute::identity,
        typename ElementC = float,
        typename ElementA = cute::half_t,
        typename ElementB = ElementA,
        typename ElementD = ElementA,
        typename Activations,
        typename Experts,
        typename Bias,
        typename Gates,
        typename MoEOut,
        typename GateOut>
    requires(aristos::Matrix<Activations> && aristos::Matrix<Experts> &&
        aristos::Matrix<Bias> && aristos::Matrix<Gates> && aristos::Matrix<MoEOut>
        && aristos::Matrix<GateOut>)
    __global__ __maxnreg__(128) void forward(
        Activations const __grid_constant__ activations,
        Experts const __grid_constant__ expertsWeights,
        Bias const __grid_constant__ bias,
        Gates const __grid_constant__ gateWeights,
        GateOut const __grid_constant__ gateOutput) {

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
            packet::encode<blocks>(activations, gateOutput, workspace);
            processor::start<
                blocks,
                Arch,
                ElementA,
                ElementB,
                ElementC,
                ElementD,
                ActivationOp,
                ActivationOpX>(CAST_TO(ElementD, workspace, rSb));
        }
        else {
            os::start<blocks>(workspace, activations, expertsWeights, bias, sb);
        }
    }

    template<
        unsigned int Arch,
        unsigned int blocks,
        GateReductionLevel g = GateReductionLevel::singleBlock
    >
    __global__ __maxnreg__(128) void backward(){

    }
}
#endif //ARISTOS_MOE_CUH
