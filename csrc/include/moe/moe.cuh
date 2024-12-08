/******************************************************************************
 * Copyright (c) 2024, Osayamen Jonathan Aimuyo.
 ******************************************************************************/

#ifndef ARISTOS_MOE_CUH
#define ARISTOS_MOE_CUH

#include "gate.cuh"
#include "packet.cuh"

namespace aristos::moe{
    template<unsigned int Arch,
        unsigned int blocks,
        unsigned int k,
        TripPredication tP = TripPredication::complete,
        typename ActivationOp = cute::identity,
        typename ActivationOpX = cute::identity,
        typename ElementC = float,
        typename ElementA = cute::half_t,
        typename ElementB = ElementA,
        typename ElementD = ElementA,
        typename MatrixA,
        typename MatrixB,
        typename MatrixBg,
        typename MatrixC,
        typename MatrixCg>
    requires(aristos::Matrix<MatrixA> && aristos::Matrix<MatrixB> &&
        aristos::Matrix<MatrixBg> && aristos::Matrix<MatrixC>
        && aristos::Matrix<MatrixCg>)
    __global__ __maxnreg__(128) void forward(
        cuda::std::byte* workspace,
        MatrixA const& activations,
        MatrixB const& expertsWeights,
        MatrixBg const& gateWeights,
        MatrixC moeOutput,
        MatrixCg gateOutput) {

        gate::forward<Arch, blocks, k, tP, ElementC>(activations,
            gateWeights, gateOutput, CAST_TO(ElementC, workspace));
        if (blockIdx.x + 1 < blocks) {
            packet::constructSend(gateOutput, workspace);
            processor::start<
                blocks,
                Arch,
                ElementA,
                ElementB,
                ElementC,
                ElementD,
                ActivationOp,
                ActivationOpX>(CAST_TO(ElementD, workspace));
        }
        else {
            if (!threadIdx.x) {
                scheduler::start<blocks>();
            }
            else {
                subscriber::start();
            }
        }
    }

    template<Matrix M, Tensor T>
    __global__ __maxnreg__(128) void backward(){

    }
}
#endif //ARISTOS_MOE_CUH
