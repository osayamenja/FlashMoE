/******************************************************************************
 * Copyright (c) 2024, Osayamen Jonathan Aimuyo.
 ******************************************************************************/

#ifndef ARISTOS_MOE_CUH
#define ARISTOS_MOE_CUH

#include "../arch.cuh"
#include "../debug.cuh"
#include "../os/os.cuh"
#include "../os/processor/processor.cuh"
#include "gate.cuh"

namespace aristos::moe{
    template<
        unsigned S = ACC::S::value,
        unsigned int P = ACC::P::value,
        unsigned int H = ACC::H::value,
        unsigned int E = ACC::E::value
    >
    __global__ __maxnreg__(ACC::PeakHardware::registers::value) void forward(
        const void* __restrict__ iP, /* A, G, B, D*/ void* __restrict__ oP /*G, O*/,
        const __grid_constant__ uint16_t sb) {
        using Config = ACC;
        using GPUType = Config::PeakHardware;
        constexpr auto blocks = GPUType::blocks::value;
        constexpr auto processors = GPUType::OS::processorBlocks::value;
        constexpr auto sharedSize = GPUType::sharedMemory::value;
        constexpr auto threads = GPUType::OS::threads::value;
        constexpr auto d = Config::DTK::value;
        constexpr auto c = Config::CM::value;
        using Element = Config::Element;
        using ActivationOp = Config::ActivationOp;
        using ActivationOpX = Config::ActivationOpX;
        using ElementC = GEA;

        // Salami slice pointers
        const auto lE = bookkeeping.nLx;
        const auto* __restrict__ gP = CONST_CAST_TO(Element, iP) + S * H;
        const auto* __restrict__ ePu = gP + H * E;
        const auto* __restrict__ ePd = ePu + lE * P * H;
        const auto* __restrict__ bU = ePd + lE * H * P;
        const auto* __restrict__ bd = bU + lE * P;
        auto* __restrict__ gOp = CAST_TO(Element, oP);
        auto* __restrict__ mOp = gOp + S * E;
        __shared__ __align__(16) cuda::std::byte workspace[sharedSize];
        // wipe buffers before the grid-wide barrier
        const auto gtQCl = bookkeeping.gtQCl;
        auto* __restrict__ gtQHeads = bookkeeping.tQH();
        for (uint i = threads * blockIdx.x + threadIdx.x; i < gtQCl; i += blocks * threads) {
            gtQHeads[i] = 0U;
        }
        if constexpr (c == CombineMode::multithreaded) {
            // clear output buffer
            constexpr auto sz = S * H;
            constexpr auto vL = sz / sizeof(uint4);
            for (uint i = threads * blockIdx.x + threadIdx.x; i < vL; i += blocks * threads) {
                CAST_TO(uint4, mOp)[i] = uint4{0U, 0U, 0U, 0U};
            }
            // residue
            for (uint i = threads * blockIdx.x + threadIdx.x + vL * sizeof(uint); i < sz; i += blocks * threads) {
                mOp[i] = Element(0);
            }
        }

        const auto activations = make_tensor(
            cute::make_gmem_ptr(CONST_CAST_TO(Element, iP)),
                    cute::Layout<cute::Shape<cute::Int<S>, cute::Int<H>>,
                            cute::Stride<cute::Int<H>, cute::_1>>{});
        const auto gateWeights = make_tensor(cute::make_gmem_ptr(gP),
            cute::Layout<cute::Shape<cute::Int<E>, cute::Int<H>>,
                cute::Stride<cute::Int<H>, cute::_1>>{});
        // Experts Weights
        const auto expertsUp = make_tensor(cute::make_gmem_ptr(ePu),
            make_layout(make_shape(lE, cute::Shape<cute::Int<P>, cute::Int<H>>{}),
                cute::LayoutRight{}));
        const auto expertsDown = make_tensor(cute::make_gmem_ptr(ePd),
            make_layout(make_shape(lE, cute::Shape<cute::Int<H>, cute::Int<P>>{}),
                cute::LayoutRight{}));
        // Bias
        // Broadcast from vector to matrix
        const auto biasUp = make_tensor(cute::make_gmem_ptr(bU),
            make_layout(make_shape(lE, cute::Shape<cute::Int<S>, cute::Int<P>>{}),
                cute::Stride<cute::Int<P>, cute::Stride<cute::_0, cute::_1>>{}));
        const auto biasDown = make_tensor(cute::make_gmem_ptr(bd),
            make_layout(make_shape(lE, cute::Shape<cute::Int<S>, cute::Int<H>>{}),
                cute::Stride<cute::Int<H>, cute::Stride<cute::_0, cute::_1>>{}));

        // Output
        const auto gateOutput = make_tensor(cute::make_gmem_ptr(gOp),
            cute::Layout<cute::Shape<cute::Int<S>, cute::Int<E>>,
                cute::Stride<cute::Int<E>, cute::_1>>{});
        const auto moeOutput = make_tensor(cute::make_gmem_ptr(mOp),
            cute::Layout<cute::Shape<cute::Int<S>, cute::Int<H>>,
                cute::Stride<cute::Int<H>, cute::_1>>{});

        gate::forward(activations, gateWeights, gateOutput, CAST_TO(ElementC, workspace));
        if (blockIdx.x + 1 < blocks) {
            constexpr auto cutoff = processors / ARISTOS_SUPER_BLOCK_SIZE * ARISTOS_SUPER_BLOCK_SIZE;
            if (blockIdx.x < cutoff) {
                packet::encode<cutoff, d, ARISTOS_SUPER_BLOCK_SIZE>(activations, workspace, sb);
            }
            processor::start<
                GPUType,
                c,
                ActivationOp,
                ActivationOpX,
                ElementC,
                Element>(workspace, gateOutput, sb);
        }
        else {
            os::start<processors, d>(workspace, moeOutput, expertsUp, expertsDown, biasUp, biasDown, sb);
        }
    }

    __host__ __forceinline__
    void forwardHost(const void* __restrict__ iP, void* __restrict__ oP){
        //reportError(isInitialized, "Not initialized");
        CHECK_ERROR_EXIT(cudaSetDevice(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE)));

        /// Consume precompiled macros
        using GPUType = aristos::Hardware<ARISTOS_ARCH, 255>;
        constexpr auto blocks = GPUType::OS::processorBlocks::value;
        constexpr auto threads = GPUType::OS::threads::value;

        // Call forward pass
        moe::forward<<<blocks, threads>>>(iP, oP, seqBit);
        if (seqBit == cuda::std::numeric_limits<decltype(seqBit)>::max()) {
            // TODO sequence bit sync logic
        }
        // The below wrapping around to zero is fine
        seqBit++;
    }
}
#endif //ARISTOS_MOE_CUH
