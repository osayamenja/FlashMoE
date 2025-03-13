/******************************************************************************
 * Copyright (c) 2024, Osayamen Jonathan Aimuyo.
 ******************************************************************************/

#ifndef ARISTOS_MOE_CUH
#define ARISTOS_MOE_CUH

#include "../arch.cuh"
#include "../debug.cuh"
#include "../os/os.cuh"
#include "../os/processor/processor.cuh"
#include "../os/sync.cuh"
#include "fffn.cuh"
#include "gate.cuh"

namespace aristos::moe{
    template<
        uint threads,
        uint blocks,
        uint processors,
        CombineMode c,
        JobType jT,
        uint OZ,
        typename Element
    >
    __device__ __forceinline__
    void clearState(Element* __restrict__ const& outP) {
        // wipe buffers before the grid-wide barrier
        auto* __restrict__ sBp = bookkeeping.sBp();
        const auto sBz = bookkeeping.sBz();
        auto* __restrict__ pDB = bookkeeping.pDB();
        auto* __restrict__ gBp = bookkeeping.gBp();
        constexpr auto gBz = Bookkeeping::gBz();
        if constexpr (c == CombineMode::multithreaded) {
            // clear output buffer
            for (uint i = threads * blockIdx.x + threadIdx.x; i < OZ; i += blocks * threads) {
                outP[i] = Element(0.0f);
            }
        }
        if constexpr (jT == JobType::training) {
            // clear loss buffers
            for (uint i = threads * blockIdx.x + threadIdx.x; i < gBz; i += blocks * threads) {
                gBp[i] = 0.0f;
            }
        }
        // clear processor doorbells
        for (uint i = threads * blockIdx.x + threadIdx.x; i < processors; i += blocks * threads) {
            pDB[i] = TQSignal{0U, 0U};
        }

        // clear sB
        for (uint i = threads * blockIdx.x + threadIdx.x; i < sBz; i += blocks * threads) {
            sBp[i] = 0U;
        }
    }
    template<
        unsigned S = ACC::S::value,
        unsigned int P = ACC::P::value,
        unsigned int H = ACC::H::value,
        unsigned int PX = ACC::PX::value
    >
    __global__ __maxnreg__(ACC::PeakHardware::registers::value) void forward(
        const void* __restrict__ iP, /* A, G, B, D*/ void* __restrict__ oP /*G, O*/,
        const __grid_constant__ uint16_t sb) {
        using GPUType = ACC::PeakHardware;
        constexpr auto blocks = GPUType::blocks::value;
        constexpr auto processors = GPUType::OS::processorBlocks::value;
        constexpr auto sharedSize = GPUType::sharedMemory::value;
        constexpr auto threads = GPUType::OS::threads::value;
        constexpr auto d = ACC::DTK::value;
        constexpr auto c = ACC::CM::value;
        using Element = ACC::Element;
        using ElementC = ACC::ElementC;

        const auto lE = bookkeeping.nLx;
        if (!blockIdx.x && !threadIdx.x) {
            printf("nLx is %u\n", lE);
        }
        // Salami slice pointers
        const auto* __restrict__ gP = CONST_CAST_TO(Element, iP) + S * H;
        const auto* __restrict__ ePu = gP + H * PX;
        const auto* __restrict__ ePd = ePu + lE * P * H;
        const auto* __restrict__ bU = ePd + lE * H * P;
        const auto* __restrict__ bd = bU + lE * P;
        auto* __restrict__ gOp = CAST_TO(Element, oP);
        auto* __restrict__ mOp = gOp + S * PX;
        __shared__ __align__(16) cuda::std::byte workspace[sharedSize];
        clearState<threads, blocks, processors, c, ACC::JT::value, S * H>(mOp);

        // prep tensors
        const auto activations = make_tensor(
            cute::make_gmem_ptr(CONST_CAST_TO(Element, iP)),
                    cute::Layout<cute::Shape<cute::Int<S>, cute::Int<H>>,
                            cute::Stride<cute::Int<H>, cute::_1>>{});
        const auto gateWeights = make_tensor(cute::make_gmem_ptr(gP),
            cute::Layout<cute::Shape<cute::Int<PX>, cute::Int<H>>,
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
            cute::Layout<cute::Shape<cute::Int<S>, cute::Int<PX>>,
                cute::Stride<cute::Int<PX>, cute::_1>>{});
        const auto moeOutput = make_tensor(cute::make_gmem_ptr(mOp),
            cute::Layout<cute::Shape<cute::Int<S>, cute::Int<H>>,
                cute::Stride<cute::Int<H>, cute::_1>>{});

        gate::forward(activations, gateWeights, gateOutput, CAST_TO(ElementC, workspace));
        if (blockIdx.x + 1 < blocks) {
            constexpr auto cutoff = processors / ARISTOS_SUPER_BLOCK_SIZE * ARISTOS_SUPER_BLOCK_SIZE;
            if (blockIdx.x < cutoff) {
                packet::encode<cutoff, d, ARISTOS_SUPER_BLOCK_SIZE>(activations, workspace, sb);
            }
            processor::start(workspace, gateOutput, sb);
        }
        else {
            os::start<processors, d>(workspace, moeOutput, expertsUp, expertsDown, biasUp, biasDown, sb);
        }
    }

    __host__ __forceinline__
    void forwardHost(const void* __restrict__ iP, void* __restrict__ oP){
        reportError(isInitialized, "Not initialized!");
        CHECK_ERROR_EXIT(cudaSetDevice(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE)));

        /// Consume precompiled macros
        constexpr auto blocks = ACC::PeakHardware::blocks::value;
        constexpr auto threads = ACC::PeakHardware::OS::threads::value;
        // Call forward pass
        if constexpr (ACC::E::value > 1) {
            forward<<<blocks, threads, 0, aristosStream>>>(iP, oP, seqBit);
        }
        else {
            // regular FFN forward
            fffn<<<blocks, threads, 0, aristosStream>>>(iP, oP);
        }
        if (seqBit == cuda::std::numeric_limits<decltype(seqBit)>::max()) {
            syncAll<<<ACC::SYB::value, ACC::PeakHardware::OS::threads::value, 0, aristosStream>>>(
                hostBookkeeping.pL(), hostBookkeeping.rank, hostBookkeeping.world);
        }
        // The below wrapping around to zero is fine due to the sync above
        seqBit++;
    }
}
#endif //ARISTOS_MOE_CUH
