/******************************************************************************
 * Copyright (c) 2024, Osayamen Jonathan Aimuyo.
 ******************************************************************************/

#ifndef ARISTOS_MOE_CUH
#define ARISTOS_MOE_CUH

#include "../arch.cuh"
#include "../debug.cuh"
#include "../os/os.cuh"
#include "../os/processor/processor.cuh"
#include "fffn.cuh"
#include "gate.cuh"
#include "../telemetry.cuh"

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
        // A barrier must occur after below otherwise, undefined behavior results.
        auto* __restrict__ pDB = bookkeeping.pDB();
        auto* __restrict__ sQ = bookkeeping.sQ();
        auto* __restrict__ gBp = bookkeeping.gBp();
        const auto gtQCl = bookkeeping.gtQCl;
        auto* __restrict__ tQH = bookkeeping.tQH();
        auto* __restrict__ tSA = bookkeeping.tSA();
        auto* __restrict__ pSA = bookkeeping.pSA();
        constexpr auto gBz = Bookkeeping::gBz();
        auto* __restrict__ eCSync = bookkeeping.eCSync();
        const auto idx = threads * blockIdx.x + threadIdx.x;
        if constexpr (c == CombineMode::multithreaded) {
            // clear output buffer
            for (uint i = idx; i < OZ; i += blocks * threads) {
                outP[i] = Element(0.0f);
            }
        }
        if constexpr (jT == JobType::training) {
            // clear loss buffers
            for (uint i = idx; i < gBz; i += blocks * threads) {
                gBp[i] = 0.0f;
            }
        }
        // clear processor doorbells
        for (uint i = idx; i < processors; i += blocks * threads) {
            pDB[i] = TQSignal{0U, 0U};
            sQ[i] = observed;
        }
        for (uint i = idx; i < gtQCl; i += blocks * threads) {
            tQH[i] = tQHeadGroundState;
            tSA[i] = 0U;
        }
        for (uint i = idx; i < ACC::E::value; i += blocks * threads) {
            pSA[i] = 0U;
        }
        if (!idx) {
            *eCSync = 0U;
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
        constexpr auto sharedSize = ACC::sharedSize::value;
        constexpr auto threads = GPUType::OS::threads::value;
        constexpr auto d = ACC::DTK::value;
        constexpr auto c = ACC::CM::value;
        using Element = ACC::Element;
        using ElementC = ACC::ElementC;

        const auto lE = bookkeeping.nLx;
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
            make_layout(cute::make_shape(lE, P),
                cute::Stride<cute::Int<P>, cute::_1>{}));
        const auto biasDown = make_tensor(cute::make_gmem_ptr(bd),
            make_layout(make_shape(lE, cute::Int<H>{}),
                cute::Stride<cute::Int<H>, cute::_1>{}));

        // Output
        const auto gateOutput = make_tensor(cute::make_gmem_ptr(gOp),
            cute::Layout<cute::Shape<cute::Int<S>, cute::Int<PX>>,
                cute::Stride<cute::Int<PX>, cute::_1>>{});
        const auto moeOutput = make_tensor(cute::make_gmem_ptr(mOp),
            cute::Layout<cute::Shape<cute::Int<S>, cute::Int<H>>,
                cute::Stride<cute::Int<H>, cute::_1>>{});

        gate::forward(activations, gateWeights, gateOutput, CAST_TO(ElementC, workspace));
        if (blockIdx.x + 1 < blocks) {
            if (blockIdx.x < ACC::DBZ::value) {
                packet::dispatch<ACC::DBZ::value, d, ACC::SBZ::value>(activations, workspace, sb);
            }
            processor::start(workspace, gateOutput, moeOutput, sb);
        }
        else {
            os::start<processors, d>(workspace, expertsUp, expertsDown, biasUp, biasDown, sb);
        }
    }

    template<uint skip = 50, uint trials = 100>
    __host__ __forceinline__
    void forwardHostBench(const void* const& __restrict__ iP, void* __restrict__ const& oP, float& duration){
        #if ARISTOS_NVTX
        aristosRange forwardRange{__PRETTY_FUNCTION__ + std::string(", seqNo: ") + std::to_string(seqBit)};
        #endif
        ARISTOS_ASSERT(isInitialized, "Not initialized!");
        ARISTOS_CHECK_CUDA(cudaSetDevice(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE)));
        /// Consume precompiled macros
        constexpr auto blocks = ACC::PeakHardware::blocks::value;
        constexpr auto threads = ACC::PeakHardware::OS::threads::value;
        #pragma unroll
        for (uint i = 0; i < skip; ++i) {
            forward<<<blocks, threads, 0, aristosStream>>>(iP, oP, seqBit);
            seqBit = sbs::next(seqBit);
        }
        cudaEvent_t start, stop;
        ARISTOS_CHECK_CUDA(cudaEventCreate(&start));
        ARISTOS_CHECK_CUDA(cudaEventCreate(&stop));
        ARISTOS_CHECK_CUDA(cudaEventRecord(start, aristos::aristosStream));
        // Call forward pass
        if constexpr (ACC::E::value > 1) {
            #pragma unroll
            for (uint i = 0; i < trials; ++i) {
                forward<<<blocks, threads, 0, aristosStream>>>(iP, oP, seqBit);
                seqBit = sbs::next(seqBit);
            }
        }
        else {
            // regular FFN forward
            fffn<<<blocks, threads, 0, aristosStream>>>(iP, oP);
        }
        ARISTOS_CHECK_CUDA(cudaEventRecord(stop, aristos::aristosStream));
        ARISTOS_CHECK_CUDA(cudaStreamSynchronize(aristosStream));
        ARISTOS_CHECK_CUDA(cudaEventElapsedTime(&duration, start, stop));
        duration = duration / trials;
        ARISTOS_CHECK_CUDA(cudaEventDestroy(start));
        ARISTOS_CHECK_CUDA(cudaEventDestroy(stop));
    }

    __host__ __forceinline__
    void forwardHost(const void* const& __restrict__ iP, void* const& __restrict__ oP){
        #if ARISTOS_NVTX
        aristosRange forwardRange{__PRETTY_FUNCTION__ + std::string(", seqNo: ") + std::to_string(seqBit)};
        #endif
        ARISTOS_ASSERT(isInitialized, "Not initialized!");
        ARISTOS_CHECK_CUDA(cudaSetDevice(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE)));
        /// Consume precompiled macros
        constexpr auto blocks = ACC::PeakHardware::blocks::value;
        constexpr auto threads = ACC::PeakHardware::OS::threads::value;
        // Call forward pass
        if constexpr (ACC::E::value > 1) {
            forward<<<blocks, threads, 0, aristosStream>>>(iP, oP, seqBit);
            seqBit = sbs::next(seqBit);
        }
        else {
            // regular FFN forward
            fffn<<<blocks, threads, 0, aristosStream>>>(iP, oP);
        }
    }
}
#endif //ARISTOS_MOE_CUH
