/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Flashmoe Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 */

#ifndef FLASHMOE_MOE_CUH
#define FLASHMOE_MOE_CUH

#include "arch.cuh"
#include "debug.cuh"
#include "os/os.cuh"
#include "os/processor/processor.cuh"
#include "moe/fffn.cuh"
#include "gate.cuh"
#include "telemetry.cuh"

namespace flashmoe::moe{
    template<
        CombineMode c,
        typename Element
    >
    __device__ __forceinline__
    void clearState(Element* __restrict__ const& outP, const int& outSize) {
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
        const int threads = gridDim.x;
        const auto idx = threads * blockIdx.x + threadIdx.x;
        const int blocks = gridDim.x;
        const int processors = blocks - 1;
        if constexpr (c == CombineMode::plural) {
            // clear output buffer
            for (uint i = idx; i < outSize; i += blocks * threads) {
                outP[i] = Element(0.0f);
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

    template<GateReductionLevel grl>
    __device__ __forceinline__
    void blockade(const int& E) {
        gridBarrier();
        const int threads = blockDim.x;
        const int blocks = gridDim.x;

        if constexpr (grl == GateReductionLevel::multiBlock) {
            // asynchronously wipe flags clean for the next iteration
            auto* __restrict__ bRsP = bookkeeping.bRsP();
            constexpr auto rSlt = Bookkeeping::rSlt();
            auto* __restrict__ rTp = bookkeeping.rTp();
            constexpr auto rTlt = Bookkeeping::rTlt();
            const auto idx = threads * blockIdx.x + threadIdx.x;
            for (unsigned int i = idx; i < rSlt; i += threads * blocks) {
                bRsP[i] = RingSoftmaxPayload{};
            }
            for (unsigned int i = idx; i < rTlt; i += threads * blocks) {
                rTp[i] = RingTopKPayload{};
            }
        }
    }
    // TODO will have to JIT below from Python
    // we intentionally limit the static inputs to the minimal needed set (tile shapes) to
    // keep our APIs flexible to users
    template<
        typename GateTile, // <M,N,K,pipeStages>
        typename GEMM1Tile, // <M,N,K,pipeStages>
        typename GEMM2Tile, // <M,N,K,pipeStages>
        int _Arch, //  GPU Architecture
        int _threads,
        bool isKG1, // is k greater than 1
        DropTokens _dTk, // yes or no,
        GateReductionLevel gRl // E > GateTileShape.K ? single : multi
    >
    struct MoEConfig {
        static_assert(cute::is_tuple_v<GateTile> && cute::rank_v<GateTile> == 4);
        using GTS = GateTile;
        static_assert(cute::is_tuple_v<GEMM1Tile> && cute::rank_v<GEMM1Tile> == 4);
        using G1TS = GEMM1Tile;
        static_assert(cute::is_tuple_v<GEMM2Tile> && cute::rank_v<GEMM2Tile> == 4);
        using G2TS = GEMM2Tile;
        using Arch = cute::Int<_Arch>;
        using Threads = cute::Int<_threads>;
        using GRL = cute::C<gRl>;
        using DTK = cute::C<_dTk>;
        using CM = cute::C<isKG1 ? CombineMode::plural : CombineMode::single>;
    };

    template<
        typename Config,
        typename Element
    >
    __global__ __maxnreg__(ACC::PeakHardware::registers::value) void forward(
        const Element* __restrict__ _activations,
        const Element* __restrict__ _gateWeights,
        const Element* __restrict__ _expertUpWeights,
        const Element* __restrict__ _biasUp,
        const Element* __restrict__ _expertDownWeights,
        const Element* __restrict__ _biasDown,
        Element* __restrict__ _gateOut,
        Element* __restrict__ _moeOut,
        const __grid_constant__ int S,
        const __grid_constant__ int H,
        const __grid_constant__ int E,
        const __grid_constant__ uint16_t sb) {
        const auto lE = bookkeeping.nLx;
        // TODO revisit this
        clearState<Config::CM::value, Config::JT::value>(_moeOut);
        gate::forward<
            Config::GTS,
            Config::Arch::value,
            Config::GRL::value>(_activations, _gateWeights, _gateOut);
        // Needed prior to tokens dispatch
        blockade<Config::JT::value>();
        if (blockIdx.x + 1 < gridDim.x) {
            if (blockIdx.x < ACC::DBZ::value) {
                // MoE dispatch
                packet::dispatch<ACC::DBZ::value, Config::DTK::value, ACC::SBZ::value>(activations, workspace, sb);
            }
            processor::start(workspace, gateOutput, moeOutput, sb);
        }
        else {
            os::start<Config::DTK::value>(expertsUp, expertsDown, biasUp, biasDown, sb);
        }
    }

    template<uint skip = 50, uint trials = 100>
    __host__ __forceinline__
    void forwardHostBench(const void* const& __restrict__ iP, void* __restrict__ const& oP, float& duration){
        #if FLASHMOE_NVTX
        flashmoeRange forwardRange{__PRETTY_FUNCTION__ + std::string(", seqNo: ") + std::to_string(seqBit)};
        #endif
        FLASHMOE_ASSERT(isInitialized, "Not initialized!");
        FLASHMOE_CHECK_CUDA(cudaSetDevice(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE)));
        cudaEvent_t start, stop;
        FLASHMOE_CHECK_CUDA(cudaEventCreate(&start));
        FLASHMOE_CHECK_CUDA(cudaEventCreate(&stop));
        /// Consume precompiled macros
        constexpr auto blocks = ACC::PeakHardware::blocks::value;
        constexpr auto threads = ACC::PeakHardware::OS::threads::value;
        #pragma unroll
        for (uint i = 0; i < skip; ++i) {
            forward<<<blocks, threads, 0, flashmoeStream>>>(iP, oP, seqBit);
            seqBit = sbs::next(seqBit);
        }
        // Call forward pass
        FLASHMOE_CHECK_CUDA(cudaEventRecord(start, flashmoe::flashmoeStream));
        if constexpr (ACC::E::value > 1) {
            #pragma unroll
            for (uint i = 0; i < trials; ++i) {
                forward<<<blocks, threads, 0, flashmoeStream>>>(iP, oP, seqBit);
                seqBit = sbs::next(seqBit);
            }
        }
        else {
            // regular FFN forward
            fffn<<<blocks, threads, 0, flashmoeStream>>>(iP, oP);
        }
        FLASHMOE_CHECK_CUDA(cudaEventRecord(stop, flashmoe::flashmoeStream));
        FLASHMOE_CHECK_CUDA(cudaStreamSynchronize(flashmoeStream));
        FLASHMOE_CHECK_CUDA(cudaEventElapsedTime(&duration, start, stop));
        duration = duration / trials;
        FLASHMOE_CHECK_CUDA(cudaEventDestroy(start));
        FLASHMOE_CHECK_CUDA(cudaEventDestroy(stop));
    }

    __host__ __forceinline__
    void forwardHost(const void* const& __restrict__ iP, void* const& __restrict__ oP){
        #if FLASHMOE_NVTX
        flashmoeRange forwardRange{__PRETTY_FUNCTION__ + std::string(", seqNo: ") + std::to_string(seqBit)};
        #endif
        FLASHMOE_ASSERT(isInitialized, "Not initialized!");
        FLASHMOE_CHECK_CUDA(cudaSetDevice(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE)));
        /// Consume precompiled macros
        constexpr auto blocks = ACC::PeakHardware::blocks::value;
        constexpr auto threads = ACC::PeakHardware::OS::threads::value;
        // Call forward pass
        if constexpr (ACC::E::value > 1) {
            forward<<<blocks, threads, 0, flashmoeStream>>>(iP, oP, seqBit);
            seqBit = sbs::next(seqBit);
        }
        else {
            // regular FFN forward
            fffn<<<blocks, threads, 0, flashmoeStream>>>(iP, oP);
        }
    }
}
#endif //FLASHMOE_MOE_CUH
