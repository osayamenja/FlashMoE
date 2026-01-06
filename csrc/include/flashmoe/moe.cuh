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
#include "os.cuh"
#include "processor.cuh"
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
        const auto gtQCl = bookkeeping.gtQCl;
        auto* __restrict__ tQH = bookkeeping.tQH();
        auto* __restrict__ tSA = bookkeeping.tSA();
        constexpr auto gBz = Bookkeeping::gBz();
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
    }

    __device__ __forceinline__
    void gridBarrier() {
        __syncthreads();
        if (!threadIdx.x) {
            __threadfence();
            bookkeeping.dB()->arrive_and_wait();
        }
        __syncthreads();
    }

    // TODO will have to JIT below from Python
    // we intentionally limit the static inputs to the minimal needed set (tile shapes) to
    // keep our APIs flexible to users
    template<
        int _Arch, //  GPU Architecture, Volta - Blackwell (700 - 1200), See cuBLASDx docs
        int _threads, // see tile::suggest_thread_count
        bool isKG1, // is k greater than 1
        DropTokens _dTk, // yes or no,
        GateReductionLevel gRl, // E > GateTileShape.N ? single : multi
        gate::InsideFusedKernel ifk, // yes or no,
        typename GateTile, // cute::Shape<M,N,K,pipeStages>, will be ignored if ifk == no
        typename GEMM1Tile, // cute::Shape<M,N,K,pipeStages>
        typename GEMM2Tile // cute::Shape<M,N,K,pipeStages>
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
        using IFK = cute::C<ifk>;
    };

    template<
        typename Config,
        typename Element
    >
    __global__ __launch_bounds__(Config::Threads::value, 1) void forward(
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
        if constexpr (Config::IFK::value == gate::InsideFusedKernel::yes) {
            gate::forward<
                Config::GTS,
                Config::Arch::value,
                Config::GRL::value>(_activations, _gateWeights, _gateOut);
            gridBarrier();
        }
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
