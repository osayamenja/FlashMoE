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
    __device__ __forceinline__
    void gridBarrier() {
        __syncthreads();
        if (!threadIdx.x) {
            __threadfence();
            bookkeeping.dB()->arrive_and_wait();
        }
        __syncthreads();
    }

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
        const __grid_constant__ int S, // sequence length
        const __grid_constant__ int H, // token hidden dimension
        const __grid_constant__ int I, // FFN intermediate size
        const __grid_constant__ int E, // total number of experts
        const __grid_constant__ int k, // top k
        const Element* __restrict__ _tokens, // [S, H]
        const Element* __restrict__ _gateWeights, // [H, E]
        const Element* __restrict__ _expertUpWeights, // [H, I]
        const Element* __restrict__ _biasUp, // [I]
        const Element* __restrict__ _expertDownWeights, // [I, H]
        const Element* __restrict__ _biasDown, // [H]
        Element* __restrict__ _gateOut, // [S, E]
        Element* __restrict__ _moeOut, //  [S, H]
        const __grid_constant__ Bookkeeping bookkeeping) {
        if constexpr (Config::IFK::value == gate::InsideFusedKernel::yes) {
            gate::forward<
                Config::GTS,
                Config::Arch::value,
                Config::GRL::value>(_tokens, _gateWeights, _gateOut);
            gridBarrier();
        }
        if (blockIdx.x + 1 < gridDim.x) {
            //if (blockIdx.x < ACC::DBZ::value) {
                // MoE dispatch
                //packet::dispatch<ACC::DBZ::value, Config::DTK::value, ACC::SBZ::value>(activations, workspace, sb);
            //}
            //processor::start(workspace, gateOutput, moeOutput, sb);
        }
        else {
            //os::start<Config::DTK::value>(expertsUp, expertsDown, biasUp, biasDown, sb);
        }
    }
}
#endif //FLASHMOE_MOE_CUH
