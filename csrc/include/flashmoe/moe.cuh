/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Flashmoe Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 */

#ifndef FLASHMOE_MOE_CUH
#define FLASHMOE_MOE_CUH

#include "infra/activation.cuh"

#include "context.cuh"
#include "gate.cuh"
#include "dispatch.cuh"
#include "processor.cuh"
#include "os.cuh"

namespace flashmoe::moe
{
  using ull_t = unsigned long long int;

  template <
    typename Element, // data type = {fp32, fp16, bf16}
    int arch, //  GPU Architecture, Volta - Blackwell (700 - 1200), See cuBLASDx docs
    int _threads, // see tile::suggest_thread_count
    bool isKG1, // is k greater than 1
    DropTokens _dTk, // yes or no,
    typename GateTile, // cute::Shape<M,N,K,pipeStages>, will be ignored if not doing gate
    typename GEMM0Tile, // cute::Shape<M,N,K,pipeStages>
    typename GEMM1Tile // cute::Shape<M,N,K,pipeStages>
  >
  struct MoEConfig {
    static_assert(cute::is_tuple_v<GateTile> && cute::rank_v<GateTile> == 4);
    using GTS = GateTile;
    static_assert(cute::is_tuple_v<GEMM0Tile> && cute::rank_v<GEMM0Tile> == 4);
    using G0TS = GEMM0Tile;
    static_assert(cute::is_tuple_v<GEMM1Tile> && cute::rank_v<GEMM1Tile> == 4);
    using G1TS = GEMM1Tile;
    static_assert(cute::get<0>(GateTile{}) == cute::get<0>(GEMM0Tile{}) &&
      cute::get<0>(GEMM0Tile{}) == cute::get<0>(GEMM1Tile{}));
    using Arch = cute::Int<arch>;
    using Threads = cute::Int<_threads>;
    using DTK = cute::C<_dTk>;
    using CM = cute::C<isKG1 ? CombineMode::plural : CombineMode::single>;
    using DType = Element;
  };

  struct KernelArgs {
    const cuda::std::byte* const tokens; // [S, H]
    const cuda::std::byte* const gateWeights = nullptr; // [H, E]
    const cuda::std::byte* const expertUpWeights; // [num_local_experts, H, I]
    const cuda::std::byte* const biasUp; // [num_local_experts, I]
    const cuda::std::byte* const expertDownWeights; // [num_local_experts, I, H]
    const cuda::std::byte* const biasDown; // [num_local_experts, H]
    cuda::std::byte* const gateOut;
    int* const expertCounts; // [E]
    cuda::std::byte* const moeOut; //  [S, H]
    const uint S; // sequence length
    const uint H; // token hidden dimension
    const uint I; // FFN intermediate size
    const uint E; // total number of experts
    const uint k; // top k
    const uint EC; // expert capacity
  };
  template <
    typename Config,
    Activation a,
    Topology topo,
    bool doGate,
    GateReductionLevel gRl = GateReductionLevel::singleBlock // ignored if doGate == false
  >
  __global__ __launch_bounds__(Config::Threads::value, 1) void forward(const __grid_constant__ KernelArgs kArgs,
    const __grid_constant__ Context ctx, const __grid_constant__ GateContext gCtx) {
    using DataType = Config::DType;
    const auto symHeap = Heap{
      ctx.symHeap, ctx.nLx, kArgs.EC, kArgs.H, sizeof(DataType)
    };
    extern __shared__ __align__(MAX_ALIGNMENT) cuda::std::byte flashWorkspace[];
    // unpack pointers
    const auto* __restrict__ tokens = reinterpret_cast<const DataType*>(kArgs.tokens);

    constexpr int bM0 = cute::get<0>(Config::G0TS{});
    constexpr int bN0 = cute::get<1>(Config::G0TS{});
    constexpr int bK0 = cute::get<2>(Config::G0TS{});
    constexpr int pS0 = cute::get<3>(Config::G0TS{});

    constexpr int bM1 = cute::get<0>(Config::G1TS{});
    constexpr int bN1 = cute::get<1>(Config::G1TS{});
    constexpr int bK1 = cute::get<2>(Config::G1TS{});
    constexpr int pS1 = cute::get<3>(Config::G1TS{});
    static_assert(bM0 == bM1);
    constexpr int bM = bM0;
    const auto roundEC = cute::ceil_div(kArgs.EC, bM) * bM;
    if constexpr (doGate) {
      const auto* __restrict__ gateWeights = reinterpret_cast<const DataType*>(kArgs.gateWeights);
      auto* __restrict__ gateOut = reinterpret_cast<DataType*>(kArgs.gateOut);
      gate::forward<
        Config::GTS,
        Config::Arch::value, gRl>(flashWorkspace, tokens, gateWeights, gateOut,
          ctx.tokenIndices, kArgs.expertCounts,
          kArgs.S, kArgs.H, kArgs.E, kArgs.k, kArgs.EC, roundEC, gridDim.x, gCtx.ecGuards, gCtx.ssp, gCtx.rtp);
      gridBarrier(gCtx.db);
    }
    constexpr int threads = Config::Threads::value;
    const auto processors = gridDim.x - 1;
    const auto superBlockSize = cute::min(cute::ceil_div(128, cute::max(kArgs.E, 4)), processors);
    const auto dispatchBlocks = (processors / superBlockSize) * superBlockSize;
    if (blockIdx.x == gridDim.x - 1) {
      // call OS
      constexpr auto subscriberCount = threads - scheduler::SCHEDULER_COUNT;
      static_assert(subscriberCount > 0 && subscriberCount % WARP_SIZE == 0);
      os::start<topo, subscriberCount, threads, bM, DataType>(flashWorkspace, kArgs.expertCounts, symHeap, ctx, kArgs.EC,
        kArgs.I / bN0, kArgs.H / bN1, dispatchBlocks, kArgs.E, kArgs.I, processors);
      return;
    }
    if (blockIdx.x < dispatchBlocks) {
      // dispatch
      dispatch<topo, Config::Threads::value, bM, bN0, Config::DTK::value>(kArgs.H, kArgs.E, symHeap, kArgs.EC, roundEC,
        ctx.epRank, ctx.world, superBlockSize, dispatchBlocks, tokens, ctx.signals, kArgs.expertCounts,
        ctx.tokenIndices, ctx.dispatchSync, ctx.pel, flashWorkspace, ctx.stateNumber);
    }
    // processor;
    const auto pA = processor::ProcessorArgs{
      ctx.statusQueue + blockIdx.x,
      ctx.tqs + blockIdx.x,
      ctx.gTqHeads,
      ctx.tQ,
      ctx.pTq,
      ctx.tileSync
    };
    using GEMM0Act = ActivationType<a, DataType>::AT;
    using AccumType = float;
    const auto tilesN0 = kArgs.I / bN0;
    const auto tilesN1 = kArgs.H / bN1;

    constexpr int arch = Config::Arch::value;
    const auto ecTilesM = cute::ceil_div(kArgs.EC, bM);
    using TileGEMM0 = tile::CollectiveMainloop<bM0, bN0, bK0, arch, DataType, AccumType, threads, pS0>;
    using TileGEMM1 = tile::CollectiveMainloop<bM1, bN1, bK1, arch, DataType, AccumType, threads, pS1>;
    static_assert(cuda::std::is_invocable_r_v<AccumType, GEMM0Act, AccumType>, "Activation should be elementwise");
    auto producerBM = cute::make_tensor(cute::make_gmem_ptr(ctx.producerCombineBitMap),
                                        cute::make_layout(cute::make_shape(ctx.world, ctx.nLx, ecTilesM, tilesN1),
                                                          cute::LayoutRight{}));
    const auto* __restrict__ expertUp = reinterpret_cast<const DataType*>(kArgs.expertUpWeights);
    const auto* __restrict__ biasUp = reinterpret_cast<const DataType*>(kArgs.biasUp);
    const auto* __restrict__ expertDown = reinterpret_cast<const DataType*>(kArgs.expertDownWeights);
    const auto* __restrict__ biasDown = reinterpret_cast<const DataType*>(kArgs.biasDown);
    auto* __restrict__ moeOut = reinterpret_cast<DataType*>(kArgs.moeOut);
    processor::start<topo, threads, Config::CM::value, TileGEMM0, TileGEMM1, GEMM0Act>
    (flashWorkspace, kArgs.S, kArgs.H, kArgs.I, kArgs.E, kArgs.k, roundEC, tilesN0, tilesN1, expertUp,
      biasUp,expertDown, biasDown, ctx.tokenIndices, moeOut, producerBM, ctx.stateNumber, symHeap, pA);
  }

  template <
    typename Config,
    Topology topo,
    bool doGate,
    Activation a,
    GateReductionLevel gRl = GateReductionLevel::singleBlock
  >
  __host__ __forceinline__
  void forwardHost(const KernelArgs& kArgs, Context& ctx, const GateContext& gCtx, const uint& blocks,
    const uint& sharedSize, cudaStream_t stream) {
    // asert(blocks >= 2)
    if constexpr (Config::CM::value == CombineMode::plural) {
      cudaMemsetAsync(kArgs.moeOut, 0, sizeof(Config::DType) * kArgs.S * kArgs.H, stream);
    }
    forward<Config, a, topo, doGate, gRl><<<blocks, Config::Threads::value, sharedSize, stream>>>
    (kArgs, ctx, gCtx);
    ctx.stateNumber = sbs::next(ctx.stateNumber);
  }
}
#endif //FLASHMOE_MOE_CUH
