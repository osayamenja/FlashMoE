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
  };

  template <
    typename Config,
    Activation a,
    Topology topo,
    bool doGate,
    GateReductionLevel gRl = GateReductionLevel::singleBlock, // ignored if doGate == false
    typename Element
  >
  __global__ __launch_bounds__(Config::Threads::value, 1) void forward(
    const __grid_constant__ uint S, // sequence length
    const __grid_constant__ uint H, // token hidden dimension
    const __grid_constant__ uint I, // FFN intermediate size
    const __grid_constant__ uint E, // total number of experts
    const __grid_constant__ int k, // top k
    const __grid_constant__ uint EC, // expert capacity
    const Element* __restrict__ tokens, // [S, H]
    const Element* __restrict__ gateWeights, // [H, E]
    const Element* __restrict__ expertUpWeights, // [num_local_experts, H, I]
    const Element* __restrict__ biasUp, // [num_local_experts, I]
    const Element* __restrict__ expertDownWeights, // [num_local_experts, I, H]
    const Element* __restrict__ biasDown, // [num_local_experts, H]
    Element* __restrict__ gateOut, // [S, E]
    int* __restrict__ expertCounts,
    Element* __restrict__ moeOut, //  [S, H]
    const __grid_constant__ Context ctx,
    const __grid_constant__ GateContext gCtx) {
    const auto symHeap = Heap{
      ctx.symHeap, ctx.nLx, EC, H, sizeof(Element)
    };
    extern __shared__ __align__(MAX_ALIGNMENT) cuda::std::byte flashWorkspace[];
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
    const auto roundEC = cute::ceil_div(EC, bM) * bM;
    if constexpr (doGate) {
      gate::forward<
        Config::GTS,
        Config::Arch::value, gRl>(flashWorkspace, tokens, gateWeights, gateOut,
          ctx.tokenIndices, expertCounts,
          S, H, E, k, EC, roundEC, gridDim.x, gCtx.ecGuards, gCtx.ssp, gCtx.rtp);
      gridBarrier(gCtx.db);
    }
    constexpr int threads = Config::Threads::value;
    const auto processors = gridDim.x - 1;
    const auto superBlockSize = cute::min(cute::ceil_div(128, cute::max(E, 4)), processors);
    const auto dispatchBlocks = (processors / superBlockSize) * superBlockSize;
    if (blockIdx.x == gridDim.x - 1) {
      // call OS
      constexpr auto subscriberCount = threads - scheduler::SCHEDULER_COUNT;
      static_assert(subscriberCount > 0 && subscriberCount % WARP_SIZE == 0);
      os::start<topo, subscriberCount, threads, bM, Element>(flashWorkspace, expertCounts, symHeap, ctx, EC,
        I / bN0, H / bN1, dispatchBlocks, E, I, processors);
      return;
    }
    if (blockIdx.x < dispatchBlocks) {
      // dispatch
      dispatch<topo, Config::Threads::value, bM, bN0, Config::DTK::value>(H, E, symHeap, EC, roundEC,
        ctx.epRank, ctx.world, superBlockSize, dispatchBlocks, tokens, ctx.signals, expertCounts,
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
    using GEMM0Act = ActivationType<a, Element>;
    using AccumType = float;
    const auto tilesN0 = I / bN0;
    const auto tilesN1 = H / bN1;

    constexpr int arch = Config::Arch::value;
    const auto ecTilesM = cute::ceil_div(EC, bM);
    using TileGEMM0 = tile::CollectiveMainloop<bM0, bN0, bK0, arch, Element, AccumType, threads, pS0>;
    using TileGEMM1 = tile::CollectiveMainloop<bM1, bN1, bK1, arch, Element, AccumType, threads, pS1>;
    static_assert(cuda::std::is_invocable_r_v<AccumType, GEMM0Act, AccumType>, "Activation should be elementwise");
    auto producerBM = cute::make_tensor(cute::make_gmem_ptr(ctx.producerCombineBitMap),
                                        cute::make_layout(cute::make_shape(ctx.world, ctx.nLx, ecTilesM, tilesN1),
                                                          cute::LayoutRight{}));
    processor::start<topo, threads, Config::CM::value, TileGEMM0, TileGEMM1, GEMM0Act>
    (flashWorkspace, S, H, I, E, k, roundEC, tilesN0, tilesN1, expertUpWeights, biasUp,
     expertDownWeights, biasDown, ctx.tokenIndices, moeOut, producerBM, ctx.stateNumber, symHeap, pA);
  }

  template <
    typename Config,
    Topology topo,
    bool doGate,
    Activation a,
    GateReductionLevel gRl = GateReductionLevel::singleBlock, // ignored if doGate == false
    typename Element
  >
  __host__ __forceinline__
  void forwardHost(const uint& S, // sequence length
                   const uint& H, // token hidden dimension
                   const uint& I, // FFN intermediate size
                   const uint& E, // total number of experts
                   const int& k, // top k
                   const uint& EC, // expert capacity
                   const Element* __restrict__ const& tokens, // [S, H]
                   const Element* __restrict__ const& gateWeights, // [H, E]
                   const Element* __restrict__ const& expertUpWeights, // [num_local_experts, H, I]
                   const Element* __restrict__ const& biasUp, // [num_local_experts, I]
                   const Element* __restrict__ const& expertDownWeights, // [num_local_experts, I, H]
                   const Element* __restrict__ const& biasDown, // [num_local_experts, H]
                   Element* __restrict__ const& gateOut, // [S, E]
                   const int* __restrict__ const& expertCounts,
                   Element* __restrict__ const& moeOut, //  [S, H]
                   Context& ctx,
                   const GateContext& gCtx,
                   const uint& blocks, const uint& sharedSize,
                   cudaStream_t stream) {
    if constexpr (Config::CM::value == CombineMode::plural) {
      cudaMemsetAsync(moeOut, 0, sizeof(Element) * S * H, stream);
    }
    forward<Config, a, topo, doGate, gRl><<<blocks, Config::Threads::value, sharedSize, stream>>>
    (S, H, I, E, k, EC, tokens, gateWeights, expertUpWeights, biasUp, expertDownWeights, biasDown, gateOut,
     expertCounts, moeOut, ctx, gCtx);
    ctx.stateNumber = sbs::next(ctx.stateNumber);
  }
}
#endif //FLASHMOE_MOE_CUH
