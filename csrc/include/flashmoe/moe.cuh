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
    typename GEMM0Tile, // cute::Shape<M,N,K,pipeStages>
    typename GEMM1Tile // cute::Shape<M,N,K,pipeStages>
  >
  struct MoEConfig {
    static_assert(cute::is_tuple_v<GEMM0Tile> && cute::rank_v<GEMM0Tile> == 4);
    using G0TS = GEMM0Tile;
    static_assert(cute::is_tuple_v<GEMM1Tile> && cute::rank_v<GEMM1Tile> == 4);
    using G1TS = GEMM1Tile;
    static_assert(cute::get<0>(GEMM0Tile{}) == cute::get<0>(GEMM1Tile{}));
    using Arch = cute::Int<arch>;
    using Threads = cute::Int<_threads>;
    using DTK = cute::C<_dTk>;
    using CM = cute::C<isKG1 ? CombineMode::plural : CombineMode::single>;
    using DType = Element;
  };

  struct KernelArgs {
    __host__
    KernelArgs(const cuda::std::byte* tokens, const cuda::std::byte* gate_weights,
      const cuda::std::byte* expert_up_weights, const cuda::std::byte* bias_up,
      const cuda::std::byte* expert_down_weights, const cuda::std::byte* bias_down, cuda::std::byte* gate_out,
      int* expert_counts, cuda::std::byte* moe_out,
      const uint s, const uint h, const uint i, const uint e, const uint k, const uint ec)
      : tokens(tokens),
        gateWeights(gate_weights),
        expertUpWeights(expert_up_weights),
        biasUp(bias_up),
        expertDownWeights(expert_down_weights),
        biasDown(bias_down),
        gateOut(gate_out),
        expertCounts(expert_counts),
        moeOut(moe_out),
        S(s),
        H(h),
        I(i),
        E(e),
        k(k),
        EC(ec) {
      require_align16(tokens);
      require_align16(gate_weights);
      require_align16(expert_up_weights);
      require_align16(expert_down_weights);
      require_align16(bias_up);
      require_align16(bias_down);
      require_align16(gate_out);
      require_align16(expert_counts);
      require_align16(moe_out);
    }

    const cuda::std::byte* tokens; // [S, H]
    const cuda::std::byte* gateWeights = nullptr; // [H, E]
    const cuda::std::byte* expertUpWeights; // [num_local_experts, H, I]
    const cuda::std::byte* biasUp; // [num_local_experts, I]
    const cuda::std::byte* expertDownWeights; // [num_local_experts, I, H]
    const cuda::std::byte* biasDown; // [num_local_experts, H]
    cuda::std::byte* gateOut;
    int* expertCounts; // [E]
    cuda::std::byte* moeOut; //  [S, H]
    const uint S; // sequence length
    const uint H; // token hidden dimension
    const uint I; // FFN intermediate size
    const uint E; // total number of experts
    const uint k; // top k
    const uint EC; // expert capacity
  };

  __host__ __device__ __forceinline__
  constexpr auto dispatchSuperBlockSize(const uint& E) {
    return cute::ceil_div(128, cute::max(E, 4));
  }
  template <
    typename Config,
    Activation a,
    Topology topo
  >
  __global__ __launch_bounds__(Config::Threads::value, 1) void forward(const __grid_constant__ KernelArgs kArgs,
    const __grid_constant__ Context ctx) {
    using DataType = Config::DType;
    extern __shared__ __align__(MAX_ALIGNMENT) cuda::std::byte flashWorkspace[];
    // unpack pointers
    const auto* __restrict__ tokens = reinterpret_cast<const DataType*>(kArgs.tokens);
    constexpr int bM0 = cute::get<0>(typename Config::G0TS{});
    constexpr int bN0 = cute::get<1>(typename Config::G0TS{});
    constexpr int bK0 = cute::get<2>(typename Config::G0TS{});
    constexpr int pS0 = cute::get<3>(typename Config::G0TS{});

    constexpr int bM1 = cute::get<0>(typename Config::G1TS{});
    constexpr int bN1 = cute::get<1>(typename Config::G1TS{});
    constexpr int bK1 = cute::get<2>(typename Config::G1TS{});
    constexpr int pS1 = cute::get<3>(typename Config::G1TS{});
    static_assert(bM0 == bM1);
    using AccumType = float;
    constexpr int bM = bM0;
    constexpr int arch = Config::Arch::value;
    constexpr int threads = Config::Threads::value;
    const auto roundEC = cute::ceil_div(kArgs.EC, bM) * bM;
    const auto symHeap = Heap{
      ctx.symHeap, ctx.nLx, roundEC, kArgs.H, sizeof(DataType)
    };
    const auto processors = gridDim.x - 1;
    const auto superBlockSize = cute::min(dispatchSuperBlockSize(kArgs.E), processors);
    const auto dispatchBlocks = (processors / superBlockSize) * superBlockSize;
    if (blockIdx.x == gridDim.x - 1) {
      // call OS
      constexpr auto subscriberCount = threads - scheduler::SCHEDULER_COUNT;
      static_assert(subscriberCount > 0 && subscriberCount % WARP_SIZE == 0);
      os::start<topo, subscriberCount, threads, bM, Config::DTK::value, DataType>(flashWorkspace, kArgs.expertCounts, symHeap, ctx, kArgs.EC,
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
    using GEMM0Act = ActivationType<DataType, a>::AT;
    const auto tilesN0 = kArgs.I / bN0;
    const auto tilesN1 = kArgs.H / bN1;

    const auto ecTilesM = cute::ceil_div(kArgs.EC, bM);
    using TileGEMM0 = tile::CollectiveMainloop<bM0, bN0, bK0, arch, DataType, AccumType, threads, pS0>;
    using TileGEMM1 = tile::CollectiveMainloop<bM1, bN1, bK1, arch, DataType, AccumType, threads, pS1>;
    static_assert(cuda::std::is_invocable_r_v<AccumType, GEMM0Act, AccumType>, "Activation should be elementwise");
    auto producerBM = cute::make_tensor(cute::make_gmem_ptr(ctx.producerCombineBitMap),
                                        cute::make_layout(cute::make_shape(static_cast<uint>(ctx.world),
                                          static_cast<uint>(ctx.nLx), ecTilesM, tilesN1), cute::LayoutRight{}));
    const auto* __restrict__ expertUp = reinterpret_cast<const DataType*>(kArgs.expertUpWeights);
    const auto* __restrict__ biasUp = reinterpret_cast<const DataType*>(kArgs.biasUp);
    const auto* __restrict__ expertDown = reinterpret_cast<const DataType*>(kArgs.expertDownWeights);
    const auto* __restrict__ biasDown = reinterpret_cast<const DataType*>(kArgs.biasDown);
    auto* __restrict__ moeOut = reinterpret_cast<DataType*>(kArgs.moeOut);
    processor::start<topo, threads, Config::CM::value, TileGEMM0, TileGEMM1, GEMM0Act>
    (flashWorkspace, kArgs.S, kArgs.H, kArgs.I, kArgs.E, roundEC, ecTilesM * kArgs.E, tilesN0, tilesN1, expertUp,
      biasUp,expertDown, biasDown, ctx.tokenIndices, moeOut, producerBM, ctx.stateNumber, symHeap, pA);
  }

  template <
    typename Config,
    Topology topo,
    Activation a
  >
  __host__ __forceinline__
  void forwardHost(const KernelArgs& kArgs, Context& ctx, const uint& sharedSize, cudaStream_t stream) {
    if constexpr (Config::CM::value == CombineMode::plural) {
      cudaMemsetAsync(kArgs.moeOut, 0, sizeof(typename Config::DType) * kArgs.S * kArgs.H, stream);
    }
    forward<Config, a, topo><<<ctx.blocks, Config::Threads::value, sharedSize, stream>>>(kArgs, ctx);
    ctx.stateNumber = sbs::next(ctx.stateNumber);
  }
}
#endif //FLASHMOE_MOE_CUH
