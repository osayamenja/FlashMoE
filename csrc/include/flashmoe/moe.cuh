/*
 * Copyright (c) 2026, Osayamen Jonathan Aimuyo
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
#include "gate.cuh"
#include "processor.cuh"
#include "os.cuh"

namespace flashmoe::heuristics {
  template<int M, int Arch>
  consteval int getTileM() {
    static_assert(M > 0 && (M == 1 || M % 2 == 0)); // allows for decode lengths, where M is typically small
    constexpr int cap = (Arch >= 900 && M >= 4096) ? 256 : 128;

    for (constexpr int t : {128, 96, 64, 48, 32, 24, 16, 8}) {
      if (t <= cap && t <= M && (M % t == 0)) {
        return t;
      }
    }
    return M < 128 ? M : 1;
  }
  template<int M, int Arch>
  consteval int getMoETileM() {
    if constexpr (Arch >= 900 && M >= 4096) {
      return cute::max(cute::min(M, 256), 16);
    }
    return cute::max(cute::min(M, 128), 16);
  }
  template<int N, typename Element>
  consteval int getTileN() {
    static_assert(N > 0 && N % 8 == 0);
    constexpr int cap = (sizeof(Element) <= 2) ? 128 : 64;

    for (constexpr int t : {128, 96, 64, 48, 32, 24, 16, 8}) {
      if (t <= cap && t <= N && (N % t == 0)) {
        return t;
      }
    }
    return 8;
  }
  template<int N>
  consteval int getGateTileN() {
    static_assert(N > 0 && N % 8 == 0);
    constexpr int cap = gate::BLOCK_N_CAP;

    for (constexpr int t : {128, 96, 64, 48, 32, 24, 16, 8}) {
      if (t <= cap && t <= N && (N % t == 0)) {
        return t;
      }
    }
    return 8;
  }

  template<int K, int Arch, MLPMatmulType mt, typename Element>
  consteval int getTileK() {
    static_assert(K > 0 && K % 16 == 0);
    constexpr int cap = cuda::std::is_same_v<Element, double> ? 32 :
    (mt == MLPMatmulType::vanilla ? 64 : (Arch >= 900 ? 64 : 32));
    for (constexpr int t : {128, 96, 64, 48, 32, 24, 16}) {
      if (t <= cap && t <= K && (K % t == 0)) {
        return t;
      }
    }
    return 16;
  }

  template<int K, int bK, int Arch>
  consteval int getPipeStages() {
    static_assert(K % bK == 0);
    constexpr int cap = (Arch>= 900 ? 3 : Arch >= 800 ? 2 : 1);
    constexpr int stages = K / bK;
    return stages >= cap ? cap : stages;
  }

}
namespace flashmoe::moe
{
  template <
    typename Element, // data type = {fp32, fp16, bf16, fp64}
    int arch, //  GPU Architecture, Volta - Blackwell (700 - 1200), See cuBLASDx docs
    int _threads, // see tile::suggest_thread_count
    CombineMode cm, // plural, if k > 1; single otherwise
    MLPMatmulType mt, // Gated MLP or Vanilla FFN
    typename GEMM0Tile, // cute::Shape<bM,bN,bK,pipeStages>
    typename GEMM1Tile, // cute::Shape<bM,bN,bK,pipeStages>
    int processor_state_size = scheduler::PROCESSOR_STATE_SIZE // ceil(#CTAS / WARP_SIZE)
  >
  struct MoEConfig {
    static_assert(cute::is_tuple_v<GEMM0Tile> && cute::rank_v<GEMM0Tile> == 4);
    using G0TS = GEMM0Tile;
    static_assert(cute::is_tuple_v<GEMM1Tile> && cute::rank_v<GEMM1Tile> == 4);
    using G1TS = GEMM1Tile;
    static_assert(cute::get<0>(GEMM0Tile{}) == cute::get<0>(GEMM1Tile{}));
    using Arch = cute::Int<arch>;
    using Threads = cute::Int<_threads>;
    using CM = cute::C<cm>;
    using MT = cute::C<mt>;
    using DType = Element;
    using AccumType = cuda::std::conditional_t<cuda::std::is_same_v<Element, double>, double, float>;
    using PSS = cute::Int<processor_state_size>;
  };

  __host__ __device__ __forceinline__
  constexpr auto dispatchSuperBlockSize(const uint& E) {
    return cute::ceil_div(128, cute::max(E, 4));
  }

  template<typename Config>
  __host__ __forceinline__
  constexpr uint kernelSMEM(const uint& E, const uint& EC, const int& world,
    const uint& numLocalExperts, const uint& tilesN1) {
    constexpr int bM0 = cute::get<0>(typename Config::G0TS{});
    constexpr int bN0 = cute::get<1>(typename Config::G0TS{});
    constexpr int bK0 = cute::get<2>(typename Config::G0TS{});
    constexpr int pSK0 = cute::get<3>(typename Config::G0TS{});

    constexpr int bM1 = cute::get<0>(typename Config::G1TS{});
    constexpr int bN1 = cute::get<1>(typename Config::G1TS{});
    constexpr int bK1 = cute::get<2>(typename Config::G1TS{});
    constexpr int pSK1 = cute::get<3>(typename Config::G1TS{});
    constexpr int threads = Config::Threads::value;
    constexpr auto mt = Config::MT::value;
    static_assert(bM0 == bM1);

    using Element = Config::DType;

    constexpr int bM = bM0;
    constexpr auto GEMM0Sz = cutlass::round_up(cute::max(sizeof(Element) * bK0 * pSK0 * (bM + bN0),
    sizeof(Element) * bM * bN0) + (mt == MLPMatmulType::gated ? sizeof(Element) * bM * bN0 : 0), MAX_ALIGNMENT);
    constexpr auto GEMM1Sz = cutlass::round_up(cute::max(sizeof(Element) * bK1 * pSK1 * (bM + bN1),
      sizeof(Element) * bM * bN1), MAX_ALIGNMENT);
    const auto dispatchSz = E * (sizeof(PEL) + sizeof(int));
    const auto OSSz = os::getSharedSize<threads, bM>(world, numLocalExperts, E, EC, tilesN1);
    const auto taskSz = sizeof(Task) * tilesN1;
    constexpr auto combineSz = cutlass::round_up(sizeof(Element) * bM * bN1, MAX_ALIGNMENT);
    return cute::max(GEMM0Sz, GEMM1Sz, dispatchSz, OSSz, taskSz, combineSz);
  }

  template<int bM, int bN0, int bN1>
  __host__ __forceinline__
  constexpr auto kernelBlocks(const uint& S, const uint& H, const uint& I, const uint& E,
    const uint& k, const int& blocksPerSM, const int& numSMs) {
    const auto processorBlocks = (cute::ceil_div(S * k, bM) * ((I / bN0) + (H / bN1))) +
    (cute::ceil_div(S * k, bM) * (H / bN1));
    const auto dispatchBlocks = dispatchSuperBlockSize(E) * E;
    return cute::max(cute::min(cute::max(processorBlocks, dispatchBlocks) + 1, blocksPerSM * numSMs), 2);
  }
  struct KernelArgs {
    __host__ __forceinline__
    KernelArgs(const cuda::std::byte* const& tokens,
      const cuda::std::byte* const& expert_up_weights, const cuda::std::byte* const& expert_up_v_weights,
      const cuda::std::byte* const& bias_up, const cuda::std::byte* const& bias_up_v,
      const cuda::std::byte* const& expert_down_weights,
      const cuda::std::byte* const& bias_down,
      const int* const& expert_counts, cuda::std::byte* const& moe_out,
      const size_t& s, const size_t& h, const size_t& i, const size_t& e, const size_t& k, const size_t& ec,
      const int& arch, const MLPMatmulType& m,
      const float& swishAlpha, const float& swishBeta, const bool check = true)
      : tokens(tokens),
        expertUpWeights(expert_up_weights),
        expertUpVWeights(expert_up_v_weights),
        biasUp(bias_up), biasUpV(bias_up_v),
        expertDownWeights(expert_down_weights),
        biasDown(bias_down),
        expertCounts(expert_counts),
        moeOut(moe_out),
        S(static_cast<uint>(s)),
        H(static_cast<uint>(h)),
        I(static_cast<uint>(i)),
        E(static_cast<uint>(e)),
        k(static_cast<uint>(k)),
        EC(static_cast<uint>(ec)), swishAlpha(swishAlpha), swishBeta(swishBeta) {
      if (check) {
        const auto supports32 = arch >= 1000;
        checkAlignment(tokens, supports32);
        checkAlignment(expert_up_weights);
        if (m == MLPMatmulType::gated) {
          checkAlignment(expert_up_v_weights);
          checkAlignment(bias_up_v);
        }
        checkAlignment(expert_down_weights);
        checkAlignment(bias_up);
        checkAlignment(bias_down);
        checkAlignment(expert_counts);
        checkAlignment(moe_out);
      }
    }

    const cuda::std::byte* const tokens; // [S, H]
    const cuda::std::byte* const expertUpWeights; // [num_local_experts, H, I]
    const cuda::std::byte* const expertUpVWeights; // [num_local_experts, H, I]
    const cuda::std::byte* const biasUp; // [num_local_experts, I]
    const cuda::std::byte* const biasUpV; // [num_local_experts, I]
    const cuda::std::byte* const expertDownWeights; // [num_local_experts, I, H]
    const cuda::std::byte* const biasDown; // [num_local_experts, H]
    const int* const expertCounts; // [E]
    cuda::std::byte* const moeOut; //  [S, H]
    const uint S; // sequence length
    const uint H; // token hidden dimension
    const uint I; // FFN intermediate size
    const uint E; // total number of experts
    const uint k; // top k
    const uint EC; // expert capacity
    const float swishAlpha = 1.f;
    const float swishBeta = 1.f;
  };

  template <
    typename Config,
    Activation a,
    Topology topo
  >
  __launch_bounds__(Config::Threads::value, 1)
  __global__  void forward(const __grid_constant__ KernelArgs kArgs, const __grid_constant__ Context ctx) {
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
      os::start<topo, subscriberCount, threads, bM, Config::PSS::value, DataType>(flashWorkspace,
        kArgs.expertCounts, symHeap, ctx, kArgs.EC,kArgs.I / bN0, kArgs.H / bN1, dispatchBlocks,
        kArgs.E, kArgs.I, processors);
      return;
    }
    const auto stateNumber = ctx.stateNumbers[blockIdx.x];
    if (blockIdx.x < dispatchBlocks) {
      // dispatch
      dispatch<topo, Config::Threads::value, bM, bN0>(kArgs.H, kArgs.E, symHeap, kArgs.EC, roundEC,
        ctx.epRank, ctx.world, superBlockSize, dispatchBlocks, tokens, ctx.signals, kArgs.expertCounts,
        ctx.tokenIndices, ctx.dispatchSync, ctx.pel, flashWorkspace, stateNumber);
    }
    const auto pA = processor::ProcessorArgs{
      ctx.statusQueue + blockIdx.x,
      ctx.tqs + blockIdx.x,
      ctx.gTqHeads,
      ctx.tQ,
      ctx.pTq,
      ctx.tileSync
    };
    using AccumType = Config::AccumType;
    using GEMM0Act = ActivationType<AccumType, a>::AT;
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
    const auto* __restrict__ expertUpV = reinterpret_cast<const DataType*>(kArgs.expertUpVWeights);
    const auto* __restrict__ biasUp = reinterpret_cast<const DataType*>(kArgs.biasUp);
    const auto* __restrict__ biasUpV = reinterpret_cast<const DataType*>(kArgs.biasUpV);
    const auto* __restrict__ expertDown = reinterpret_cast<const DataType*>(kArgs.expertDownWeights);
    const auto* __restrict__ biasDown = reinterpret_cast<const DataType*>(kArgs.biasDown);
    auto* __restrict__ moeOut = reinterpret_cast<DataType*>(kArgs.moeOut);
    // processor
    processor::start<Config::MT::value, topo, threads, Config::CM::value, TileGEMM0, TileGEMM1, GEMM0Act>
    (flashWorkspace, kArgs.S, kArgs.H, kArgs.I, kArgs.E, roundEC, ecTilesM * kArgs.E, tilesN0, tilesN1,
      expertUp, expertUpV,  biasUp, biasUpV,
      static_cast<AccumType>(kArgs.swishAlpha), static_cast<AccumType>(kArgs.swishBeta),
      expertDown, biasDown, ctx.tokenIndices, moeOut, producerBM, stateNumber, symHeap, pA);
    __syncthreads();
    if (!threadIdx.x) {
      // for the next epoch
      ctx.stateNumbers[blockIdx.x] = sbs::next(stateNumber);
    }
  }

  template <
    typename Config,
    Topology topo,
    Activation a
  >
  __host__ __forceinline__
  void forwardHost(const KernelArgs& kArgs, const Context& ctx, cudaStream_t stream) {
    if constexpr (Config::CM::value == CombineMode::plural) {
      cudaMemsetAsync(kArgs.moeOut, 0, sizeof(typename Config::DType) * kArgs.S * static_cast<size_t>(kArgs.H), stream);
    }
    forward<Config, a, topo><<<ctx.blocks, Config::Threads::value, ctx.smemSize, stream>>>(kArgs, ctx);
  }
}
#endif //FLASHMOE_MOE_CUH
