/******************************************************************************
 * Copyright (c) 2026, Osayamen Jonathan Aimuyo.
 ******************************************************************************/

#ifndef FLASHMOE_MOE_CUH
#define FLASHMOE_MOE_CUH

#include "infra/activation.cuh"

#include "context.cuh"
#include "dispatch.cuh"
#include "gate.cuh"
#include "processor.cuh"
#include "os.cuh"

namespace flashmoe {
  __device__ __host__ constexpr int MAX_DISPATCH_BLOCKS = 128;
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
    int processor_state_size = scheduler::PROCESSOR_STATE_SIZE, // ceil(#CTAS / WARP_SIZE)
    QuantRecipe qr = QuantRecipe::none // mxfp8: E4M3 weights and activations, bias, output and Element stay bf16
  >
  struct MoEConfig {
    static_assert(cute::is_tuple_v<GEMM0Tile> && cute::rank_v<GEMM0Tile> == 4);
    using G0TS = GEMM0Tile;
    static_assert(cute::is_tuple_v<GEMM1Tile> && cute::rank_v<GEMM1Tile> == 4);
    using G1TS = GEMM1Tile;
    static_assert(cute::get<0>(GEMM0Tile{}) == cute::get<0>(GEMM1Tile{}));
    static_assert(qr == QuantRecipe::none || (cuda::std::is_same_v<Element, __nv_bfloat16> &&
      cute::get<0>(GEMM0Tile{}) == mx::SF_ROW_TILE), "MXFP8 requires bf16 activations and bM == 128");
    using Arch = cute::Int<arch>;
    using Threads = cute::Int<_threads>;
    using CM = cute::C<cm>;
    using MT = cute::C<mt>;
    using QR = cute::C<qr>;
    using DType = Element;
    // element type of GEMM0Staging, i.e., of the GEMM0 output and the GEMM1 input
    using StagingType = cuda::std::conditional_t<qr == QuantRecipe::mxfp8, __nv_fp8_e4m3, Element>;
    using AccumType = cuda::std::conditional_t<cuda::std::is_same_v<Element, double>, double, float>;
    using PSS = cute::Int<processor_state_size>;
  };

  __host__ __device__ __forceinline__
  constexpr auto dispatchSuperBlockSize(const uint& E) {
    return cute::ceil_div(MAX_DISPATCH_BLOCKS, cute::max(E, 4));
  }

  __host__ __forceinline__
  auto checkAlignment(const void *const&p, const bool supports32 = false) {
    const auto alignment = supports32 ? 32 : 16;
    if (p == nullptr || !cuda::is_aligned(p, alignment)) {
      printf("Pointer is not %d-byte aligned\n", alignment);
      cuda::std::terminate();
    }
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
    // MXFP8: the block-scaled mainloops have their own pipelines, see processor::start
    constexpr size_t GEMM0Sz = [] {
      if constexpr (Config::QR::value == QuantRecipe::mxfp8) {
        return static_cast<size_t>(tile::sm120::MXFP8GEMM0<bM, bN0, threads, mt == MLPMatmulType::gated>::sharedSize);
      }
      else {
        return static_cast<size_t>(cutlass::round_up(cute::max(sizeof(Element) * bK0 * pSK0 * (bM + bN0),
          sizeof(Element) * bM * bN0) + (mt == MLPMatmulType::gated ? sizeof(Element) * bM * bN0 : 0),
          MAX_ALIGNMENT));
      }
    }();
    constexpr size_t GEMM1Sz = [] {
      if constexpr (Config::QR::value == QuantRecipe::mxfp8) {
        return static_cast<size_t>(tile::sm120::MXFP8GEMM1<bM, bN1, threads>::sharedSize);
      }
      else {
        return static_cast<size_t>(cutlass::round_up(cute::max(sizeof(Element) * bK1 * pSK1 * (bM + bN1),
          sizeof(Element) * bM * bN1), MAX_ALIGNMENT));
      }
    }();
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
    const auto processorBlocks = cute::ceil_div(S * k, bM) * ((I / bN0) + (H / bN1));
    const auto dispatchBlocks = cute::min(dispatchSuperBlockSize(E) * E, MAX_DISPATCH_BLOCKS);
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
      const size_t& s, const size_t& h, const size_t& i, const size_t& e, const size_t& ec,
      const int& arch, const MLPMatmulType& m, const int& bM, const float& swishAlpha, const float& swishBeta,
      const bool check = true,
      const cuda::std::byte* const& expert_up_sf = nullptr, const cuda::std::byte* const& expert_up_v_sf = nullptr,
      const cuda::std::byte* const& expert_down_sf = nullptr)
      : tokens(tokens),
        expertUpWeights(expert_up_weights),
        expertUpVWeights(expert_up_v_weights),
        expertUpSF(expert_up_sf), expertUpVSF(expert_up_v_sf),
        biasUp(bias_up), biasUpV(bias_up_v),
        expertDownWeights(expert_down_weights),
        expertDownSF(expert_down_sf),
        biasDown(bias_down),
        expertCounts(expert_counts),
        moeOut(moe_out),
        S(static_cast<uint>(s)),
        H(static_cast<uint>(h)),
        I(static_cast<uint>(i)),
        E(static_cast<uint>(e)),
        EC(static_cast<uint>(ec)),
        flagColStride(cute::ceil_div(EC, bM) * E),
        swishAlpha(swishAlpha), swishBeta(swishBeta) {
      if (check) {
        const auto supports32 = arch >= 1000;
        checkAlignment(tokens, supports32);
        checkAlignment(expert_up_weights);
        if (m == MLPMatmulType::gated) {
          checkAlignment(expert_up_v_weights);
          checkAlignment(bias_up_v);
        }
        checkAlignment(expert_down_weights);
        // MXFP8: the weights are e4m3 and these are their scale factors
        if (expert_up_sf != nullptr) {
          checkAlignment(expert_up_sf);
          if (m == MLPMatmulType::gated) {
            checkAlignment(expert_up_v_sf);
          }
          checkAlignment(expert_down_sf);
        }
        checkAlignment(bias_up);
        checkAlignment(bias_down);
        checkAlignment(expert_counts);
        checkAlignment(moe_out);
      }
    }

    const cuda::std::byte* const tokens; // [S, H]
    // MXFP8: the weights are e4m3 (1 byte per element), with the same shapes. Each [N, K] weight matrix is K-major,
    // i.e., row-major with N rows, and has one UE8M0 scale per 32 consecutive K elements, stored in the
    // canonical layout of a [N, K / 32] matrix (mx::sfOffset) right after one another.
    const cuda::std::byte* const expertUpWeights; // [num_local_experts, H, I]
    const cuda::std::byte* const expertUpVWeights; // [num_local_experts, H, I]
    const cuda::std::byte* const expertUpSF; // MXFP8: [num_local_experts, mx::sfBytes(I, H / 32)]
    const cuda::std::byte* const expertUpVSF; // MXFP8: [num_local_experts, mx::sfBytes(I, H / 32)]
    const cuda::std::byte* const biasUp; // [num_local_experts, I]
    const cuda::std::byte* const biasUpV; // [num_local_experts, I]
    const cuda::std::byte* const expertDownWeights; // [num_local_experts, I, H]
    const cuda::std::byte* const expertDownSF; // MXFP8: [num_local_experts, mx::sfBytes(H, I / 32)]
    const cuda::std::byte* const biasDown; // [num_local_experts, H]
    const int* const expertCounts; // [E]
    cuda::std::byte* const moeOut; //  [S, H]
    const uint S; // sequence length
    const uint H; // token hidden dimension
    const uint I; // FFN intermediate size
    const uint E; // total number of experts
    const uint EC; // expert capacity
    const uint flagColStride; // ceil(EC / bM) * E
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
    const auto roundEC = cute::ceil_div(ctx.EC, bM) * bM;
    constexpr auto recipe = Config::QR::value;
    const auto symHeap = Heap{
      ctx.symHeap, ctx.nLx, roundEC, kArgs.H, sizeof(DataType),
      recipe == QuantRecipe::mxfp8 ? ctx.symHeapSFOffset : 0
    };
    const auto processors = gridDim.x - 1;
    const auto dispatchProcessors = cute::max((processors * 3) / 4, 1); // 75%
    const auto superBlockSize = min(dispatchSuperBlockSize(kArgs.E), dispatchProcessors);
    const auto dispatchBlocks = cuda::round_down(cute::min(superBlockSize * kArgs.E, MAX_DISPATCH_BLOCKS, dispatchProcessors), superBlockSize);
    if (blockIdx.x == gridDim.x - 1) {
      // call OS
      constexpr auto subscriberCount = threads - scheduler::SCHEDULER_COUNT;
      static_assert(subscriberCount > 0 && subscriberCount % WARP_SIZE == 0);
      os::start<topo, subscriberCount, threads, bM, Config::PSS::value, typename Config::StagingType>(flashWorkspace,
        kArgs.expertCounts, symHeap, ctx, kArgs.EC,kArgs.I / bN0, kArgs.H / bN1, dispatchBlocks,
        kArgs.E, kArgs.I, processors);
      return;
    }
    const auto stateNumber = ctx.stateNumbers[blockIdx.x];
    if (blockIdx.x < dispatchBlocks) {
      // dispatch
      dispatch<topo, Config::Threads::value, bM, bN0, recipe>(kArgs.H, kArgs.E, symHeap, kArgs.EC, roundEC,
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
    const auto mxA = processor::MXFP8Args{
      reinterpret_cast<const uint8_t*>(kArgs.expertUpSF),
      reinterpret_cast<const uint8_t*>(kArgs.expertUpVSF),
      reinterpret_cast<const uint8_t*>(kArgs.expertDownSF),
      reinterpret_cast<uint8_t*>(ctx.GEMM0StagingSF),
      ctx.nLx
    };
    // processor
    processor::start<Config::MT::value, topo, threads, Config::CM::value, TileGEMM0, TileGEMM1, GEMM0Act, recipe>
    (flashWorkspace, kArgs.S, kArgs.H, kArgs.I, roundEC, kArgs.flagColStride, tilesN0, tilesN1,
      expertUp, expertUpV,  biasUp, biasUpV,
      static_cast<AccumType>(kArgs.swishAlpha), static_cast<AccumType>(kArgs.swishBeta),
      expertDown, biasDown, ctx.tokenIndices, moeOut, producerBM, stateNumber, symHeap, pA, mxA);
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
