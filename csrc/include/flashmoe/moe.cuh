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
    // mxfp8: E4M3 weights and activations; mxfp4, nvfp4: E2M1 weights and activations.
    // Bias, output and Element stay bf16
    QuantRecipe qr = QuantRecipe::none
  >
  struct MoEConfig {
    static_assert(cute::is_tuple_v<GEMM0Tile> && cute::rank_v<GEMM0Tile> == 4);
    using G0TS = GEMM0Tile;
    static_assert(cute::is_tuple_v<GEMM1Tile> && cute::rank_v<GEMM1Tile> == 4);
    using G1TS = GEMM1Tile;
    static_assert(cute::get<0>(GEMM0Tile{}) == cute::get<0>(GEMM1Tile{}));
    static_assert(!isBlockScaled(qr) || (cuda::std::is_same_v<Element, __nv_bfloat16> &&
      cute::get<0>(GEMM0Tile{}) == mx::SF_ROW_TILE), "Block-scaled recipes require bf16 activations and bM == 128");
    using Arch = cute::Int<arch>;
    using Threads = cute::Int<_threads>;
    using CM = cute::C<cm>;
    using MT = cute::C<mt>;
    using QR = cute::C<qr>;
    using DType = Element;
    // element type of GEMM0Staging, i.e., of the GEMM0 output and the GEMM1 input
    using StagingType = cuda::std::conditional_t<qr == QuantRecipe::mxfp8, __nv_fp8_e4m3,
      cuda::std::conditional_t<isFP4(qr), cutlass::float_e2m1_t, Element>>;
    // GEMM0Staging is sized in bits, see bootstrap::initialize and the subscriber
    static_assert(!isBlockScaled(qr) || cute::sizeof_bits_v<StagingType> == quantElementBits(qr));
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

  /// For arrays that only need their natural alignment, e.g., the FP32 per-expert scales
  __host__ __forceinline__
  auto checkAlignmentBytes(const void *const&p, const size_t& alignment) {
    if (p == nullptr || !cuda::is_aligned(p, alignment)) {
      printf("Pointer is not %zu-byte aligned\n", alignment);
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
    // Block-scaled recipes: the mainloops have their own pipelines, see processor::start
    constexpr size_t GEMM0Sz = [] {
      if constexpr (Config::QR::value == QuantRecipe::mxfp8) {
        return static_cast<size_t>(tile::sm120::MXFP8GEMM0<bM, bN0, threads, mt == MLPMatmulType::gated>::sharedSize);
      }
      else if constexpr (isFP4(Config::QR::value)) {
        return static_cast<size_t>(tile::sm120::FP4GEMM0<Config::QR::value, bM, bN0, threads,
          mt == MLPMatmulType::gated>::sharedSize);
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
      else if constexpr (isFP4(Config::QR::value)) {
        return static_cast<size_t>(tile::sm120::FP4GEMM1<Config::QR::value, bM, bN1, threads>::sharedSize);
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
      const cuda::std::byte* const& expert_down_sf = nullptr,
      const float* const& expert_up_gs = nullptr, const float* const& expert_up_v_gs = nullptr,
      const float* const& expert_down_gs = nullptr,
      const float& input_scale = 1.f, const float& interim_scale = 1.f)
      : tokens(tokens),
        expertUpWeights(expert_up_weights),
        expertUpVWeights(expert_up_v_weights),
        expertUpSF(expert_up_sf), expertUpVSF(expert_up_v_sf),
        expertUpGS(expert_up_gs), expertUpVGS(expert_up_v_gs),
        biasUp(bias_up), biasUpV(bias_up_v),
        expertDownWeights(expert_down_weights),
        expertDownSF(expert_down_sf),
        expertDownGS(expert_down_gs),
        biasDown(bias_down),
        expertCounts(expert_counts),
        moeOut(moe_out),
        S(static_cast<uint>(s)),
        H(static_cast<uint>(h)),
        I(static_cast<uint>(i)),
        E(static_cast<uint>(e)),
        EC(static_cast<uint>(ec)),
        flagColStride(cute::ceil_div(EC, bM) * E),
        swishAlpha(swishAlpha), swishBeta(swishBeta),
        inputScale(input_scale), interimScale(interim_scale) {
      if (check) {
        const auto supports32 = arch >= 1000;
        checkAlignment(tokens, supports32);
        checkAlignment(expert_up_weights);
        if (m == MLPMatmulType::gated) {
          checkAlignment(expert_up_v_weights);
          checkAlignment(bias_up_v);
        }
        checkAlignment(expert_down_weights);
        // Block-scaled recipes: the weights are quantized and these are their scale factors
        if (expert_up_sf != nullptr) {
          checkAlignment(expert_up_sf);
          if (m == MLPMatmulType::gated) {
            checkAlignment(expert_up_v_sf);
          }
          checkAlignment(expert_down_sf);
        }
        // nvfp4: per-expert FP32 decode scales of the weights
        if (expert_up_gs != nullptr) {
          checkAlignmentBytes(expert_up_gs, alignof(float));
          if (m == MLPMatmulType::gated) {
            checkAlignmentBytes(expert_up_v_gs, alignof(float));
          }
          checkAlignmentBytes(expert_down_gs, alignof(float));
        }
        checkAlignment(bias_up);
        checkAlignment(bias_down);
        checkAlignment(expert_counts);
        checkAlignment(moe_out);
      }
    }

    const cuda::std::byte* const tokens; // [S, H]
    // Block-scaled recipes: the weights are quantized, with the same shapes: e4m3 (1 byte per element) for mxfp8,
    // e2m1 (2 elements per byte, the even K element in the low nibble) for mxfp4 and nvfp4.
    // Each [N, K] weight matrix is K-major, i.e., row-major with N rows, and has one scale per sfVec (quantSFVecSize)
    // consecutive K elements: UE8M0 per 32 (mxfp8, mxfp4) or UE4M3 per 16 (nvfp4), stored in the canonical layout of a
    // [N, K / sfVec] matrix (mx::sfOffset) right after one another.
    const cuda::std::byte* const expertUpWeights; // [num_local_experts, H, I]
    const cuda::std::byte* const expertUpVWeights; // [num_local_experts, H, I]
    const cuda::std::byte* const expertUpSF; // [num_local_experts, mx::sfBytes(I, H / sfVec)]
    const cuda::std::byte* const expertUpVSF; // [num_local_experts, mx::sfBytes(I, H / sfVec)]
    // nvfp4 only: FP32 decode scale of each expert's weight matrix: a dequantized element is
    // e2m1 * float(scale factor) * gs
    const float* const expertUpGS; // [num_local_experts]
    const float* const expertUpVGS; // [num_local_experts]
    const cuda::std::byte* const biasUp; // [num_local_experts, I]
    const cuda::std::byte* const biasUpV; // [num_local_experts, I]
    const cuda::std::byte* const expertDownWeights; // [num_local_experts, I, H]
    const cuda::std::byte* const expertDownSF; // [num_local_experts, mx::sfBytes(H, I / sfVec)]
    const float* const expertDownGS; // nvfp4 only: [num_local_experts]
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
    // nvfp4 only: static per-layer decode scales of the activations, i.e., a dequantized element is
    // e2m1 * float(scale factor) * inputScale (dispatched tokens) or * interimScale (GEMM0 output)
    const float inputScale = 1.f;
    const float interimScale = 1.f;
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
      isBlockScaled(recipe) ? ctx.symHeapSFOffset : 0,
      isBlockScaled(recipe) ? quantSFVecSize(recipe) : mx::BLOCK_SIZE
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
        ctx.tokenIndices, ctx.dispatchSync, ctx.pel, flashWorkspace, stateNumber,
        recipe == QuantRecipe::nvfp4 ? fp4::nvfp4EncodeScale(kArgs.inputScale) : 1.f);
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
    const auto mxA = processor::QuantArgs{
      reinterpret_cast<const uint8_t*>(kArgs.expertUpSF),
      reinterpret_cast<const uint8_t*>(kArgs.expertUpVSF),
      reinterpret_cast<const uint8_t*>(kArgs.expertDownSF),
      reinterpret_cast<uint8_t*>(ctx.GEMM0StagingSF),
      ctx.nLx,
      kArgs.expertUpGS, kArgs.expertUpVGS, kArgs.expertDownGS,
      kArgs.inputScale, kArgs.interimScale
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
