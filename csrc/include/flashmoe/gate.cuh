/******************************************************************************
 * Copyright (c) 2026, Osayamen Jonathan Aimuyo.
 ******************************************************************************/
//
// Created by Osayamen on 11/25/24.
//

#ifndef GATE_CUH
#define GATE_CUH

#include <cub/cub.cuh>

#include "context.cuh"
#include "infra/atomics.cuh"
#include "infra/packed.cuh"
#include "tile.cuh"
#include "infra/vt.cuh"

namespace flashmoe
{
  enum class SoftMaxOptimizationLevel {
    none,
    // use fast exponential and float division
    highest,
  };

  enum class GateReductionLevel {
    singleBlock,
    multiBlock
  };
}

namespace flashmoe::gate
{
  constexpr int BLOCK_N_CAP = 32;
  enum class ReturnLogits {
    yes,
    no
  };
  template <SoftMaxOptimizationLevel level = SoftMaxOptimizationLevel::none>
  __device__ __forceinline__
  float fexp(const float& x) {
    static_assert(level == SoftMaxOptimizationLevel::none);
    return expf(x);
  }

  template <>
  __device__ __forceinline__
  float fexp<SoftMaxOptimizationLevel::highest>(const float& x) {
    return __expf(x);
  }

  template <SoftMaxOptimizationLevel level = SoftMaxOptimizationLevel::none>
  __device__ __forceinline__
  float fdiv(const float& x, const float& y) {
    static_assert(level == SoftMaxOptimizationLevel::none);
    return fdividef(x, y);
  }

  template <>
  __device__ __forceinline__
  float fdiv<SoftMaxOptimizationLevel::highest>(const float& x, const float& y) {
    return __fdividef(x, y);
  }

  template <int threads>
  using BlockScan = cub::BlockScan<int, threads, cub::BLOCK_SCAN_WARP_SCANS>;
  using SoftType = float;
  template<int bM, int bN, int bK, int pSK, typename Element>
  constexpr auto kernelSMEM() {
    return cutlass::round_up(cute::max(sizeof(Element) * bK * pSK * (bM + bN),
    sizeof(flashmoe::gate::SoftType) * bM * bN), flashmoe::MAX_ALIGNMENT);
  }

  struct GateKernelArgs {
    const cuda::std::byte* const tokens = nullptr;
    const cuda::std::byte* const weights = nullptr;
    cuda::std::byte* const routing = nullptr;
    int* const expertCounts = nullptr;
    TPS* const tokenIds = nullptr;
    const int S = 0;
    const int H = 0;
    const int E = 0;
    const int k = 0;
    const int EC = 0;
    const int roundEC = 0;
  };

  template<int bM, int bN>
  __device__ __forceinline__
  void stageTile(const GateKernelArgs& kArgs, const GateContext& ctx,
                 const int tileM, const int tileN, const int (&selected)[bN],
                 const SoftType (&probabilities)[bN], const SoftType combineWeight) {
    const int token = tileM * bM + threadIdx.x;
    if (threadIdx.x >= bM || token >= kArgs.S) {
      return;
    }
    #pragma unroll
    for (int i = 0; i < bN; ++i) {
      const int expert = tileN * bN + i;
      ctx.staging[static_cast<size_t>(expert) * kArgs.S + token] = selected[i]
        ? TPS{static_cast<uint32_t>(token), fdividef(probabilities[i], combineWeight)}
        : TPS{cuda::std::numeric_limits<uint32_t>::max(), 0.f};
    }
  }

  constexpr int capacityThreads = 128;
  constexpr int capacityItems = 8;

  static __global__ void capacityKernel(const __grid_constant__ GateKernelArgs kArgs,
                                       const TPS* staging) {
    __shared__ BlockScan<capacityThreads>::TempStorage scan;
    const int expert = blockIdx.x;
    int total = 0;
    for (int base = 0; base < kArgs.S; base += capacityThreads * capacityItems) {
      TPS records[capacityItems];
      int selected[capacityItems], positions[capacityItems];
      #pragma unroll
      for (int i = 0; i < capacityItems; ++i) {
        const int token = base + threadIdx.x * capacityItems + i;
        records[i] = token < kArgs.S
          ? staging[static_cast<size_t>(expert) * kArgs.S + token]
          : TPS{cuda::std::numeric_limits<uint32_t>::max(), 0.f};
        // A selected record may legitimately have probability zero.
        selected[i] = records[i].tokenIdx != cuda::std::numeric_limits<uint32_t>::max();
      }
      int aggregate;
      BlockScan<capacityThreads>(scan).ExclusiveSum(selected, positions, aggregate);
      #pragma unroll
      for (int i = 0; i < capacityItems; ++i) {
        const int slot = total + positions[i];
        if (selected[i] && slot < kArgs.EC) {
          kArgs.tokenIds[static_cast<size_t>(expert) * kArgs.roundEC + slot] = records[i];
        }
      }
      total += aggregate;
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      kArgs.expertCounts[expert] = total;
    }
  }

  /// Fused GEMM, softmax, topKMask, assuming blocks >= tiles.N
  template <
    GateReductionLevel g,
    typename TileGEMM,
    SoftMaxOptimizationLevel sro,
    ReturnLogits r,
    typename Element,
    typename ElementC
  >
  struct GateMainloop {
    static_assert(g == GateReductionLevel::multiBlock);
    __device__ __forceinline__
    void operator()(void* __restrict__ const& gateScratch, const GateKernelArgs& kArgs, const GateContext& ctx, const int& tileIdx) {
      constexpr int threads = TileGEMM::Threads::value;
      constexpr TileGEMM tileMainloop{};
      auto accumulator = TileGEMM::BLAS::suggest_accumulator();
      constexpr auto bM = cute::get<0>(typename TileGEMM::TileShape{});
      constexpr auto bN = cute::get<1>(typename TileGEMM::TileShape{});
      constexpr auto bK = cute::get<2>(typename TileGEMM::TileShape{});
      const auto tilesM = cute::ceil_div(kArgs.S, bM);
      // assert(E % bN == 0)
      const auto tilesN = kArgs.E / bN;
      // assert(H % bK == 0)
      const auto tilesK = kArgs.H / bK;
      const auto tileCoord = tile::idx2Coord(tilesM, tilesN, tileIdx);
      static_assert(cuda::std::numeric_limits<SoftType>::has_infinity);
      // compute tile
      tileMainloop(gateScratch,
        reinterpret_cast<const Element*>(kArgs.tokens),
        reinterpret_cast<const Element*>(kArgs.weights), accumulator, kArgs.S, kArgs.E, kArgs.H, tileCoord);
      __syncthreads();

      /// Pointers for flags needed in epilogue
      /// col-major indexing to facilitate coalescing
      constexpr int phases = 2;
      const auto myTileOffset = bM * (cute::get<0>(tileCoord) + cute::get<1>(tileCoord) * tilesM) + threadIdx.x;
      const auto nextTileOffset = bM * (cute::get<0>(tileCoord) +
        (cute::get<1>(tileCoord) + 1 == tilesN ? 0 : cute::get<1>(tileCoord) + 1) * tilesM) + threadIdx.x;

      // Block Ring SoftMax pointers
      auto* __restrict__ brsMailbox = ctx.ssp + myTileOffset;
      auto* __restrict__ brsXMailbox = ctx.ssp + nextTileOffset;

      // Block Ring top k pointers
      const auto myTileOffsetP = bM * (cute::get<0>(tileCoord) + phases * cute::get<1>(tileCoord) * tilesM) +
        threadIdx.x;
      const auto nextTileOffsetP = bM * (cute::get<0>(tileCoord) +
          phases * (cute::get<1>(tileCoord) + 1 == tilesN ? 0 : cute::get<1>(tileCoord) + 1) * tilesM) +
        threadIdx.x;
      auto* __restrict__ tkMailbox = ctx.rtp + myTileOffsetP;
      auto* __restrict__ tkXMailbox = ctx.rtp + nextTileOffsetP;

      /// Epilogue -> RingSoftmax + topk + routing table construction
      // assumes row-major as we would be performing the softmax on the row
      using AccumType = decltype(accumulator)::value_type;
      auto sC = cublasdx::make_tensor(
        cute::make_smem_ptr(static_cast<AccumType*>(gateScratch)),
        cute::Layout<cute::Shape<cute::Int<bM>, cute::Int<bN>>,
                     cute::Stride<cute::Int<bN>, cute::_1>>{});
      // vectorized smem layout
      using VTD = VectorTypeDescriptor<AccumType, ElementAlignment<AccumType, bN>>;
      using VT = VTD::VectorType;
      constexpr int vectorWidth = VTD::VectorWidth::value;
      static_assert(bN % vectorWidth == 0);
      constexpr int vbN = bN / VTD::VectorWidth::value;
      // assumes row-major
      auto vsC = cublasdx::make_tensor(
        cute::make_smem_ptr(static_cast<VT*>(gateScratch)),
        cute::Layout<cute::Shape<cute::Int<bM>, cute::Int<vbN>>,
                     cute::Stride<cute::Int<vbN>, cute::_1>>{});
      const auto c_frag = accumulator.get_results();
      constexpr Converter<SoftType, AccumType> softLoad{};
      constexpr Converter<ElementC, AccumType> softStore{};
      // rmem -> smem
      cublasdx::copy_fragment<TileGEMM::CAlign::value>(c_frag, sC, accumulator);
      __syncthreads();
      // TODO relax this
      static_assert(threads >= bM);
      SoftType reginald[bN];
      // using notation from https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf
      auto dI = static_cast<SoftType>(0.f);
      /// Below needed for assigning -infinity
      /// See https://stackoverflow.com/a/20016972
      static_assert(cuda::std::numeric_limits<SoftType>::is_iec559, "IEEE 754 required");
      static_assert(cuda::std::numeric_limits<SoftType>::has_infinity);
      auto mI = -cuda::std::numeric_limits<SoftType>::infinity();
      // Begin Ring softmax
      const auto predicate = cute::min(kArgs.S, bM);
      if (threadIdx.x < predicate) {
        #pragma unroll
        for (int i = 0; i < vbN; ++i) {
          // smem -> rmem in blocked format
          // each thread owns a row
          const auto v = vsC(threadIdx.x, i);
          // unpack
          #pragma unroll
          for (int j = 0; j < vectorWidth; ++j) {
            reginald[j + i * vectorWidth] = softLoad(v[j]);
          }
        }
        if (cute::get<1>(tileCoord) > 0) {
          // await payload
          auto payload = cuda::std::bit_cast<SoftmaxStatePacked>(
            atomicExch(reinterpret_cast<ull_t*>(brsMailbox), 0UL));
          while (!has_payload_arrived(payload)) {
            payload = cuda::std::bit_cast<SoftmaxStatePacked>(
              atomicExch(reinterpret_cast<ull_t*>(brsMailbox), 0UL));
          }
          unpack_state(payload, mI, dI);
        }
        /// Softmax Reduction
        #pragma unroll
        for (int j = 0; j < bN; ++j) {
          const auto pM = mI;
          mI = max(mI, reginald[j]);
          dI = fmaf(dI, fexp<sro>(pM - mI), fexp<sro>(reginald[j] - mI));
        }
        if (cute::get<1>(tileCoord) + 1 < tilesN) {
          const auto sP = pack_state(mI, dI);
          // signal payload
          atomicExch(reinterpret_cast<ull_t*>(brsXMailbox), cuda::std::bit_cast<ull_t>(sP));
          // await payload
          auto* __restrict__ receiveBox = reinterpret_cast<ull_t*>(brsMailbox);
          auto payload = cuda::std::bit_cast<SoftmaxStatePacked>(atomicExch(receiveBox, 0UL));
          while (!has_payload_arrived(payload)) {
            payload = cuda::std::bit_cast<SoftmaxStatePacked>(atomicExch(receiveBox, 0UL));
          }
          unpack_state(payload, mI, dI);
        }
        else {
          // Ring ends with me, let's unblock everyone else
          const auto sP = pack_state(mI, dI);
          for (int j = 0; j < tilesN - 1; ++j) {
            // signal payload
            atomicExch(reinterpret_cast<ull_t*>(brsXMailbox + bM * j * tilesM),
                       cuda::std::bit_cast<ull_t>(sP));
          }
        }
        #pragma unroll
        for (int j = 0; j < bN; ++j) {
          reginald[j] = fdiv<sro>(fexp<sro>(reginald[j] - mI), dI);
        }
        #pragma unroll
        for (int i = 0; i < vbN; ++i) {
          auto v = VT{};
          // pack
          // ideally the compiler would not emit any MOV instructions here
          #pragma unroll
          for (int j = 0; j < vectorWidth; ++j) {
            v[j] = softStore(reginald[j + i * vectorWidth]);
          }
          // rmem -> smem
          vsC(threadIdx.x, i) = v;
        }
      }
      __syncthreads();
      // Ring Softmax is complete
      // smem -> gmem
      if constexpr (r == ReturnLogits::yes) {
        auto gC = tile::getC<bM, bN, TileGEMM::CArr::value>(reinterpret_cast<ElementC*>(kArgs.routing), kArgs.S, kArgs.E,
        cute::select<0, 1>(tileCoord));
        const auto tsC = cublasdx::make_tensor(
          static_cast<ElementC*>(gateScratch), TileGEMM::BLAS::get_layout_smem_c());
        static_assert(cute::is_compatible<decltype(gC.layout()), decltype(tsC.layout())>::value);
        cublasdx::copy<TileGEMM::BLAS, ElementAlignment<ElementC, bN>>(tsC, gC);
      }
      int rTK[bN];
      #pragma unroll
      for (int i = 0; i < bN; ++i) {
        rTK[i] = 0;
      }
      // sum of the combine weights per token
      auto mCw = static_cast<SoftType>(0.f);
      auto sV = -cuda::std::numeric_limits<SoftType>::infinity();
      uint32_t sIdx = 0;
      auto lSV = sV;
      auto lSIdx = sIdx;
      // Now do online top-k mask
      if (threadIdx.x < predicate) {
        bool shouldSweep = true;
        for (int i = 0; i < kArgs.k; ++i) {
          // needed as we alternate between two buffers
          const auto flagPrefix = i % phases * bM * tilesM;
          if (shouldSweep) {
            #pragma unroll
            for (int j = 0; j < bN; ++j) {
              // find local maximum
              if (!rTK[j] && reginald[j] > lSV) {
                lSIdx = cute::get<1>(tileCoord) * bN + j;
                lSV = reginald[j];
              }
            }
            shouldSweep = false;
          }
          // Strictly applies to the most westwards peer, as they initiate the proposal per round
          sV = lSV;
          sIdx = lSIdx;
          if (cute::get<1>(tileCoord) > 0) {
            // await payload
            auto* __restrict__ receiveBox = reinterpret_cast<ull_t*>(tkMailbox + flagPrefix);
            auto payload = cuda::std::bit_cast<RingTopKPayload>(atomicExch(receiveBox, 0UL));
            while (!has_payload_arrived(payload)) {
              payload = cuda::std::bit_cast<RingTopKPayload>(atomicExch(receiveBox, 0UL));
            }
            unpack_tk_payload(payload, sV, sIdx);
            if (lSV > sV) {
              //we either relay the received values or propagate our proposal
              sV = lSV;
              sIdx = lSIdx;
            }
          }
          // Every tile except the most eastwards
          if (cute::get<1>(tileCoord) + 1 < tilesN) {
            // propagate our proposal
            // Now we pass our proposal through the ring
            atomicExch(reinterpret_cast<ull_t*>(tkXMailbox + flagPrefix),
                       cuda::std::bit_cast<ull_t>(pack_tk_payload(sV, sIdx)));
            // Now we await the final results to return
            auto* __restrict__ receiveBox = reinterpret_cast<ull_t*>(tkMailbox + flagPrefix);
            auto payload = cuda::std::bit_cast<RingTopKPayload>(atomicExch(receiveBox, 0UL));
            while (!has_payload_arrived(payload)) {
              payload = cuda::std::bit_cast<RingTopKPayload>(atomicExch(receiveBox, 0UL));
            }
            unpack_tk_payload(payload, sV, sIdx);
          }
          else {
            // Phase 0 ends with me, let's unblock everyone else in one go
            const auto payload = cuda::std::bit_cast<ull_t>(pack_tk_payload(sV, sIdx));
            auto* __restrict__ mailboxes = tkXMailbox;
            for (int j = 0; j < tilesN - 1; ++j) {
              atomicExch(reinterpret_cast<ull_t*>(mailboxes + flagPrefix), payload);
              mailboxes += phases * bM * tilesM;
            }
          }
          if (sIdx / bN == cute::get<1>(tileCoord)) {
            // Our proposal won in this round!
            // let's update the mask
            const int sIdxIntra = sIdx % bN;
            #pragma unroll
            for (int j = 0; j < bN; ++j) {
              if (j == sIdxIntra) {
                rTK[j] = 1;
              }
            }
            // We need to sweep in the next round
            shouldSweep = true;
            lSV = -cuda::std::numeric_limits<SoftType>::infinity();
          }
          mCw += sV;
        }
      }
      stageTile<bM, bN>(kArgs, ctx, cute::get<0>(tileCoord),
                       cute::get<1>(tileCoord), rTK, reginald, mCw);
    }
  };

  // Special, nice case where N == BLOCK_N
  template <
    typename TileGEMM,
    SoftMaxOptimizationLevel sro,
    ReturnLogits r,
    typename Element,
    typename ElementC
  >
  struct GateMainloop<GateReductionLevel::singleBlock, TileGEMM, sro, r, Element, ElementC> {
    __device__ __forceinline__
    void operator()(void* __restrict__ const& gateScratch, const GateKernelArgs& kArgs, const GateContext& ctx, const int& tileIdx) {
      constexpr int threads = TileGEMM::Threads::value;
      constexpr TileGEMM tileMainloop{};
      auto accumulator = TileGEMM::BLAS::suggest_accumulator();
      constexpr auto bM = cute::get<0>(typename TileGEMM::TileShape{});
      constexpr auto bN = cute::get<1>(typename TileGEMM::TileShape{});
      constexpr auto bK = cute::get<2>(typename TileGEMM::TileShape{});
      const auto tilesM = cute::ceil_div(kArgs.S, bM);
      constexpr int tilesN = 1;
      const auto tilesK = kArgs.H / bK;
      const auto tileCoord = tile::idx2Coord(tilesM, tilesN, tileIdx);
      static_assert(cuda::std::numeric_limits<ElementC>::has_infinity);
      // compute tile
      tileMainloop(gateScratch,
        reinterpret_cast<const Element*>(kArgs.tokens),
        reinterpret_cast<const Element*>(kArgs.weights), accumulator, kArgs.S, kArgs.E, kArgs.H, tileCoord);
      __syncthreads();

      /// Epilogue -> softmax + topk + routing table construction
      // Begin online softmax
      // using notation from https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf
      // do all computation in float for stability
      auto dI = static_cast<SoftType>(0.f);
      auto mI = -cuda::std::numeric_limits<SoftType>::infinity();
      // assumes row-major as we would be performing the softmax on the row
      using AccumType = decltype(accumulator)::value_type;
      auto sC = cublasdx::make_tensor(
        cute::make_smem_ptr(static_cast<AccumType*>(gateScratch)),
        cute::Layout<cute::Shape<cute::Int<bM>, cute::Int<bN>>,
                     cute::Stride<cute::Int<bN>, cute::_1>>{});
      // vectorized smem layout
      using VTD = VectorTypeDescriptor<AccumType, ElementAlignment<AccumType, bN>>;
      using VT = VTD::VectorType;
      constexpr int vectorWidth = VTD::VectorWidth::value;
      static_assert(bN % vectorWidth == 0);
      constexpr int vbN = bN / VTD::VectorWidth::value;
      // assumes row-major
      auto vsC = cublasdx::make_tensor(
        cute::make_smem_ptr(static_cast<VT*>(gateScratch)),
        cute::Layout<cute::Shape<cute::Int<bM>, cute::Int<vbN>>,
                     cute::Stride<cute::Int<vbN>, cute::_1>>{});
      const auto c_frag = accumulator.get_results();
      constexpr Converter<SoftType, AccumType> softLoad{};
      constexpr Converter<ElementC, AccumType> softStore{};
      // rmem -> smem
      cublasdx::copy_fragment<TileGEMM::CAlign::value>(c_frag, sC, accumulator);
      __syncthreads();
      // TODO relax this
      static_assert(threads >= bM);
      SoftType reginald[bN];
      const auto predicate = cute::min(kArgs.S, bM);
      if (threadIdx.x < predicate) {
        #pragma unroll
        for (int i = 0; i < vbN; ++i) {
          // smem -> rmem in blocked format
          // each thread owns a row
          const auto v = vsC(threadIdx.x, i);
          // unpack
          #pragma unroll
          for (int j = 0; j < vectorWidth; ++j) {
            reginald[j + i * vectorWidth] = softLoad(v[j]);
          }
        }
        /// Softmax Reduction
        #pragma unroll
        for (int j = 0; j < bN; ++j) {
          const auto pM = mI;
          mI = max(mI, reginald[j]);
          dI = fmaf(dI, fexp<sro>(pM - mI), fexp<sro>(reginald[j] - mI));
        }
        #pragma unroll
        for (int j = 0; j < bN; ++j) {
          reginald[j] = fdiv<sro>(fexp<sro>(reginald[j] - mI), dI);
        }
        #pragma unroll
        for (int i = 0; i < vbN; ++i) {
          auto v = VT{};
          // pack
          // ideally the compiler would not emit any MOV instructions here
          #pragma unroll
          for (int j = 0; j < vectorWidth; ++j) {
            v[j] = softStore(reginald[j + i * vectorWidth]);
          }
          // rmem -> smem
          vsC(threadIdx.x, i) = v;
        }
      }
      __syncthreads();
      if constexpr (r == ReturnLogits::yes) {
        // smem -> gmem
        auto gC = tile::getC<bM, bN, TileGEMM::CArr::value>(reinterpret_cast<ElementC*>(kArgs.routing),
          kArgs.S, kArgs.E,cute::select<0, 1>(tileCoord));
        const auto tsC = cublasdx::make_tensor(
          static_cast<ElementC*>(gateScratch), TileGEMM::BLAS::get_layout_smem_c());
        static_assert(cute::is_compatible<decltype(gC.layout()), decltype(tsC.layout())>::value);
        cublasdx::copy<TileGEMM::BLAS, ElementAlignment<ElementC, bN>>(tsC, gC);
      }
      int rTK[bN];
      #pragma unroll
      for (int i = 0; i < bN; ++i) {
        rTK[i] = 0;
      }
      // sum of the combine weights per token
      auto mCw = static_cast<SoftType>(0.f);
      // Now do online top-k mask
      if (threadIdx.x < predicate) {
        for (int i = 0; i < kArgs.k; ++i) {
          auto sV = -cuda::std::numeric_limits<SoftType>::infinity();
          int sIdx = 0;
          #pragma unroll
          for (int j = 0; j < bN; ++j) {
            if (reginald[j] > sV && !rTK[j]) {
              sIdx = j;
              sV = reginald[j];
            }
          }
          #pragma unroll
          for (int j = 0; j < bN; ++j) {
            if (j == sIdx) {
              rTK[j] = 1; // set the bit
            }
          }
          mCw += sV;
        }
      }
      stageTile<bM, bN>(kArgs, ctx, cute::get<0>(tileCoord),
                       cute::get<1>(tileCoord), rTK, reginald, mCw);
    }
  };

  template <
    typename TileGEMM,
    GateReductionLevel grl = GateReductionLevel::singleBlock,
    SoftMaxOptimizationLevel sro = SoftMaxOptimizationLevel::none,
    ReturnLogits r = ReturnLogits::yes,
    typename Element,
    typename ElementR
  >
  __device__ __forceinline__
  void stage(void* __restrict__ const& workspace, const GateKernelArgs& kArgs, const GateContext& ctx) {
    // assert(blocks >= E / bN)
    using TileShape = TileGEMM::TileShape;
    constexpr int bM = cute::get<0>(TileShape{});
    constexpr int bN = cute::get<1>(TileShape{});
    GateMainloop<grl, TileGEMM, sro, r, Element, ElementR> gateMainLoop{};
    const auto nT = cute::ceil_div(kArgs.S, bM) * (kArgs.E / bN);
    for (int tileIdx = static_cast<int>(blockIdx.x); tileIdx < nT; tileIdx += static_cast<int>(gridDim.x)) {
      gateMainLoop(workspace, kArgs, ctx, tileIdx);
    }
  }

  template<
        typename TileShape,
        int Arch,
        int threads,
        GateReductionLevel grl = GateReductionLevel::singleBlock,
        SoftMaxOptimizationLevel sro = SoftMaxOptimizationLevel::none,
        ReturnLogits r = ReturnLogits::yes,
        typename AccumType,
        typename Element,
        typename ElementR = float
    >
  __launch_bounds__(threads, 1)
  __global__ void stageKernel(const __grid_constant__ GateKernelArgs kArgs, const __grid_constant__ GateContext ctx) {
    constexpr int bM = cute::get<0>(TileShape{});
    constexpr int bN = cute::get<1>(TileShape{});
    constexpr int bK = cute::get<2>(TileShape{});
    constexpr int pipeStages = cute::get<3>(TileShape{});
    using TileGEMM = tile::CollectiveMainloop<bM, bN, bK, Arch, Element, AccumType, threads, pipeStages>;
    extern __shared__ __align__(TileGEMM::GeneralAlignment::value) cuda::std::byte gateWorkspace[];
    gate::stage<TileGEMM, grl, sro, r, Element, ElementR>(gateWorkspace, kArgs, ctx);
  }

  template<typename TileShape, int Arch, int threads,
           GateReductionLevel grl = GateReductionLevel::singleBlock,
           SoftMaxOptimizationLevel sro = SoftMaxOptimizationLevel::none,
           ReturnLogits r = ReturnLogits::yes,
           typename AccumType, typename Element, typename ElementR = float>
  __host__ __forceinline__
  cudaError_t forward(const GateKernelArgs& kArgs, const GateContext& ctx,
                      int blocks, int sharedBytes, cudaStream_t stream) {
    stageKernel<TileShape, Arch, threads, grl, sro, r, AccumType, Element, ElementR>
      <<<blocks, threads, sharedBytes, stream>>>(kArgs, ctx);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) {
      return error;
    }
    capacityKernel<<<kArgs.E, capacityThreads, 0, stream>>>(kArgs, ctx.staging);
    return cudaGetLastError();
  }
}
#endif //GATE_CUH
