//
// Created by osayamen on 1/11/26.
//

#ifndef FLASHMOE_COMBINE_CUH
#define FLASHMOE_COMBINE_CUH
#include "infra/packed.cuh"
#include "tile.cuh"
#include "infra/rvt.cuh"
#include "infra/vt.cuh"

namespace flashmoe
{
  enum class CombineMode {
    single, // top k = 1
    plural // top k > 1
  };

  __device__ __forceinline__
  float2 float2Mul(const float2& a, const float2& b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    return __fmul2_rn(a, b);
#else
    return float2{a.x * b.x, a.y * b.y};
#endif
  }

  // MoE combine at tile granularity
  template <
    int bM,
    int bN,
    int Arch,
    int threads,
    CombineMode c,
    cublasdx::arrangement cArr = cublasdx::row_major,
    typename Element,
    typename TileCoord
  >
  __device__ __forceinline__
  void combine(const int& S, const int& H,
               void* __restrict__ const& workspace,
               const TPS* __restrict__ const& tokenIndices, // [bM]
               Element* __restrict__ const& moeOutput, // [S, H] in local HBM
               const Element* __restrict__ const& tokens, // [bM, H] in local HBM
               const uint& tileSize, const TileCoord& tileCoord) {
    static_assert(cute::rank_v<TileCoord> == 2);
    static_assert(decltype(cute::get<0>(tileCoord))::value == 0);
    static_assert(cArr == cublasdx::row_major);
    __shared__ TPS stIds[bM];
    using SCS = cuda::std::conditional_t<cArr == cublasdx::row_major, cute::Stride<cute::Int<bN>, cute::_1>,
                                         cute::Stride<cute::_1, cute::Int<bM>>>;
    using SCL = cute::Layout<cute::Shape<cute::Int<bM>, cute::Int<bN>>, SCS>;
    auto sC = cute::make_tensor(cute::make_smem_ptr(static_cast<Element*>(workspace)),
                                SCL{});
    const auto gC = tile::get<bM, bN, cArr>(tokens, bM, H, tileCoord);
    static_assert(cute::is_compatible<decltype(gC.layout()), decltype(sC.layout())>::value);

    #pragma unroll
    for (int i = threadIdx.x; i < bM; i += threads) {
      stIds[i] = tokenIndices[i];
    }
    // copy processed tile from gmem -> smem
    cublasdx::copy<threads, ElementAlignment<Element, bN>>(threadIdx.x, gC, sC);
    cublasdx::copy_wait();
    __syncthreads();
    if constexpr (c == CombineMode::single) {
      using VTD = VectorTypeDescriptor<Element, ElementAlignment<Element, bN>>;
      using VT = VTD::VectorType;
      constexpr auto vw = VTD::VectorWidth::value;
      constexpr auto vbN = bN / vw;
      constexpr auto nElems = vbN * bM;
      const auto actualElems = tileSize * vbN;
      constexpr auto elemsPerThread = nElems / threads;
      uint cache[cute::ceil_div(nElems, threads)]; // use ceil_div to avoid 0
      // row major output
      const auto vH = H / vw;
      auto mC = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<VT*>(moeOutput)),
                                  cute::make_layout(cute::make_shape(S, vH), cute::LayoutRight{}));
      // we enforce H % bN == 0 cpu-side
      auto tC = cute::local_tile(mC, cute::make_shape(S, cute::Int<vbN>{}), tileCoord);
      // row major layout
      using VSL = cute::Layout<cute::Shape<cute::Int<bM>, cute::Int<vbN>>,
                               cute::Stride<cute::Int<vbN>, cute::_1>>;
      const auto vsC = cute::make_tensor(cute::make_smem_ptr(static_cast<const VT*>(workspace)),
                                         VSL{});
      // vector copy results to gmem and call it a day
      #pragma unroll
      for (int i = 0; i < elemsPerThread; ++i) {
        const auto idx = threadIdx.x + i * threads;
        const auto rowIdx = idx / vbN;
        cache[i] = stIds[rowIdx].tokenIdx;
      }
      if (tileSize == bM) {
        #pragma unroll
        for (int i = 0; i < elemsPerThread; ++i) {
          const auto idx = threadIdx.x + i * threads;
          const auto rowIdx = idx / vbN;
          const auto colIdx = idx % vbN;
          const auto tokenIdx = cache[i];
          // smem -> gmem.
          tC(tokenIdx, colIdx) = vsC(rowIdx, colIdx);
        }
      }
      else {
        #pragma unroll
        for (int i = 0; i < elemsPerThread; ++i) {
          const auto idx = threadIdx.x + i * threads;
          const auto rowIdx = idx / vbN;
          const auto colIdx = idx % vbN;
          const auto tokenIdx = cache[i];
          // smem -> gmem.
          if (idx < actualElems) {
            tC(tokenIdx, colIdx) = vsC(rowIdx, colIdx);
          }
        }
      }
      constexpr auto residue = nElems - (elemsPerThread * threads);
      if constexpr (residue) {
        if (threadIdx.x < residue) {
          const auto idx = threadIdx.x + elemsPerThread * threads;
          const auto rowIdx = idx / vbN;
          const auto colIdx = idx % vbN;
          const auto tokenIdx = stIds[rowIdx].tokenIdx;
          // smem -> gmem.
          if (idx < actualElems) {
            tC(tokenIdx, colIdx) = vsC(rowIdx, colIdx);
          }
        }
      }
    }
    else {
      // we need to atomically reduce to the output buffer here
      // promotes to fp16x2 or bf16x2, if it can.
      using RAD = RedAddType<Element, ElementAlignment<Element, bN>>;
      using RAT = RAD::Type;
      constexpr int bNp = bN / RAD::Width::value;
      static_assert(RAD::Width::value == 1 || RAD::Width::value == 2);

      constexpr int maxVectorWidth = ElementAlignment<RAT, bNp> / sizeof(RAT);
      using RedAddOp = RedAdd<RedArch<Arch>, RAT, maxVectorWidth>;
      using RVD = VectorTypeDescriptor<RAT, RedAddOp::VectorWidth::value * sizeof(RAT)>;
      using RV = RVD::VectorType;

      constexpr int totalVecWidth = RVD::VectorWidth::value * RAD::Width::value;
      constexpr auto rbN = bN / totalVecWidth;
      using RSL = cute::Layout<cute::Shape<cute::Int<bM>, cute::Int<rbN>>,
                               cute::Stride<cute::Int<rbN>, cute::_1>>;
      const auto vsC = cute::make_tensor(cute::make_smem_ptr(static_cast<const RV*>(workspace)),
                                         RSL{});
      // row major output
      const auto vHo = H / RAD::Width::value;
      auto mC = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<RAT*>(moeOutput)),
                                  cute::make_layout(cute::make_shape(S, vHo), cute::LayoutRight{}));
      auto tC = cute::local_tile(mC, cute::make_shape(S, cute::Int<bNp>{}), tileCoord);

      constexpr auto totalElems = bM * rbN; // cublasdx::cosize(RSL{})
      const auto actualElems = tileSize * rbN;
      constexpr auto redElemsPerThread = totalElems / threads;
      constexpr int packWidth = RVD::VectorWidth::value;
      if (tileSize == bM) {
        for (int i = 0; i < redElemsPerThread; ++i) {
          const auto idx = threadIdx.x + i * threads;
          const auto rowIdx = idx / rbN;
          const auto colIdx = idx % rbN;
          const auto indexAndScale = stIds[rowIdx];
          auto tokenValue = vsC(rowIdx, colIdx);
          const auto tokIdx = indexAndScale.tokenIdx;
          if constexpr (RAD::Width::value == 2) {
            constexpr Converter<float2, RAT> loadOp{};
            constexpr Converter<RAT, float2> storeOp{};
            // fp16x2 or bf16x2
            const auto scale2 = float2{indexAndScale.probability, indexAndScale.probability};
            #pragma unroll
            for (int j = 0; j < packWidth; ++j) {
              // convert to float2 -> multiply -> convert back
              tokenValue[j] = storeOp(float2Mul(loadOp(tokenValue[j]), scale2));
            }
          }
          else {
            constexpr Converter<float, RAT> loadOp{};
            constexpr Converter<RAT, float> storeOp{};
            #pragma unroll
            for (int j = 0; j < packWidth; ++j) {
              // convert to float -> multiply -> convert back
              tokenValue[j] = storeOp(loadOp(tokenValue[j]) * indexAndScale.probability);
            }
          }
          // account for the fact that the type of the below tile is either Element or RAT.
          // Since we read 'packWidth' per iteration we need to advance the colIx by that much.
          // The row index is preserved because we only apply this vectorization within a row
          // which is contiguous in memory.
          auto* __restrict__ tCp = (&tC(tokIdx, colIdx * packWidth));
          constexpr RedAddOp op{};
          op(tCp, tokenValue);
        }
      }
      else {
        for (int i = 0; i < redElemsPerThread; ++i) {
          const auto idx = threadIdx.x + i * threads;
          if (idx < actualElems) {
            const auto rowIdx = idx / rbN;
            const auto colIdx = idx % rbN;
            const auto indexAndScale = stIds[rowIdx];
            auto tokenValue = vsC(rowIdx, colIdx);
            const auto tokIdx = indexAndScale.tokenIdx;
            if constexpr (RAD::Width::value == 2) {
              constexpr Converter<float2, RAT> loadOp{};
              constexpr Converter<RAT, float2> storeOp{};
              // fp16x2 or bf16x2
              const auto scale2 = float2{indexAndScale.probability, indexAndScale.probability};
              #pragma unroll
              for (int j = 0; j < packWidth; ++j) {
                // convert to float2 -> multiply -> convert back
                tokenValue[j] = storeOp(float2Mul(loadOp(tokenValue[j]), scale2));
              }
            }
            else {
              constexpr Converter<float, RAT> loadOp{};
              constexpr Converter<RAT, float> storeOp{};
              #pragma unroll
              for (int j = 0; j < packWidth; ++j) {
                // convert to float -> multiply -> convert back
                tokenValue[j] = storeOp(loadOp(tokenValue[j]) * indexAndScale.probability);
              }
            }
            auto* __restrict__ tCp = &tC(tokIdx, colIdx * packWidth);
            constexpr RedAddOp op{};
            op(tCp, tokenValue);
          }
        }
      }
      constexpr auto residue = totalElems - (redElemsPerThread * threads);
      if constexpr (residue) {
        if (threadIdx.x < residue) {
          const auto idx = threadIdx.x + redElemsPerThread * threads;
          const auto rowIdx = idx / rbN;
          const auto colIdx = idx % rbN;
          const auto indexAndScale = stIds[rowIdx];
          auto tokenValue = vsC(rowIdx, colIdx);
          const auto tokIdx = indexAndScale.tokenIdx;
          if (idx < actualElems) {
            if constexpr (RAD::Width::value == 2) {
              constexpr Converter<float2, RAT> loadOp{};
              constexpr Converter<RAT, float2> storeOp{};
              // fp16x2 or bf16x2
              const auto scale2 = float2{indexAndScale.probability, indexAndScale.probability};
              #pragma unroll
              for (int j = 0; j < packWidth; ++j) {
                // convert to float2 -> multiply -> convert back
                tokenValue[j] = storeOp(float2Mul(loadOp(tokenValue[j]), scale2));
              }
            }
            else {
              constexpr Converter<float, RAT> loadOp{};
              constexpr Converter<RAT, float> storeOp{};
              #pragma unroll
              for (int j = 0; j < packWidth; ++j) {
                // convert to float -> multiply -> convert back
                tokenValue[j] = storeOp(loadOp(tokenValue[j]) * indexAndScale.probability);
              }
            }
            auto* __restrict__ tCp = &tC(tokIdx, colIdx * packWidth);
            constexpr RedAddOp op{};
            op(tCp, tokenValue);
          }
        }
      }
    }
  }
}
#endif //FLASHMOE_COMBINE_CUH
