/******************************************************************************
 * Copyright (c) 2026, Osayamen Jonathan Aimuyo.
 ******************************************************************************/
//
// Created by FlashMoE contributors on 10/5/26.
//
// FP4 helpers for the two W4A4 recipes of QuantRecipe, both with E2M1 elements (two per byte, the even k in the low
// nibble, K-major, a row of K elements is K / 2 bytes) and the canonical 128x4 scale-factor (SF) layout of infra/mx.cuh:
//  - mxfp4: OCP microscaling, one UE8M0 scale per 32 elements along K,
//  - nvfp4: one UE4M3 scale per 16 elements along K and one FP32 scale per tensor gs, the decode scale: a tensor is
//           x ~ e2m1 * float(sf) * gs. st = 1 / gs is the encode scale. gs = amax(tensor) / (6 * 448) maps the largest
//           block scale to 448, the largest UE4M3 value; activations use a calibrated, static gs.
// Scale computation, element conversions and standalone quantize / dequantize kernels.

#ifndef FLASHMOE_FP4_CUH
#define FLASHMOE_FP4_CUH

#include <cfloat>
#include <cstdint>
#include <stdexcept>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cuda/std/bit>
#include <cuda/std/type_traits>

#include "mx.cuh"
#include "structures.cuh"

namespace flashmoe::fp4 {
  /// Largest finite E2M1 value
  constexpr float E2M1_MAX = 6.f;
  /// Bits of one element
  constexpr int ELEMENT_BITS = 4;

  /// Number of consecutive K elements that share one scale factor: 32 (mxfp4) or 16 (nvfp4)
  template <QuantRecipe r>
    requires(isFP4(r))
  consteval int sfVecSize() {
    return quantSFVecSize(r);
  }

  // ------------------------------------------------------------------------------------------------------------
  // E2M1 conversions
  // ------------------------------------------------------------------------------------------------------------

  /// Two floats -> two E2M1, round-to-nearest-even on the code and saturating to +-6
  /// (cvt.rn.satfinite.e2m1x2.f32 on sm_100a+/sm_120a, an equivalent software conversion elsewhere).
  /// lo is stored in the low nibble of the result and hi in the high nibble.
  __host__ __device__ __forceinline__
  uint8_t packE2M1x2(const float& lo, const float& hi) {
    return static_cast<uint8_t>(__nv_cvt_float2_to_fp4x2(make_float2(lo, hi), __NV_E2M1, cudaRoundNearest));
  }

  /// Two E2M1 -> two floats (exact, via cvt.rn.f16x2.e2m1x2 on sm_100a+/sm_120a). The low nibble becomes .x.
  __host__ __device__ __forceinline__
  float2 unpackE2M1x2(const uint8_t& v) {
    const __half2 h = __half2(__nv_cvt_fp4x2_to_halfraw2(static_cast<__nv_fp4x2_storage_t>(v), __NV_E2M1));
    return __half22float2(h);
  }

  // ------------------------------------------------------------------------------------------------------------
  // Scales
  // ------------------------------------------------------------------------------------------------------------

  /// mxfp4 scale of a block with absolute maximum amax, the recipe of mx::ue8m0FromAmax with E2M1_MAX instead of 448:
  /// 2^ceil(log2(amax / 6)), so that amax / scale <= 6 and the block does not saturate, encoded as UE8M0
  /// (value = 2^(byte - 127)). Pure integer arithmetic on the bits of amax, exact and portable. Because 6 = 1.5 * 2^2,
  ///   byte = biasedExponent(amax) - 2 + (mantissa(amax) > 0.5 ? 1 : 0).
  /// The sign of amax is ignored. Results below mx::UE8M0_MIN (amax == 0, denormals, ...) clamp to mx::UE8M0_MIN;
  /// NaN and +-Inf return mx::UE8M0_NAN (0xFF).
  __host__ __device__ __forceinline__
  constexpr uint8_t ue8m0FromAmaxFP4(const float& amax) {
    const auto bits = cuda::std::bit_cast<uint32_t>(amax) & 0x7FFFFFFFU;
    const uint32_t biased = bits >> 23;
    if (biased == 0xFFU) {
      return mx::UE8M0_NAN;
    }
    const uint32_t mantissa = bits & 0x7FFFFFU;
    const int e = static_cast<int>(biased) - 2 + (mantissa > 0x400000U ? 1 : 0);
    return static_cast<uint8_t>(e < mx::UE8M0_MIN ? mx::UE8M0_MIN : e);
  }

  /// UE4M3 (the ordinary E4M3, positive) -> float, exact
  __host__ __device__ __forceinline__
  float ue4m3ToFloat(const uint8_t& sf) {
    return mx::unpackE4M3x2(static_cast<uint16_t>(sf)).x;
  }

  /// Encode scale of a tensor with decode scale gs: st = 1 / gs (1 for gs == 0, i.e., an all-zero tensor)
  __host__ __device__ __forceinline__
  float nvfp4EncodeScale(const float& gs) {
    return gs > 0.f ? 1.0f / gs : 1.0f;
  }

  /// Decode scale of a tensor with absolute maximum amax: amax / (6 * 448)
  __host__ __device__ __forceinline__
  constexpr float nvfp4GlobalScale(const float& amax) {
    return amax / (E2M1_MAX * mx::E4M3_MAX);
  }

  /// nvfp4 scale of a block with absolute maximum amax, st = encode scale of the tensor:
  /// UE4M3 = e4m3_rn_sat(amax * (st * (1 / 6))), the recipe of CUTLASS compute_1d_scaling_factor_and_quantized_output
  /// (cutlass/util/reference/host/gett.hpp). NaN stays NaN (0x7F); the sign of amax is ignored.
  __host__ __device__ __forceinline__
  uint8_t nvfp4ScaleFromAmax(const float& amax, const float& st) {
    constexpr float invMax = 1.0f / E2M1_MAX;
    return static_cast<uint8_t>(mx::packE4M3x2(fabsf(amax) * (st * invMax), 0.f) & 0x7FU);
  }

  /// Factor the elements of a block are multiplied by before the E2M1 conversion.
  ///  mxfp4: 2^(127 - sf), exact; 1 for a NaN / Inf block (the payload is irrelevant there).
  ///  nvfp4: min(st / float(sf), FLT_MAX), IEEE division; sf == 0 gives FLT_MAX, which the zero scale cancels on decode.
  template <QuantRecipe r>
    requires(isFP4(r))
  __host__ __device__ __forceinline__
  float blockEncodeScale(const uint8_t& sf, const float& st) {
    if constexpr (r == QuantRecipe::mxfp4) {
      return sf == mx::UE8M0_NAN ? 1.0f : mx::ue8m0ToRcpFloat(sf);
    }
    else {
      return fminf(st / ue4m3ToFloat(sf), FLT_MAX);
    }
  }

  /// Factor of the decoded elements of a block: x ~ e2m1 * blockDecodeScale<r>(sf, gs).
  ///  mxfp4: 2^(sf - 127) (NaN for sf == 0xFF), gs is ignored; nvfp4: float(sf) * gs.
  template <QuantRecipe r>
    requires(isFP4(r))
  __host__ __device__ __forceinline__
  float blockDecodeScale(const uint8_t& sf, const float& gs) {
    if constexpr (r == QuantRecipe::mxfp4) {
      return mx::ue8m0ToFloat(sf);
    }
    else {
      return ue4m3ToFloat(sf) * gs;
    }
  }

  // ------------------------------------------------------------------------------------------------------------
  // Quantize / dequantize kernels
  // ------------------------------------------------------------------------------------------------------------

  /// Quantizes 8 values held by each lane. sfVecSize<r>() / 8 adjacent lanes (4 for mxfp4, 2 for nvfp4, aligned to
  /// that count) cooperate on one scale block: the shuffles are executed by the whole warp, so every lane of the warp
  /// has to call this (a lane without data passes zeros, which cannot raise the amax). Returns the 4 bytes of the 8
  /// E2M1 (element 2 i in the low nibble of byte i) and the UE8M0 / UE4M3 byte of the block in sfByte.
  /// st is the encode scale of an nvfp4 tensor and unused for mxfp4.
  template <QuantRecipe r>
    requires(isFP4(r))
  __device__ __forceinline__
  uint32_t quantize8(const float (&x)[8], const float& st, uint8_t& sfByte) {
    constexpr int lanesPerBlock = sfVecSize<r>() / 8;
    uint32_t amaxBits = 0U; // |x| as an integer is monotonic in |x| and orders NaN > Inf > finite
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      amaxBits = max(amaxBits, __float_as_uint(x[i]) & 0x7FFFFFFFU);
    }
    #pragma unroll
    for (int o = lanesPerBlock / 2; o > 0; o >>= 1) {
      amaxBits = max(amaxBits, __shfl_xor_sync(0xFFFFFFFFU, amaxBits, o));
    }
    const float amax = __uint_as_float(amaxBits);
    if constexpr (r == QuantRecipe::mxfp4) {
      sfByte = ue8m0FromAmaxFP4(amax);
    }
    else {
      sfByte = nvfp4ScaleFromAmax(amax, st);
    }
    const float enc = blockEncodeScale<r>(sfByte, st);
    uint32_t w = 0U;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      w |= static_cast<uint32_t>(packE2M1x2(x[2 * i] * enc, x[2 * i + 1] * enc)) << (8 * i);
    }
    return w;
  }

  /// Quantizes a row-major [rows, K] matrix (K % 32 == 0 for mxfp4, K % 16 == 0 for nvfp4) to E2M1 (q: rows * K / 2
  /// bytes) and scales along K. One thread handles 8 input elements; the lanes past the end of the tensor contribute 0.
  /// The launcher guarantees rows * K / 8 < 2^32.
  template <QuantRecipe r, typename Element>
    requires(isFP4(r) && mx::MXInput<Element>)
  __global__ void quantizeKernel(const Element* __restrict__ in, uint8_t* __restrict__ q, uint8_t* __restrict__ sf,
                                 const __grid_constant__ int rows, const __grid_constant__ int K,
                                 const __grid_constant__ float st) {
    constexpr int vec = 8;
    constexpr int lanesPerBlock = sfVecSize<r>() / vec;
    const uint32_t vecPerRow = K / vec;
    const uint32_t total = static_cast<uint32_t>(rows) * vecPerRow;
    const uint32_t u = blockIdx.x * blockDim.x + threadIdx.x;
    const bool valid = u < total;
    const uint32_t row = valid ? u / vecPerRow : 0U;
    const uint32_t col = u - row * vecPerRow;

    float x[vec] = {};
    if (valid) {
      alignas(16) Element e[vec];
      constexpr int chunks = vec * sizeof(Element) / 16;
      const auto* src = reinterpret_cast<const uint4*>(in + static_cast<size_t>(row) * K + col * vec);
      #pragma unroll
      for (int i = 0; i < chunks; ++i) {
        reinterpret_cast<uint4*>(e)[i] = src[i];
      }
      #pragma unroll
      for (int i = 0; i < vec; ++i) {
        x[i] = mx::detail::toFloat(e[i]);
      }
    }
    uint8_t sfByte = 0;
    const uint32_t w = quantize8<r>(x, st, sfByte);
    if (!valid) {
      return;
    }
    *reinterpret_cast<uint32_t*>(q + static_cast<size_t>(row) * (K / 2) + col * 4) = w;
    if (col % lanesPerBlock == 0) {
      sf[mx::sfOffset(row, col / lanesPerBlock, K / sfVecSize<r>())] = sfByte;
    }
  }

  /// Dequantizes E2M1 + scales into a row-major [rows, K] matrix: out = e2m1 * blockDecodeScale<r>(sf, gs).
  /// One thread handles 16 elements.
  template <QuantRecipe r, typename Out>
    requires(isFP4(r) && mx::MXOutput<Out>)
  __global__ void dequantizeKernel(const uint8_t* __restrict__ q, const uint8_t* __restrict__ sf,
                                   Out* __restrict__ out, const __grid_constant__ int rows,
                                   const __grid_constant__ int K, const __grid_constant__ float gs) {
    constexpr int vec = 16;
    const uint32_t vecPerRow = K / vec;
    const uint32_t total = static_cast<uint32_t>(rows) * vecPerRow;
    const uint32_t u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= total) {
      return;
    }
    const uint32_t row = u / vecPerRow;
    const uint32_t col = u - row * vecPerRow;
    const size_t base = static_cast<size_t>(row) * K + col * vec;
    const float scale = blockDecodeScale<r>(sf[mx::sfOffset(row, col * vec / sfVecSize<r>(), K / sfVecSize<r>())], gs);
    const uint2 raw = *reinterpret_cast<const uint2*>(q + static_cast<size_t>(row) * (K / 2) + col * 8);
    const uint32_t words[2] = {raw.x, raw.y};
    float f[vec];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float2 v = unpackE2M1x2(static_cast<uint8_t>((words[i / 4] >> (8 * (i % 4))) & 0xFFU));
      f[2 * i] = v.x * scale;
      f[2 * i + 1] = v.y * scale;
    }
    if constexpr (cuda::std::is_same_v<Out, float>) {
      auto* dst = reinterpret_cast<float4*>(out + base);
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        dst[i] = make_float4(f[4 * i], f[4 * i + 1], f[4 * i + 2], f[4 * i + 3]);
      }
    }
    else {
      alignas(16) __nv_bfloat162 o[vec / 2];
      #pragma unroll
      for (int i = 0; i < vec / 2; ++i) {
        o[i] = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
      }
      auto* dst = reinterpret_cast<uint4*>(out + base);
      dst[0] = *reinterpret_cast<const uint4*>(&o[0]);
      dst[1] = *reinterpret_cast<const uint4*>(&o[4]);
    }
  }

  /// Absolute maximum of n elements: atomicMax of the bits of |x| (monotonic) into out, which must hold 0 (or any
  /// smaller amax) at the start. NaN wins over every number.
  template <typename Element>
    requires(mx::MXInput<Element>)
  __global__ void amaxKernel(const Element* __restrict__ in, uint32_t* __restrict__ out, const __grid_constant__ size_t n) {
    uint32_t amaxBits = 0U;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
      amaxBits = max(amaxBits, __float_as_uint(mx::detail::toFloat(in[i])) & 0x7FFFFFFFU);
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      amaxBits = max(amaxBits, __shfl_xor_sync(0xFFFFFFFFU, amaxBits, o));
    }
    if ((threadIdx.x & 31U) == 0U) {
      atomicMax(out, amaxBits);
    }
  }

  // ------------------------------------------------------------------------------------------------------------
  // Host launchers
  // ------------------------------------------------------------------------------------------------------------

  /// Bytes of the E2M1 data of a [rows, K] matrix
  __host__ __forceinline__
  constexpr size_t dataBytes(const size_t& rows, const size_t& K) {
    return rows * K / 2;
  }

  /// Bytes of the scale factor buffer of a [rows, K] matrix (layout of mx::sfOffset())
  template <QuantRecipe r>
    requires(isFP4(r))
  __host__ __forceinline__
  constexpr size_t sfBytes(const size_t& rows, const size_t& K) {
    return mx::sfBytes(rows, K / sfVecSize<r>());
  }

  /// Quantizes in[rows, K] (row-major, K % 32 == 0 for mxfp4, K % 16 == 0 for nvfp4) to q[rows, K / 2] (E2M1, row-major)
  /// and sf, the scale factors in the canonical layout of mx::sfOffset() with sfBytes<r>(rows, K) bytes.
  /// gs: the decode scale of the tensor for nvfp4 (see nvfp4GlobalScale), ignored for mxfp4.
  /// All pointers are device pointers; q needs 4-byte alignment and in 16-byte alignment.
  /// The padding of sf (rows up to a multiple of 128, k-blocks up to a multiple of 4) is zero-filled.
  template <QuantRecipe r, typename Element>
    requires(isFP4(r) && mx::MXInput<Element>)
  __host__ __forceinline__
  void quantizeFP4(const Element* const& in, uint8_t* const& q, uint8_t* const& sf, const int& rows, const int& K,
                   const float& gs = 1.f, cudaStream_t stream = nullptr) {
    constexpr int vs = sfVecSize<r>();
    if (rows <= 0 || K <= 0 || K % vs != 0) {
      throw std::runtime_error("quantizeFP4: K must be a positive multiple of the scale block size");
    }
    const size_t units = static_cast<size_t>(rows) * (K / 8);
    if (units >= (1ULL << 32)) {
      throw std::runtime_error("quantizeFP4: tensor too large");
    }
    const int numKBlocks = K / vs;
    if (rows % mx::SF_ROW_TILE != 0 || numKBlocks % mx::SF_K_TILE != 0) {
      cudaMemsetAsync(sf, 0, mx::sfBytes(rows, numKBlocks), stream);
    }
    constexpr uint threads = 256;
    const auto blocks = static_cast<uint>((units + threads - 1) / threads);
    quantizeKernel<r, Element><<<blocks, threads, 0, stream>>>(in, q, sf, rows, K, r == QuantRecipe::nvfp4 ?
      nvfp4EncodeScale(gs) : 1.0f);
  }

  /// Inverse of quantizeFP4: out[rows, K] = e2m1 * blockDecodeScale<r>(sf, gs), exact for Out = float and mxfp4.
  template <QuantRecipe r, typename Out>
    requires(isFP4(r) && mx::MXOutput<Out>)
  __host__ __forceinline__
  void dequantizeFP4(const uint8_t* const& q, const uint8_t* const& sf, Out* const& out, const int& rows, const int& K,
                     const float& gs = 1.f, cudaStream_t stream = nullptr) {
    if (rows <= 0 || K <= 0 || K % sfVecSize<r>() != 0 || K % 16 != 0) {
      throw std::runtime_error("dequantizeFP4: K must be a positive multiple of 32 (mxfp4) or 16 (nvfp4)");
    }
    const size_t units = static_cast<size_t>(rows) * (K / 16);
    if (units >= (1ULL << 32)) {
      throw std::runtime_error("dequantizeFP4: tensor too large");
    }
    constexpr uint threads = 256;
    const auto blocks = static_cast<uint>((units + threads - 1) / threads);
    dequantizeKernel<r, Out><<<blocks, threads, 0, stream>>>(q, sf, out, rows, K, gs);
  }

  /// *out = max |in[i]| over n elements (device pointers; out is overwritten). Feed it to nvfp4GlobalScale.
  template <typename Element>
    requires(mx::MXInput<Element>)
  __host__ __forceinline__
  void amax(const Element* const& in, const size_t& n, float* const& out, cudaStream_t stream = nullptr) {
    cudaMemsetAsync(out, 0, sizeof(float), stream);
    constexpr uint threads = 256;
    const auto blocks = static_cast<uint>(std::min<size_t>((n + threads - 1) / threads, 4096));
    amaxKernel<Element><<<blocks, threads, 0, stream>>>(in, reinterpret_cast<uint32_t*>(out), n);
  }
}
#endif //FLASHMOE_FP4_CUH
