/******************************************************************************
 * Copyright (c) 2026, Osayamen Jonathan Aimuyo.
 ******************************************************************************/
//
// Created by FlashMoE contributors on 10/5/26.
//
// MXFP8 (OCP Microscaling, E4M3 elements + one UE8M0 scale per 32 elements along K) helpers:
// scale computation, element conversions, the canonical swizzled scale-factor (SF) layout
// and standalone quantize / dequantize kernels.

#ifndef FLASHMOE_MX_CUH
#define FLASHMOE_MX_CUH

#include <cstdint>
#include <stdexcept>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cuda/std/bit>
#include <cuda/std/type_traits>

// The hardware float -> UE8M0 conversion (cvt.rp.satfinite.ue8m0x2.f32) needs the architecture-specific
// (sm_100a / sm_103a / sm_110a / sm_120a / sm_121a) or the family-specific (sm_100f ...) instruction set.
#if defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM101_ALL) || \
    defined(__CUDA_ARCH_FEAT_SM103_ALL) || defined(__CUDA_ARCH_FEAT_SM110_ALL) || \
    defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL) || \
    (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ >= 1000))
#  define FLASHMOE_MX_UE8M0_CVT 1
#endif

namespace flashmoe::mx {
  /// Number of consecutive K elements that share one scale factor
  constexpr int BLOCK_SIZE = 32;
  /// Largest finite E4M3 value
  constexpr float E4M3_MAX = 448.f;
  /// UE8M0 encoding of NaN, i.e., a block whose amax is NaN or Inf
  constexpr uint8_t UE8M0_NAN = 0xFF;
  /// Smallest scale produced by the quantizer, 2^-126: the smallest normal FP32 value, so no scale is ever a
  /// denormal (UE8M0 0 would be 2^-127). Blocks with amax < 448 * 2^-126 (~5e-37) are scaled by it as well.
  constexpr uint8_t UE8M0_MIN = 1;
  /// Scale factors are stored in atoms of 128 rows x 4 k-blocks = 512 B
  constexpr int SF_ROW_TILE = 128;
  constexpr int SF_K_TILE = 4;
  constexpr int SF_ATOM_BYTES = SF_ROW_TILE * SF_K_TILE;

  // ------------------------------------------------------------------------------------------------------------
  // Scale factor layout
  // ------------------------------------------------------------------------------------------------------------

  /// Bytes of the scale factor buffer of a [rows, numKBlocks] scale matrix:
  /// rows are padded to a multiple of 128 and k-blocks to a multiple of 4.
  __host__ __device__ __forceinline__
  constexpr size_t sfBytes(const size_t& rows, const size_t& numKBlocks) {
    return ((rows + SF_ROW_TILE - 1) / SF_ROW_TILE) * ((numKBlocks + SF_K_TILE - 1) / SF_K_TILE) * SF_ATOM_BYTES;
  }

  /// Byte offset of the scale for (row, kBlock), where kBlock = k / 32, inside the buffer of sfBytes() bytes.
  /// This is the canonical "128x4 swizzled" layout of cuBLASLt CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0 and
  /// CUTLASS Sm1xxBlockScaledConfig<32>::tile_atom_to_shape_SFA/SFB:
  /// atoms are ordered k-block-tile fastest, then row-tile; inside an atom, offset = (row % 32) * 16 +
  /// (row % 128) / 32 * 4 + kBlock % 4.
  __host__ __device__ __forceinline__
  constexpr size_t sfOffset(const uint32_t& row, const uint32_t& kBlock, const uint32_t& numKBlocks) {
    const size_t kTiles = (static_cast<size_t>(numKBlocks) + SF_K_TILE - 1) / SF_K_TILE;
    const size_t atom = static_cast<size_t>(row / SF_ROW_TILE) * kTiles + kBlock / SF_K_TILE;
    return atom * SF_ATOM_BYTES + (row % 32) * 16 + ((row % SF_ROW_TILE) / 32) * 4 + (kBlock % SF_K_TILE);
  }

  // ------------------------------------------------------------------------------------------------------------
  // UE8M0 scales
  // ------------------------------------------------------------------------------------------------------------

  /// Portable, exact version of ue8m0FromAmax. Pure integer arithmetic on the bits of amax.
  ///
  /// Returns the biased exponent e of the smallest power of two 2^(e - 127) with amax / 2^(e - 127) <= 448, i.e.,
  /// e = 127 + ceil(log2(amax / 448)). Because 448 = 1.75 * 2^8, that is
  ///   e = biasedExponent(amax) - 8 + (mantissa(amax) > 0.75 ? 1 : 0).
  /// The sign of amax is ignored. Results below UE8M0_MIN (including amax == 0 and denormals) clamp to UE8M0_MIN;
  /// NaN and +-Inf return UE8M0_NAN (0xFF). Finite amax never exceeds 247.
  __host__ __device__ __forceinline__
  constexpr uint8_t ue8m0FromAmaxPortable(const float& amax) {
    const auto bits = cuda::std::bit_cast<uint32_t>(amax) & 0x7FFFFFFFU;
    const uint32_t biased = bits >> 23;
    if (biased == 0xFFU) {
      return UE8M0_NAN;
    }
    const uint32_t mantissa = bits & 0x7FFFFFU;
    const int e = static_cast<int>(biased) - 8 + (mantissa > 0x600000U ? 1 : 0);
    // covers amax == 0 and denormals, for which biased == 0
    return static_cast<uint8_t>(e < UE8M0_MIN ? UE8M0_MIN : e);
  }

#if defined(FLASHMOE_MX_UE8M0_CVT)
  /// Hardware version of ue8m0FromAmaxPortable: the float amax / 448 rounded up (towards +inf) to a power of two with
  /// cvt.rp.satfinite.ue8m0x2.f32, clamped to UE8M0_MIN. Bit-identical to the portable version for every finite
  /// amax: the product amax * (1 / 448) in FP32 cannot move across a power of two (verified exhaustively around the
  /// boundaries 448 * 2^k by the unit test), and where it is a denormal the clamp makes the difference moot.
  /// NaN and Inf are NOT handled here, see ue8m0FromAmax.
  __device__ __forceinline__
  uint8_t ue8m0FromAmaxHw(const float& amax) {
    constexpr float invMax = 1.0f / E4M3_MAX;
    uint16_t packed;
    // The two packed results are {hi = 0, lo = scale}.
    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, 0f00000000, %1;" : "=h"(packed) : "f"(amax * invMax));
    return static_cast<uint8_t>(max(packed & 0xFFU, static_cast<uint32_t>(UE8M0_MIN)));
  }
#endif

  /// UE8M0 scale of a block with absolute maximum amax, following the NVIDIA MXFP8 recipe:
  /// scale = 2^ceil(log2(amax / 448)), so that amax / scale <= 448 and the block does not saturate.
  /// Encoded as UE8M0, i.e., value = 2^(byte - 127).
  ///  - amax == 0 returns UE8M0_MIN (scale 2^-126): all elements of the block are zero, so any finite scale is valid.
  ///  - results below the range clamp to UE8M0_MIN (2^-126, the smallest normal FP32); finite inputs cannot exceed it
  ///    on the upper end (largest result 247).
  ///  - NaN and +-Inf return UE8M0_NAN (0xFF): every element of that block dequantizes to NaN, as in the MX spec.
  ///  - the sign of amax is ignored.
  /// Uses cvt.rp.satfinite.ue8m0x2.f32 where available (sm_100a+/sm_120a), bit-identical to the portable path.
  __device__ __forceinline__
  uint8_t ue8m0FromAmax(const float& amax) {
#if defined(FLASHMOE_MX_UE8M0_CVT)
    const float a = fabsf(amax);
    const uint8_t e = ue8m0FromAmaxHw(a);
    // NaN and Inf are resolved here and not by the hardware conversion.
    return !(a < __int_as_float(0x7F800000)) ? UE8M0_NAN : e;
#else
    return ue8m0FromAmaxPortable(amax);
#endif
  }

  /// 2^(e - 127). e == 0 gives the FP32 denormal 2^-127, e == 255 gives NaN.
  __host__ __device__ __forceinline__
  constexpr float ue8m0ToFloat(const uint8_t& e) {
    if (e == UE8M0_NAN) {
      return cuda::std::bit_cast<float>(0x7FFFFFFFU);
    }
    return cuda::std::bit_cast<float>(e == 0 ? 0x00400000U : (static_cast<uint32_t>(e) << 23));
  }

  /// 2^(127 - e), the exact reciprocal of ue8m0ToFloat(e). e == 255 gives NaN.
  __host__ __device__ __forceinline__
  constexpr float ue8m0ToRcpFloat(const uint8_t& e) {
    if (e == UE8M0_NAN) {
      return cuda::std::bit_cast<float>(0x7FFFFFFFU);
    }
    return cuda::std::bit_cast<float>(e == 254 ? 0x00400000U : (static_cast<uint32_t>(254 - e) << 23));
  }

  // ------------------------------------------------------------------------------------------------------------
  // E4M3 conversions
  // ------------------------------------------------------------------------------------------------------------

  /// Two floats -> two E4M3, round-to-nearest-even and saturating to +-448 (cvt.rn.satfinite.e4m3x2.f32 on sm_89+).
  /// lo is stored in the low byte of the result and hi in the high byte.
  __host__ __device__ __forceinline__
  uint16_t packE4M3x2(const float& lo, const float& hi) {
    return static_cast<uint16_t>(__nv_cvt_float2_to_fp8x2(make_float2(lo, hi), __NV_SATFINITE, __NV_E4M3));
  }

  /// Two E4M3 -> two floats (exact, via cvt.rn.f16x2.e4m3x2 on sm_89+). The low byte becomes .x.
  __host__ __device__ __forceinline__
  float2 unpackE4M3x2(const uint16_t& v) {
    const __half2 h = __half2(__nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(v), __NV_E4M3));
    return __half22float2(h);
  }

  // ------------------------------------------------------------------------------------------------------------
  // Quantize / dequantize kernels
  // ------------------------------------------------------------------------------------------------------------

  template<typename Element>
  concept MXInput = cuda::std::is_same_v<Element, __nv_bfloat16> ||
                    cuda::std::is_same_v<Element, __half> ||
                    cuda::std::is_same_v<Element, float>;

  template<typename Element>
  concept MXOutput = cuda::std::is_same_v<Element, __nv_bfloat16> || cuda::std::is_same_v<Element, float>;

  namespace detail {
    __device__ __forceinline__
    float toFloat(const float& x) {
      return x;
    }

    __device__ __forceinline__
    float toFloat(const __half& x) {
      return __half2float(x);
    }

    __device__ __forceinline__
    float toFloat(const __nv_bfloat16& x) {
      return __bfloat162float(x);
    }
  }

  /// Quantizes a row-major [rows, K] matrix (K % 32 == 0) to E4M3 and UE8M0 scales along K.
  /// One thread handles 16 bytes of input; 32 / vec adjacent lanes cooperate on one 32-element block.
  /// Every thread of a warp executes the shuffles; the lanes past the end of the tensor contribute 0.
  /// The launcher guarantees rows * K / vec < 2^32.
  template<typename Element>
    requires(MXInput<Element>)
  __global__ void quantizeKernel(const Element* __restrict__ in, uint8_t* __restrict__ q, uint8_t* __restrict__ sf,
                                 const __grid_constant__ int rows, const __grid_constant__ int K) {
    constexpr int vec = 16 / sizeof(Element); // elements per thread
    constexpr int lanesPerBlock = BLOCK_SIZE / vec;
    static_assert(vec % 4 == 0 && BLOCK_SIZE % vec == 0 && lanesPerBlock >= 1);
    const uint32_t vecPerRow = K / vec;
    const uint32_t total = static_cast<uint32_t>(rows) * vecPerRow;
    const uint32_t u = blockIdx.x * blockDim.x + threadIdx.x;
    const bool valid = u < total;
    const uint32_t row = valid ? u / vecPerRow : 0U;
    const uint32_t col = u - row * vecPerRow;

    float x[vec];
    uint32_t amaxBits = 0U; // |x| as an integer is monotonic in |x| and orders NaN > Inf > finite
    if (valid) {
      alignas(16) Element e[vec];
      *reinterpret_cast<uint4*>(e) = *reinterpret_cast<const uint4*>(in + static_cast<size_t>(row) * K + col * vec);
      #pragma unroll
      for (int i = 0; i < vec; ++i) {
        x[i] = detail::toFloat(e[i]);
        amaxBits = max(amaxBits, __float_as_uint(x[i]) & 0x7FFFFFFFU);
      }
    }
    #pragma unroll
    for (int o = lanesPerBlock / 2; o > 0; o >>= 1) {
      amaxBits = max(amaxBits, __shfl_xor_sync(0xFFFFFFFFU, amaxBits, o));
    }
    if (!valid) {
      return;
    }
    const uint8_t sfByte = ue8m0FromAmax(__uint_as_float(amaxBits));
    // NaN / Inf block: the scale is NaN, the payload is irrelevant but kept finite-saturated
    const float inv = sfByte == UE8M0_NAN ? 1.0f : ue8m0ToRcpFloat(sfByte);
    uint32_t w[vec / 4];
    #pragma unroll
    for (int i = 0; i < vec / 4; ++i) {
      const uint32_t lo = packE4M3x2(x[4 * i] * inv, x[4 * i + 1] * inv);
      const uint32_t hi = packE4M3x2(x[4 * i + 2] * inv, x[4 * i + 3] * inv);
      w[i] = lo | (hi << 16);
    }
    uint8_t* dst = q + static_cast<size_t>(row) * K + col * vec;
    if constexpr (vec == 8) {
      *reinterpret_cast<uint2*>(dst) = make_uint2(w[0], w[1]);
    }
    else {
      *reinterpret_cast<uint32_t*>(dst) = w[0];
    }
    if (col % lanesPerBlock == 0) {
      sf[sfOffset(row, col / lanesPerBlock, K / BLOCK_SIZE)] = sfByte;
    }
  }

  /// Dequantizes E4M3 + UE8M0 (K % 32 == 0) into a row-major [rows, K] matrix. One thread handles 16 elements.
  template<typename Out>
    requires(MXOutput<Out>)
  __global__ void dequantizeKernel(const uint8_t* __restrict__ q, const uint8_t* __restrict__ sf,
                                   Out* __restrict__ out,
                                   const __grid_constant__ int rows, const __grid_constant__ int K) {
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
    const float scale = ue8m0ToFloat(sf[sfOffset(row, col / (BLOCK_SIZE / vec), K / BLOCK_SIZE)]);
    const uint4 raw = *reinterpret_cast<const uint4*>(q + base);
    const uint32_t words[4] = {raw.x, raw.y, raw.z, raw.w};
    float f[vec];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 a = unpackE4M3x2(static_cast<uint16_t>(words[i] & 0xFFFFU));
      const float2 b = unpackE4M3x2(static_cast<uint16_t>(words[i] >> 16));
      f[4 * i] = a.x * scale;
      f[4 * i + 1] = a.y * scale;
      f[4 * i + 2] = b.x * scale;
      f[4 * i + 3] = b.y * scale;
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

  // ------------------------------------------------------------------------------------------------------------
  // Host launchers
  // ------------------------------------------------------------------------------------------------------------

  /// Quantizes in[rows, K] (row-major, K % 32 == 0) to q[rows, K] (E4M3, row-major) and sf, the scale factors in the
  /// canonical layout of sfOffset() with sfBytes(rows, K / 32) bytes.
  /// All pointers are device pointers; q needs 8-byte alignment and in 16-byte alignment.
  /// The padding of sf (rows up to a multiple of 128, k-blocks up to a multiple of 4) is zero-filled.
  template<typename Element>
    requires(MXInput<Element>)
  __host__ __forceinline__
  void quantizeMXFP8(const Element* const& in, uint8_t* const& q, uint8_t* const& sf,
                     const int& rows, const int& K, cudaStream_t stream = nullptr) {
    constexpr int vec = 16 / sizeof(Element);
    if (rows <= 0 || K <= 0 || K % BLOCK_SIZE != 0) {
      throw std::runtime_error("quantizeMXFP8: K must be a positive multiple of 32");
    }
    const size_t units = static_cast<size_t>(rows) * (K / vec);
    if (units >= (1ULL << 32)) {
      throw std::runtime_error("quantizeMXFP8: tensor too large");
    }
    const int numKBlocks = K / BLOCK_SIZE;
    if (rows % SF_ROW_TILE != 0 || numKBlocks % SF_K_TILE != 0) {
      cudaMemsetAsync(sf, 0, sfBytes(rows, numKBlocks), stream);
    }
    constexpr uint threads = 256;
    const auto blocks = static_cast<uint>((units + threads - 1) / threads);
    quantizeKernel<Element><<<blocks, threads, 0, stream>>>(in, q, sf, rows, K);
  }

  /// Inverse of quantizeMXFP8: out[rows, K] = q * 2^(sf - 127), exact for Out = float.
  template<typename Out>
    requires(MXOutput<Out>)
  __host__ __forceinline__
  void dequantizeMXFP8(const uint8_t* const& q, const uint8_t* const& sf, Out* const& out,
                       const int& rows, const int& K, cudaStream_t stream = nullptr) {
    if (rows <= 0 || K <= 0 || K % BLOCK_SIZE != 0) {
      throw std::runtime_error("dequantizeMXFP8: K must be a positive multiple of 32");
    }
    const size_t units = static_cast<size_t>(rows) * (K / 16);
    if (units >= (1ULL << 32)) {
      throw std::runtime_error("dequantizeMXFP8: tensor too large");
    }
    constexpr uint threads = 256;
    const auto blocks = static_cast<uint>((units + threads - 1) / threads);
    dequantizeKernel<Out><<<blocks, threads, 0, stream>>>(q, sf, out, rows, K);
  }
}
#endif //FLASHMOE_MX_CUH
