//
// Created by FlashMoE contributors on 10/5/26.
//
// correctness test and benchmark for MXFP8 (E4M3 elements + one UE8M0 scale per 32 K-elements).
//  1. unit checks: UE8M0 scale conversion (hardware vs portable vs double precision model), E4M3 conversion,
//     the scale factor layout against CUTLASS's Sm1xxBlockScaledConfig<32>
//  2. quantize / dequantize: bit-exact comparison with a host model, invariants, round-trip error, throughput
//  3. cuBLASLt MXFP8 GEMM (C = a @ b^T, a is [M, K], b is [N, K], both K-major) fed with our quantized operands and
//     our scale factors, checked against an FP32 MatX GEMM on the dequantized operands and timed against the
//     BF16 GEMMs of MatX and cuBLASLt.
//  4. the block-scaled MXFP8 tile mainloop and epilogues of flashmoe::tile::sm120::MXFP8Mainloop (tile_sm120_mxfp8.cuh)
//     in a persistent kernel (grid = min(#tiles, #SMs), every CTA loops over tiles and reuses its workspace):
//     vanilla GEMM + BF16 epilogue, vanilla GEMM + MXFP8 epilogue, gated GEMM + gated MXFP8 epilogue. Checked against
//     an FP32 GEMM of the dequantized operands (+ bias, activation, gating, then the same MXFP8 quantization) and
//     timed against cuBLASLt MXFP8 and BF16 (two GEMMs for the gated case).
//
// ./gemmMXFP8 [<M> <N> <K>] [--unit] [--quant] [--gemm] [--tile] [--tilecheck] [--runs=<n>] [--cfg=<substring>]
// no flags: everything, over the default shapes; --tile: correctness + benchmark of the tile kernels over the MoE
// shapes (or the given one), --tilecheck: correctness only, over small shapes with fewer CTAs than tiles and a
// workspace that is overwritten between tiles; --cfg=<s>: only the tile configurations whose name contains s

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cublasLt.h>
#include <cuda_fp8.h>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>

#include "common.cuh"
#include "debug.cuh"
#include "../include/flashmoe/infra/activation.cuh"
#include "../include/flashmoe/infra/mx.cuh"
#include "../include/flashmoe/tile.cuh"
#include "../include/flashmoe/tile_sm120_mxfp8.cuh"

#define CHECK_CUBLAS(e)                                    \
do {                                                         \
    cublasStatus_t status = (e);                             \
    if (status != CUBLAS_STATUS_SUCCESS) {                   \
        fprintf(stderr, "<%s:%d> %s:\n    cublas status %d\n", \
            __FILE__, __LINE__, #e, static_cast<int>(status)); \
        fflush(stderr);                                      \
        exit(1);                                             \
    }                                                        \
} while (0)

namespace mx = flashmoe::mx;
constexpr int arch = FLASHMOE_ARCH;

// ------------------------------------------------------------------------------------------------------------
// host models, independent of the code under test
// ------------------------------------------------------------------------------------------------------------

__host__ __forceinline__
float hostToFloat(const float x) { return x; }

__host__ __forceinline__
float hostToFloat(const __half x) { return __half2float(x); }

__host__ __forceinline__
float hostToFloat(const __nv_bfloat16 x) { return __bfloat162float(x); }

// E4M3 (OCP FP8): 1 sign, 4 exponent (bias 7), 3 mantissa bits. 0x7F / 0xFF are NaN, there is no Inf.
__host__ __forceinline__
float hostE4M3Decode(const uint8_t b) {
  const int e = (b >> 3) & 0xF;
  const int m = b & 7;
  if (e == 15 && m == 7) {
    return std::nanf("");
  }
  const double v = e == 0 ? std::ldexp(m / 8.0, -6) : std::ldexp(1.0 + m / 8.0, e - 7);
  return static_cast<float>((b & 0x80) ? -v : v);
}

// brute force round-to-nearest-even with saturation to +-448
__host__ __forceinline__
uint8_t hostE4M3Encode(const float x) {
  const uint8_t sign = std::signbit(x) ? 0x80 : 0;
  const double ax = std::fabs(static_cast<double>(x));
  uint8_t best = 0;
  double bestDist = std::fabs(ax - 0.0);
  for (int c = 1; c <= 0x7E; ++c) {
    const double d = std::fabs(ax - hostE4M3Decode(static_cast<uint8_t>(c)));
    if (d < bestDist || (d == bestDist && (c % 2 == 0))) {
      bestDist = d;
      best = static_cast<uint8_t>(c);
    }
  }
  return static_cast<uint8_t>(sign | best);
}

// ue8m0 = 127 + ceil(log2(amax / 448)), double precision
__host__ __forceinline__
uint8_t hostUe8m0(const float amax) {
  if (std::isnan(amax) || std::isinf(amax)) {
    return mx::UE8M0_NAN;
  }
  const double a = std::fabs(static_cast<double>(amax));
  if (a == 0.0) {
    return mx::UE8M0_MIN;
  }
  int ex = 0;
  const double m = std::frexp(a / 448.0, &ex); // a / 448 = m * 2^ex, m in [0.5, 1)
  const int ceilLog2 = m == 0.5 ? ex - 1 : ex;
  return static_cast<uint8_t>(std::clamp(127 + ceilLog2, static_cast<int>(mx::UE8M0_MIN), 254));
}

// ------------------------------------------------------------------------------------------------------------
// unit checks
// ------------------------------------------------------------------------------------------------------------

__global__ void ue8m0Kernel(const float *__restrict__ in, uint8_t *__restrict__ api,
                            uint8_t *__restrict__ portable, uint8_t *__restrict__ hw, const size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const float x = in[i];
  api[i] = mx::ue8m0FromAmax(x);
  portable[i] = mx::ue8m0FromAmaxPortable(x);
#if defined(FLASHMOE_MX_UE8M0_CVT)
  hw[i] = mx::ue8m0FromAmaxHw(fabsf(x));
#else
  hw[i] = portable[i];
#endif
}

// the hardware conversion exists only in the arch-specific (sm_xxxa) device compilation
__global__ void hwFlagKernel(int *__restrict__ flag) {
#if defined(FLASHMOE_MX_UE8M0_CVT)
  *flag = 1;
#else
  *flag = 0;
#endif
}

__host__ __forceinline__
void checkUe8m0(cudaStream_t stream) {
  int *dFlag = nullptr;
  int hwCompiled = 0;
  cudaMallocAsync(&dFlag, sizeof(int), stream);
  hwFlagKernel<<<1, 1, 0, stream>>>(dFlag);
  cudaMemcpyAsync(&hwCompiled, dFlag, sizeof(int), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  cudaFreeAsync(dFlag, stream);
  std::vector<float> in;
  std::mt19937_64 rng(0x5eed);
  // 1. uniformly random bit patterns: every exponent, denormals, Inf, NaN, both signs
  constexpr size_t randomCount = 1ULL << 24;
  for (size_t i = 0; i < randomCount; ++i) {
    in.push_back(cuda::std::bit_cast<float>(static_cast<uint32_t>(rng())));
  }
  // 2. neighbourhoods of the decision boundary amax = 448 * 2^k (mantissa 0x600000 and the exponent edges)
  for (uint32_t biased = 0; biased < 256; ++biased) {
    for (int d = -80; d <= 80; ++d) {
      for (const uint32_t center : {0x600000U, 0x000000U, 0x7FFFFFU, 0x400000U}) {
        const int64_t mant = static_cast<int64_t>(center) + d;
        if (mant < 0 || mant > 0x7FFFFF) {
          continue;
        }
        in.push_back(cuda::std::bit_cast<float>((biased << 23) | static_cast<uint32_t>(mant)));
      }
    }
  }
  // 3. special values
  for (const float v : {0.f, -0.f, INFINITY, -INFINITY, NAN, -NAN, 1.f, 448.f, 449.f, 447.999f, 3.4028235e38f,
                        1.17549435e-38f, 1.4e-45f, 448.f * 0x1p-127f, 448.f * 0x1p-127f * 1.0001f}) {
    in.push_back(v);
  }
  const size_t n = in.size();
  float *dIn = nullptr;
  uint8_t *dOut = nullptr;
  cudaMallocAsync(&dIn, n * sizeof(float), stream);
  cudaMallocAsync(&dOut, 3 * n, stream);
  cudaMemcpyAsync(dIn, in.data(), n * sizeof(float), cudaMemcpyHostToDevice, stream);
  ue8m0Kernel<<<(n + 255) / 256, 256, 0, stream>>>(dIn, dOut, dOut + n, dOut + 2 * n, n);
  std::vector<uint8_t> out(3 * n);
  cudaMemcpyAsync(out.data(), dOut, 3 * n, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  CHECK_CUDA(cudaPeekAtLastError());
  cudaFreeAsync(dIn, stream);
  cudaFreeAsync(dOut, stream);

  size_t hwVsPortable = 0, apiVsModel = 0, portableVsModel = 0, nvVsPortable = 0, nvCases = 0;
  size_t specialOk = 0;
  for (size_t i = 0; i < n; ++i) {
    const uint8_t api = out[i], portable = out[n + i], hw = out[2 * n + i];
    const uint8_t model = hostUe8m0(in[i]);
    // the raw hardware conversion is not relied upon for NaN / Inf (ue8m0FromAmax resolves those itself)
    const bool special = std::isnan(in[i]) || std::isinf(in[i]);
    if (!special && hw != portable) {
      ++hwVsPortable;
      if (hwVsPortable <= 8) {
        printf("# hw != portable: amax bits 0x%08x (%.9e) hw %u portable %u model %u\n",
               cuda::std::bit_cast<uint32_t>(in[i]), in[i], hw, portable, model);
      }
    }
    if (api != model) {
      ++apiVsModel;
      if (apiVsModel <= 8) {
        printf("# api != model: amax bits 0x%08x (%.9e) api %u model %u\n",
               cuda::std::bit_cast<uint32_t>(in[i]), in[i], api, model);
      }
    }
    if (portable != model) {
      ++portableVsModel;
    }
    if (special) {
      ++specialOk;
    }
    // NVIDIA's own library conversion (software on the host): ceil to a power of two of amax * (1 / 448)
    // (its range goes down to UE8M0 0, ours to UE8M0_MIN: compare the normal range only)
    if (!special && std::fabs(in[i]) >= 0x1p-100f) {
      ++nvCases;
      const uint8_t nv = __nv_cvt_float_to_e8m0(std::fabs(in[i]) * (1.0f / 448.f), __NV_SATFINITE, cudaRoundPosInf);
      if (nv != portable) {
        ++nvVsPortable;
      }
    }
  }
  printf("test, cases, mismatches, detail\n");
  printf("ue8m0 hardware vs portable, %zu, %zu, hardware cvt %s; NaN/Inf excluded (%zu)\n",
         n - specialOk, hwVsPortable, hwCompiled ? "compiled in" : "NOT compiled in (hw == portable)", specialOk);
  printf("ue8m0 api vs double model, %zu, %zu, including zero/denormal/NaN/Inf\n", n, apiVsModel);
  printf("ue8m0 portable vs double model, %zu, %zu, including zero/denormal/NaN/Inf\n", n, portableVsModel);
  printf("ue8m0 portable vs cuda_fp8.h __nv_cvt_float_to_e8m0, %zu, %zu, finite |amax| >= 2^-100\n",
         nvCases, nvVsPortable);
}

__global__ void e4m3Kernel(const float *__restrict__ in, uint8_t *__restrict__ enc, float *__restrict__ dec,
                           const size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  enc[i] = static_cast<uint8_t>(mx::packE4M3x2(in[i], in[i]) & 0xFFU);
  // decode the byte pattern i % 256 (only meaningful for the first 256 entries)
  const auto b = static_cast<uint16_t>(i % 256);
  dec[i] = mx::unpackE4M3x2(static_cast<uint16_t>(b | (b << 8))).x;
}

__host__ __forceinline__
void checkE4M3(cudaStream_t stream) {
  constexpr size_t n = 1 << 20;
  std::vector<float> in(n);
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> logMag(-12.f, 10.f);
  for (size_t i = 0; i < n; ++i) {
    in[i] = std::exp2(logMag(rng)) * ((rng() & 1) ? -1.f : 1.f);
  }
  float *dIn, *dDec;
  uint8_t *dEnc;
  cudaMallocAsync(&dIn, n * sizeof(float), stream);
  cudaMallocAsync(&dDec, n * sizeof(float), stream);
  cudaMallocAsync(&dEnc, n, stream);
  cudaMemcpyAsync(dIn, in.data(), n * sizeof(float), cudaMemcpyHostToDevice, stream);
  e4m3Kernel<<<n / 256, 256, 0, stream>>>(dIn, dEnc, dDec, n);
  std::vector<uint8_t> enc(n);
  std::vector<float> dec(256);
  cudaMemcpyAsync(enc.data(), dEnc, n, cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(dec.data(), dDec, 256 * sizeof(float), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  CHECK_CUDA(cudaPeekAtLastError());
  cudaFreeAsync(dIn, stream);
  cudaFreeAsync(dDec, stream);
  cudaFreeAsync(dEnc, stream);
  size_t encMismatch = 0, decMismatch = 0, roundTrip = 0;
  for (size_t i = 0; i < n; ++i) {
    if (enc[i] != hostE4M3Encode(in[i])) {
      ++encMismatch;
    }
  }
  for (int b = 0; b < 256; ++b) {
    const float h = hostE4M3Decode(static_cast<uint8_t>(b));
    const bool bothNaN = std::isnan(h) && std::isnan(dec[b]);
    if (!bothNaN && std::memcmp(&h, &dec[b], sizeof(float)) != 0) {
      ++decMismatch;
    }
    if (!std::isnan(h) && hostE4M3Encode(dec[b]) != b) {
      ++roundTrip;
    }
  }
  printf("e4m3 float->e4m3 (RN, satfinite) vs brute force, %zu, %zu, log-uniform magnitudes 2^-12..2^10\n",
         n, encMismatch);
  printf("e4m3 e4m3->float vs formula, 256, %zu, all byte patterns\n", decMismatch);
  printf("e4m3 decode -> encode round trip, 254, %zu, all non-NaN byte patterns\n", roundTrip);
}

__host__ __forceinline__
size_t layoutMismatches(const int rows, const int K, const bool isSFA, size_t &cosizeMismatch) {
  using Config = cutlass::detail::Sm1xxBlockScaledConfig<32>;
  const int rp = (rows + 127) / 128 * 128;
  const int kp = (K + 127) / 128 * 128;
  const int numKBlocks = K / 32;
  size_t bad = 0;
  auto count = [&](const auto &layout) {
    if (static_cast<size_t>(cute::cosize(layout)) != mx::sfBytes(rp, kp / 32)) {
      ++cosizeMismatch;
    }
    for (int r = 0; r < rows; ++r) {
      for (int kb = 0; kb < numKBlocks; ++kb) {
        if (static_cast<size_t>(layout(r, kb * 32)) != mx::sfOffset(r, kb, numKBlocks)) {
          ++bad;
        }
      }
    }
  };
  if (isSFA) {
    count(Config::tile_atom_to_shape_SFA(cute::make_shape(rp, 1, kp)));
  }
  else {
    count(Config::tile_atom_to_shape_SFB(cute::make_shape(1, rp, kp)));
  }
  return bad;
}

__host__ __forceinline__
void checkSfLayout() {
  // (rows, K); rows / K are padded to 128 for the CuTe layout, which requires full atoms
  const int shapes[][2] = {{128, 128}, {256, 512}, {129, 96}, {300, 160}, {768, 2048}, {2048, 768},
                           {5120, 8192}, {8192, 5120}, {1, 32}, {127, 4096}};
  size_t cases = 0, bad = 0, cosizeBad = 0;
  for (const auto &s : shapes) {
    for (const bool isSFA : {true, false}) {
      bad += layoutMismatches(s[0], s[1], isSFA, cosizeBad);
      cases += static_cast<size_t>(s[0]) * (s[1] / 32);
    }
  }
  printf("sfOffset vs CUTLASS Sm1xxBlockScaledConfig<32> SFA+SFB, %zu, %zu, "
         "%zu shapes; cosize != sfBytes in %zu\n", cases, bad, std::size(shapes) * 2, cosizeBad);
}

// ------------------------------------------------------------------------------------------------------------
// quantize / dequantize
// ------------------------------------------------------------------------------------------------------------

// multiplies every 32-element block by 2^r, r in [-10, 10], to exercise the scales
template<typename Element>
__global__ void scaleBlocksKernel(Element *__restrict__ x, const size_t n, const uint64_t seed) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const int r = static_cast<int>(mix64((i / 32) ^ seed) % 21) - 10;
  constexpr Converter<float, Element> load{};
  constexpr Converter<Element, float> store{};
  x[i] = store(ldexpf(load(x[i]), r));
}

template<typename Element>
__host__ __forceinline__
void fillInput(Element *const &x, const size_t &n, const bool wide, const uint64_t seed, cudaStream_t stream) {
  randUniform<arch>(x, n, seed, -1.f, 1.f, stream);
  if (wide) {
    scaleBlocksKernel<Element><<<(n + 255) / 256, 256, 0, stream>>>(x, n, seed ^ 0x9e3779b97f4a7c15ULL);
  }
}

struct QuantStats {
  double maxErrVsAmax = 0; // max |deq - x| / amax(block)
  double meanRelErr = 0; // sum |deq - x| / sum |x|
  double relFrob = 0; // ||deq - x||_F / ||x||_F
  float maxAbsQ = 0;
  size_t scaleMismatch = 0;
  size_t dataMismatch = 0;
  size_t dequantMismatch = 0;
  size_t badQ = 0; // NaN or |q| > 448
  size_t paddingNonzero = 0;
  size_t offsetCollisions = 0;
};

template<typename Element>
__host__ __forceinline__
QuantStats verifyQuant(const std::vector<Element> &x, const std::vector<uint8_t> &q, const std::vector<uint8_t> &sf,
                       const std::vector<float> &deqF, const std::vector<__nv_bfloat16> &deqB,
                       const int rows, const int K) {
  QuantStats s;
  const int numKBlocks = K / 32;
  float lut[256];
  for (int b = 0; b < 256; ++b) {
    lut[b] = hostE4M3Decode(static_cast<uint8_t>(b));
  }
  std::vector<uint8_t> used(sf.size(), 0);
  double sumErr = 0, sumAbs = 0, sumSqErr = 0, sumSq = 0;
  for (int r = 0; r < rows; ++r) {
    for (int kb = 0; kb < numKBlocks; ++kb) {
      const size_t base = static_cast<size_t>(r) * K + static_cast<size_t>(kb) * 32;
      float xf[32];
      float amax = 0.f;
      for (int j = 0; j < 32; ++j) {
        xf[j] = hostToFloat(x[base + j]);
        amax = std::max(amax, std::fabs(xf[j]));
      }
      const size_t off = mx::sfOffset(r, kb, numKBlocks);
      if (used[off]) {
        ++s.offsetCollisions;
      }
      used[off] = 1;
      const uint8_t expect = hostUe8m0(amax);
      if (sf[off] != expect) {
        ++s.scaleMismatch;
      }
      const float inv = mx::ue8m0ToRcpFloat(expect);
      const float scale = mx::ue8m0ToFloat(sf[off]);
      for (int j = 0; j < 32; ++j) {
        const uint8_t qe = static_cast<uint8_t>(__nv_cvt_float_to_fp8(xf[j] * inv, __NV_SATFINITE, __NV_E4M3));
        const uint8_t qg = q[base + j];
        if (qe != qg) {
          ++s.dataMismatch;
        }
        const float v = lut[qg];
        if (std::isnan(v) || std::fabs(v) > 448.f) {
          ++s.badQ;
        }
        else {
          s.maxAbsQ = std::max(s.maxAbsQ, std::fabs(v));
        }
        const float dq = v * scale;
        if (std::memcmp(&dq, &deqF[base + j], sizeof(float)) != 0) {
          ++s.dequantMismatch;
        }
        const __nv_bfloat16 db = __float2bfloat16_rn(dq);
        if (std::memcmp(&db, &deqB[base + j], sizeof(__nv_bfloat16)) != 0) {
          ++s.dequantMismatch;
        }
        const double err = std::fabs(static_cast<double>(dq) - xf[j]);
        if (amax > 0.f) {
          s.maxErrVsAmax = std::max(s.maxErrVsAmax, err / amax);
        }
        sumErr += err;
        sumAbs += std::fabs(xf[j]);
        sumSqErr += err * err;
        sumSq += static_cast<double>(xf[j]) * xf[j];
      }
    }
  }
  // every byte of sf that is not the scale of a (row, k-block) is padding and must be zero
  for (size_t i = 0; i < sf.size(); ++i) {
    if (!used[i] && sf[i] != 0) {
      ++s.paddingNonzero;
    }
  }
  s.meanRelErr = sumErr / std::max(sumAbs, 1e-300);
  s.relFrob = std::sqrt(sumSqErr / std::max(sumSq, 1e-300));
  return s;
}

template<typename Element>
__host__ __forceinline__
void quantCase(const int rows, const int K, const bool wide, void *const &scratch, const size_t &scratchBytes,
               cudaStream_t stream) {
  const size_t n = static_cast<size_t>(rows) * K;
  const int numKBlocks = K / 32;
  const size_t sfSize = mx::sfBytes(rows, numKBlocks);
  Element *x = nullptr;
  uint8_t *q = nullptr, *sf = nullptr;
  float *deqF = nullptr;
  __nv_bfloat16 *deqB = nullptr;
  CHECK_CUDA(cudaMallocAsync(&x, n * sizeof(Element), stream));
  CHECK_CUDA(cudaMallocAsync(&q, n, stream));
  CHECK_CUDA(cudaMallocAsync(&sf, sfSize, stream));
  CHECK_CUDA(cudaMallocAsync(&deqF, n * sizeof(float), stream));
  CHECK_CUDA(cudaMallocAsync(&deqB, n * sizeof(__nv_bfloat16), stream));
  std::random_device rd;
  fillInput(x, n, wide, rd(), stream);
  // poison the outputs: the quantizer must overwrite every valid entry and zero the padding of sf
  cudaMemsetAsync(q, 0xAA, n, stream);
  cudaMemsetAsync(sf, 0xAA, sfSize, stream);
  mx::quantizeMXFP8<Element>(x, q, sf, rows, K, stream);
  mx::dequantizeMXFP8<float>(q, sf, deqF, rows, K, stream);
  mx::dequantizeMXFP8<__nv_bfloat16>(q, sf, deqB, rows, K, stream);
  std::vector<Element> hx(n);
  std::vector<uint8_t> hq(n), hsf(sfSize);
  std::vector<float> hdF(n);
  std::vector<__nv_bfloat16> hdB(n);
  cudaMemcpyAsync(hx.data(), x, n * sizeof(Element), cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(hq.data(), q, n, cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(hsf.data(), sf, sfSize, cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(hdF.data(), deqF, n * sizeof(float), cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(hdB.data(), deqB, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  CHECK_CUDA(cudaPeekAtLastError());
  const auto s = verifyQuant<Element>(hx, hq, hsf, hdF, hdB, rows, K);

  // timing, with the L2 flushed before every iteration
  constexpr int runs = 20;
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  float quantMs = 0.f, dequantMs = 0.f;
  for (int i = 0; i < runs + 2; ++i) {
    float ms = 0.f;
    cudaMemsetAsync(scratch, i, scratchBytes, stream);
    cudaEventRecord(e0, stream);
    mx::quantizeMXFP8<Element>(x, q, sf, rows, K, stream);
    cudaEventRecord(e1, stream);
    cudaEventSynchronize(e1);
    cudaEventElapsedTime(&ms, e0, e1);
    quantMs += i >= 2 ? ms : 0.f; // first two iterations are warmup
    cudaMemsetAsync(scratch, i, scratchBytes, stream);
    cudaEventRecord(e0, stream);
    mx::dequantizeMXFP8<__nv_bfloat16>(q, sf, deqB, rows, K, stream);
    cudaEventRecord(e1, stream);
    cudaEventSynchronize(e1);
    cudaEventElapsedTime(&ms, e0, e1);
    dequantMs += i >= 2 ? ms : 0.f;
  }
  quantMs /= runs;
  dequantMs /= runs;
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  CHECK_CUDA(cudaPeekAtLastError());
  const double quantGB = static_cast<double>(n * sizeof(Element) + n + sfSize) * 1e-9;
  const double dequantGB = static_cast<double>(n + sfSize + n * sizeof(__nv_bfloat16)) * 1e-9;
  printf("%d, %d, %s, %s, %.4f, %.4f, %.4f, %.1f, %zu, %zu, %zu, %zu, %zu, %zu, %.4f, %.1f, %.4f, %.1f\n",
         rows, K, element_string<Element>(), wide ? "wide" : "uniform",
         s.maxErrVsAmax * 100, s.meanRelErr * 100, s.relFrob * 100, s.maxAbsQ,
         s.scaleMismatch, s.dataMismatch, s.dequantMismatch, s.badQ, s.paddingNonzero, s.offsetCollisions,
         quantMs, quantGB / (quantMs * 1e-3), dequantMs, dequantGB / (dequantMs * 1e-3));
  cudaFreeAsync(x, stream);
  cudaFreeAsync(q, stream);
  cudaFreeAsync(sf, stream);
  cudaFreeAsync(deqF, stream);
  cudaFreeAsync(deqB, stream);
  cudaStreamSynchronize(stream);
}

__host__ __forceinline__
void quantSection(cudaStream_t stream) {
  constexpr size_t scratchBytes = 256ULL << 20; // larger than the 96 MB L2
  void *scratch = nullptr;
  CHECK_CUDA(cudaMalloc(&scratch, scratchBytes));
  printf("rows, K, dtype, data, max_err_vs_blockamax(%%), mean_rel_err(%%), rel_frobenius(%%), max|q|, "
         "scale_mismatch, data_mismatch, dequant_mismatch, bad_q, padding_nonzero, offset_collisions, "
         "Quant_Time(ms), Quant_GB/s, Dequant_bf16_Time(ms), Dequant_GB/s\n");
  const int shapes[][2] = {{130, 96}, {257, 1536}, {1024, 4096}, {8192, 5120}};
  for (const auto &s : shapes) {
    for (const bool wide : {false, true}) {
      quantCase<__nv_bfloat16>(s[0], s[1], wide, scratch, scratchBytes, stream);
    }
    quantCase<__half>(s[0], s[1], false, scratch, scratchBytes, stream);
    quantCase<float>(s[0], s[1], false, scratch, scratchBytes, stream);
  }
  cudaFree(scratch);
}

// ------------------------------------------------------------------------------------------------------------
// GEMM
// ------------------------------------------------------------------------------------------------------------

__global__ void bf16ToFloatKernel(const __nv_bfloat16 *__restrict__ in, float *__restrict__ out, const size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = __bfloat162float(in[i]);
  }
}

// out[0] = sum (x - y)^2, out[1] = sum y^2, out[2] = max |x - y| (as a double bit pattern), out[3] = #(|x - y| > atol + rtol |y|)
__global__ void diffKernel(const float *__restrict__ x, const float *__restrict__ y, const size_t n,
                           const float rtol, const float atol, double *__restrict__ out) {
  double sumSqDiff = 0, sumSqRef = 0, maxAbs = 0;
  unsigned long long fails = 0;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const double d = static_cast<double>(x[i]) - y[i];
    sumSqDiff += d * d;
    sumSqRef += static_cast<double>(y[i]) * y[i];
    maxAbs = fmax(maxAbs, fabs(d));
    // NaN fails the comparison
    fails += !(fabs(d) <= static_cast<double>(atol) + static_cast<double>(rtol) * fabs(static_cast<double>(y[i])));
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    sumSqDiff += __shfl_down_sync(0xFFFFFFFFU, sumSqDiff, o);
    sumSqRef += __shfl_down_sync(0xFFFFFFFFU, sumSqRef, o);
    maxAbs = fmax(maxAbs, __shfl_down_sync(0xFFFFFFFFU, maxAbs, o));
    fails += __shfl_down_sync(0xFFFFFFFFU, fails, o);
  }
  if ((threadIdx.x & 31) == 0) {
    atomicAdd(out, sumSqDiff);
    atomicAdd(out + 1, sumSqRef);
    atomicMax(reinterpret_cast<unsigned long long *>(out + 2), static_cast<unsigned long long>(__double_as_longlong(maxAbs)));
    atomicAdd(reinterpret_cast<unsigned long long *>(out + 3), fails);
  }
}

struct DiffStats {
  double relFrob; // ||x - y||_F / ||y||_F
  double rms; // rms of y
  double maxAbs;
  double failPct;
};

__host__ __forceinline__
DiffStats diffStats(const float *const &x, const float *const &y, const size_t &n, const float &rtol,
                    const float &atol, double *const &scratch, cudaStream_t stream) {
  cudaMemsetAsync(scratch, 0, 4 * sizeof(double), stream);
  diffKernel<<<1024, 256, 0, stream>>>(x, y, n, rtol, atol, scratch);
  double h[4];
  cudaMemcpyAsync(h, scratch, sizeof(h), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  unsigned long long fails;
  std::memcpy(&fails, &h[3], sizeof(fails));
  return {std::sqrt(h[0] / std::max(h[1], 1e-300)), std::sqrt(h[1] / static_cast<double>(n)), h[2],
          100.0 * static_cast<double>(fails) / static_cast<double>(n)};
}

// reference of the reference: C[r, :] for a few rows, one warp per output, double accumulation
__global__ void naiveRowsKernel(const float *__restrict__ a, const float *__restrict__ b, const int *__restrict__ rowIds,
                                double *__restrict__ out, const int N, const int K) {
  const int warp = static_cast<int>((static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32);
  const int lane = threadIdx.x % 32;
  const int rowSlot = warp / N;
  const int col = warp % N;
  if (rowSlot >= 4) {
    return;
  }
  const float *ar = a + static_cast<size_t>(rowIds[rowSlot]) * K;
  const float *br = b + static_cast<size_t>(col) * K;
  double acc = 0;
  for (int k = lane; k < K; k += 32) {
    acc += static_cast<double>(ar[k]) * br[k];
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    acc += __shfl_down_sync(0xFFFFFFFFU, acc, o);
  }
  if (lane == 0) {
    out[static_cast<size_t>(rowSlot) * N + col] = acc;
  }
}

// C[M, N] (row-major) = A[M, K] @ B[N, K]^T through cuBLASLt, which is column-major:
// cuBLAS operand "A" is our B (a K x N column-major matrix, TRANSA = T), operand "B" is our A (K x M, TRANSB = N),
// m = N, n = M, k = K and the column-major D (N x M) is our row-major C.
struct LtGemm {
  cublasLtHandle_t handle{};
  cublasLtMatmulDesc_t op{};
  cublasLtMatrixLayout_t aL{}, bL{}, cL{};
  cublasLtMatmulAlgo_t algo{};
  void *workspace = nullptr;
  size_t workspaceBytes = 0;
  bool valid = false;

  // sfA / sfB: the scale factors of our A [M, K] and B [N, K]. mx == false: BF16 operands
  __host__ __forceinline__
  void init(cublasLtHandle_t h, const bool mxfp8, const int M, const int N, const int K,
            const void *const &sfA, const void *const &sfB, void *const &ws, const size_t &wsBytes) {
    handle = h;
    workspace = ws;
    workspaceBytes = wsBytes;
    const auto inType = mxfp8 ? CUDA_R_8F_E4M3 : CUDA_R_16BF;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t transA = CUBLAS_OP_T;
    const cublasOperation_t transB = CUBLAS_OP_N;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));
    if (mxfp8) {
      const cublasLtMatmulMatrixScale_t mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
      CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode)));
      CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode)));
      // cuBLAS A = our B, cuBLAS B = our A
      const void *scaleA = sfB;
      const void *scaleB = sfA;
      CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scaleA, sizeof(scaleA)));
      CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scaleB, sizeof(scaleB)));
    }
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&aL, inType, K, N, K));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&bL, inType, K, M, K));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&cL, CUDA_R_16BF, N, M, N));
    cublasLtMatmulPreference_t pref{};
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&pref));
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &workspaceBytes, sizeof(workspaceBytes)));
    cublasLtMatmulHeuristicResult_t result{};
    int found = 0;
    const auto status = cublasLtMatmulAlgoGetHeuristic(handle, op, aL, bL, cL, cL, pref, 1, &result, &found);
    cublasLtMatmulPreferenceDestroy(pref);
    if (status != CUBLAS_STATUS_SUCCESS || found == 0) {
      fprintf(stderr, "cublasLtMatmulAlgoGetHeuristic found no %s algorithm for M=%d N=%d K=%d (status %d)\n",
              mxfp8 ? "MXFP8" : "BF16", M, N, K, static_cast<int>(status));
      return;
    }
    algo = result.algo;
    valid = true;
  }

  // a: our A, b: our B (both K-major), c: our row-major C
  __host__ __forceinline__
  void run(const void *const &a, const void *const &b, void *const &c, cudaStream_t stream) const {
    const float alpha = 1.f;
    const float beta = 0.f;
    CHECK_CUBLAS(cublasLtMatmul(handle, op, &alpha, b, aL, a, bL, &beta, c, cL, c, cL, &algo,
      workspace, workspaceBytes, stream));
  }

  __host__ __forceinline__
  void destroy() {
    cublasLtMatmulDescDestroy(op);
    cublasLtMatrixLayoutDestroy(aL);
    cublasLtMatrixLayoutDestroy(bL);
    cublasLtMatrixLayoutDestroy(cL);
  }
};

template<typename F>
__host__ __forceinline__
float timeMs(F &&f, cudaStream_t stream, const int warmup, const int runs) {
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  for (int i = 0; i < warmup; ++i) {
    f();
  }
  cudaEventRecord(e0, stream);
  for (int i = 0; i < runs; ++i) {
    f();
  }
  cudaEventRecord(e1, stream);
  cudaEventSynchronize(e1);
  float ms = 0.f;
  cudaEventElapsedTime(&ms, e0, e1);
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  CHECK_CUDA(cudaPeekAtLastError());
  return ms / static_cast<float>(runs);
}

__host__ __forceinline__
double tflops(const int M, const int N, const int K, const double ms) {
  return 2.0 * M * N * K / (ms * 1e-3) * 1e-12;
}

__host__ __forceinline__
void gemmCase(const int M, const int N, const int K, const bool wide, const int runs, cublasLtHandle_t lt,
              void *const &workspace, const size_t &workspaceBytes, matx::cudaExecutor &exec) {
  auto stream = exec.getStream();
  const size_t nA = static_cast<size_t>(M) * K;
  const size_t nB = static_cast<size_t>(N) * K;
  const size_t nC = static_cast<size_t>(M) * N;
  using bf16 = __nv_bfloat16;
  bf16 *a, *b, *cLt, *cBf;
  uint8_t *aq, *bq, *aSF, *bSF;
  float *aF, *bF, *cRef, *cF, *cG;
  double *stats;
  int *rowIds;
  double *naiveOut;
  const size_t aSFBytes = mx::sfBytes(M, K / 32);
  const size_t bSFBytes = mx::sfBytes(N, K / 32);
  CHECK_CUDA(cudaMallocAsync(&a, nA * sizeof(bf16), stream));
  CHECK_CUDA(cudaMallocAsync(&b, nB * sizeof(bf16), stream));
  CHECK_CUDA(cudaMallocAsync(&cLt, nC * sizeof(bf16), stream));
  CHECK_CUDA(cudaMallocAsync(&cBf, nC * sizeof(bf16), stream));
  CHECK_CUDA(cudaMallocAsync(&aq, nA, stream));
  CHECK_CUDA(cudaMallocAsync(&bq, nB, stream));
  CHECK_CUDA(cudaMallocAsync(&aSF, aSFBytes, stream));
  CHECK_CUDA(cudaMallocAsync(&bSF, bSFBytes, stream));
  CHECK_CUDA(cudaMallocAsync(&aF, nA * sizeof(float), stream));
  CHECK_CUDA(cudaMallocAsync(&bF, nB * sizeof(float), stream));
  CHECK_CUDA(cudaMallocAsync(&cRef, nC * sizeof(float), stream));
  CHECK_CUDA(cudaMallocAsync(&cF, nC * sizeof(float), stream));
  CHECK_CUDA(cudaMallocAsync(&cG, nC * sizeof(float), stream));
  CHECK_CUDA(cudaMallocAsync(&stats, 4 * sizeof(double), stream));
  CHECK_CUDA(cudaMallocAsync(&rowIds, 4 * sizeof(int), stream));
  CHECK_CUDA(cudaMallocAsync(&naiveOut, 4 * static_cast<size_t>(N) * sizeof(double), stream));

  std::random_device rd;
  fillInput(a, nA, wide, rd(), stream);
  fillInput(b, nB, wide, rd(), stream);
  // quantize, then dequantize to FP32 for the reference
  mx::quantizeMXFP8<bf16>(a, aq, aSF, M, K, stream);
  mx::quantizeMXFP8<bf16>(b, bq, bSF, N, K, stream);
  mx::dequantizeMXFP8<float>(aq, aSF, aF, M, K, stream);
  mx::dequantizeMXFP8<float>(bq, bSF, bF, N, K, stream);

  // reference: FP32 GEMM of the dequantized operands (MatX -> cuBLASLt SGEMM)
  auto tAF = matx::make_tensor<float>(aF, {M, K});
  auto tBF = matx::make_tensor<float>(bF, {N, K});
  auto tRef = matx::make_tensor<float>(cRef, {M, N});
  (tRef = matx::matmul(tAF, tBF.PermuteMatrix())).run(exec);
  // BF16 baseline of the original operands (MatX -> cuBLASLt)
  auto tA16 = matx::make_tensor<matx::matxBf16>(reinterpret_cast<matx::matxBf16 *>(a), {M, K});
  auto tB16 = matx::make_tensor<matx::matxBf16>(reinterpret_cast<matx::matxBf16 *>(b), {N, K});
  auto tC16 = matx::make_tensor<matx::matxBf16>(reinterpret_cast<matx::matxBf16 *>(cBf), {M, N});
  (tC16 = matx::matmul(tA16, tB16.PermuteMatrix())).run(exec);
  // the same through our own cuBLASLt wrapper, so that both BF16 baselines are reported
  LtGemm bf16Gemm;
  bf16Gemm.init(lt, false, M, N, K, nullptr, nullptr, workspace, workspaceBytes);
  LtGemm mxGemm;
  mxGemm.init(lt, true, M, N, K, aSF, bSF, workspace, workspaceBytes);
  if (!mxGemm.valid) {
    cudaStreamSynchronize(stream);
    printf("%d, %d, %d, %s, no cuBLASLt MXFP8 algorithm\n", M, N, K, wide ? "wide" : "uniform");
    return;
  }
  mxGemm.run(aq, bq, cLt, stream);
  exec.sync();
  CHECK_CUDA(cudaPeekAtLastError());

  // sanity of the reference itself: 4 rows against a double precision dot product
  const int hRows[4] = {0, M / 3, M / 2 + 1, M - 1};
  cudaMemcpyAsync(rowIds, hRows, sizeof(hRows), cudaMemcpyHostToDevice, stream);
  naiveRowsKernel<<<static_cast<unsigned>((4ULL * N * 32 + 255) / 256), 256, 0, stream>>>(aF, bF, rowIds, naiveOut, N, K);
  std::vector<double> hNaive(4 * static_cast<size_t>(N));
  std::vector<float> hRefRows(4 * static_cast<size_t>(N));
  cudaMemcpyAsync(hNaive.data(), naiveOut, hNaive.size() * sizeof(double), cudaMemcpyDeviceToHost, stream);
  for (int i = 0; i < 4; ++i) {
    cudaMemcpyAsync(hRefRows.data() + static_cast<size_t>(i) * N, cRef + static_cast<size_t>(hRows[i]) * N,
                    static_cast<size_t>(N) * sizeof(float), cudaMemcpyDeviceToHost, stream);
  }
  cudaStreamSynchronize(stream);
  double dd = 0, rr = 0;
  for (size_t i = 0; i < hNaive.size(); ++i) {
    dd += (hNaive[i] - hRefRows[i]) * (hNaive[i] - hRefRows[i]);
    rr += hNaive[i] * hNaive[i];
  }
  const double refVsNaive = std::sqrt(dd / std::max(rr, 1e-300));

  // compare cuBLASLt MXFP8 (bf16 output) with the FP32 reference
  const size_t convBlocks = (nC + 255) / 256;
  bf16ToFloatKernel<<<convBlocks, 256, 0, stream>>>(cLt, cF, nC);
  bf16ToFloatKernel<<<convBlocks, 256, 0, stream>>>(cBf, cG, nC);
  // tolerances: the output is rounded to BF16 (relative error <= 2^-9 per rounding, two roundings of values that
  // differ by FP32 accumulation order -> at most 1 BF16 ulp = 2^-7 of the value), so rtol = 1e-2 is > 1 ulp.
  // atol covers cancellation: 1e-3 of the rms of the reference output, far above FP32 accumulation-order noise.
  constexpr float rtol = 1e-2f;
  const auto base = diffStats(cF, cRef, nC, rtol, 0.f, stats, stream);
  const float atol = static_cast<float>(1e-3 * base.rms);
  auto tCx = matx::make_tensor<float>(cF, {M, N});
  auto numMatches = matx::make_tensor<long int>({});
  (numMatches = matx::sum(matx::isclose(tCx, tRef, rtol, atol))).run(exec);
  exec.sync();
  const double errorPct = (1.0 - static_cast<double>(numMatches()) / static_cast<double>(nC)) * 100.0;
  const auto vsRef = diffStats(cF, cRef, nC, rtol, atol, stats, stream);
  // quantization error of MXFP8 relative to the BF16 GEMM of the original operands
  const auto vsBf16 = diffStats(cF, cG, nC, rtol, atol, stats, stream);

  // timing
  const int warmup = 10;
  const float mxMs = timeMs([&] { mxGemm.run(aq, bq, cLt, stream); }, stream, warmup, runs);
  const float matxMs = timeMs([&] { (tC16 = matx::matmul(tA16, tB16.PermuteMatrix())).run(exec); }, stream,
                              warmup, runs);
  float ltBf16Ms = -1.f;
  if (bf16Gemm.valid) {
    ltBf16Ms = timeMs([&] { bf16Gemm.run(a, b, cBf, stream); }, stream, warmup, runs);
  }
  printf("%d, %d, %d, %s, %.1e, %.3e, %.4f, %.4f, %.3e, %.4f, %.4f, %.3f, %.1f, %.3f, %.1f, %.3f, %.1f, %.2f, %.2f\n",
         M, N, K, wide ? "wide" : "uniform", rtol, atol, errorPct, vsRef.relFrob * 100, refVsNaive,
         vsBf16.relFrob * 100, vsRef.maxAbs / std::max(base.rms, 1e-30),
         static_cast<double>(mxMs), tflops(M, N, K, mxMs), static_cast<double>(matxMs), tflops(M, N, K, matxMs),
         static_cast<double>(ltBf16Ms), ltBf16Ms > 0 ? tflops(M, N, K, ltBf16Ms) : -1.0,
         static_cast<double>(matxMs / mxMs), ltBf16Ms > 0 ? static_cast<double>(ltBf16Ms / mxMs) : -1.0);
  fflush(stdout);

  mxGemm.destroy();
  if (bf16Gemm.valid) {
    bf16Gemm.destroy();
  }
  cudaFreeAsync(a, stream);
  cudaFreeAsync(b, stream);
  cudaFreeAsync(cLt, stream);
  cudaFreeAsync(cBf, stream);
  cudaFreeAsync(aq, stream);
  cudaFreeAsync(bq, stream);
  cudaFreeAsync(aSF, stream);
  cudaFreeAsync(bSF, stream);
  cudaFreeAsync(aF, stream);
  cudaFreeAsync(bF, stream);
  cudaFreeAsync(cRef, stream);
  cudaFreeAsync(cF, stream);
  cudaFreeAsync(cG, stream);
  cudaFreeAsync(stats, stream);
  cudaFreeAsync(rowIds, stream);
  cudaFreeAsync(naiveOut, stream);
  cudaStreamSynchronize(stream);
}

__host__ __forceinline__
void gemmSection(const std::vector<std::array<int, 3>> &shapes, const int runs, matx::cudaExecutor &exec) {
  cublasLtHandle_t lt{};
  CHECK_CUBLAS(cublasLtCreate(&lt));
  constexpr size_t workspaceBytes = 32ULL << 20;
  void *workspace = nullptr;
  CHECK_CUDA(cudaMalloc(&workspace, workspaceBytes));
  printf("M, N, K, data, rtol, atol, error(%%), relFrob_vs_ref(%%), ref_vs_naive_relFrob, mxfp8_vs_bf16_relFrob(%%), "
         "maxAbsErr/rms, MXFP8_Time(ms), MXFP8_TFLOPS, MatX_BF16_Time(ms), MatX_BF16_TFLOPS, "
         "cublasLt_BF16_Time(ms), cublasLt_BF16_TFLOPS, Speedup_vs_MatX_BF16, Speedup_vs_cublasLt_BF16\n");
  for (const auto &s : shapes) {
    for (const bool wide : {false, true}) {
      gemmCase(s[0], s[1], s[2], wide, runs, lt, workspace, workspaceBytes, exec);
    }
  }
  cudaFree(workspace);
  cublasLtDestroy(lt);
}

// ------------------------------------------------------------------------------------------------------------
// MXFP8 tile mainloop and epilogues (tile_sm120_mxfp8.cuh) in a persistent kernel
// ------------------------------------------------------------------------------------------------------------

template<int bN, int bK, int stages, bool gated>
using TileMainloop = flashmoe::tile::sm120::MXFP8Mainloop<128, bN, bK, stages, 256, gated>;


using IdentityAct = flashmoe::ActivationType<float, flashmoe::Activation::identity>::AT;
using SiluAct = flashmoe::ActivationType<float, flashmoe::Activation::silu>::AT;
using ReluAct = flashmoe::ActivationType<float, flashmoe::Activation::relu>::AT;
using GeluAct = flashmoe::ActivationType<float, flashmoe::Activation::gelu>::AT;

// stands in for whatever else uses the workspace between two GEMM tasks of a persistent processor
template<typename Mainloop>
__device__ __forceinline__
void poisonWorkspace(cuda::std::byte *__restrict__ const &workspace) {
  auto *w = reinterpret_cast<uint4 *>(workspace);
  for (int i = static_cast<int>(threadIdx.x); i < Mainloop::sharedSize / 16; i += 256) {
    w[i] = make_uint4(0xDEADBEEFU, 0xFFFFFFFFU, 0x7F7F7F7FU, 0x01010101U);
  }
  __syncthreads();
}

// epilogue of the vanilla tile kernel; sink: no epilogue, the accumulator is only reduced and written conditionally
// (impossible condition), i.e., the mainloop alone is timed
enum class TileEpi {
  bf16,
  mxfp8,
  sink
};

// persistent: CTA b computes tiles b, b + gridDim.x, ... with the same workspace
template<typename Mainloop, typename Act, TileEpi epi>
__global__ __launch_bounds__(256, 1)
void tileVanillaKernel(const uint8_t *__restrict__ a, const uint8_t *__restrict__ sfa,
                       const uint8_t *__restrict__ b, const uint8_t *__restrict__ sfb,
                       __nv_bfloat16 *__restrict__ outC, uint8_t *__restrict__ outQ, uint8_t *__restrict__ outSF,
                       const __nv_bfloat16 *__restrict__ bias, const int M, const int N, const int K,
                       const int poison) {
  extern __shared__ __align__(1024) cuda::std::byte workspace[];
  constexpr int bM = cute::get<0>(typename Mainloop::TileShape{});
  constexpr int bN = cute::get<1>(typename Mainloop::TileShape{});
  constexpr Mainloop mainloop{};
  const int tilesM = M / bM;
  const int tilesN = N / bN;
  for (int tile = static_cast<int>(blockIdx.x); tile < tilesM * tilesN; tile += static_cast<int>(gridDim.x)) {
    const auto tileCoord = flashmoe::tile::idx2Coord(tilesM, tilesN, tile);
    auto acc = Mainloop::makeAccumulator();
    mainloop(workspace, a, sfa, b, sfb, acc, M, N, K, tileCoord);
    if constexpr (epi == TileEpi::bf16) {
      Mainloop::template epilogueBF16<Act>(acc, outC, bias, M, N, tileCoord);
    }
    else if constexpr (epi == TileEpi::mxfp8) {
      Mainloop::template epilogueMXFP8<Act>(acc, outQ, outSF, bias, M, N, tileCoord);
    }
    else {
      float s = 0.f;
      #pragma unroll
      for (int i = 0; i < cute::size(acc); ++i) {
        s += acc(i);
      }
      if (s == 1234567.f) {
        outC[threadIdx.x] = __float2bfloat16(s);
      }
    }
    if (poison) {
      poisonWorkspace<Mainloop>(workspace);
    }
  }
}

template<typename Mainloop, typename Act>
__global__ __launch_bounds__(256, 1)
void tileGatedKernel(const uint8_t *__restrict__ a, const uint8_t *__restrict__ sfa,
                     const uint8_t *__restrict__ b, const uint8_t *__restrict__ sfb,
                     const uint8_t *__restrict__ b2, const uint8_t *__restrict__ sfb2,
                     uint8_t *__restrict__ outQ, uint8_t *__restrict__ outSF,
                     const __nv_bfloat16 *__restrict__ bias, const __nv_bfloat16 *__restrict__ biasV,
                     const float alpha, const float beta, const int M, const int N, const int K, const int poison) {
  extern __shared__ __align__(1024) cuda::std::byte workspace[];
  constexpr int bM = cute::get<0>(typename Mainloop::TileShape{});
  constexpr int bN = cute::get<1>(typename Mainloop::TileShape{});
  constexpr Mainloop mainloop{};
  const int tilesM = M / bM;
  const int tilesN = N / bN;
  for (int tile = static_cast<int>(blockIdx.x); tile < tilesM * tilesN; tile += static_cast<int>(gridDim.x)) {
    const auto tileCoord = flashmoe::tile::idx2Coord(tilesM, tilesN, tile);
    auto acc = Mainloop::makeAccumulator();
    auto acc2 = Mainloop::makeAccumulator();
    mainloop(workspace, a, sfa, b, sfb, b2, sfb2, acc, acc2, M, N, K, tileCoord);
    Mainloop::template epilogueGatedMXFP8<Act>(acc, acc2, outQ, outSF, bias, biasV, alpha, beta, M, N, tileCoord);
    if (poison) {
      poisonWorkspace<Mainloop>(workspace);
    }
  }
}

// reference epilogues in FP32 on the FP32 GEMM result
template<typename Act>
__global__ void refVanillaKernel(const float *__restrict__ c, const __nv_bfloat16 *__restrict__ bias,
                                 float *__restrict__ y, const size_t n, const int N) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    constexpr Act act{};
    y[i] = act(c[i] + __bfloat162float(bias[i % N]));
  }
}

template<typename Act>
__global__ void refGatedKernel(const float *__restrict__ g, const float *__restrict__ u,
                               const __nv_bfloat16 *__restrict__ bg, const __nv_bfloat16 *__restrict__ bu,
                               float *__restrict__ y, const float alpha, const float beta, const size_t n,
                               const int N) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    constexpr Act act{};
    y[i] = alpha * act((g[i] + __bfloat162float(bg[i % N])) * beta) * (u[i] + __bfloat162float(bu[i % N]));
  }
}

__global__ void neqBytesKernel(const uint8_t *__restrict__ x, const uint8_t *__restrict__ y, const size_t n,
                               unsigned long long *__restrict__ count) {
  unsigned long long c = 0;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    c += x[i] != y[i];
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    c += __shfl_down_sync(0xFFFFFFFFU, c, o);
  }
  if ((threadIdx.x & 31) == 0) {
    atomicAdd(count, c);
  }
}

__host__ __forceinline__
double neqBytesPct(const uint8_t *const &x, const uint8_t *const &y, const size_t &n, unsigned long long *const &dCount,
                   cudaStream_t stream) {
  cudaMemsetAsync(dCount, 0, sizeof(unsigned long long), stream);
  neqBytesKernel<<<1024, 256, 0, stream>>>(x, y, n, dCount);
  unsigned long long h = 0;
  cudaMemcpyAsync(&h, dCount, sizeof(h), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  return 100.0 * static_cast<double>(h) / static_cast<double>(n);
}

// operands, FP32 reference GEMMs and cuBLASLt baselines of one problem; the gated problem has two B matrices
struct TileProblem {
  int M = 0, N = 0, K = 0;
  bool gated = false, wide = false;
  __nv_bfloat16 *a = nullptr, *b = nullptr, *b2 = nullptr, *bias = nullptr, *biasV = nullptr;
  uint8_t *aq = nullptr, *bq = nullptr, *b2q = nullptr, *aSF = nullptr, *bSF = nullptr, *b2SF = nullptr;
  float *refC = nullptr, *refC2 = nullptr; // FP32 A @ B^T, A @ B2^T of the dequantized operands
  float alpha = 1.25f, beta = 0.8f; // gated: y = alpha * act(beta * (g + bias)) * (u + biasV)
  double ltMxMs = -1, ltBf16Ms = -1; // cuBLASLt of the whole problem (gated: two GEMMs)
};

struct TileCtx {
  cudaStream_t stream{};
  matx::cudaExecutor *exec = nullptr;
  int runs = 50;
  int gridOverride = 0; // 0: min(#tiles, #SMs)
  int numSMs = 0;
  bool bench = true;
  std::string filter;
  double *stats = nullptr; // 4 doubles
  unsigned long long *count = nullptr;
};

__host__ __forceinline__
void makeTileProblem(TileProblem &p, const int M, const int N, const int K, const bool gated, const bool wide,
                     const TileCtx &ctx, cublasLtHandle_t lt, void *const &ltWorkspace, const size_t &ltWsBytes) {
  using bf16 = __nv_bfloat16;
  auto stream = ctx.stream;
  p.M = M;
  p.N = N;
  p.K = K;
  p.gated = gated;
  p.wide = wide;
  const size_t nA = static_cast<size_t>(M) * K;
  const size_t nB = static_cast<size_t>(N) * K;
  const size_t nC = static_cast<size_t>(M) * N;
  CHECK_CUDA(cudaMalloc(&p.a, nA * sizeof(bf16)));
  CHECK_CUDA(cudaMalloc(&p.b, nB * sizeof(bf16)));
  CHECK_CUDA(cudaMalloc(&p.bias, N * sizeof(bf16)));
  CHECK_CUDA(cudaMalloc(&p.aq, nA));
  CHECK_CUDA(cudaMalloc(&p.bq, nB));
  CHECK_CUDA(cudaMalloc(&p.aSF, mx::sfBytes(M, K / 32)));
  CHECK_CUDA(cudaMalloc(&p.bSF, mx::sfBytes(N, K / 32)));
  CHECK_CUDA(cudaMalloc(&p.refC, nC * sizeof(float)));
  float *aF, *bF, *b2F = nullptr;
  CHECK_CUDA(cudaMalloc(&aF, nA * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&bF, nB * sizeof(float)));
  std::random_device rd;
  fillInput(p.a, nA, wide, rd(), stream);
  fillInput(p.b, nB, wide, rd(), stream);
  fillInput(p.bias, static_cast<size_t>(N), false, rd(), stream);
  mx::quantizeMXFP8<bf16>(p.a, p.aq, p.aSF, M, K, stream);
  mx::quantizeMXFP8<bf16>(p.b, p.bq, p.bSF, N, K, stream);
  mx::dequantizeMXFP8<float>(p.aq, p.aSF, aF, M, K, stream);
  mx::dequantizeMXFP8<float>(p.bq, p.bSF, bF, N, K, stream);
  if (gated) {
    CHECK_CUDA(cudaMalloc(&p.b2, nB * sizeof(bf16)));
    CHECK_CUDA(cudaMalloc(&p.biasV, N * sizeof(bf16)));
    CHECK_CUDA(cudaMalloc(&p.b2q, nB));
    CHECK_CUDA(cudaMalloc(&p.b2SF, mx::sfBytes(N, K / 32)));
    CHECK_CUDA(cudaMalloc(&p.refC2, nC * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&b2F, nB * sizeof(float)));
    fillInput(p.b2, nB, wide, rd(), stream);
    fillInput(p.biasV, static_cast<size_t>(N), false, rd(), stream);
    mx::quantizeMXFP8<bf16>(p.b2, p.b2q, p.b2SF, N, K, stream);
    mx::dequantizeMXFP8<float>(p.b2q, p.b2SF, b2F, N, K, stream);
  }
  // reference: FP32 GEMM of the dequantized operands (MatX -> cuBLASLt SGEMM)
  auto tAF = matx::make_tensor<float>(aF, {M, K});
  auto tBF = matx::make_tensor<float>(bF, {N, K});
  auto tRef = matx::make_tensor<float>(p.refC, {M, N});
  (tRef = matx::matmul(tAF, tBF.PermuteMatrix())).run(*ctx.exec);
  if (gated) {
    auto tB2F = matx::make_tensor<float>(b2F, {N, K});
    auto tRef2 = matx::make_tensor<float>(p.refC2, {M, N});
    (tRef2 = matx::matmul(tAF, tB2F.PermuteMatrix())).run(*ctx.exec);
  }
  ctx.exec->sync();
  cudaFree(aF);
  cudaFree(bF);
  if (gated) {
    cudaFree(b2F);
  }
  if (!ctx.bench) {
    return;
  }
  // cuBLASLt baselines, same quantized operands and scale factors (gated: gate and up GEMM)
  bf16 *c;
  CHECK_CUDA(cudaMalloc(&c, nC * sizeof(bf16)));
  LtGemm mx1, mx2, bf1;
  mx1.init(lt, true, M, N, K, p.aSF, p.bSF, ltWorkspace, ltWsBytes);
  bf1.init(lt, false, M, N, K, nullptr, nullptr, ltWorkspace, ltWsBytes);
  if (gated) {
    mx2.init(lt, true, M, N, K, p.aSF, p.b2SF, ltWorkspace, ltWsBytes);
  }
  if (mx1.valid && (!gated || mx2.valid)) {
    p.ltMxMs = timeMs([&] {
      mx1.run(p.aq, p.bq, c, stream);
      if (gated) {
        mx2.run(p.aq, p.b2q, c, stream);
      }
    }, stream, 10, ctx.runs);
  }
  if (bf1.valid) {
    p.ltBf16Ms = timeMs([&] {
      bf1.run(p.a, p.b, c, stream);
      if (gated) {
        bf1.run(p.a, p.b2, c, stream);
      }
    }, stream, 10, ctx.runs);
  }
  for (auto *g : {&mx1, &mx2, &bf1}) {
    if (g->valid) {
      g->destroy();
    }
  }
  cudaFree(c);
}

__host__ __forceinline__
void destroyTileProblem(TileProblem &p) {
  for (void *x : {static_cast<void *>(p.a), static_cast<void *>(p.b), static_cast<void *>(p.b2),
                  static_cast<void *>(p.bias), static_cast<void *>(p.biasV), static_cast<void *>(p.aq),
                  static_cast<void *>(p.bq), static_cast<void *>(p.b2q), static_cast<void *>(p.aSF),
                  static_cast<void *>(p.bSF), static_cast<void *>(p.b2SF), static_cast<void *>(p.refC),
                  static_cast<void *>(p.refC2)}) {
    cudaFree(x);
  }
  p = TileProblem{};
}

// Compares the output of a tile kernel with the reference y, the FP32 epilogue of the FP32 GEMM of the dequantized
// operands.
//  - BF16 output: error(%) = share of elements with |out - y| > atol + rtol |y| (rtol 1e-2 > one BF16 ulp,
//    atol = 1e-3 rms(y) covers cancellation), relFrob = ||out - y|| / ||y||.
//  - MXFP8 output: y is quantized with the library recipe (mx::quantizeMXFP8 of the FP32 y) and both are dequantized.
//    error(%) = share of elements that differ from the dequantized reference (a different FP32 accumulation order
//    can flip a rounding decision or, rarely, a scale), relFrob = ||deq(out) - deq(ref)|| / ||deq(ref)||,
//    codeMismatch / scaleMismatch = share of differing E4M3 / UE8M0 bytes, quantErr = ||deq(out) - y|| / ||y||
//    next to refQuantErr = ||deq(ref) - y|| / ||y||, the unavoidable quantization error.
struct TileResult {
  double errorPct = -1, relFrobPct = -1, codeMismatchPct = -1, scaleMismatchPct = -1, quantErrPct = -1,
         refQuantErrPct = -1;
};

__host__ __forceinline__
TileResult tileCompare(const TileProblem &p, const TileCtx &ctx, const bool toBF16, const float *const &refY,
                       const __nv_bfloat16 *const &outC, const uint8_t *const &outQ, const uint8_t *const &outSF) {
  auto stream = ctx.stream;
  const int M = p.M, N = p.N;
  const size_t nC = static_cast<size_t>(M) * N;
  TileResult r;
  float *x;
  CHECK_CUDA(cudaMalloc(&x, nC * sizeof(float)));
  if (toBF16) {
    bf16ToFloatKernel<<<(nC + 255) / 256, 256, 0, stream>>>(outC, x, nC);
    constexpr float rtol = 1e-2f;
    const auto base = diffStats(x, refY, nC, rtol, 0.f, ctx.stats, stream);
    const auto d = diffStats(x, refY, nC, rtol, static_cast<float>(1e-3 * base.rms), ctx.stats, stream);
    r.errorPct = d.failPct;
    r.relFrobPct = d.relFrob * 100;
  }
  else {
    uint8_t *refQ, *refSF;
    float *refDeq;
    const size_t sfBytes = mx::sfBytes(M, N / 32);
    CHECK_CUDA(cudaMalloc(&refQ, nC));
    CHECK_CUDA(cudaMalloc(&refSF, sfBytes));
    CHECK_CUDA(cudaMalloc(&refDeq, nC * sizeof(float)));
    mx::quantizeMXFP8<float>(refY, refQ, refSF, M, N, stream);
    mx::dequantizeMXFP8<float>(refQ, refSF, refDeq, M, N, stream);
    mx::dequantizeMXFP8<float>(outQ, outSF, x, M, N, stream);
    const auto exact = diffStats(x, refDeq, nC, 0.f, 0.f, ctx.stats, stream);
    r.errorPct = exact.failPct;
    r.relFrobPct = exact.relFrob * 100;
    r.codeMismatchPct = neqBytesPct(outQ, refQ, nC, ctx.count, stream);
    r.scaleMismatchPct = neqBytesPct(outSF, refSF, sfBytes, ctx.count, stream);
    r.quantErrPct = diffStats(x, refY, nC, 0.f, 0.f, ctx.stats, stream).relFrob * 100;
    r.refQuantErrPct = diffStats(refDeq, refY, nC, 0.f, 0.f, ctx.stats, stream).relFrob * 100;
    cudaFree(refQ);
    cudaFree(refSF);
    cudaFree(refDeq);
  }
  cudaFree(x);
  return r;
}

__host__ __forceinline__
void tileHeader() {
  printf("mode, mainloop, M, N, K, data, activation, smemBytes, regs, localBytes, gridCheck, grid, error(%%), "
         "relFrob(%%), codeMismatch(%%), scaleMismatch(%%), quantErr(%%), refQuantErr(%%), Time(ms), TFLOPS, "
         "cublasLt_MXFP8_TFLOPS, cublasLt_BF16_TFLOPS, Ratio_vs_cublasLt_MXFP8, Speedup_vs_cublasLt_BF16\n");
}

__host__ __forceinline__
void tilePrint(const TileProblem &p, const char *const mode, const char *const cfg, const char *const act,
               const int smemBytes, const cudaFuncAttributes &attr, const int gridCheck, const int grid,
               const TileResult &r, const double ms) {
  const double flops = (p.gated ? 4.0 : 2.0) * p.M * p.N * p.K;
  const double tf = ms > 0 ? flops / (ms * 1e-3) * 1e-12 : -1;
  const double ltMx = p.ltMxMs > 0 ? flops / (p.ltMxMs * 1e-3) * 1e-12 : -1;
  const double ltBf = p.ltBf16Ms > 0 ? flops / (p.ltBf16Ms * 1e-3) * 1e-12 : -1;
  printf("%s, %s, %d, %d, %d, %s, %s, %d, %d, %zu, %d, %d, %.5f, %.5f, %.5f, %.5f, %.4f, %.4f, %.4f, %.1f, %.1f, %.1f, "
         "%.3f, %.3f\n", mode, cfg, p.M, p.N, p.K, p.wide ? "wide" : "uniform", act, smemBytes, attr.numRegs,
         attr.localSizeBytes, gridCheck, grid, r.errorPct, r.relFrobPct, r.codeMismatchPct, r.scaleMismatchPct,
         r.quantErrPct, r.refQuantErrPct, ms, tf, ltMx, ltBf, ltMx > 0 && tf > 0 ? tf / ltMx : -1.0,
         ltBf > 0 && tf > 0 ? tf / ltBf : -1.0);
  fflush(stdout);
}

__host__ __forceinline__
bool tileWanted(const TileCtx &ctx, const char *const cfg) {
  return ctx.filter.empty() || std::strstr(cfg, ctx.filter.c_str()) != nullptr;
}

// opt-in dynamic shared memory per CTA of sm_120 / sm_121
constexpr int maxSharedMemory = 101376;

// vanilla GEMM (one B matrix) with the BF16 or the MXFP8 epilogue
template<typename Mainloop, typename Act, TileEpi epi>
__host__ __forceinline__
void runVanilla(const TileProblem &p, const TileCtx &ctx, const char *const cfg, const char *const actName) {
  using bf16 = __nv_bfloat16;
  constexpr int bM = cute::get<0>(typename Mainloop::TileShape{});
  constexpr int bN = cute::get<1>(typename Mainloop::TileShape{});
  constexpr bool toBF16 = epi == TileEpi::bf16;
  constexpr bool sink = epi == TileEpi::sink;
  if (!tileWanted(ctx, cfg) || (sink && !ctx.bench)) {
    return;
  }
  const char *const mode = toBF16 ? "vanilla_bf16" : (sink ? "vanilla_mainloop" : "vanilla_mxfp8");
  const int M = p.M, N = p.N, K = p.K;
  if (Mainloop::sharedSize > maxSharedMemory || p.gated || M % bM != 0 || N % bN != 0 || K % 128 != 0) {
    return; // does not fit into the shared memory of an sm_120 CTA / shape not supported by this configuration
  }
  auto stream = ctx.stream;
  constexpr int ctaThreads = Mainloop::Threads::value;
  auto kernel = tileVanillaKernel<Mainloop, Act, epi>;
  CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Mainloop::sharedSize));
  cudaFuncAttributes attr{};
  CHECK_CUDA(cudaFuncGetAttributes(&attr, kernel));
  const size_t nC = static_cast<size_t>(M) * N;
  float *refY = nullptr;
  bf16 *outC = nullptr;
  uint8_t *outQ = nullptr, *outSF = nullptr;
  if constexpr (sink) {
    CHECK_CUDA(cudaMalloc(&outC, 1024 * sizeof(bf16)));
  }
  else {
    CHECK_CUDA(cudaMalloc(&refY, nC * sizeof(float)));
    if constexpr (toBF16) {
      CHECK_CUDA(cudaMalloc(&outC, nC * sizeof(bf16)));
    }
    else {
      CHECK_CUDA(cudaMalloc(&outQ, nC));
      CHECK_CUDA(cudaMalloc(&outSF, mx::sfBytes(M, N / 32)));
    }
    refVanillaKernel<Act><<<(nC + 255) / 256, 256, 0, stream>>>(p.refC, p.bias, refY, nC, N);
  }
  const int tiles = (M / bM) * (N / bN);
  const int grid = std::min(tiles, ctx.numSMs);
  const int gridCheck = ctx.gridOverride > 0 ? std::min(tiles, ctx.gridOverride) : grid;
  auto launch = [&](const int g, const int poison) {
    kernel<<<g, ctaThreads, Mainloop::sharedSize, stream>>>(p.aq, p.aSF, p.bq, p.bSF, outC, outQ, outSF, p.bias,
                                                            M, N, K, poison);
  };
  TileResult r;
  if constexpr (!sink) {
    if constexpr (toBF16) {
      cudaMemsetAsync(outC, 0xFF, nC * sizeof(bf16), stream); // NaN
    }
    else {
      cudaMemsetAsync(outQ, 0xFF, nC, stream);
      cudaMemsetAsync(outSF, 0xFF, mx::sfBytes(M, N / 32), stream);
    }
    launch(gridCheck, 1);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaGetLastError());
    r = tileCompare(p, ctx, toBF16, refY, outC, outQ, outSF);
  }
  double ms = -1;
  if (ctx.bench) {
    ms = timeMs([&] { launch(grid, 0); }, stream, 10, ctx.runs);
  }
  tilePrint(p, mode, cfg, actName, Mainloop::sharedSize, attr, gridCheck, grid, r, ms);
  cudaFree(refY);
  cudaFree(outC);
  cudaFree(outQ);
  cudaFree(outSF);
}

// gated GEMM, gated MXFP8 epilogue; bN is the width of each of the two B tiles
template<typename Mainloop, typename Act>
__host__ __forceinline__
void runGated(const TileProblem &p, const TileCtx &ctx, const char *const cfg, const char *const actName) {
  using bf16 = __nv_bfloat16;
  constexpr int bM = cute::get<0>(typename Mainloop::TileShape{});
  constexpr int bN = cute::get<1>(typename Mainloop::TileShape{});
  if (!tileWanted(ctx, cfg)) {
    return;
  }
  const int M = p.M, N = p.N, K = p.K;
  if (Mainloop::sharedSize > maxSharedMemory || !p.gated || M % bM != 0 || N % bN != 0 || K % 128 != 0) {
    return;
  }
  auto stream = ctx.stream;
  constexpr int ctaThreads = Mainloop::Threads::value;
  auto kernel = tileGatedKernel<Mainloop, Act>;
  CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Mainloop::sharedSize));
  cudaFuncAttributes attr{};
  CHECK_CUDA(cudaFuncGetAttributes(&attr, kernel));
  const size_t nC = static_cast<size_t>(M) * N;
  float *refY;
  uint8_t *outQ, *outSF;
  CHECK_CUDA(cudaMalloc(&refY, nC * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&outQ, nC));
  CHECK_CUDA(cudaMalloc(&outSF, mx::sfBytes(M, N / 32)));
  refGatedKernel<Act><<<(nC + 255) / 256, 256, 0, stream>>>(p.refC, p.refC2, p.bias, p.biasV, refY, p.alpha, p.beta,
                                                            nC, N);
  const int tiles = (M / bM) * (N / bN);
  const int grid = std::min(tiles, ctx.numSMs);
  const int gridCheck = ctx.gridOverride > 0 ? std::min(tiles, ctx.gridOverride) : grid;
  auto launch = [&](const int g, const int poison) {
    kernel<<<g, ctaThreads, Mainloop::sharedSize, stream>>>(p.aq, p.aSF, p.bq, p.bSF, p.b2q, p.b2SF, outQ, outSF,
                                                            p.bias, p.biasV, p.alpha, p.beta, M, N, K, poison);
  };
  cudaMemsetAsync(outQ, 0xFF, nC, stream);
  cudaMemsetAsync(outSF, 0xFF, mx::sfBytes(M, N / 32), stream);
  launch(gridCheck, 1);
  CHECK_CUDA(cudaStreamSynchronize(stream));
  CHECK_CUDA(cudaGetLastError());
  const auto r = tileCompare(p, ctx, false, refY, nullptr, outQ, outSF);
  double ms = -1;
  if (ctx.bench) {
    ms = timeMs([&] { launch(grid, 0); }, stream, 10, ctx.runs);
  }
  tilePrint(p, "gated_mxfp8", cfg, actName, Mainloop::sharedSize, attr, gridCheck, grid, r, ms);
  cudaFree(refY);
  cudaFree(outQ);
  cudaFree(outSF);
}

// all configurations of the tile kernels on one problem
__host__ __forceinline__
void runTileConfigs(const TileProblem &p, const TileCtx &ctx, const bool extra) {
  // vanilla: GEMM1-like (BF16 epilogue, identity) and GEMM0-like (MXFP8 epilogue, SiLU)
  runVanilla<TileMainloop<128, 64, 4, false>, IdentityAct, TileEpi::bf16>(p, ctx, "v128x128x64s4", "identity");
  runVanilla<TileMainloop<128, 64, 4, false>, SiluAct, TileEpi::mxfp8>(p, ctx, "v128x128x64s4", "silu");
  runVanilla<TileMainloop<128, 64, 4, false>, IdentityAct, TileEpi::sink>(p, ctx, "v128x128x64s4", "none");
  runVanilla<TileMainloop<128, 64, 5, false>, IdentityAct, TileEpi::bf16>(p, ctx, "v128x128x64s5", "identity");
  runVanilla<TileMainloop<128, 64, 5, false>, SiluAct, TileEpi::mxfp8>(p, ctx, "v128x128x64s5", "silu");
  runVanilla<TileMainloop<128, 64, 5, false>, IdentityAct, TileEpi::sink>(p, ctx, "v128x128x64s5", "none");
  runVanilla<TileMainloop<128, 128, 2, false>, IdentityAct, TileEpi::bf16>(p, ctx, "v128x128x128s2", "identity");
  runVanilla<TileMainloop<128, 128, 2, false>, SiluAct, TileEpi::mxfp8>(p, ctx, "v128x128x128s2", "silu");
  runVanilla<TileMainloop<128, 128, 2, false>, IdentityAct, TileEpi::sink>(p, ctx, "v128x128x128s2", "none");
  // gated: two B tiles of 64 or 128 columns
  runGated<TileMainloop<64, 64, 4, true>, SiluAct>(p, ctx, "g128x(64+64)x64s4", "silu");
  runGated<TileMainloop<64, 64, 5, true>, SiluAct>(p, ctx, "g128x(64+64)x64s5", "silu");
  runGated<TileMainloop<128, 64, 3, true>, SiluAct>(p, ctx, "g128x(128+128)x64s3", "silu");
  if (extra) {
    // the other activations and the 128 x 256 tile
    runVanilla<TileMainloop<128, 64, 4, false>, ReluAct, TileEpi::bf16>(p, ctx, "v128x128x64s4", "relu");
    runVanilla<TileMainloop<128, 64, 4, false>, GeluAct, TileEpi::mxfp8>(p, ctx, "v128x128x64s4", "gelu");
    runVanilla<TileMainloop<128, 64, 4, false>, IdentityAct, TileEpi::mxfp8>(p, ctx, "v128x128x64s4", "identity");
    runVanilla<TileMainloop<256, 64, 3, false>, IdentityAct, TileEpi::bf16>(p, ctx, "v128x256x64s3", "identity");
    runVanilla<TileMainloop<256, 64, 3, false>, SiluAct, TileEpi::mxfp8>(p, ctx, "v128x256x64s3", "silu");
    runVanilla<TileMainloop<256, 64, 3, false>, IdentityAct, TileEpi::sink>(p, ctx, "v128x256x64s3", "none");
    runGated<TileMainloop<64, 64, 4, true>, GeluAct>(p, ctx, "g128x(64+64)x64s4", "gelu");
    runGated<TileMainloop<64, 64, 4, true>, ReluAct>(p, ctx, "g128x(64+64)x64s4", "relu");
  }
}

__host__ __forceinline__
void tileSection(const std::vector<std::array<int, 3>> &shapes, const bool allShapesGated, const bool check,
                 const int runs, const std::string &filter, matx::cudaExecutor &exec) {
  cublasLtHandle_t lt{};
  CHECK_CUBLAS(cublasLtCreate(&lt));
  constexpr size_t workspaceBytes = 32ULL << 20;
  void *workspace = nullptr;
  CHECK_CUDA(cudaMalloc(&workspace, workspaceBytes));
  int dev = 0;
  cudaGetDevice(&dev);
  TileCtx ctx;
  ctx.stream = exec.getStream();
  ctx.exec = &exec;
  ctx.runs = runs;
  ctx.bench = !check;
  ctx.filter = filter;
  CHECK_CUDA(cudaDeviceGetAttribute(&ctx.numSMs, cudaDevAttrMultiProcessorCount, dev));
  CHECK_CUDA(cudaMalloc(&ctx.stats, 4 * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&ctx.count, sizeof(unsigned long long)));
  tileHeader();
  // vanilla problems, then gated problems; --tilecheck uses a persistent grid of 3 CTAs
  ctx.gridOverride = check ? 3 : 0;
  for (const bool gated : {false, true}) {
    for (const auto &s : shapes) {
      if (!allShapesGated && gated && !((s[0] == 8192 && s[1] == 8192 && s[2] == 5120) ||
                                        (s[0] == 4096 && s[1] == 768 && s[2] == 2048))) {
        continue; // the default benchmark shapes of the gated GEMM
      }
      for (const bool wide : check ? std::vector<bool>{false, true} : std::vector<bool>{false}) {
        TileProblem p;
        makeTileProblem(p, s[0], s[1], s[2], gated, wide, ctx, lt, workspace, workspaceBytes);
        runTileConfigs(p, ctx, check || allShapesGated);
        destroyTileProblem(p);
      }
    }
  }
  cudaFree(ctx.stats);
  cudaFree(ctx.count);
  cudaFree(workspace);
  cublasLtDestroy(lt);
}

int main(const int argc, char **argv) {
  bool unit = false, quant = false, gemm = false, tile = false, tileCheck = false;
  int runs = 50;
  std::string cfgFilter;
  std::vector<int> dims;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--unit") {
      unit = true;
    }
    else if (arg == "--quant") {
      quant = true;
    }
    else if (arg == "--gemm") {
      gemm = true;
    }
    else if (arg == "--tile") {
      tile = true;
    }
    else if (arg == "--tilecheck") {
      tileCheck = true;
    }
    else if (arg.rfind("--runs=", 0) == 0) {
      runs = std::stoi(arg.substr(7));
    }
    else if (arg.rfind("--cfg=", 0) == 0) {
      cfgFilter = arg.substr(6);
    }
    else {
      dims.push_back(std::stoi(arg));
    }
  }
  if (!unit && !quant && !gemm && !tile && !tileCheck) {
    unit = quant = gemm = tile = tileCheck = true;
  }
  std::vector<std::array<int, 3>> shapes;
  if (dims.size() == 3) {
    shapes.push_back({dims[0], dims[1], dims[2]});
  }
  else {
    for (const int n : {4096, 8192}) {
      for (const int m : {128, 1024, 4096, 8192}) {
        shapes.push_back({m, n, n});
      }
    }
    shapes.push_back({8192, 8192, 5120});
    shapes.push_back({8192, 5120, 8192});
    shapes.push_back({4096, 768, 2048});
    shapes.push_back({4096, 2048, 768});
  }
  cudaSetDevice(0);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  matx::cudaExecutor exec{stream};
  if (unit) {
    checkUe8m0(stream);
    checkE4M3(stream);
    checkSfLayout();
    printf("\n");
  }
  if (quant) {
    quantSection(stream);
    printf("\n");
  }
  if (gemm) {
    gemmSection(shapes, runs, exec);
    printf("\n");
  }
  if (tileCheck) {
    // small shapes, 3 persistent CTAs, workspace overwritten between tiles
    std::vector<std::array<int, 3>> checkShapes = {{128, 128, 128}, {256, 512, 256}, {384, 640, 1152},
                                                   {1024, 1024, 768}, {512, 768, 2048}};
    if (dims.size() == 3) {
      checkShapes = {{dims[0], dims[1], dims[2]}};
    }
    tileSection(checkShapes, true, true, runs, cfgFilter, exec);
    printf("\n");
  }
  if (tile) {
    // MoE shapes of the benchmark; the gated GEMM is benchmarked on the first two GEMM0-like shapes only
    std::vector<std::array<int, 3>> tileShapes = {{4096, 4096, 4096}, {8192, 8192, 8192}, {8192, 8192, 5120},
                                                  {8192, 5120, 8192}, {4096, 768, 2048}, {4096, 2048, 768}};
    if (dims.size() == 3) {
      tileShapes = {{dims[0], dims[1], dims[2]}};
    }
    tileSection(tileShapes, dims.size() == 3, false, runs, cfgFilter, exec);
  }
  cudaStreamDestroy(stream);
  return 0;
}
