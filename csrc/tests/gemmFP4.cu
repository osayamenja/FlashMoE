//
// Created by FlashMoE contributors on 10/5/26.
//
// correctness test and benchmark for the FP4 formats MXFP4 (E2M1 + one UE8M0 scale per 32 K-elements) and NVFP4
// (E2M1 + one UE4M3 scale per 16 K-elements + one FP32 scale per tensor).
//  --check:
//   1. unit checks of infra/fp4.cuh against independent host models: E2M1 conversions (all codes, the round-to-nearest-
//      even ties), the MXFP4 / NVFP4 scale recipes, the scale factor layout against CUTLASS's Sm1xxBlockScaledConfig<32>
//      and <16>, the quantize / dequantize kernels (bit exact, zeroed SF padding).
//   2. the tile mainloop and epilogues of flashmoe::tile::sm120::FP4Mainloop (tile_sm120_fp4.cuh) in a persistent kernel
//      (grid = 3 CTAs < #tiles, workspace poisoned between tiles): vanilla + BF16 epilogue, vanilla + FP4 epilogue,
//      gated + FP4 epilogue, checked against an FP32 GEMM of the dequantized operands (+ bias, activation, gating, then
//      the same FP4 quantization of the output); next to it cuBLASLt on the same data.
//  --bench: 8192x8192x8192 vanilla, 8192x8192x5120 gated, 4096x2048x768 vanilla; TFLOPS of the FP4 mainloop alone, FP4
//   tile with epilogue, the MXFP8 tile of PR2 with epilogue, cuBLASLt NVFP4 / MXFP4 and MXFP8 (GEMM only).
//
// ./gemmFP4 [--check] [--bench] [--fmt=mxfp4|nvfp4] [--runs=<n>]

#include <algorithm>
#include <array>
#include <cfloat>
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
#include "../include/flashmoe/infra/fp4.cuh"
#include "../include/flashmoe/infra/mx.cuh"
#include "../include/flashmoe/tile.cuh"
#include "../include/flashmoe/tile_sm120_fp4.cuh"
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
namespace fp4 = flashmoe::fp4;
using flashmoe::QuantRecipe;
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

// E2M1: 1 sign, 2 exponent (bias 1), 1 mantissa bit; magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6}, no Inf / NaN
constexpr float hostE2M1Values[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};

__host__ __forceinline__
float hostE2M1Decode(const uint8_t nibble) {
  const float v = hostE2M1Values[nibble & 7];
  return (nibble & 8) ? -v : v;
}

// brute force round-to-nearest, ties to the even code, saturation to +-6; NaN is not part of the contract
__host__ __forceinline__
uint8_t hostE2M1Encode(const float x) {
  const uint8_t sign = std::signbit(x) ? 8 : 0;
  const double ax = std::fabs(static_cast<double>(x));
  if (ax >= 6.0) { // saturation, also keeps the distances below finite for huge values and Inf
    return static_cast<uint8_t>(sign | 7);
  }
  uint8_t best = 0;
  double bestDist = std::fabs(ax - 0.0);
  for (int c = 1; c < 8; ++c) {
    const double d = std::fabs(ax - hostE2M1Values[c]);
    if (d < bestDist || (d == bestDist && (c % 2 == 0))) {
      bestDist = d;
      best = static_cast<uint8_t>(c);
    }
  }
  return static_cast<uint8_t>(sign | best);
}

// ue8m0 = 127 + ceil(log2(amax / 6)), double precision
__host__ __forceinline__
uint8_t hostUe8m0FP4(const float amax) {
  if (std::isnan(amax) || std::isinf(amax)) {
    return mx::UE8M0_NAN;
  }
  const double a = std::fabs(static_cast<double>(amax));
  if (a == 0.0) {
    return mx::UE8M0_MIN;
  }
  int ex = 0;
  const double m = std::frexp(a / 6.0, &ex); // a / 6 = m * 2^ex, m in [0.5, 1)
  const int ceilLog2 = m == 0.5 ? ex - 1 : ex;
  return static_cast<uint8_t>(std::clamp(127 + ceilLog2, static_cast<int>(mx::UE8M0_MIN), 254));
}

// the scale pair (sf byte, encode factor) of a block with absolute maximum amax
template<QuantRecipe r>
__host__ __forceinline__
std::pair<uint8_t, float> hostBlockScale(const float amax, const float st) {
  if constexpr (r == QuantRecipe::mxfp4) {
    const uint8_t e = hostUe8m0FP4(amax);
    return {e, e == mx::UE8M0_NAN ? 1.0f : static_cast<float>(std::ldexp(1.0, 127 - e))};
  }
  else {
    const float p = amax * (st * (1.0f / 6.0f));
    const uint8_t sf = hostE4M3Encode(p);
    return {sf, std::min(st / hostE4M3Decode(sf), FLT_MAX)};
  }
}

template<QuantRecipe r>
__host__ __forceinline__
float hostDecodeScale(const uint8_t sf, const float gs) {
  if constexpr (r == QuantRecipe::mxfp4) {
    return sf == mx::UE8M0_NAN ? std::nanf("") : static_cast<float>(std::ldexp(1.0, static_cast<int>(sf) - 127));
  }
  else {
    return hostE4M3Decode(sf) * gs;
  }
}

template<QuantRecipe r>
constexpr const char *fmtName() {
  return r == QuantRecipe::mxfp4 ? "mxfp4" : "nvfp4";
}

// ------------------------------------------------------------------------------------------------------------
// unit checks
// ------------------------------------------------------------------------------------------------------------

__global__ void e2m1Kernel(const float *__restrict__ in, uint8_t *__restrict__ enc, float *__restrict__ dec,
                           const size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  // pair (in[2 i], in[2 i + 1]) -> one byte, low nibble = in[2 i]
  enc[i] = fp4::packE2M1x2(in[2 * i], in[2 * i + 1]);
  // decode the byte pattern i % 256 (only meaningful for the first 256 entries)
  const float2 d = fp4::unpackE2M1x2(static_cast<uint8_t>(i % 256));
  dec[2 * i] = d.x;
  dec[2 * i + 1] = d.y;
}

__host__ __forceinline__
void checkE2M1(cudaStream_t stream) {
  std::vector<float> in;
  // ties and their neighbours, every code value, saturation, signed zeros
  const float centers[] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.f, 0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, 7.f,
                           100.f, 1e30f, FLT_MAX, 1e-30f};
  for (const float c : centers) {
    for (const float s : {1.f, -1.f}) {
      in.push_back(s * c);
      in.push_back(s * std::nextafter(c, 0.f));
      in.push_back(s * std::nextafter(c, 1e38f));
    }
  }
  in.push_back(INFINITY);
  in.push_back(-INFINITY);
  std::mt19937 rng(11);
  std::uniform_real_distribution<float> logMag(-6.f, 4.f);
  while (in.size() < (1U << 20)) {
    in.push_back(std::exp2(logMag(rng)) * ((rng() & 1) ? -1.f : 1.f));
  }
  const size_t n = in.size() / 2; // pairs
  float *dIn, *dDec;
  uint8_t *dEnc;
  cudaMallocAsync(&dIn, 2 * n * sizeof(float), stream);
  cudaMallocAsync(&dDec, 2 * n * sizeof(float), stream);
  cudaMallocAsync(&dEnc, n, stream);
  cudaMemcpyAsync(dIn, in.data(), 2 * n * sizeof(float), cudaMemcpyHostToDevice, stream);
  e2m1Kernel<<<(n + 255) / 256, 256, 0, stream>>>(dIn, dEnc, dDec, n);
  std::vector<uint8_t> enc(n);
  std::vector<float> dec(512);
  cudaMemcpyAsync(enc.data(), dEnc, n, cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(dec.data(), dDec, 512 * sizeof(float), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  CHECK_CUDA(cudaPeekAtLastError());
  cudaFreeAsync(dIn, stream);
  cudaFreeAsync(dDec, stream);
  cudaFreeAsync(dEnc, stream);
  size_t encMismatch = 0, decMismatch = 0, hostApiMismatch = 0;
  for (size_t i = 0; i < n; ++i) {
    const uint8_t expect = static_cast<uint8_t>(hostE2M1Encode(in[2 * i]) | (hostE2M1Encode(in[2 * i + 1]) << 4));
    if (enc[i] != expect) {
      if (++encMismatch <= 4) {
        printf("# e2m1 encode: %.9g, %.9g -> 0x%02x, model 0x%02x\n", in[2 * i], in[2 * i + 1], enc[i], expect);
      }
    }
    // the software path of cuda_fp4.h on the host
    if (fp4::packE2M1x2(in[2 * i], in[2 * i + 1]) != expect) {
      ++hostApiMismatch;
    }
  }
  // thread t decodes the byte t; compared bit-wise, so +0 and -0 (code 8) are told apart
  for (int t = 0; t < 256; ++t) {
    const float lo = hostE2M1Decode(t & 0xF), hi = hostE2M1Decode(t >> 4);
    if (std::memcmp(&lo, &dec[2 * t], 4) != 0 || std::memcmp(&hi, &dec[2 * t + 1], 4) != 0) {
      ++decMismatch;
    }
  }
  size_t roundTrip = 0;
  for (int c = 0; c < 16; ++c) {
    if (hostE2M1Encode(hostE2M1Decode(static_cast<uint8_t>(c))) != c) {
      ++roundTrip;
    }
  }
  printf("test, cases, mismatches, detail\n");
  printf("e2m1 float->e2m1 (RN-even, satfinite) device vs brute force, %zu, %zu, ties 0.25 .. 5 + neighbours, "
         "log-uniform 2^-6..2^4\n", 2 * n, encMismatch);
  printf("e2m1 float->e2m1 host path of cuda_fp4.h vs brute force, %zu, %zu, same inputs\n", n, hostApiMismatch);
  printf("e2m1 e2m1->float vs table, 512, %zu, all byte patterns (both nibbles)\n", decMismatch);
  printf("e2m1 decode -> encode round trip, 16, %zu, all codes\n", roundTrip);
}

__global__ void scaleKernel(const float *__restrict__ amax, const float *__restrict__ st, uint8_t *__restrict__ ue8m0,
                            uint8_t *__restrict__ ue4m3, float *__restrict__ encMx, float *__restrict__ encNv,
                            const size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  ue8m0[i] = fp4::ue8m0FromAmaxFP4(amax[i]);
  ue4m3[i] = fp4::nvfp4ScaleFromAmax(amax[i], st[i]);
  encMx[i] = fp4::blockEncodeScale<QuantRecipe::mxfp4>(ue8m0[i], st[i]);
  encNv[i] = fp4::blockEncodeScale<QuantRecipe::nvfp4>(ue4m3[i], st[i]);
}

__host__ __forceinline__
void checkScales(cudaStream_t stream) {
  std::vector<float> amax, st;
  std::mt19937_64 rng(0x5eed);
  std::uniform_real_distribution<float> logSt(-12.f, 12.f), logAmax(-14.f, 8.f);
  // 1. uniformly random bit patterns: every exponent, denormals, Inf, NaN
  for (size_t i = 0; i < (1U << 22); ++i) {
    amax.push_back(cuda::std::bit_cast<float>(static_cast<uint32_t>(rng())));
  }
  // 2. neighbourhoods of the MXFP4 decision boundary amax = 6 * 2^k (mantissa 0x400000) and the exponent edges
  for (uint32_t biased = 0; biased < 256; ++biased) {
    for (int d = -40; d <= 40; ++d) {
      for (const uint32_t center : {0x400000U, 0x000000U, 0x7FFFFFU}) {
        const int64_t mant = static_cast<int64_t>(center) + d;
        if (mant >= 0 && mant <= 0x7FFFFF) {
          amax.push_back(cuda::std::bit_cast<float>((biased << 23) | static_cast<uint32_t>(mant)));
        }
      }
    }
  }
  for (const float v : {0.f, -0.f, INFINITY, -INFINITY, NAN, 6.f, 6.0000005f, 5.9999995f, 3.f, 12.f, 1.4e-45f}) {
    amax.push_back(v);
  }
  // 3. NVFP4: random amax around the working range, random encode scales
  const size_t nSpecial = amax.size();
  for (size_t i = 0; i < (1U << 18); ++i) {
    amax.push_back(std::exp2(logAmax(rng)));
  }
  st.resize(amax.size());
  for (size_t i = 0; i < st.size(); ++i) {
    st[i] = std::exp2(logSt(rng));
  }
  const size_t n = amax.size();
  float *dIn, *dSt, *dEnc;
  uint8_t *dOut;
  cudaMallocAsync(&dIn, n * sizeof(float), stream);
  cudaMallocAsync(&dSt, n * sizeof(float), stream);
  cudaMallocAsync(&dEnc, 2 * n * sizeof(float), stream);
  cudaMallocAsync(&dOut, 2 * n, stream);
  cudaMemcpyAsync(dIn, amax.data(), n * sizeof(float), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(dSt, st.data(), n * sizeof(float), cudaMemcpyHostToDevice, stream);
  scaleKernel<<<(n + 255) / 256, 256, 0, stream>>>(dIn, dSt, dOut, dOut + n, dEnc, dEnc + n, n);
  std::vector<uint8_t> out(2 * n);
  std::vector<float> enc(2 * n);
  cudaMemcpyAsync(out.data(), dOut, 2 * n, cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(enc.data(), dEnc, 2 * n * sizeof(float), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  CHECK_CUDA(cudaPeekAtLastError());
  cudaFreeAsync(dIn, stream);
  cudaFreeAsync(dSt, stream);
  cudaFreeAsync(dEnc, stream);
  cudaFreeAsync(dOut, stream);
  size_t mxBad = 0, mxEncBad = 0, mxHostBad = 0, nvBad = 0, nvEncBad = 0, nvCases = 0, saturated = 0;
  for (size_t i = 0; i < nSpecial; ++i) {
    const auto [e, encModel] = hostBlockScale<QuantRecipe::mxfp4>(amax[i], 1.f);
    mxBad += out[i] != e;
    mxHostBad += fp4::ue8m0FromAmaxFP4(amax[i]) != e; // the host instantiation of the same function
    mxEncBad += std::memcmp(&enc[i], &encModel, 4) != 0;
  }
  for (size_t i = nSpecial; i < n; ++i) {
    const auto [sf, encModel] = hostBlockScale<QuantRecipe::nvfp4>(amax[i], st[i]);
    ++nvCases;
    saturated += sf == 0x7E;
    nvBad += out[n + i] != sf;
    nvEncBad += std::memcmp(&enc[n + i], &encModel, 4) != 0;
  }
  printf("mxfp4 ue8m0 scale (device) vs double model, %zu, %zu, random bits, 6 * 2^k boundaries, zero/denormal/NaN/Inf\n",
         nSpecial, mxBad);
  printf("mxfp4 ue8m0 scale (host instantiation) vs double model, %zu, %zu, same inputs\n", nSpecial, mxHostBad);
  printf("mxfp4 encode factor 2^(127 - sf) vs model, %zu, %zu, same inputs\n", nSpecial, mxEncBad);
  printf("nvfp4 ue4m3 scale amax * (st / 6) vs brute-force E4M3 RN, %zu, %zu, log-uniform amax / st; %zu saturated\n",
         nvCases, nvBad, saturated);
  printf("nvfp4 encode factor min(st / sf, FLT_MAX) vs host float math, %zu, %zu, same inputs\n", nvCases, nvEncBad);
}

template<int VS>
__host__ __forceinline__
size_t layoutMismatches(const int rows, const int K, const bool isSFA, size_t &cosizeMismatch) {
  using Config = cutlass::detail::Sm1xxBlockScaledConfig<VS>;
  const int rp = (rows + 127) / 128 * 128;
  const int kp = (K + 4 * VS - 1) / (4 * VS) * (4 * VS);
  const int numKBlocks = K / VS;
  size_t bad = 0;
  auto count = [&](const auto &layout) {
    if (static_cast<size_t>(cute::cosize(layout)) != mx::sfBytes(rp, kp / VS)) {
      ++cosizeMismatch;
    }
    for (int r = 0; r < rows; ++r) {
      for (int kb = 0; kb < numKBlocks; ++kb) {
        if (static_cast<size_t>(layout(r, kb * VS)) != mx::sfOffset(r, kb, numKBlocks)) {
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

template<int VS>
__host__ __forceinline__
void checkSfLayout() {
  const int shapes[][2] = {{128, 128}, {256, 512}, {129, 96}, {300, 192}, {768, 2048}, {2048, 768}, {5120, 8192},
                           {8192, 5120}, {1, 32}, {127, 4096}};
  size_t cases = 0, bad = 0, cosizeBad = 0;
  for (const auto &s : shapes) {
    if (s[1] % VS != 0) {
      continue;
    }
    for (const bool isSFA : {true, false}) {
      bad += layoutMismatches<VS>(s[0], s[1], isSFA, cosizeBad);
      cases += static_cast<size_t>(s[0]) * (s[1] / VS);
    }
  }
  printf("sfOffset vs CUTLASS Sm1xxBlockScaledConfig<%d> SFA+SFB, %zu, %zu, cosize != sfBytes in %zu shapes\n", VS,
         cases, bad, cosizeBad);
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
  size_t scaleMismatch = 0;
  size_t dataMismatch = 0;
  size_t dequantMismatch = 0;
  size_t paddingNonzero = 0;
  size_t offsetCollisions = 0;
};

template<QuantRecipe r, typename Element>
__host__ __forceinline__
QuantStats verifyQuant(const std::vector<Element> &x, const std::vector<uint8_t> &q, const std::vector<uint8_t> &sf,
                       const std::vector<float> &deqF, const std::vector<__nv_bfloat16> &deqB,
                       const int rows, const int K, const float gs) {
  QuantStats s;
  constexpr int vs = fp4::sfVecSize<r>();
  const int numKBlocks = K / vs;
  const float st = r == QuantRecipe::nvfp4 ? fp4::nvfp4EncodeScale(gs) : 1.f;
  std::vector<uint8_t> used(sf.size(), 0);
  double sumErr = 0, sumAbs = 0, sumSqErr = 0, sumSq = 0;
  for (int row = 0; row < rows; ++row) {
    for (int kb = 0; kb < numKBlocks; ++kb) {
      const size_t base = static_cast<size_t>(row) * K + static_cast<size_t>(kb) * vs;
      float xf[vs];
      float amax = 0.f;
      for (int j = 0; j < vs; ++j) {
        xf[j] = hostToFloat(x[base + j]);
        amax = std::max(amax, std::fabs(xf[j]));
      }
      const size_t off = mx::sfOffset(row, kb, numKBlocks);
      if (used[off]) {
        ++s.offsetCollisions;
      }
      used[off] = 1;
      const auto [expect, enc] = hostBlockScale<r>(amax, st);
      if (sf[off] != expect) {
        ++s.scaleMismatch;
      }
      const float scale = hostDecodeScale<r>(sf[off], gs);
      for (int j = 0; j < vs; ++j) {
        const size_t k = base + j; // element index
        const uint8_t byte = q[k / 2];
        const uint8_t got = (k % 2) ? static_cast<uint8_t>(byte >> 4) : static_cast<uint8_t>(byte & 0xF);
        if (got != hostE2M1Encode(xf[j] * enc)) {
          ++s.dataMismatch;
        }
        const float dq = hostE2M1Decode(got) * scale;
        if (std::memcmp(&dq, &deqF[k], sizeof(float)) != 0) {
          ++s.dequantMismatch;
        }
        const __nv_bfloat16 db = __float2bfloat16_rn(dq);
        if (std::memcmp(&db, &deqB[k], sizeof(__nv_bfloat16)) != 0) {
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

template<QuantRecipe r, typename Element>
__host__ __forceinline__
void quantCase(const int rows, const int K, const bool wide, cudaStream_t stream) {
  const size_t n = static_cast<size_t>(rows) * K;
  const size_t sfSize = fp4::sfBytes<r>(rows, K);
  Element *x = nullptr;
  uint8_t *q = nullptr, *sf = nullptr;
  float *deqF = nullptr, *dAmax = nullptr;
  __nv_bfloat16 *deqB = nullptr;
  CHECK_CUDA(cudaMallocAsync(&x, n * sizeof(Element), stream));
  CHECK_CUDA(cudaMallocAsync(&q, fp4::dataBytes(rows, K), stream));
  CHECK_CUDA(cudaMallocAsync(&sf, sfSize, stream));
  CHECK_CUDA(cudaMallocAsync(&deqF, n * sizeof(float), stream));
  CHECK_CUDA(cudaMallocAsync(&deqB, n * sizeof(__nv_bfloat16), stream));
  CHECK_CUDA(cudaMallocAsync(&dAmax, sizeof(float), stream));
  std::random_device rd;
  fillInput(x, n, wide, rd(), stream);
  // the decode scale of the tensor (nvfp4), from the device amax, which is compared with the host
  fp4::amax<Element>(x, n, dAmax, stream);
  float amaxDev = 0.f;
  std::vector<Element> hx(n);
  cudaMemcpyAsync(&amaxDev, dAmax, sizeof(float), cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(hx.data(), x, n * sizeof(Element), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  float amaxHost = 0.f;
  for (const auto &v : hx) {
    amaxHost = std::max(amaxHost, std::fabs(hostToFloat(v)));
  }
  const float gs = r == QuantRecipe::nvfp4 ? fp4::nvfp4GlobalScale(amaxDev) : 1.f;
  // poison the outputs: the quantizer must overwrite every valid entry and zero the padding of sf
  cudaMemsetAsync(q, 0xAA, fp4::dataBytes(rows, K), stream);
  cudaMemsetAsync(sf, 0xAA, sfSize, stream);
  fp4::quantizeFP4<r, Element>(x, q, sf, rows, K, gs, stream);
  fp4::dequantizeFP4<r, float>(q, sf, deqF, rows, K, gs, stream);
  fp4::dequantizeFP4<r, __nv_bfloat16>(q, sf, deqB, rows, K, gs, stream);
  std::vector<uint8_t> hq(fp4::dataBytes(rows, K)), hsf(sfSize);
  std::vector<float> hdF(n);
  std::vector<__nv_bfloat16> hdB(n);
  cudaMemcpyAsync(hq.data(), q, hq.size(), cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(hsf.data(), sf, sfSize, cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(hdF.data(), deqF, n * sizeof(float), cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(hdB.data(), deqB, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  CHECK_CUDA(cudaPeekAtLastError());
  const auto s = verifyQuant<r, Element>(hx, hq, hsf, hdF, hdB, rows, K, gs);
  printf("%s, %d, %d, %s, %s, %.4e, %d, %.4f, %.4f, %.4f, %zu, %zu, %zu, %zu, %zu\n", fmtName<r>(), rows, K,
         element_string<Element>(), wide ? "wide" : "uniform", gs, amaxDev != amaxHost, s.maxErrVsAmax * 100,
         s.meanRelErr * 100, s.relFrob * 100, s.scaleMismatch, s.dataMismatch, s.dequantMismatch, s.paddingNonzero,
         s.offsetCollisions);
  cudaFreeAsync(x, stream);
  cudaFreeAsync(q, stream);
  cudaFreeAsync(sf, stream);
  cudaFreeAsync(deqF, stream);
  cudaFreeAsync(deqB, stream);
  cudaFreeAsync(dAmax, stream);
  cudaStreamSynchronize(stream);
}

template<QuantRecipe r>
__host__ __forceinline__
void quantSection(cudaStream_t stream) {
  for (const auto &s : {std::array<int, 2>{130, 96}, std::array<int, 2>{257, 1536}}) {
    for (const bool wide : {false, true}) {
      quantCase<r, __nv_bfloat16>(s[0], s[1], wide, stream);
    }
  }
  quantCase<r, __half>(130, 96, false, stream);
  quantCase<r, float>(130, 96, false, stream);
}

// ------------------------------------------------------------------------------------------------------------
// cuBLASLt (column-major): C[M, N] (row-major) = A[M, K] @ B[N, K]^T, with block-scaled operands for the MX / NV kinds
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
  return {std::sqrt(h[0] / std::max(h[1], 1e-300)), std::sqrt(h[1] / static_cast<double>(n)),
          100.0 * static_cast<double>(fails) / static_cast<double>(n)};
}

enum class LtKind {
  mxfp8,
  mxfp4,
  nvfp4
};

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

  // sfA / sfB: the scale factors of our A [M, K] and B [N, K]
  __host__ __forceinline__
  void init(cublasLtHandle_t h, const LtKind kind, const int M, const int N, const int K, const void *const &sfA,
            const void *const &sfB, void *const &ws, const size_t &wsBytes) {
    handle = h;
    workspace = ws;
    workspaceBytes = wsBytes;
    const auto inType = kind == LtKind::mxfp8 ? CUDA_R_8F_E4M3 : CUDA_R_4F_E2M1;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t transA = CUBLAS_OP_T;
    const cublasOperation_t transB = CUBLAS_OP_N;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));
    const cublasLtMatmulMatrixScale_t mode = kind == LtKind::nvfp4 ? CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3 :
      CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode)));
    // cuBLAS A = our B, cuBLAS B = our A
    const void *scaleA = sfB;
    const void *scaleB = sfA;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scaleA, sizeof(scaleA)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scaleB, sizeof(scaleB)));
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
      return; // valid stays false: no algorithm, reported as N/A
    }
    algo = result.algo;
    valid = true;
  }

  // a: our A, b: our B (both K-major), c: our row-major C; alpha scales the product (the tensor scales of nvfp4)
  __host__ __forceinline__
  void run(const void *const &a, const void *const &b, void *const &c, const float &alpha, cudaStream_t stream) const {
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
double tflops(const double flops, const double ms) {
  return ms > 0 ? flops / (ms * 1e-3) * 1e-12 : -1.0;
}

// ------------------------------------------------------------------------------------------------------------
// tile mainloops and epilogues in a persistent kernel: grid = min(#tiles, #SMs), every CTA loops over tiles and
// reuses its workspace
// ------------------------------------------------------------------------------------------------------------

template<QuantRecipe r, int bN, int bK, int stages, bool gated>
using FP4Tile = flashmoe::tile::sm120::FP4Mainloop<r, 128, bN, bK, stages, 256, gated>;

template<int bN, int bK, int stages, bool gated>
using MX8Tile = flashmoe::tile::sm120::MXFP8Mainloop<128, bN, bK, stages, 256, gated>;

using IdentityAct = flashmoe::ActivationType<float, flashmoe::Activation::identity>::AT;
using SiluAct = flashmoe::ActivationType<float, flashmoe::Activation::silu>::AT;

template<typename Mainloop>
concept IsFP4Loop = requires { Mainloop::recipe; }; // the MXFP8 mainloop of PR2 has no recipe

// everything a tile kernel reads; a, b, b2: E2M1 bytes (FP4) or E4M3 (MXFP8) with their scale factors
struct Ops {
  const uint8_t *a, *sfa, *b, *sfb, *b2, *sfb2;
  const __nv_bfloat16 *bias, *biasV;
  float alpha, alphaG, alphaU; // FP4 only: the accumulator factors undoing the tensor scales (nvfp4)
  float swishA, swishB; // gated: y = swishA * act(swishB * (g + bias)) * (u + biasV)
  float st; // FP4 only: encode scale of the output tensor (nvfp4)
  int M, N, K;
};

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
  quant,
  sink
};

template<typename Mainloop, typename Act, TileEpi epi>
__global__ __launch_bounds__(256, 1)
void tileVanillaKernel(const __grid_constant__ Ops ops, __nv_bfloat16 *__restrict__ outC, uint8_t *__restrict__ outQ,
                       uint8_t *__restrict__ outSF, const int poison) {
  extern __shared__ __align__(1024) cuda::std::byte workspace[];
  constexpr int bM = cute::get<0>(typename Mainloop::TileShape{});
  constexpr int bN = cute::get<1>(typename Mainloop::TileShape{});
  constexpr Mainloop mainloop{};
  const int tilesM = ops.M / bM;
  const int tilesN = ops.N / bN;
  for (int tile = static_cast<int>(blockIdx.x); tile < tilesM * tilesN; tile += static_cast<int>(gridDim.x)) {
    const auto tileCoord = flashmoe::tile::idx2Coord(tilesM, tilesN, tile);
    auto acc = Mainloop::makeAccumulator();
    mainloop(workspace, ops.a, ops.sfa, ops.b, ops.sfb, acc, ops.M, ops.N, ops.K, tileCoord);
    if constexpr (epi == TileEpi::bf16) {
      if constexpr (IsFP4Loop<Mainloop>) {
        Mainloop::template epilogueBF16<Act>(acc, outC, ops.bias, ops.alpha, ops.M, ops.N, tileCoord);
      }
      else {
        Mainloop::template epilogueBF16<Act>(acc, outC, ops.bias, ops.M, ops.N, tileCoord);
      }
    }
    else if constexpr (epi == TileEpi::quant) {
      if constexpr (IsFP4Loop<Mainloop>) {
        Mainloop::template epilogueFP4<Act>(acc, outQ, outSF, ops.bias, ops.alpha, ops.st, ops.M, ops.N, tileCoord);
      }
      else {
        Mainloop::template epilogueMXFP8<Act>(acc, outQ, outSF, ops.bias, ops.M, ops.N, tileCoord);
      }
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

template<typename Mainloop, typename Act, TileEpi epi>
__global__ __launch_bounds__(256, 1)
void tileGatedKernel(const __grid_constant__ Ops ops, __nv_bfloat16 *__restrict__ outC, uint8_t *__restrict__ outQ,
                     uint8_t *__restrict__ outSF, const int poison) {
  extern __shared__ __align__(1024) cuda::std::byte workspace[];
  constexpr int bM = cute::get<0>(typename Mainloop::TileShape{});
  constexpr int bN = cute::get<1>(typename Mainloop::TileShape{});
  constexpr Mainloop mainloop{};
  const int tilesM = ops.M / bM;
  const int tilesN = ops.N / bN;
  for (int tile = static_cast<int>(blockIdx.x); tile < tilesM * tilesN; tile += static_cast<int>(gridDim.x)) {
    const auto tileCoord = flashmoe::tile::idx2Coord(tilesM, tilesN, tile);
    auto acc = Mainloop::makeAccumulator();
    auto acc2 = Mainloop::makeAccumulator();
    mainloop(workspace, ops.a, ops.sfa, ops.b, ops.sfb, ops.b2, ops.sfb2, acc, acc2, ops.M, ops.N, ops.K, tileCoord);
    if constexpr (epi == TileEpi::quant) {
      if constexpr (IsFP4Loop<Mainloop>) {
        Mainloop::template epilogueGatedFP4<Act>(acc, acc2, outQ, outSF, ops.bias, ops.biasV, ops.alphaG, ops.alphaU,
                                                 ops.swishA, ops.swishB, ops.st, ops.M, ops.N, tileCoord);
      }
      else {
        Mainloop::template epilogueGatedMXFP8<Act>(acc, acc2, outQ, outSF, ops.bias, ops.biasV, ops.swishA,
                                                   ops.swishB, ops.M, ops.N, tileCoord);
      }
    }
    else {
      float s = 0.f;
      #pragma unroll
      for (int i = 0; i < cute::size(acc); ++i) {
        s += acc(i) + acc2(i);
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

// opt-in dynamic shared memory per CTA of sm_120 / sm_121
constexpr int maxSharedMemory = 101376;

struct Ctx {
  cudaStream_t stream{};
  matx::cudaExecutor *exec = nullptr;
  cublasLtHandle_t lt{};
  void *ltWorkspace = nullptr;
  size_t ltWorkspaceBytes = 0;
  int runs = 50;
  int numSMs = 0;
  double *stats = nullptr; // 4 doubles
  unsigned long long *count = nullptr;
};

// operands of one problem in format r; the gated problem has two B matrices
template<QuantRecipe r>
struct Problem {
  int M = 0, N = 0, K = 0;
  bool gated = false, wide = false;
  __nv_bfloat16 *a = nullptr, *b = nullptr, *b2 = nullptr, *bias = nullptr, *biasV = nullptr;
  uint8_t *aq = nullptr, *bq = nullptr, *b2q = nullptr, *aSF = nullptr, *bSF = nullptr, *b2SF = nullptr;
  float gsA = 1.f, gsB = 1.f, gsB2 = 1.f; // nvfp4 decode scales of the tensors
  float *refC = nullptr, *refC2 = nullptr; // check: FP32 A @ B^T, A @ B2^T of the dequantized operands
  float swishA = 1.25f, swishB = 0.8f;
  // --bench: the MXFP8 operands (PR2 quantizer) of the same bf16 data
  uint8_t *a8 = nullptr, *b8 = nullptr, *b28 = nullptr, *a8SF = nullptr, *b8SF = nullptr, *b28SF = nullptr;

  __host__ __forceinline__
  Ops ops(const float st) const {
    return {aq, aSF, bq, bSF, b2q, b2SF, bias, biasV, gsA * gsB, gsA * gsB, gsA * gsB2, swishA, swishB, st, M, N, K};
  }

  __host__ __forceinline__
  Ops ops8() const {
    return {a8, a8SF, b8, b8SF, b28, b28SF, bias, biasV, 1.f, 1.f, 1.f, swishA, swishB, 1.f, M, N, K};
  }
};

template<QuantRecipe r>
__host__ __forceinline__
float quantizeOperand(const __nv_bfloat16 *const &x, uint8_t *&q, uint8_t *&sf, const int rows, const int K,
                      const Ctx &ctx) {
  CHECK_CUDA(cudaMalloc(&q, fp4::dataBytes(rows, K)));
  CHECK_CUDA(cudaMalloc(&sf, fp4::sfBytes<r>(rows, K)));
  float gs = 1.f;
  if constexpr (r == QuantRecipe::nvfp4) {
    float *dAmax;
    CHECK_CUDA(cudaMalloc(&dAmax, sizeof(float)));
    fp4::amax<__nv_bfloat16>(x, static_cast<size_t>(rows) * K, dAmax, ctx.stream);
    float amax = 0.f;
    cudaMemcpyAsync(&amax, dAmax, sizeof(float), cudaMemcpyDeviceToHost, ctx.stream);
    cudaStreamSynchronize(ctx.stream);
    cudaFree(dAmax);
    gs = fp4::nvfp4GlobalScale(amax);
  }
  fp4::quantizeFP4<r, __nv_bfloat16>(x, q, sf, rows, K, gs, ctx.stream);
  return gs;
}

// reference: the dequantized operands (FP32, including the tensor scales) and their FP32 GEMMs (MatX -> cuBLAS SGEMM)
template<QuantRecipe r>
__host__ __forceinline__
void makeReference(Problem<r> &p, const Ctx &ctx) {
  const size_t nA = static_cast<size_t>(p.M) * p.K, nB = static_cast<size_t>(p.N) * p.K,
    nC = static_cast<size_t>(p.M) * p.N;
  float *aF, *bF, *b2F = nullptr;
  CHECK_CUDA(cudaMalloc(&aF, nA * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&bF, nB * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&p.refC, nC * sizeof(float)));
  fp4::dequantizeFP4<r, float>(p.aq, p.aSF, aF, p.M, p.K, p.gsA, ctx.stream);
  fp4::dequantizeFP4<r, float>(p.bq, p.bSF, bF, p.N, p.K, p.gsB, ctx.stream);
  auto tAF = matx::make_tensor<float>(aF, {p.M, p.K});
  auto tBF = matx::make_tensor<float>(bF, {p.N, p.K});
  auto tRef = matx::make_tensor<float>(p.refC, {p.M, p.N});
  (tRef = matx::matmul(tAF, tBF.PermuteMatrix())).run(*ctx.exec);
  if (p.gated) {
    CHECK_CUDA(cudaMalloc(&b2F, nB * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&p.refC2, nC * sizeof(float)));
    fp4::dequantizeFP4<r, float>(p.b2q, p.b2SF, b2F, p.N, p.K, p.gsB2, ctx.stream);
    auto tB2F = matx::make_tensor<float>(b2F, {p.N, p.K});
    auto tRef2 = matx::make_tensor<float>(p.refC2, {p.M, p.N});
    (tRef2 = matx::matmul(tAF, tB2F.PermuteMatrix())).run(*ctx.exec);
  }
  ctx.exec->sync();
  cudaFree(aF);
  cudaFree(bF);
  cudaFree(b2F);
}

template<QuantRecipe r>
__host__ __forceinline__
void makeProblem(Problem<r> &p, const int M, const int N, const int K, const bool gated, const bool wide,
                 const uint64_t seed, const bool reference, const Ctx &ctx) {
  using bf16 = __nv_bfloat16;
  p.M = M;
  p.N = N;
  p.K = K;
  p.gated = gated;
  p.wide = wide;
  const size_t nA = static_cast<size_t>(M) * K, nB = static_cast<size_t>(N) * K;
  CHECK_CUDA(cudaMalloc(&p.a, nA * sizeof(bf16)));
  CHECK_CUDA(cudaMalloc(&p.b, nB * sizeof(bf16)));
  CHECK_CUDA(cudaMalloc(&p.bias, N * sizeof(bf16)));
  fillInput(p.a, nA, wide, seed, ctx.stream);
  fillInput(p.b, nB, wide, seed + 1, ctx.stream);
  fillInput(p.bias, static_cast<size_t>(N), false, seed + 2, ctx.stream);
  p.gsA = quantizeOperand<r>(p.a, p.aq, p.aSF, M, K, ctx);
  p.gsB = quantizeOperand<r>(p.b, p.bq, p.bSF, N, K, ctx);
  if (gated) {
    CHECK_CUDA(cudaMalloc(&p.b2, nB * sizeof(bf16)));
    CHECK_CUDA(cudaMalloc(&p.biasV, N * sizeof(bf16)));
    fillInput(p.b2, nB, wide, seed + 3, ctx.stream);
    fillInput(p.biasV, static_cast<size_t>(N), false, seed + 4, ctx.stream);
    p.gsB2 = quantizeOperand<r>(p.b2, p.b2q, p.b2SF, N, K, ctx);
  }
  if (reference) {
    makeReference(p, ctx);
  }
  CHECK_CUDA(cudaStreamSynchronize(ctx.stream));
}

template<QuantRecipe r>
__host__ __forceinline__
void destroyProblem(Problem<r> &p) {
  for (void *x : {static_cast<void *>(p.a), static_cast<void *>(p.b), static_cast<void *>(p.b2),
                  static_cast<void *>(p.bias), static_cast<void *>(p.biasV), static_cast<void *>(p.aq),
                  static_cast<void *>(p.bq), static_cast<void *>(p.b2q), static_cast<void *>(p.aSF),
                  static_cast<void *>(p.bSF), static_cast<void *>(p.b2SF), static_cast<void *>(p.refC),
                  static_cast<void *>(p.refC2), static_cast<void *>(p.a8), static_cast<void *>(p.b8),
                  static_cast<void *>(p.b28), static_cast<void *>(p.a8SF), static_cast<void *>(p.b8SF),
                  static_cast<void *>(p.b28SF)}) {
    cudaFree(x);
  }
  p = Problem<r>{};
}

template<QuantRecipe r>
__host__ __forceinline__
void quantizeMx8(uint8_t *&q, uint8_t *&sf, const __nv_bfloat16 *const &x, const int rows, const int K,
                 const Ctx &ctx) {
  CHECK_CUDA(cudaMalloc(&q, static_cast<size_t>(rows) * K));
  CHECK_CUDA(cudaMalloc(&sf, mx::sfBytes(rows, K / 32)));
  mx::quantizeMXFP8<__nv_bfloat16>(x, q, sf, rows, K, ctx.stream);
}

template<QuantRecipe r>
__host__ __forceinline__
void makeMx8Operands(Problem<r> &p, const Ctx &ctx) {
  quantizeMx8<r>(p.a8, p.a8SF, p.a, p.M, p.K, ctx);
  quantizeMx8<r>(p.b8, p.b8SF, p.b, p.N, p.K, ctx);
  if (p.gated) {
    quantizeMx8<r>(p.b28, p.b28SF, p.b2, p.N, p.K, ctx);
  }
  CHECK_CUDA(cudaStreamSynchronize(ctx.stream));
}

// Compares the output of a tile kernel with the reference y, the FP32 epilogue of the FP32 GEMM of the dequantized
// operands.
//  - BF16 output: error(%) = share of elements with |out - y| > atol + rtol |y| (rtol 1e-2 > one BF16 ulp,
//    atol = 1e-3 rms(y) covers cancellation), relFrob = ||out - y|| / ||y||.
//  - FP4 output: y is quantized with the library recipe (fp4::quantizeFP4 of the FP32 y) and both are dequantized.
//    error(%) = share of elements that differ from the dequantized reference (a different FP32 accumulation order
//    can flip a rounding decision or, rarely, a scale), relFrob = ||deq(out) - deq(ref)|| / ||deq(ref)||,
//    codeMismatch / scaleMismatch = share of differing E2M1 / scale bytes, quantErr = ||deq(out) - y|| / ||y||
//    next to refQuantErr = ||deq(ref) - y|| / ||y||, the unavoidable quantization error.
struct TileResult {
  double errorPct = -1, relFrobPct = -1, codeMismatchPct = -1, scaleMismatchPct = -1, quantErrPct = -1,
         refQuantErrPct = -1;
};

__host__ __forceinline__
double neqBytesPct(const uint8_t *const &x, const uint8_t *const &y, const size_t &n, const Ctx &ctx) {
  cudaMemsetAsync(ctx.count, 0, sizeof(unsigned long long), ctx.stream);
  neqBytesKernel<<<1024, 256, 0, ctx.stream>>>(x, y, n, ctx.count);
  unsigned long long h = 0;
  cudaMemcpyAsync(&h, ctx.count, sizeof(h), cudaMemcpyDeviceToHost, ctx.stream);
  cudaStreamSynchronize(ctx.stream);
  return 100.0 * static_cast<double>(h) / static_cast<double>(n);
}

template<QuantRecipe r>
__host__ __forceinline__
TileResult compareBF16(const Problem<r> &p, const Ctx &ctx, const float *const &refY,
                       const __nv_bfloat16 *const &outC) {
  const size_t nC = static_cast<size_t>(p.M) * p.N;
  float *x;
  CHECK_CUDA(cudaMalloc(&x, nC * sizeof(float)));
  bf16ToFloatKernel<<<(nC + 255) / 256, 256, 0, ctx.stream>>>(outC, x, nC);
  constexpr float rtol = 1e-2f;
  const auto base = diffStats(x, refY, nC, rtol, 0.f, ctx.stats, ctx.stream);
  const auto d = diffStats(x, refY, nC, rtol, static_cast<float>(1e-3 * base.rms), ctx.stats, ctx.stream);
  cudaFree(x);
  TileResult res;
  res.errorPct = d.failPct;
  res.relFrobPct = d.relFrob * 100;
  return res;
}

// gsOut: decode scale of the output tensor (nvfp4)
template<QuantRecipe r>
__host__ __forceinline__
TileResult compareFP4(const Problem<r> &p, const Ctx &ctx, const float *const &refY, const float &gsOut,
                      const uint8_t *const &outQ, const uint8_t *const &outSF) {
  const int M = p.M, N = p.N;
  const size_t nC = static_cast<size_t>(M) * N;
  const size_t sfSize = fp4::sfBytes<r>(M, N);
  uint8_t *refQ, *refSF;
  float *refDeq, *x;
  CHECK_CUDA(cudaMalloc(&refQ, fp4::dataBytes(M, N)));
  CHECK_CUDA(cudaMalloc(&refSF, sfSize));
  CHECK_CUDA(cudaMalloc(&refDeq, nC * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&x, nC * sizeof(float)));
  fp4::quantizeFP4<r, float>(refY, refQ, refSF, M, N, gsOut, ctx.stream);
  fp4::dequantizeFP4<r, float>(refQ, refSF, refDeq, M, N, gsOut, ctx.stream);
  fp4::dequantizeFP4<r, float>(outQ, outSF, x, M, N, gsOut, ctx.stream);
  TileResult res;
  const auto exact = diffStats(x, refDeq, nC, 0.f, 0.f, ctx.stats, ctx.stream);
  res.errorPct = exact.failPct;
  res.relFrobPct = exact.relFrob * 100;
  res.codeMismatchPct = neqBytesPct(outQ, refQ, fp4::dataBytes(M, N), ctx);
  res.scaleMismatchPct = neqBytesPct(outSF, refSF, sfSize, ctx);
  res.quantErrPct = diffStats(x, refY, nC, 0.f, 0.f, ctx.stats, ctx.stream).relFrob * 100;
  res.refQuantErrPct = diffStats(refDeq, refY, nC, 0.f, 0.f, ctx.stats, ctx.stream).relFrob * 100;
  cudaFree(refQ);
  cudaFree(refSF);
  cudaFree(refDeq);
  cudaFree(x);
  return res;
}

__host__ __forceinline__
void checkHeader() {
  printf("mode, fmt, mainloop, M, N, K, data, activation, smemBytes, regs, localBytes, grid, error(%%), relFrob(%%), "
         "codeMismatch(%%), scaleMismatch(%%), quantErr(%%), refQuantErr(%%)\n");
}

template<QuantRecipe r>
__host__ __forceinline__
void checkPrint(const Problem<r> &p, const char *const mode, const char *const cfg, const char *const act,
                const int smemBytes, const cudaFuncAttributes &attr, const int grid, const TileResult &t) {
  printf("%s, %s, %s, %d, %d, %d, %s, %s, %d, %d, %zu, %d, %.5f, %.5f, %.5f, %.5f, %.4f, %.4f\n", mode, fmtName<r>(),
         cfg, p.M, p.N, p.K, p.wide ? "wide" : "uniform", act, smemBytes, attr.numRegs, attr.localSizeBytes, grid,
         t.errorPct, t.relFrobPct, t.codeMismatchPct, t.scaleMismatchPct, t.quantErrPct, t.refQuantErrPct);
  fflush(stdout);
}

// the amax of y, i.e., the calibration of the static scale of the output tensor (nvfp4), as decode scale
__host__ __forceinline__
float outputScale(const float *const &y, const size_t &n, const Ctx &ctx) {
  float *dAmax;
  CHECK_CUDA(cudaMalloc(&dAmax, sizeof(float)));
  fp4::amax<float>(y, n, dAmax, ctx.stream);
  float amax = 0.f;
  cudaMemcpyAsync(&amax, dAmax, sizeof(float), cudaMemcpyDeviceToHost, ctx.stream);
  cudaStreamSynchronize(ctx.stream);
  cudaFree(dAmax);
  return fp4::nvfp4GlobalScale(amax);
}

// vanilla GEMM (one B matrix) with the BF16 or the FP4 epilogue on 3 persistent CTAs and a poisoned workspace
template<QuantRecipe r, typename Mainloop, typename Act, TileEpi epi>
__host__ __forceinline__
void checkVanilla(const Problem<r> &p, const Ctx &ctx, const char *const cfg, const char *const actName) {
  using bf16 = __nv_bfloat16;
  constexpr int bM = cute::get<0>(typename Mainloop::TileShape{});
  constexpr int bN = cute::get<1>(typename Mainloop::TileShape{});
  constexpr int bK = cute::get<2>(typename Mainloop::TileShape{});
  constexpr bool toBF16 = epi == TileEpi::bf16;
  const int M = p.M, N = p.N, K = p.K;
  if (Mainloop::sharedSize > maxSharedMemory || p.gated || M % bM != 0 || N % bN != 0 || K % bK != 0) {
    return; // does not fit into the shared memory of an sm_120 CTA / shape not supported by this configuration
  }
  auto stream = ctx.stream;
  auto kernel = tileVanillaKernel<Mainloop, Act, epi>;
  CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Mainloop::sharedSize));
  cudaFuncAttributes attr{};
  CHECK_CUDA(cudaFuncGetAttributes(&attr, kernel));
  const size_t nC = static_cast<size_t>(M) * N;
  float *refY;
  bf16 *outC = nullptr;
  uint8_t *outQ = nullptr, *outSF = nullptr;
  CHECK_CUDA(cudaMalloc(&refY, nC * sizeof(float)));
  refVanillaKernel<Act><<<(nC + 255) / 256, 256, 0, stream>>>(p.refC, p.bias, refY, nC, N);
  float gsOut = 1.f;
  if constexpr (toBF16) {
    CHECK_CUDA(cudaMalloc(&outC, nC * sizeof(bf16)));
    cudaMemsetAsync(outC, 0xFF, nC * sizeof(bf16), stream); // NaN
  }
  else {
    CHECK_CUDA(cudaMalloc(&outQ, fp4::dataBytes(M, N)));
    CHECK_CUDA(cudaMalloc(&outSF, fp4::sfBytes<r>(M, N)));
    cudaMemsetAsync(outQ, 0xFF, fp4::dataBytes(M, N), stream);
    cudaMemsetAsync(outSF, 0xFF, fp4::sfBytes<r>(M, N), stream);
    if constexpr (r == QuantRecipe::nvfp4) {
      gsOut = outputScale(refY, nC, ctx);
    }
  }
  const int grid = std::min((M / bM) * (N / bN), 3);
  kernel<<<grid, 256, Mainloop::sharedSize, stream>>>(p.ops(fp4::nvfp4EncodeScale(gsOut)), outC, outQ, outSF, 1);
  CHECK_CUDA(cudaStreamSynchronize(stream));
  CHECK_CUDA(cudaGetLastError());
  const auto res = toBF16 ? compareBF16(p, ctx, refY, outC) : compareFP4(p, ctx, refY, gsOut, outQ, outSF);
  checkPrint(p, toBF16 ? "vanilla_bf16" : "vanilla_fp4", cfg, actName, Mainloop::sharedSize, attr, grid, res);
  cudaFree(refY);
  cudaFree(outC);
  cudaFree(outQ);
  cudaFree(outSF);
}

// gated GEMM, gated FP4 epilogue; bN is the width of each of the two B tiles
template<QuantRecipe r, typename Mainloop, typename Act>
__host__ __forceinline__
void checkGated(const Problem<r> &p, const Ctx &ctx, const char *const cfg, const char *const actName) {
  constexpr int bM = cute::get<0>(typename Mainloop::TileShape{});
  constexpr int bN = cute::get<1>(typename Mainloop::TileShape{});
  constexpr int bK = cute::get<2>(typename Mainloop::TileShape{});
  const int M = p.M, N = p.N, K = p.K;
  if (Mainloop::sharedSize > maxSharedMemory || !p.gated || M % bM != 0 || N % bN != 0 || K % bK != 0) {
    return;
  }
  auto stream = ctx.stream;
  auto kernel = tileGatedKernel<Mainloop, Act, TileEpi::quant>;
  CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Mainloop::sharedSize));
  cudaFuncAttributes attr{};
  CHECK_CUDA(cudaFuncGetAttributes(&attr, kernel));
  const size_t nC = static_cast<size_t>(M) * N;
  float *refY;
  uint8_t *outQ, *outSF;
  CHECK_CUDA(cudaMalloc(&refY, nC * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&outQ, fp4::dataBytes(M, N)));
  CHECK_CUDA(cudaMalloc(&outSF, fp4::sfBytes<r>(M, N)));
  refGatedKernel<Act><<<(nC + 255) / 256, 256, 0, stream>>>(p.refC, p.refC2, p.bias, p.biasV, refY, p.swishA, p.swishB,
                                                            nC, N);
  const float gsOut = r == QuantRecipe::nvfp4 ? outputScale(refY, nC, ctx) : 1.f;
  cudaMemsetAsync(outQ, 0xFF, fp4::dataBytes(M, N), stream);
  cudaMemsetAsync(outSF, 0xFF, fp4::sfBytes<r>(M, N), stream);
  const int grid = std::min((M / bM) * (N / bN), 3);
  kernel<<<grid, 256, Mainloop::sharedSize, stream>>>(p.ops(fp4::nvfp4EncodeScale(gsOut)), nullptr, outQ, outSF, 1);
  CHECK_CUDA(cudaStreamSynchronize(stream));
  CHECK_CUDA(cudaGetLastError());
  checkPrint(p, "gated_fp4", cfg, actName, Mainloop::sharedSize, attr, grid,
             compareFP4(p, ctx, refY, gsOut, outQ, outSF));
  cudaFree(refY);
  cudaFree(outQ);
  cudaFree(outSF);
}

// cuBLASLt on the same quantized operands (alpha = the tensor scales), against the FP32 reference
template<QuantRecipe r>
__host__ __forceinline__
void checkCublasLt(const Problem<r> &p, const Ctx &ctx) {
  using bf16 = __nv_bfloat16;
  const size_t nC = static_cast<size_t>(p.M) * p.N;
  LtGemm gemm;
  gemm.init(ctx.lt, r == QuantRecipe::nvfp4 ? LtKind::nvfp4 : LtKind::mxfp4, p.M, p.N, p.K, p.aSF, p.bSF,
            ctx.ltWorkspace, ctx.ltWorkspaceBytes);
  if (!gemm.valid) {
    printf("cublasLt_%s, %s, %d, %d, %d, %s, N/A (no cuBLASLt algorithm), , , \n", fmtName<r>(), fmtName<r>(), p.M,
           p.N, p.K, p.wide ? "wide" : "uniform");
    return;
  }
  bf16 *c;
  float *x;
  CHECK_CUDA(cudaMalloc(&c, nC * sizeof(bf16)));
  CHECK_CUDA(cudaMalloc(&x, nC * sizeof(float)));
  cudaMemsetAsync(c, 0xFF, nC * sizeof(bf16), ctx.stream);
  gemm.run(p.aq, p.bq, c, p.gsA * p.gsB, ctx.stream);
  const auto res = compareBF16(p, ctx, p.refC, c);
  CHECK_CUDA(cudaStreamSynchronize(ctx.stream));
  printf("cublasLt, %s, GEMM only, %d, %d, %d, %s, -, -, -, -, -, -, -, -, %.5f, %.5f, -, -, -, -\n", fmtName<r>(), p.M,
         p.N, p.K, p.wide ? "wide" : "uniform", res.errorPct, res.relFrobPct);
  gemm.destroy();
  cudaFree(c);
  cudaFree(x);
}

// all candidate configurations on one small problem
template<QuantRecipe r>
__host__ __forceinline__
void checkTileConfigs(const Problem<r> &p, const Ctx &ctx) {
  // vanilla: GEMM1-like (BF16 epilogue, identity) and GEMM0-like (FP4 epilogue, SiLU)
  checkVanilla<r, FP4Tile<r, 128, 128, 3, false>, IdentityAct, TileEpi::bf16>(p, ctx, "v128x128x128s3", "identity");
  checkVanilla<r, FP4Tile<r, 128, 128, 3, false>, SiluAct, TileEpi::quant>(p, ctx, "v128x128x128s3", "silu");
  checkVanilla<r, FP4Tile<r, 128, 128, 4, false>, IdentityAct, TileEpi::bf16>(p, ctx, "v128x128x128s4", "identity");
  checkVanilla<r, FP4Tile<r, 128, 128, 4, false>, SiluAct, TileEpi::quant>(p, ctx, "v128x128x128s4", "silu");
  checkVanilla<r, FP4Tile<r, 128, 256, 2, false>, IdentityAct, TileEpi::bf16>(p, ctx, "v128x128x256s2", "identity");
  checkVanilla<r, FP4Tile<r, 128, 256, 2, false>, SiluAct, TileEpi::quant>(p, ctx, "v128x128x256s2", "silu");
  // gated: two B tiles of 64 or 128 columns
  checkGated<r, FP4Tile<r, 128, 128, 3, true>, SiluAct>(p, ctx, "g128x(128+128)x128s3", "silu");
  checkGated<r, FP4Tile<r, 64, 128, 4, true>, SiluAct>(p, ctx, "g128x(64+64)x128s4", "silu");
}

template<QuantRecipe r>
__host__ __forceinline__
void checkSection(const Ctx &ctx) {
  // 3 persistent CTAs for 15 tiles; the second shape has K % 256 != 0
  const std::array<int, 4> shapes[] = {{384, 640, 1536, 0}, {384, 640, 1536, 1}, {256, 384, 640, 0}};
  for (const bool gated : {false, true}) {
    for (const auto &s : shapes) {
      Problem<r> p;
      makeProblem(p, s[0], s[1], s[2], gated, s[3] != 0, 0x1234 + s[0] + s[3], true, ctx);
      checkTileConfigs<r>(p, ctx);
      if (!gated && s[3] == 0 && s[2] == 1536) {
        checkCublasLt<r>(p, ctx);
      }
      destroyProblem(p);
    }
  }
}

__host__ __forceinline__
void unitSection(cudaStream_t stream) {
  checkE2M1(stream);
  checkScales(stream);
  checkSfLayout<32>();
  checkSfLayout<16>();
  printf("\nfmt, rows, K, dtype, data, gs, amax_mismatch, max_err_vs_blockamax(%%), mean_rel_err(%%), "
         "rel_frobenius(%%), scale_mismatch, data_mismatch, dequant_mismatch, padding_nonzero, offset_collisions\n");
  quantSection<QuantRecipe::mxfp4>(stream);
  quantSection<QuantRecipe::nvfp4>(stream);
}

// ------------------------------------------------------------------------------------------------------------
// benchmark: the tile kernels (mainloop alone with a sink epilogue, mainloop + epilogue) and cuBLASLt, GEMM only
// ------------------------------------------------------------------------------------------------------------

// the CTA-wide output buffers of the tile kernels (large enough for every format) and the best candidates so far
struct Bufs {
  __nv_bfloat16 *c = nullptr;
  uint8_t *q = nullptr, *sf = nullptr;
};

struct Best {
  double tf = -1;
  std::string cfg;

  __host__ __forceinline__
  void offer(const double t, const char *const name) {
    if (t > tf) {
      tf = t;
      cfg = name;
    }
  }
};

template<typename Mainloop, typename Act, TileEpi epi, bool gated>
__host__ __forceinline__
auto tileKernel() {
  if constexpr (gated) {
    return tileGatedKernel<Mainloop, Act, epi>;
  }
  else {
    return tileVanillaKernel<Mainloop, Act, epi>;
  }
}

// one candidate: prints a row, updates the best mainloop-only and with-epilogue throughput. Skipped when it needs
// more shared memory than an sm_120 CTA has or the shape is no multiple of the tile.
template<typename Mainloop, typename Act, bool gated>
__host__ __forceinline__
void benchCandidate(const int M, const int N, const int K, const Ops &ops, const Bufs &bufs, const Ctx &ctx,
                    const char *const fmt, const char *const cfg, Best &bestMain, Best &bestEpi) {
  constexpr int bM = cute::get<0>(typename Mainloop::TileShape{});
  constexpr int bN = cute::get<1>(typename Mainloop::TileShape{});
  constexpr int bK = cute::get<2>(typename Mainloop::TileShape{});
  if (Mainloop::sharedSize > maxSharedMemory || M % bM != 0 || N % bN != 0 || K % bK != 0) {
    printf("%s, %s, %d, %d, %d, skipped (smem %d B or K %% %d)\n", fmt, cfg, M, N, K, Mainloop::sharedSize, bK);
    return;
  }
  auto stream = ctx.stream;
  auto kSink = tileKernel<Mainloop, Act, TileEpi::sink, gated>();
  auto kEpi = tileKernel<Mainloop, Act, TileEpi::quant, gated>();
  CHECK_CUDA(cudaFuncSetAttribute(kSink, cudaFuncAttributeMaxDynamicSharedMemorySize, Mainloop::sharedSize));
  CHECK_CUDA(cudaFuncSetAttribute(kEpi, cudaFuncAttributeMaxDynamicSharedMemorySize, Mainloop::sharedSize));
  cudaFuncAttributes attrSink{}, attrEpi{};
  CHECK_CUDA(cudaFuncGetAttributes(&attrSink, kSink));
  CHECK_CUDA(cudaFuncGetAttributes(&attrEpi, kEpi));
  const int grid = std::min((M / bM) * (N / bN), ctx.numSMs);
  const double flops = (gated ? 4.0 : 2.0) * M * N * K;
  const float sinkMs = timeMs([&] { kSink<<<grid, 256, Mainloop::sharedSize, stream>>>(ops, bufs.c, bufs.q, bufs.sf, 0); },
                              stream, 10, ctx.runs);
  const float epiMs = timeMs([&] { kEpi<<<grid, 256, Mainloop::sharedSize, stream>>>(ops, bufs.c, bufs.q, bufs.sf, 0); },
                             stream, 10, ctx.runs);
  const double tfSink = tflops(flops, sinkMs), tfEpi = tflops(flops, epiMs);
  printf("%s, %s, %d, %d, %d, %d, %d/%d, %zu/%zu, %d, %.4f, %.1f, %.4f, %.1f\n", fmt, cfg, M, N, K, Mainloop::sharedSize,
         attrSink.numRegs, attrEpi.numRegs, attrSink.localSizeBytes, attrEpi.localSizeBytes, grid, sinkMs, tfSink, epiMs,
         tfEpi);
  fflush(stdout);
  bestMain.offer(tfSink, cfg);
  bestEpi.offer(tfEpi, cfg);
}

// cuBLASLt of the whole problem (gated: the gate and the up GEMM), -1 if it has no algorithm
__host__ __forceinline__
double benchCublasLt(const LtKind kind, const int M, const int N, const int K, const bool gated, const void *const &a,
                     const void *const &sfa, const void *const &b, const void *const &sfb, const void *const &b2,
                     const void *const &sfb2, const float alpha, const Ctx &ctx) {
  void *c;
  CHECK_CUDA(cudaMalloc(&c, static_cast<size_t>(M) * N * sizeof(__nv_bfloat16)));
  LtGemm g1, g2;
  g1.init(ctx.lt, kind, M, N, K, sfa, sfb, ctx.ltWorkspace, ctx.ltWorkspaceBytes);
  if (gated) {
    g2.init(ctx.lt, kind, M, N, K, sfa, sfb2, ctx.ltWorkspace, ctx.ltWorkspaceBytes);
  }
  double tf = -1;
  if (g1.valid && (!gated || g2.valid)) {
    const float ms = timeMs([&] {
      g1.run(a, b, c, alpha, ctx.stream);
      if (gated) {
        g2.run(a, b2, c, alpha, ctx.stream);
      }
    }, ctx.stream, 10, ctx.runs);
    tf = tflops((gated ? 4.0 : 2.0) * M * N * K, ms);
  }
  for (auto *g : {&g1, &g2}) {
    if (g->valid) {
      g->destroy();
    }
  }
  cudaFree(c);
  return tf;
}

template<QuantRecipe r>
__host__ __forceinline__
void benchShape(const int M, const int N, const int K, const bool gated, const Ctx &ctx) {
  Problem<r> p;
  makeProblem(p, M, N, K, gated, false, 0xbe7c4 + M + K, false, ctx);
  makeMx8Operands(p, ctx);
  Bufs bufs;
  CHECK_CUDA(cudaMalloc(&bufs.c, 4096 * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&bufs.q, static_cast<size_t>(M) * N));
  CHECK_CUDA(cudaMalloc(&bufs.sf, static_cast<size_t>(M) * N / 8 + (1 << 20)));
  // timing only: the output encode scale is 1 (nvfp4), whatever the data are the work is the same
  const Ops ops = p.ops(1.f);
  const Ops ops8 = p.ops8();
  Best fp4Main, fp4Epi, mx8Main, mx8Epi;
  printf("\n# %s %s M=%d N=%d K=%d\n", gated ? "gated (GEMM0 of a gated MLP, 2 B matrices)" : "vanilla", fmtName<r>(), M, N, K);
  printf("fmt, mainloop, M, N, K, smemBytes, regs(sink/epi), localBytes(sink/epi), grid, sink_ms, mainloop_TFLOPS, "
         "epi_ms, withEpilogue_TFLOPS\n");
  if (gated) {
    benchCandidate<FP4Tile<r, 128, 128, 3, true>, SiluAct, true>(M, N, K, ops, bufs, ctx, fmtName<r>(),
                                                                    "g128x(128+128)x128s3", fp4Main, fp4Epi);
    benchCandidate<FP4Tile<r, 64, 128, 4, true>, SiluAct, true>(M, N, K, ops, bufs, ctx, fmtName<r>(),
                                                                   "g128x(64+64)x128s4", fp4Main, fp4Epi);
    benchCandidate<MX8Tile<128, 64, 3, true>, SiluAct, true>(M, N, K, ops8, bufs, ctx, "mxfp8", "g128x(128+128)x64s3",
                                                             mx8Main, mx8Epi);
    benchCandidate<MX8Tile<64, 64, 4, true>, SiluAct, true>(M, N, K, ops8, bufs, ctx, "mxfp8", "g128x(64+64)x64s4",
                                                            mx8Main, mx8Epi);
  }
  else {
    benchCandidate<FP4Tile<r, 128, 128, 3, false>, SiluAct, false>(M, N, K, ops, bufs, ctx, fmtName<r>(),
                                                                      "v128x128x128s3", fp4Main, fp4Epi);
    benchCandidate<FP4Tile<r, 128, 128, 4, false>, SiluAct, false>(M, N, K, ops, bufs, ctx, fmtName<r>(),
                                                                      "v128x128x128s4", fp4Main, fp4Epi);
    if (K % 256 == 0) {
      benchCandidate<FP4Tile<r, 128, 256, 2, false>, SiluAct, false>(M, N, K, ops, bufs, ctx, fmtName<r>(),
                                                                        "v128x128x256s2", fp4Main, fp4Epi);
    }
    benchCandidate<MX8Tile<128, 64, 4, false>, SiluAct, false>(M, N, K, ops8, bufs, ctx, "mxfp8", "v128x128x64s4",
                                                               mx8Main, mx8Epi);
    benchCandidate<MX8Tile<128, 128, 2, false>, SiluAct, false>(M, N, K, ops8, bufs, ctx, "mxfp8", "v128x128x128s2",
                                                                mx8Main, mx8Epi);
  }
  const double ltFp4 = benchCublasLt(r == QuantRecipe::nvfp4 ? LtKind::nvfp4 : LtKind::mxfp4, M, N, K, gated, p.aq,
                                     p.aSF, p.bq, p.bSF, p.b2q, p.b2SF, p.gsA * p.gsB, ctx);
  const double ltMx8 = benchCublasLt(LtKind::mxfp8, M, N, K, gated, p.a8, p.a8SF, p.b8, p.b8SF, p.b28, p.b28SF, 1.f, ctx);
  printf("summary, mode, fmt, M, N, K, FP4_mainloop_TFLOPS(best cfg), FP4_withEpilogue_TFLOPS(best cfg), "
         "MXFP8_mainloop_TFLOPS, MXFP8_withEpilogue_TFLOPS(best cfg), cublasLt_%s_TFLOPS, cublasLt_MXFP8_TFLOPS, "
         "FP4_epi/MXFP8_epi, FP4_epi/cublasLt_%s, MXFP8_epi/cublasLt_MXFP8\n", fmtName<r>(), fmtName<r>());
  printf("summary, %s, %s, %d, %d, %d, %.1f (%s), %.1f (%s), %.1f, %.1f (%s), %.1f, %.1f, %.3f, %.3f, %.3f\n",
         gated ? "gated" : "vanilla", fmtName<r>(), M, N, K, fp4Main.tf, fp4Main.cfg.c_str(), fp4Epi.tf,
         fp4Epi.cfg.c_str(), mx8Main.tf, mx8Epi.tf, mx8Epi.cfg.c_str(), ltFp4, ltMx8,
         mx8Epi.tf > 0 ? fp4Epi.tf / mx8Epi.tf : -1.0, ltFp4 > 0 ? fp4Epi.tf / ltFp4 : -1.0,
         ltMx8 > 0 ? mx8Epi.tf / ltMx8 : -1.0);
  fflush(stdout);
  cudaFree(bufs.c);
  cudaFree(bufs.q);
  cudaFree(bufs.sf);
  destroyProblem(p);
}

template<QuantRecipe r>
__host__ __forceinline__
void benchSection(const Ctx &ctx) {
  benchShape<r>(8192, 8192, 8192, false, ctx);
  benchShape<r>(8192, 8192, 5120, true, ctx); // GEMM0 of Llama4-Scout: M = 8192 tokens, N = I = 8192, K = H = 5120
  benchShape<r>(4096, 2048, 768, false, ctx); // small shape: K = 768 is 6 k-tiles of 128 (3 of 256)
}

int main(const int argc, char **argv) {
  bool check = false, bench = false;
  int runs = 50;
  std::string fmt;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--check") {
      check = true;
    }
    else if (arg == "--bench") {
      bench = true;
    }
    else if (arg.rfind("--fmt=", 0) == 0) {
      fmt = arg.substr(6);
    }
    else if (arg.rfind("--runs=", 0) == 0) {
      runs = std::stoi(arg.substr(7));
    }
  }
  if (!check && !bench) {
    check = bench = true;
  }
  const bool mxfp4 = fmt.empty() || fmt == "mxfp4";
  const bool nvfp4 = fmt.empty() || fmt == "nvfp4";
  cudaSetDevice(0);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  matx::cudaExecutor exec{stream};
  Ctx ctx;
  ctx.stream = stream;
  ctx.exec = &exec;
  ctx.runs = runs;
  CHECK_CUBLAS(cublasLtCreate(&ctx.lt));
  ctx.ltWorkspaceBytes = 32ULL << 20;
  CHECK_CUDA(cudaMalloc(&ctx.ltWorkspace, ctx.ltWorkspaceBytes));
  int dev = 0;
  cudaGetDevice(&dev);
  CHECK_CUDA(cudaDeviceGetAttribute(&ctx.numSMs, cudaDevAttrMultiProcessorCount, dev));
  CHECK_CUDA(cudaMalloc(&ctx.stats, 4 * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&ctx.count, sizeof(unsigned long long)));
  if (check) {
    unitSection(stream);
    printf("\n");
    checkHeader();
    if (mxfp4) {
      checkSection<QuantRecipe::mxfp4>(ctx);
    }
    if (nvfp4) {
      checkSection<QuantRecipe::nvfp4>(ctx);
    }
    printf("\n");
  }
  if (bench) {
    if (mxfp4) {
      benchSection<QuantRecipe::mxfp4>(ctx);
    }
    if (nvfp4) {
      benchSection<QuantRecipe::nvfp4>(ctx);
    }
  }
  cudaFree(ctx.stats);
  cudaFree(ctx.count);
  cudaFree(ctx.ltWorkspace);
  cublasLtDestroy(ctx.lt);
  cudaStreamDestroy(stream);
  return 0;
}
