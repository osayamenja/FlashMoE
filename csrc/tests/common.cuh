//
// Created by osayamen on 12/29/25.
//

#ifndef FLASHMOE_COMMON_CUH
#define FLASHMOE_COMMON_CUH

#include <algorithm>
#include <cassert>
#include <vector>
#include <tuple>
#include <random>

#include <cublasdx.hpp>
#include <curanddx.hpp>
#include <matx.h>
#include <cuda_runtime.h>

#include "../include/flashmoe/infra/packed.cuh"

template<typename T, typename S>
    struct Converter {
    __device__ auto operator()(const S& x) const {
        return static_cast<T>(x);
    }
};
template<>
struct Converter<__half, float> {
    __device__ auto operator()(const float& x) const {
        return  __float2half(x);
    }
};
template<>
struct Converter<__nv_bfloat16, float> {
    __device__ auto operator()(const float& x) const {
        return  __float2bfloat16(x);
    }
};
template<>
struct Converter<matx::matxFp16, float> {
    __device__ auto operator()(const float& x) const {
        return  __float2half(x);
    }
};
template<>
struct Converter<matx::matxBf16, float> {
    __device__ auto operator()(const float& x) const {
        return  __float2bfloat16(x);
    }
};

template<>
struct Converter<float, matx::matxFp16> {
    __device__ auto operator()(const matx::matxFp16& x) const {
        return  __half2float(x.x);
    }
};
template<>
struct Converter<float, matx::matxBf16> {
    __device__ auto operator()(const matx::matxBf16& x) const {
        return  __bfloat162float(x.x);
    }
};

// Deterministic 64-bit mix hash (fast, good enough for tie-breaking)
__device__ __forceinline__ uint64_t mix64(uint64_t x) {
    // SplitMix64 finalizer (good diffusion, fast)
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

// Convert to float in [0,1) using the top 24 bits (exactly representable in fp32)
__device__ __forceinline__ float u01_from_u64(const uint64_t& x) {
    const uint32_t m = static_cast<uint32_t>(x >> 40); // top 24 bits
    return __uint_as_float(m) * (1.0f / 16777216.0f); // 2^24
}

// jitter in [-eps/2, +eps/2], deterministic from 64-bit idx + seed
__device__ __forceinline__ float tie_jitter64(const uint64_t& linear_idx,
    const float& eps, const uint64_t seed = 0x12345678abcdefULL) {
    const uint64_t h = mix64(linear_idx ^ seed);
    const float u = u01_from_u64(h);     // [0,1)
    return (u - 0.5f) * eps;       // [-eps/2, +eps/2]
}

template<typename T, int Alignment = 16>
    struct VectorTypeDescriptor {
    using VectorWidth = cute::C<Alignment / sizeof(T)>;
    using VectorType = cutlass::AlignedArray<T, VectorWidth::value, Alignment>;
};

template <int Arch, bool predicate, bool addJitter = false, typename Element>
__global__ void generateRandUniform(
    Element* __restrict__ out,
    const __grid_constant__ size_t n,
    const __grid_constant__ long int seed,
    const __grid_constant__ float  minv,
    const __grid_constant__ float  maxv,
    const __grid_constant__ unsigned long long global_offset = 0ULL
) {
    // Describe a thread-level Philox generator specialized to a target SM arch.
    using RNG = decltype(curanddx::Generator<curanddx::philox4_32>() +
                         curanddx::SM<Arch>() +
                         curanddx::Thread());
    // Thread id in the launch
    const auto tid = static_cast<unsigned long long int>(blockIdx.x)
    * blockDim.x + threadIdx.x;

    constexpr int vF = 4;
    const size_t out_base = static_cast<size_t>(tid) * vF;

    RNG rng(seed, tid, global_offset);

    curanddx::uniform<float> dist(minv, maxv);

    auto v_raw = dist.generate4(rng);
    auto v_mixed = v_raw;
    if constexpr (addJitter) {
        constexpr auto eps = 1e-7f;
        constexpr auto jitter_seed = 0xCAFEBABEDEADBEEFULL;
        v_mixed = float4{
            v_raw.x + tie_jitter64(out_base, eps, jitter_seed),
            v_raw.y + tie_jitter64(out_base + 1, eps, jitter_seed),
            v_raw.z + tie_jitter64(out_base + 2, eps, jitter_seed),
            v_raw.w + tie_jitter64(out_base + 3, eps, jitter_seed)
        };
    }
    const auto v = v_mixed;

    constexpr Converter<Element, float> storeOp{};

    if constexpr (predicate) {
        if (out_base + (vF - 1) >= n) return;
        // n % 4 == 0
        using VTD = VectorTypeDescriptor<Element, vF * sizeof(Element)>;
        using VT = VTD::VectorType;
        static_assert(VTD::VectorWidth::value == vF);
        auto* __restrict__ vo = reinterpret_cast<VT*>(out);
        VT vt{};
        vt[0] = storeOp(v.x);
        vt[1] = storeOp(v.y);
        vt[2] = storeOp(v.z);
        vt[3] = storeOp(v.w);
        vo[tid] = vt;
    }
    else {
        if (out_base >= n) return;
        out[out_base + 0] = storeOp(v.x);
        if (out_base + 1 < n) out[out_base + 1] = storeOp(v.y);
        if (out_base + 2 < n) out[out_base + 2] = storeOp(v.z);
        if (out_base + 3 < n) out[out_base + 3] = storeOp(v.w);
    }
}

template<int Arch, bool addJitter = false, typename Element>
__host__ __forceinline__
void randUniform(Element* __restrict__ const& out,
    const  size_t& n, const size_t& seed, const float& minv,
    const float& maxv, cudaStream_t stream) {
    constexpr uint threads = 128;
    const auto blocks = static_cast<uint>(cute::ceil_div(n, threads * 4));
    if (n % 4 == 0) {
        generateRandUniform<Arch, true, addJitter><<<blocks, threads, 0, stream>>>(out, n, seed, minv, maxv);
    }
    else {
        generateRandUniform<Arch, false, addJitter><<<blocks, threads, 0, stream>>>(out, n, seed, minv, maxv);
    }
}

__host__ __forceinline__
float random_float(const float& min, const float& max, const uint& seed) {
    static std::mt19937 gen(seed);      // Mersenne Twister RNG
    std::uniform_real_distribution<float> dist(min, max);
    return dist(gen);
}

enum class WeightDistribution {
  Uniform,      // U(0,1]
  Exponential1,   // Exp(lambda=1)
  LogNormal       // logN(mu=0, sigma=1)
};

template<WeightDistribution Dist>
__host__ __forceinline__
float sample_positive_weight(std::mt19937& rng, const float d_a = -1.0f, const float d_b = 1.0f) {
  float w = 0.0f;
  if constexpr (Dist == WeightDistribution::Uniform) {
    std::uniform_real_distribution<float> uni01(d_a, d_b);
    w = uni01(rng);
  } else if constexpr (Dist == WeightDistribution::Exponential1) {
    std::exponential_distribution<float> exp1(1.0f);
    w = exp1(rng);
  } else if constexpr (Dist == WeightDistribution::LogNormal) {
    std::lognormal_distribution<float> logn(0.0f, 1.0f);
    w = logn(rng);
  } else {
    static_assert(Dist == WeightDistribution::Uniform ||
                  Dist == WeightDistribution::Exponential1 ||
                  Dist == WeightDistribution::LogNormal,
                  "Unsupported distribution");
  }
  if (w == 0.0f) w = 1e-7f;
  return w;
}

__host__ __forceinline__
std::vector<int> choose_topk_experts_from_scores(std::mt19937& rng, const int& E, const int& k,
    const float d_a = -1.0f, const float d_b = 1.0f) {
  assert(E > 0 && k > 0 && k <= E);

  std::uniform_real_distribution<float> score_dist(d_a, d_b);
  std::vector<std::pair<float,int>> scored;
  scored.reserve(static_cast<size_t>(E));
  for (int e = 0; e < E; ++e) scored.emplace_back(score_dist(rng), e);

  auto nth = scored.begin() + (E - k);
  std::ranges::nth_element(
      scored.begin(), nth, scored.end(),
      [](auto const& a, auto const& b) { return a.first < b.first; });

  std::vector<int> experts;
  experts.reserve(static_cast<size_t>(k));
  for (int r = 0; r < k; ++r) experts.push_back(scored[E - k + r].second);
  return experts;
}

// Generates a routing assignment:
// - each token appears exactly k times globally
// - within each expert row, each token appears at most once (guaranteed by construction)
// - expertCounts[e] <= S always
//
// Returns: (expertCounts[E], tokenIds[E*EC] in row-major [e, j])
template<WeightDistribution Dist = WeightDistribution::Uniform>
__host__ __forceinline__
auto generate_token_ids_and_expert_counts(const int& S, const int& E, const int& EC, const int& roundEC, const int& k,
    const float d_a = -1.0f, const float d_b = 1.0f) {
  assert(S > 0 && E > 0 && k > 0);
  assert(k <= E && "Need k <= E to pick k unique experts per token");

  std::random_device rd;
  std::mt19937 rng(rd());

  std::vector<int> expertCounts(static_cast<size_t>(E), 0);
  std::vector<int> writePos(static_cast<size_t>(E), 0);
  std::vector<int> seen(static_cast<size_t>(S), 0);

  std::vector<flashmoe::TPS> tokenIds(
      static_cast<size_t>(E) * static_cast<size_t>(roundEC), flashmoe::TPS{0, 0.0f});

  // For each token, select k unique experts, generate k weights, normalize, append.
  for (uint t = 0; t < S; ++t) {
    const std::vector<int> experts = choose_topk_experts_from_scores(rng, E, k, d_a, d_b);

    // Raw weights
    std::vector<float> w(static_cast<size_t>(k), 0.0f);
    float sumW = 0.0f;
    for (int r = 0; r < k; ++r) {
      w[static_cast<size_t>(r)] = sample_positive_weight<Dist>(rng, d_a, d_b);
      sumW += w[static_cast<size_t>(r)];
    }
    const float invSum = 1.0f / sumW;

    // Append to experts
    for (int r = 0; r < k; ++r) {
      const int e = experts[static_cast<size_t>(r)];
      const int j = writePos[static_cast<size_t>(e)]++;

      if (j < EC) {
        tokenIds[static_cast<size_t>(e) * static_cast<size_t>(roundEC) + static_cast<size_t>(j)] =
          flashmoe::TPS{t, w[static_cast<size_t>(r)] * invSum};
      }

      expertCounts[static_cast<size_t>(e)] = writePos[static_cast<size_t>(e)];
      seen[static_cast<size_t>(t)] += 1;
    }
  }

  // Sanity checks
  {
    long long total = 0;
    for (int e = 0; e < E; ++e) {
      assert(expertCounts[static_cast<size_t>(e)] <= EC);
      total += expertCounts[static_cast<size_t>(e)];
    }
    assert(total == 1LL * S * k);
    for (int t = 0; t < S; ++t) assert(seen[static_cast<size_t>(t)] == k);
  }

  return std::make_tuple(std::move(expertCounts), std::move(tokenIds));
}

__host__ __forceinline__
void printMetadata(const std::vector<int>& counts, const std::vector<flashmoe::TPS>& ids,
    const int& roundEC, const int& E) {
  const auto t0 = cute::make_tensor(counts.data(), cute::make_layout(
      cute::make_shape(1, E), cute::LayoutRight{}));
  print_tensor(t0);
  void* p = std::malloc(ids.size() * sizeof(uint));
  auto* indices = static_cast<uint*>(p);
  std::ranges::transform(ids.begin(), ids.end(), indices, [](const flashmoe::TPS t) {
      return t.tokenIdx;
  });
  const auto t1 = cute::make_tensor(indices, cute::make_layout(
      cute::make_shape(E, roundEC), cute::LayoutRight{}));
  print_tensor(t1);
  auto* scales = static_cast<float*>(p);
  std::ranges::transform(ids.begin(), ids.end(), scales, [](const flashmoe::TPS t) {
      return t.probability;
  });
  const auto t2 = cute::make_tensor(scales, cute::make_layout(
      cute::make_shape(E, roundEC), cute::LayoutRight{}));
  print_tensor(t2);
  std::free(p);
}

template<typename Element>
using MXE = cuda::std::conditional_t<cuda::std::is_same_v<Element, __half>, matx::matxFp16,
        cuda::std::conditional_t<cuda::std::is_same_v<Element, __nv_bfloat16>, matx::matxBf16,
        cuda::std::conditional_t<cuda::std::is_same_v<Element, cublasdx::tfloat32_t>, float, Element>>>;
#endif //FLASHMOE_COMMON_CUH