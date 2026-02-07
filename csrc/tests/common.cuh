//
// Created by osayamen on 12/29/25.
//

#ifndef FLASHMOE_COMMON_CUH
#define FLASHMOE_COMMON_CUH

#include <random>

#include <cublasdx.hpp>
#include <curanddx.hpp>
#include <matx.h>
#include <cuda_runtime.h>


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

template<typename Element>
using MXE = cuda::std::conditional_t<cuda::std::is_same_v<Element, __half>, matx::matxFp16,
        cuda::std::conditional_t<cuda::std::is_same_v<Element, __nv_bfloat16>, matx::matxBf16,
        cuda::std::conditional_t<cuda::std::is_same_v<Element, cublasdx::tfloat32_t>, float, Element>>>;
#endif //FLASHMOE_COMMON_CUH