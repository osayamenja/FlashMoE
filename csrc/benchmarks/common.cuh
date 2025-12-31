//
// Created by osayamen on 12/29/25.
//

#ifndef FLASHMOE_COMMON_CUH
#define FLASHMOE_COMMON_CUH

#include <cublasdx.hpp>
#include <curanddx.hpp>
#include <matx.h>
#include <cuda_runtime.h>

#if !defined(CHECK_CUDA)
#  define CHECK_CUDA(e)                                      \
do {                                                         \
    cudaError_t code = (e);                                  \
    if (code != cudaSuccess) {                               \
        fprintf(stderr, "<%s:%d> %s:\n    %s: %s\n",         \
            __FILE__, __LINE__, #e,                          \
            cudaGetErrorName(code),                          \
            cudaGetErrorString(code));                       \
        fflush(stderr);                                      \
        exit(1);                                             \
    }                                                        \
} while (0)
#endif

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
__device__ __forceinline__ float tie_jitter64(const int64_t& linear_idx,
    const float& eps, const uint64_t seed = 0x12345678abcdefULL) {
    const uint64_t h = mix64(static_cast<uint64_t>(linear_idx) ^ seed);
    const float u = u01_from_u64(h);     // [0,1)
    return (u - 0.5f) * eps;       // [-eps/2, +eps/2]
}

template<typename T>
    struct VectorTypeDescriptor {
    using VectorWidth = cute::C<16 / sizeof(T)>;
    using VectorType = cutlass::AlignedArray<T, VectorWidth::value, 16>;
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

    const size_t out_base = static_cast<size_t>(tid) * 4;

    if (out_base >= n) return;

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

    if constexpr (!predicate) {
        // n % 4 == 0
        using VTD = VectorTypeDescriptor<Element>;
        using VT = VectorTypeDescriptor<Element>::VectorType;
        static_assert(VTD::VectorWidth::value == 4);
        auto* __restrict__ vo = reinterpret_cast<VT*>(out);
        VT vt{storeOp(v.x), storeOp(v.y), storeOp(v.z), storeOp(v.w)};
        vo[tid] = vt;
    }
    else {
        out[out_base + 0] = storeOp(v.x);
        if (out_base + 1 < n) out[out_base + 1] = storeOp(v.y);
        if (out_base + 2 < n) out[out_base + 2] = storeOp(v.z);
        if (out_base + 3 < n) out[out_base + 3] = storeOp(v.w);
    }
}

template<int Arch, bool addJitter = false, typename Element>
__host__ __forceinline__
void randUniform(Element* __restrict__ const& out,
    const  size_t& n, const long int& seed, const float& minv,
    const float& maxv, cudaStream_t stream) {
    constexpr int threads = 256;
    const int blocks = cute::ceil_div(n, 256 * 4);
    if (n % 4 == 0) {
        generateRandUniform<Arch, true, addJitter><<<blocks, threads, 0, stream>>>(out, n, seed, minv, maxv);
    }
    else {
        generateRandUniform<Arch, false, addJitter><<<blocks, threads, 0, stream>>>(out, n, seed, minv, maxv);
    }

}
template<typename Element>
using MXE = cuda::std::conditional_t<cuda::std::is_same_v<Element, __half>, matx::matxFp16,
        cuda::std::conditional_t<cuda::std::is_same_v<Element, __nv_bfloat16>, matx::matxBf16,
        cuda::std::conditional_t<cuda::std::is_same_v<Element, cublasdx::tfloat32_t>, float, Element>>>;
#endif //FLASHMOE_COMMON_CUH