//
// Created by osayamen on 12/29/25.
//

#ifndef FLASHMOE_COMMON_CUH
#define FLASHMOE_COMMON_CUH
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

template <int Arch, typename Element>
__global__ void generateRandUniform(
    Element* __restrict__ out,
    const __grid_constant__ size_t n,                // number of floats to write
    const __grid_constant__ long int seed,           // RNG seed
    const __grid_constant__ float  minv,
    const __grid_constant__ float  maxv,
    const __grid_constant__ unsigned long long global_offset = 0ULL
) {
    // Describe a thread-level Philox generator specialized to a target SM arch.
    using RNG = decltype(curanddx::Generator<curanddx::philox4_32>() +
                         curanddx::SM<Arch>() +
                         curanddx::Thread());
    // Thread id in the launch
    const auto tid = static_cast<unsigned long long int>(blockIdx.x) * blockDim.x + threadIdx.x;

    // Each thread produces 4 floats at a time.
    const size_t out_base = static_cast<size_t>(tid) * 4;

    if (out_base >= n) return;

    RNG rng(seed, tid, global_offset);

    // Uniform distribution in [minv, maxv)
    curanddx::uniform<float> dist(minv, maxv);

    // Generate 4 floats (Philox path)
    // Note: For Philox4_32 in cuRANDDx, generate4() is the supported form.
    const auto v = dist.generate4(rng);

    constexpr Converter<Element, float> storeOp{};

    if (n % 4 == 0) {
        // n % 4 == 0
        out[out_base + 0] = storeOp(v.x);
        out[out_base + 1] = storeOp(v.y);
        out[out_base + 2] = storeOp(v.z);
        out[out_base + 3] = storeOp(v.w);
    }
    else {
        // Store, handling the tail if n is not a multiple of 4
        out[out_base + 0] = storeOp(v.x);
        if (out_base + 1 < n) out[out_base + 1] = storeOp(v.y);
        if (out_base + 2 < n) out[out_base + 2] = storeOp(v.z);
        if (out_base + 3 < n) out[out_base + 3] = storeOp(v.w);
    }
}
#endif //FLASHMOE_COMMON_CUH