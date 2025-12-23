//
// Created by osayamen on 12/22/25.
//

#include <cublasdx.hpp>
#include <cutlass/epilogue/thread/activation.h>
#include <matx.h>
#include "cdx_gemm.cuh"
#include "../include/flashmoe/debug.cuh"

#define RTOL 1e-2
#define ATOL 1e-3

template<typename T>
struct ConvFunctor {
    __device__ auto operator()(const float& x) const {
        return static_cast<T>(x);
    }
};
template<>
struct ConvFunctor<matx::matxFp16> {
    __device__ auto operator()(const float& x) const {
        return  __float2half(x);
    }
};
template<>
struct ConvFunctor<matx::matxBf16> {
    __device__ auto operator()(const float& x) const {
        return  __float2bfloat16(x);
    }
};

template<typename Element>
using MXE = cuda::std::conditional_t<cuda::std::is_same_v<Element, __half>, matx::matxFp16,
        cuda::std::conditional_t<cuda::std::is_same_v<Element, __nv_bfloat16>, matx::matxBf16,
        cuda::std::conditional_t<cuda::std::is_same_v<Element, cublasdx::tfloat32_t>, float, Element>>>;

__host__ __forceinline__
void test_cdx() {
    const int M = 8192;
    const int N = 2048;
    const int K = 2048;
    constexpr int bM = 128;
    constexpr int bN = 128;
    constexpr int bK = 32;
    constexpr int pipeStages = 2;
    constexpr int alignment = 16;
    using Element = __half;
    using ElementC = Element;
    using MMA_C = float;
    static_assert(alignment >= sizeof(Element) && alignment <= 16 && cuda::is_power_of_two(alignment));

    cudaSetDevice(0);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    Element* a = nullptr;
    Element* b = nullptr;
    ElementC* c = nullptr;
    ElementC* c_ref = nullptr;
    ElementC* bias = nullptr;

    cudaMallocAsync(&a, M * K * sizeof(Element), stream);
    cudaMallocAsync(&b, K * N * sizeof(Element), stream);
    cudaMallocAsync(&c, M * N * sizeof(ElementC), stream);
    cudaMallocAsync(&c_ref, M * N * sizeof(ElementC), stream);
    cudaMallocAsync(&bias, N * sizeof(ElementC), stream);
    matx::cudaExecutor exec{stream};
    using MX = MXE<Element>;
    using MXC = MXE<ElementC>;
    auto* __restrict__ a_m = reinterpret_cast<MX*>(a);
    auto* __restrict__ b_m = reinterpret_cast<MX*>(b);
    auto* __restrict__ c_m = reinterpret_cast<MXC*>(c);
    auto* __restrict__ c_ref_m = reinterpret_cast<MXC*>(c_ref);
    auto* __restrict__ bias_m = reinterpret_cast<MXC*>(bias);
    auto tA = matx::make_tensor<MX>(a_m, {M, K});
    (tA = matx::apply(ConvFunctor<MX>{}, matx::random<float>(tA.Shape(), matx::NORMAL, 7))).run(exec);
    auto tB = matx::make_tensor<MX>(b_m, {N, K});
    (tB = matx::apply(ConvFunctor<MX>{}, matx::random<float>(tB.Shape(), matx::NORMAL, 17))).run(exec);
    auto tC = matx::make_tensor<MXC>(c_m, {M, N});
    auto tC_ref = matx::make_tensor<MXC>(c_ref_m, {M, N});
    auto tBias = matx::make_tensor<MXC>(bias_m, {N});
    (tBias = matx::apply(ConvFunctor<MXC>{}, matx::random<float>(tBias.Shape(), matx::NORMAL))).run(exec);
    using Act = cutlass::epilogue::thread::ReLU<MMA_C>;
    using ActM = cutlass::epilogue::thread::ReLU<MX>;
    using BLAS = decltype(
            cublasdx::Size<bM, bN, bK>() +
            cublasdx::Precision<Element, Element, MMA_C>() +
            cublasdx::Type<cublasdx::type::real>() +
            cublasdx::Function<cublasdx::function::MM>() +
            cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major, cublasdx::row_major>() +
            cublasdx::Block() +
            cublasdx::MaxAlignment() +
            cublasdx::StaticBlockDim() +
            cublasdx::SM<FLASHMOE_ARCH>());
    auto kernel = v1::gk<BLAS, pipeStages, Act, Element, ElementC>;
    int bps = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kernel, BLAS::max_threads_per_block, 0);
    const int blocks = min((M / bM) * (N / bN), bps * NUM_SMS);
    kernel<<<blocks, BLAS::suggested_block_dim, 0, stream>>>(a, b, c, bias, M, N, K);
    (tC_ref = matx::apply(ActM{}, (matx::matmul(tA, tB.PermuteMatrix()) + tBias))).run(exec);
    FLASHMOE_CHECK_CUDA(cudaPeekAtLastError());
    exec.sync();
    auto num_matches = matx::make_tensor<long int>({});
    (num_matches = matx::sum(matx::isclose(tC, tC_ref, RTOL, ATOL))).run(exec);
    exec.sync();
    const auto ep =  (1.0 - (static_cast<double>(num_matches()) / static_cast<double>(M*N))) * 100;
    printf("error_pct: %lf%%\n", ep);
    cudaFreeAsync(a, stream);
    cudaFreeAsync(b, stream);
    cudaFreeAsync(c, stream);
    cudaFreeAsync(c_ref, stream);
    cudaFreeAsync(bias, stream);
    FLASHMOE_CHECK_CUDA(cudaPeekAtLastError());
    cudaStreamDestroy(stream);
}

int main() {
    test_cdx();
}