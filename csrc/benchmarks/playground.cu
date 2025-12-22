//
// Created by osayamen on 12/22/25.
//

#include <cublasdx.hpp>
#include <cutlass/epilogue/thread/activation.h>
#include <matx.h>
#include "cdx_gemm.cuh"
#include "../include/flashmoe/debug.cuh"

#define RTOL 1e-3
#define ATOL 1e-4

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
        cuda::std::conditional_t<cuda::std::is_same_v<Element, __nv_bfloat16>, matx::matxBf16, Element>>;
__host__ __forceinline__
void test_cdx() {
    const int M = 16;
    const int N = 16;
    const int K = 16;
    constexpr int bM = 16;
    constexpr int bN = 16;
    constexpr int bK = 16;
    constexpr int pipeStages = 1;
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
    Element* c = nullptr;
    Element* c_ref = nullptr;
    Element* bias = nullptr;

    cudaMallocManaged(&a, M * K * sizeof(Element));
    cudaMallocManaged(&b, K * N * sizeof(Element));
    cudaMallocManaged(&c, M * N * sizeof(ElementC));
    cudaMallocManaged(&c_ref, M * N * sizeof(ElementC));
    cudaMallocManaged(&bias, N * sizeof(ElementC));
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
    /*tA(0,0) = static_cast<Element>(1.f);
    tA(0,1) = static_cast<Element>(2.f);
    tA(1,0) = static_cast<Element>(3.f);
    tA(1,1) = static_cast<Element>(4.f);*/
    auto tB = matx::make_tensor<MX>(b_m, {N, K});
    (tB = matx::apply(ConvFunctor<MX>{}, matx::random<float>(tB.Shape(), matx::NORMAL, 17))).run(exec);
    /*tB(0,0) = static_cast<Element>(2.f);
    tB(0,1) = static_cast<Element>(3.f);
    tB(1,0) = static_cast<Element>(4.f);
    tB(1,1) = static_cast<Element>(5.f);*/
    auto tC = matx::make_tensor<MXC>(c_m, {M, N});
    auto tC_ref = matx::make_tensor<MXC>(c_ref_m, {M, N});
    auto tBias = matx::make_tensor<MXC>(bias_m, {N});
    //(tBias = matx::apply(ConvFunctor<MXC>{}, matx::random<float>(tBias.Shape(), matx::NORMAL))).run(exec);
    (tBias = matx::zeros<MXC>(tBias.Shape())).run(exec);
    using Act = cutlass::epilogue::thread::Identity<ElementC>;
    using ActM = cutlass::epilogue::thread::Identity<MX>;
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
    exec.sync();
    print(tC);
    print(tC_ref);
    auto result = matx::make_tensor<int>({});
    matx::allclose(result, tC, tC_ref, RTOL, ATOL, exec);
    //print(matx::isclose(tC, tC_ref));
    exec.sync();
    FLASHMOE_CHECK_CUDA(cudaPeekAtLastError());
    printf("Correct? %s\n", result() == 1 ? "Yes" : "No");
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(bias);
    cudaStreamDestroy(stream);
}

int main() {
    test_cdx();
}