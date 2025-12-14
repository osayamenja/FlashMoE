//
// Created by osayamen on 12/13/25.
//
#include <matx.h>
#include <cutlass/epilogue/thread/activation.h>
#include "../include/flashmoe/debug.cuh"
#include "../include/flashmoe/os/processor/gemm.cuh"
#include "../include/flashmoe/os/processor/mmaConfig.cuh"

#define RTOL 1e-5
#define ATOL 1e-8
template<typename Element>
__global__ void gemm_v1(const Element* __restrict__ a,
    const Element* __restrict__ b,
    const Element* c, Element* bias) {

}

template<typename Element, typename ElementC>
auto gk(const Element* __restrict__ a, const Element* __restrict__ b,
    ElementC* __restrict__ c_ext, ElementC* __restrict__ c,
    const int M, const int N, const int K, cudaStream_t stream) {

}

template<typename T>
__device__
T activation(const T& v) {
    using CutlassReLU = cutlass::epilogue::thread::ReLU<T>;
    return CutlassReLU{}(v);
}

template<typename T>
__device__
T converter (const float& v) {
    if constexpr (cuda::std::is_same_v<T, matx::matxFp16>) {
        return static_cast<T>(__float2half(v));
    }
    else if (cuda::std::is_same_v<T, matx::matxBf16>) {
        return static_cast<T>(__float2bfloat16(v));
    }
    else {
        return v;
    }
}

template<typename Element, typename ElementC>
auto reference(const Element* __restrict__ a, const Element* __restrict__ b,
    const Element* __restrict__ bias, ElementC* __restrict__ c_ext,
    ElementC* __restrict__ c,
    const int M, const int N, const int K, cudaStream_t stream) {
    matx::cudaExecutor exec{stream, true};
    auto tA = matx::make_tensor<Element>(a, {M, K});
    auto tB = matx::make_tensor<Element>(b, {K, N});
    auto tC = matx::make_tensor<ElementC>(c, {M, N});
    auto tBias = matx::make_tensor<Element>(bias, {N});
    using ReLU = cutlass::epilogue::thread::ReLU<ElementC>;
    // ReLU((a @ b) + bias)
    (tC = matx::apply(ReLU{}, (matx::matmul(tA, tB) + tBias))).run(exec);
    auto result = matx::make_tensor<int>({});
    auto tCx = matx::make_tensor<Element>(c_ext, {M, N});
    matx::allclose(result, tCx, tC, RTOL, ATOL, exec);
    // warmup
    for (int i = 0; i < 128; ++i) {
        (tC = matx::apply(ReLU{}, (matx::matmul(tA, tB) + tBias))).run(exec);
    }
    exec.sync();
    exec.start_timer();
    for (int i = 0; i < 128; ++i) {
        (tC = matx::apply(ReLU{}, (matx::matmul(tA, tB) + tBias))).run(exec);
    }
    exec.stop_timer();
    exec.sync();
    FLASHMOE_CHECK_CUDA(cudaPeekAtLastError());
    return std::make_tuple(result(), exec.get_time_ms() / 32.f);
}

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

void work() {
    cudaSetDevice(0);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    using Element = matx::matxFp16;
    Element* a = nullptr;
    Element* b = nullptr;
    using ElementC = Element;
    ElementC* c = nullptr;
    const int M = 8192;
    const int N = 2048;
    const int K = 2048;
    matx::cudaExecutor exec{stream, true};
    cudaMallocAsync(&a, M * K * sizeof(Element), stream);
    cudaMallocAsync(&b, N * K * sizeof(Element), stream);
    cudaMallocAsync(&c, M * N * sizeof(ElementC), stream);
    cudaStreamSynchronize(stream);
    auto tA = matx::make_tensor<Element>(a, {M, K});
    (tA = matx::apply(ConvFunctor<Element>{}, matx::random<float>(tA.Shape(), matx::UNIFORM))).run(exec);
    auto tB = matx::make_tensor<Element>(b, {K, N});
    (tB = matx::apply(ConvFunctor<Element>{}, matx::random<float>(tB.Shape(), matx::UNIFORM))).run(exec);
    auto tC = matx::make_tensor<Element>(c, {M, N});
    auto tBias = matx::make_tensor<ElementC>(b, {N});
    (tBias = matx::ones<ElementC>(tBias.Shape())).run(exec);
    // let's see if relu works
    // activation((a @ b) + bias)
    using ReLU = cutlass::epilogue::thread::ReLU<Element>;
    (tC = matx::apply(ReLU{}, (matx::matmul(tA, tB) + tBias))).run(exec);
    for (int i = 0; i < 128; ++i) {
        (tC = matx::apply(ReLU{}, (matx::matmul(tA, tB) + tBias))).run(exec);
    }
    (tC = matx::apply(ReLU{}, (matx::matmul(tA, tB) + tBias))).run(exec);
    FLASHMOE_CHECK_CUDA(cudaPeekAtLastError());
    cudaFreeAsync(a, stream);
    cudaFreeAsync(b, stream);
    cudaFreeAsync(c, stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}

int main() {
    work();
}