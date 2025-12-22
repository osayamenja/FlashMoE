//
// Created by osayamen on 12/13/25.
//
#include <matx.h>
#include <cutlass/epilogue/thread/activation.h>

#include "cdx_gemm.cuh"
#include "../include/flashmoe/debug.cuh"

#define RTOL 1e-3
#define ATOL 1e-4

template<typename Element>
using MXE = cuda::std::conditional_t<cuda::std::is_same_v<Element, cute::half_t>, matx::matxFp16,
        cuda::std::conditional_t<cuda::std::is_same_v<Element, cute::bfloat16_t>, matx::matxBf16, Element>>;

template<int warmup, int runs, typename Activation, typename Element, typename ElementC>
__host__ __forceinline__
auto reference(void* const& a, void* const& b,
    void* const& bias, void* const& c_ref, void* const& c_ext,
    const int& M, const int& N, const int& K, matx::cudaExecutor& exec) {
    auto* mx_a = static_cast<Element*>(a);
    auto* mx_b = static_cast<Element*>(b);
    auto* mx_bias = static_cast<ElementC*>(bias);
    auto* mx_c_ref = static_cast<ElementC*>(c_ref);
    auto* mx_c_ext = static_cast<ElementC*>(c_ext);

    auto tA = matx::make_tensor<Element>(mx_a, {M, K});
    auto tB = matx::make_tensor<Element>(mx_b, {N, K});
    auto tC = matx::make_tensor<ElementC>(mx_c_ref, {M, N});
    auto tBias = matx::make_tensor<ElementC>(mx_bias, {N});
    // ReLU((a @ b) + bias)
    (tC = matx::apply(Activation{}, (matx::matmul(tA, tB.PermuteMatrix()) + tBias))).run(exec);
    auto result = matx::make_tensor<int>({});
    exec.sync();
    auto tCx = matx::make_tensor<ElementC>(mx_c_ext, {M, N});
    matx::allclose(result, tCx, tC, RTOL, ATOL, exec);
    exec.sync();
    // warmup
    for (int i = 0; i < warmup; ++i) {
        (tC = matx::apply(Activation{}, (matx::matmul(tA, tB) + tBias))).run(exec);
    }
    exec.sync();
    exec.start_timer();
    for (int i = 0; i < runs; ++i) {
        (tC = matx::apply(Activation{}, (matx::matmul(tA, tB) + tBias))).run(exec);
    }
    exec.stop_timer();
    exec.sync();
    FLASHMOE_CHECK_CUDA(cudaPeekAtLastError());
    return std::make_tuple(result(), exec.get_time_ms() / static_cast<float>(runs));
}

template<int threads, typename Act, typename Kernel, typename Element, typename ElementC>
__host__ __forceinline__
auto gk_test(Kernel& kernel, Element* const& a, Element* const& b,
    ElementC* const& c, ElementC* const& c_ref, ElementC* const& bias,
    const int& M, const int& N, const int& K, const int& blocks, matx::cudaExecutor& exec) {
    constexpr auto runs = 64;
    constexpr auto warmup = 32;
    kernel<<<blocks, threads, 0, exec.getStream()>>>(a, b, c, bias, M, N, K);
    const auto [isCorrect, ref_time_ms] = reference<warmup, runs, Act, MXE<Element>, MXE<ElementC>>(
        a, b, bias, c_ref, c, M, N, K, exec);
    // warmup
    for (int i = 0; i < warmup; ++i) {
        kernel<<<blocks, threads, 0, exec.getStream()>>>(a, b, c, bias, M, N, K);
    }
    exec.sync();
    exec.start_timer();
    for (int i = 0; i < runs; ++i) {
        kernel<<<blocks, threads, 0, exec.getStream()>>>(a, b, c, bias, M, N, K);
    }
    exec.stop_timer();
    exec.sync();
    FLASHMOE_CHECK_CUDA(cudaPeekAtLastError());
    const auto k_time_ms = exec.get_time_ms() / static_cast<float>(runs);
    printf("%d, %d, %d, %s, %f, %f\n", M, N, K, isCorrect ? "Yes" : "No", k_time_ms, ref_time_ms);
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

template<int bM, int bN, int bK, int pipeStages, int threads, typename MMA_C, typename Element, typename ElementC>
__host__ __forceinline__
void test_driver(const int& M, const int& N, const int& K, matx::cudaExecutor& exec) {
    Element* a = nullptr;
    Element* b = nullptr;
    ElementC* c = nullptr;
    ElementC* c_ref = nullptr;
    ElementC* bias = nullptr;
    auto stream = exec.getStream();
    cudaMallocAsync(&a, M * K * sizeof(Element), stream);
    cudaMallocAsync(&b, N * K * sizeof(Element), stream);
    cudaMallocAsync(&c, M * N * sizeof(ElementC), stream);
    cudaMallocAsync(&c_ref, M * N * sizeof(ElementC), stream);
    cudaMallocAsync(&bias, N * sizeof(ElementC), stream);

    using Act = cutlass::epilogue::thread::ReLU<Element>;
    using BLAS = decltype(
            cublasdx::Size<bM, bN, bK>() +
            cublasdx::Precision<Element, Element, MMA_C>() +
            cublasdx::Type<cublasdx::type::real>() +
            cublasdx::Function<cublasdx::function::MM>() +
            cublasdx::Arrangement<cublasdx::row_major, cublasdx::row_major, cublasdx::row_major>() +
            cublasdx::Block() +
            cublasdx::MaxAlignment() +
            cublasdx::StaticBlockDim() +
            cublasdx::SM<FLASHMOE_ARCH>());
    auto kernel = v1::gk<BLAS, pipeStages, Act, Element, ElementC>;
    int bps = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kernel, threads, 0);
    const int blocks = min((M / bM) * (N / bN), bps * NUM_SMS);
    using MX = MXE<Element>;
    using MXC = MXE<ElementC>;
    using ActM = cutlass::epilogue::thread::ReLU<MX>;
    auto tA = matx::make_tensor<MX>(reinterpret_cast<MX*>(a), {M, K});
    (tA = matx::apply(ConvFunctor<MX>{}, matx::random<float>(tA.Shape(), matx::NORMAL))).run(exec);
    auto tB = matx::make_tensor<MX>(reinterpret_cast<MX*>(b), {N, K});
    (tB = matx::apply(ConvFunctor<MX>{}, matx::random<float>(tB.Shape(), matx::NORMAL))).run(exec);
    auto tBias = matx::make_tensor<MXC>(reinterpret_cast<MXC*>(bias), {N});
    (tBias = matx::apply(ConvFunctor<MXC>{}, matx::random<float>(tBias.Shape(), matx::NORMAL))).run(exec);
    gk_test<threads, ActM, Element, ElementC>(kernel, a, b, c, c_ref, bias, M, N, K, blocks, exec);

    FLASHMOE_CHECK_CUDA(cudaPeekAtLastError());
    cudaFreeAsync(a, stream);
    cudaFreeAsync(b, stream);
    cudaFreeAsync(c, stream);
    cudaFreeAsync(c_ref, stream);
    cudaFreeAsync(bias, stream);
    cudaStreamSynchronize(stream);
}

__host__ __forceinline__
void test_parser(const int argc, char** argv) {
    int MNK = 128;
    using Element = float;
    using ElementC = float;
    using MMA_C = float;
    printf("M, N, K, Correct?, Kernel_Time(ms), Matx_Time(ms)\n");
    constexpr auto bK = 8 * (4 / sizeof(ElementC));
    if (argc > 1) {
        MNK = std::stoi(argv[1]);
    }
    cudaSetDevice(0);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    matx::cudaExecutor exec{stream, true};
    for (int i = MNK; i <= 4*1024; i *= 2) {
        test_driver<128, 64, bK, 2, 128, MMA_C, Element, ElementC>(i, i, i, exec);
    }
    cudaStreamDestroy(stream);
}

int main(int argc, char** argv) {
    test_parser(argc, argv);
}