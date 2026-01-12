//
// Created by osayamen on 12/13/25.
//
// Benchmark and correctness test for the tiled GEMM underlying FlashMoE.
// We compare against the numerical library MatX which calls cuBLASLt underneath.

#include <random>

#include <cutlass/epilogue/thread/activation.h>
#include <cublasdx.hpp>

#include "common.cuh"
#include "debug.cuh"
#include "../include/flashmoe/tile.cuh"

template<typename TileGEMM, typename Activation, typename ElementC, typename Element>
__device__ __forceinline__
void gemmMainloop(void* __restrict__ const& workspace,
    const Element* __restrict__ const& a,
    const Element* __restrict__ const& b,
    ElementC* __restrict__ const& c,
    const ElementC* __restrict__ const& bias,
    const int& M, const int& N, const int& K, const int& tileIdx) {
    using BLAS = TileGEMM::BLAS;
    auto accumulator = BLAS::suggest_accumulator();
    using BM = cute::Int<cublasdx::size_of<BLAS>::m>;
    using BN = cute::Int<cublasdx::size_of<BLAS>::n>;
    const auto tileCoord = flashmoe::tile::idx2Coord(M / BM{}, N / BN{}, tileIdx);
    // gmem -> rmem: prefetch bias
    const auto gD = flashmoe::tile::getBias<BM{}, BN{}>(bias, M, N, cute::select<0, 1>(tileCoord));
    auto d_frag = cublasdx::make_fragment_like<ElementC>(accumulator.get_results());
    cublasdx::copy_fragment<cublasdx::alignment_of<BLAS>::c>(gD, d_frag, accumulator);
    // compute Tile
    constexpr TileGEMM tileMainloop{};
    tileMainloop(workspace, a, b, accumulator, M, N, K, tileCoord);
    // Epilogue
    constexpr Activation act{}; // activation function like relu, etc
    // ElementC -> accum type
    constexpr flashmoe::Converter<typename decltype(accumulator)::value_type, ElementC> loadConv{};
    // accum type -> ElementC
    constexpr flashmoe::Converter<ElementC, typename decltype(accumulator)::value_type> storeConv{};
    const auto c_frag = accumulator.get_results();
    constexpr int accum_size = cublasdx::size(c_frag);
    cute::for_each(cute::make_int_sequence<accum_size>{}, [&c_frag, &d_frag](auto i) {
        d_frag(i) = storeConv(act(c_frag(i) + loadConv(d_frag(i))));
    });
    auto gC = flashmoe::tile::getC<BM{}, BN{}, cublasdx::arrangement_of_v_c<BLAS>>(c, M, N,
        cute::select<0, 1>(tileCoord));
    // rmem -> gmem
    cublasdx::copy_fragment<cublasdx::alignment_of<BLAS>::c>(d_frag, gC, accumulator);
}

#define SC(T, v) static_cast<T>(v)
// Fused kernel for Act((a @ b) + bias)
template<typename TileGEMM, typename Activation, typename Element, typename ElementC>
requires(cublasdx::is_blas_execution_v<typename TileGEMM::BLAS>)
__launch_bounds__(TileGEMM::BLAS::max_threads_per_block, 1)
__global__ void gk(const Element* __restrict__ a, const Element* __restrict__ b,
    ElementC* __restrict__ c, const ElementC* __restrict__ bias,
    const __grid_constant__ int M, const __grid_constant__ int N, const int __grid_constant__ K) {
    using BLAS = TileGEMM::BLAS;
    constexpr int bM = cublasdx::size_of<BLAS>::m;
    constexpr int bN = cublasdx::size_of<BLAS>::n;
    constexpr int bK = cublasdx::size_of<BLAS>::k;
    const int nTiles = (M / bM) * (N / bN);
    __shared__ __align__(16) cuda::std::byte workspace[bK * TileGEMM::PipeStages::value * (bM + bN) * sizeof(Element)];
    // simple row-major tile scheduling,
    // TODO try threadblock swizzling
    for (int tileIdx = SC(int, blockIdx.x); tileIdx < nTiles; tileIdx += SC(int, gridDim.x)) {
        gemmMainloop<TileGEMM, Activation>(workspace, a, b, c, bias, M, N, K, tileIdx);
    }
}

template<int warmup, int runs, typename Activation, typename Element, typename ElementC>
__host__ __forceinline__
auto reference(void* const& a, void* const& b,
    void* const& bias, void* const& c_ref, void* const& c_ext,
    const int& M, const int& N, const int& K, const float& rtol, const float& atol,
    matx::cudaExecutor& exec) {
    auto* mx_a = static_cast<Element*>(a);
    auto* mx_b = static_cast<Element*>(b);
    auto* mx_bias = static_cast<ElementC*>(bias);
    auto* mx_c_ref = static_cast<ElementC*>(c_ref);
    auto* mx_c_ext = static_cast<ElementC*>(c_ext);

    auto tA = matx::make_tensor<Element>(mx_a, {M, K});
    auto tB = matx::make_tensor<Element>(mx_b, {N, K});
    auto tC = matx::make_tensor<ElementC>(mx_c_ref, {M, N});
    auto tCx = matx::make_tensor<ElementC>(mx_c_ext, {M, N});
    auto tBias = matx::make_tensor<ElementC>(mx_bias, {N});
    // Act((a @ b) + bias)
    (tC = matx::apply(Activation{}, (matx::matmul(tA, tB.PermuteMatrix()) + tBias))).run(exec);
    exec.sync();
    matx::cudaExecutor exec1{exec.getStream()};
    auto num_matches = matx::make_tensor<long int>({});
    (num_matches = matx::sum(matx::isclose(tCx, tC, rtol, atol))).run(exec1);
    exec1.sync();
    // calculate error percentage
    const auto ep =  (1.0 - (static_cast<double>(num_matches()) / static_cast<double>(M*N))) * 100;
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
    CHECK_CUDA(cudaPeekAtLastError());
    return std::make_tuple(ep, exec.get_time_ms() / static_cast<float>(runs));
}

template<int threads, typename Act, typename Kernel, typename Element, typename ElementC>
__host__ __forceinline__
auto gk_run(Kernel& kernel, Element* const& a, Element* const& b,
    ElementC* const& c, ElementC* const& c_ref, ElementC* const& bias,
    const int& M, const int& N, const int& K, const int& blocks,
    const float& rtol, const float& atol, matx::cudaExecutor& exec) {
    constexpr auto runs = 128;
    constexpr auto warmup = 32;
    kernel<<<blocks, threads, 0, exec.getStream()>>>(a, b, c, bias, M, N, K);
    const auto [error_pct, ref_time_ms] = reference<warmup, runs, Act, MXE<Element>, MXE<ElementC>>(
        a, b, bias, c_ref, c, M, N, K, rtol, atol, exec);
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
    CHECK_CUDA(cudaPeekAtLastError());
    const auto k_time_ms = exec.get_time_ms() / static_cast<float>(runs);
    return std::make_tuple(k_time_ms, error_pct, ref_time_ms);
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

template<int bM, int bN, int bK, int pipeStages, typename MMA_C, typename Element, typename ElementC>
__host__ __forceinline__
void driver(const int& M, const int& N, const int& K, const float& rtol, const float& atol, matx::cudaExecutor& exec) {
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

    using MX = MXE<Element>;
    using MXC = MXE<ElementC>;
    using Act = cutlass::epilogue::thread::ReLU<MMA_C>;
    using ActM = cutlass::epilogue::thread::ReLU<MX>;
    constexpr int threads = flashmoe::tile::suggest_thread_count<bM, bN, bK, FLASHMOE_ARCH, Element, MMA_C>();
    using TileGEMM = flashmoe::tile::CollectiveMainloop<
            bM, bN, bK, FLASHMOE_ARCH, Element, MMA_C, threads, pipeStages
    >;
    auto kernel = gk<TileGEMM, Act, Element, ElementC>;
    int bps = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kernel, threads, 0);
    const int blocks = min((M / bM) * (N / bN), bps * NUM_SMS);
    auto tA = matx::make_tensor<MX>(reinterpret_cast<MX*>(a), {M, K});
    std::random_device rd; // random seed provider
    (tA = matx::apply(ConvFunctor<MX>{}, matx::random<float>(tA.Shape(), matx::UNIFORM,rd()))).run(exec);
    auto tB = matx::make_tensor<MX>(reinterpret_cast<MX*>(b), {N, K});
    (tB = matx::apply(ConvFunctor<MX>{}, matx::random<float>(tB.Shape(), matx::UNIFORM, rd()))).run(exec);
    auto tBias = matx::make_tensor<MXC>(reinterpret_cast<MXC*>(bias), {N});
    (tBias = matx::apply(ConvFunctor<MXC>{}, matx::random<float>(tBias.Shape(), matx::UNIFORM, rd()))).run(exec);
    const auto [k_ms, e_p, r_ms] = gk_run<threads, ActM>(kernel, a, b, c, c_ref, bias, M, N, K, blocks, rtol, atol, exec);

    printf("%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %.1e, %.1e, %lf, %f, %f\n",
        M, N, K, bM, bN, bK, pipeStages, threads, bps, NUM_SMS, blocks, rtol, atol, e_p, k_ms, r_ms);

    CHECK_CUDA(cudaPeekAtLastError());
    cudaFreeAsync(a, stream);
    cudaFreeAsync(b, stream);
    cudaFreeAsync(c, stream);
    cudaFreeAsync(c_ref, stream);
    cudaFreeAsync(bias, stream);
    cudaStreamSynchronize(stream);
}

__host__ __forceinline__
void kickStart(const int argc, char** argv) {
    int MNK = 2;
    int MNK_max = 8192;
    float rtol = 2e-2f;
    float atol = 2e-3f;
    using Element = __half;
    using ElementC = Element;
    using MMA_C = float;
    printf("M, N, K, bM, bN, bK, pipeStages, threads, blocks/SM, SMs, blocks, rtol, atol, error(%%), Kernel_Time(ms), "
           "Matx_Time(ms)\n");
    if (argc > 1) {
        MNK = std::stoi(argv[1]);
    }
    if (argc > 2) {
        MNK_max = std::stoi(argv[2]);
    }
    if (argc > 3) {
        rtol = std::stof(argv[3]);
    }
    if (argc > 4) {
        atol = std::stof(argv[4]);
    }
    cudaSetDevice(0);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    matx::cudaExecutor exec{stream, true};
    for (int i = MNK; i <= MNK_max; i *= 2) {
        switch (i) {
        case 2:
            driver<2, 2, 2, 1, MMA_C, Element, ElementC>(i, i, i, rtol, atol, exec);
            break;
        case 4:
            driver<4, 4, 4, 1, MMA_C, Element, ElementC>(i, i, i, rtol, atol, exec);
            break;
        case 8:
            driver<8, 8, 8, 1, MMA_C, Element, ElementC>(i, i, i, rtol, atol, exec);
            break;
        case 16:
            driver<16, 16, 16, 1, MMA_C, Element, ElementC>(i, i, i, rtol, atol, exec);
            break;
        case 32:
            driver<32, 32, 32, 1, MMA_C, Element, ElementC>(i, i, i, rtol, atol, exec);
            break;
        case 64:
            driver<64, 64, 64, 1, MMA_C, Element, ElementC>(i, i, i, rtol, atol, exec);
            break;
        default:
            {
                if (i >= 128 && i <= 2048) {
                    driver<128, 64, 32, 2, MMA_C, Element, ElementC>(i, i, i, rtol, atol,exec);
                }
                else if (i > 2048) {
                    driver<128, 64 * (4 / sizeof(Element)), 32, 2, MMA_C, Element, ElementC>(i, i, i, rtol, atol,exec);
                }
            }

        }
    }
    cudaStreamDestroy(stream);
}

int main(const int argc, char** argv) {
    kickStart(argc, argv);
}