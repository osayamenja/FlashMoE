//
// Created by osayamen on 1/19/26.
//
// correctness test for the tiled GEMM underlying FlashMoE.

#include <random>
#include <stdexcept>
#include <string>
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
    using AccumType = decltype(accumulator)::value_type;
    // ElementC -> accum type
    constexpr flashmoe::Converter<AccumType, ElementC> loadConv{};
    // accum type -> ElementC
    constexpr flashmoe::Converter<ElementC, AccumType> storeConv{};
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
    const int nTiles = (M / bM) * (N / bN);
    extern __shared__ __align__(TileGEMM::GeneralAlignment::value) cuda::std::byte gemmWorkspace[];
    // simple row-major tile scheduling,
    // TODO try threadblock swizzling
    for (int tileIdx = SC(int, blockIdx.x); tileIdx < nTiles; tileIdx += SC(int, gridDim.x)) {
        gemmMainloop<TileGEMM, Activation>(gemmWorkspace, a, b, c, bias, M, N, K, tileIdx);
    }
}

template<int warmup, int runs, typename AccumType, typename Activation, typename Element, typename ElementC>
__host__ __forceinline__
auto reference(void* const& a, void* const& b,
    void* const& bias, void* const& bias_interim, void* const& c_ref,
    void* const& c_interim, void* const& c_ext,
    const int& M, const int& N, const int& K, const float& rtol, const float& atol,
    matx::cudaExecutor& exec) {
    auto* mx_a = static_cast<Element*>(a);
    auto* mx_b = static_cast<Element*>(b);
    auto* mx_bias = static_cast<ElementC*>(bias);
    auto* mx_c_ref = static_cast<ElementC*>(c_ref);
    auto* mx_c_interim = static_cast<AccumType*>(c_interim);
    auto* mx_bias_interim = static_cast<AccumType*>(bias_interim);
    auto* mx_c_ext = static_cast<ElementC*>(c_ext);

    auto tA = matx::make_tensor<Element>(mx_a, {M, K});
    auto tB = matx::make_tensor<Element>(mx_b, {N, K});
    auto tC = matx::make_tensor<ElementC>(mx_c_ref, {M, N});
    auto tC_interim = matx::make_tensor<AccumType>(mx_c_interim, {M, N});
    auto tCx = matx::make_tensor<ElementC>(mx_c_ext, {M, N});
    auto tBias = matx::make_tensor<ElementC>(mx_bias, {N});
    auto tBias_interim = matx::make_tensor<AccumType>(mx_bias_interim, {N});
    // Act((a @ b) + bias), b is in row-major -> as PyTorch does
    (tC_interim = matx::matmul(tA, tB.PermuteMatrix())).run(exec);
    // convert bias to AccumType
    (tBias_interim = matx::apply(Converter<AccumType, ElementC>{}, tBias)).run(exec);
    // add in AccumType -> apply activation -> convert back to Element
    (tC = matx::apply(Converter<ElementC, AccumType>{}, matx::apply(Activation{}, (tC_interim + tBias_interim)))).run(exec);
    exec.sync();
    matx::cudaExecutor exec1{exec.getStream()};
    auto num_matches = matx::make_tensor<long int>({});
    (num_matches = matx::sum(matx::isclose(tCx, tC, rtol, atol))).run(exec1);
    exec1.sync();
    // calculate error percentage
    const auto ep =  (1.0 - (static_cast<double>(num_matches()) / static_cast<double>(M*N))) * 100;
    // warmup
    for (int i = 0; i < warmup; ++i) {
        (tC = matx::apply(Activation{}, (matx::matmul(tA, tB.PermuteMatrix()) + tBias))).run(exec);
    }
    exec.sync();
    exec.start_timer();
    for (int i = 0; i < runs; ++i) {
        (tC = matx::apply(Activation{}, (matx::matmul(tA, tB.PermuteMatrix()) + tBias))).run(exec);
    }
    exec.stop_timer();
    exec.sync();
    CHECK_CUDA(cudaPeekAtLastError());
    return std::make_tuple(ep, exec.get_time_ms() / static_cast<float>(runs));
}

template<typename TileGEMM, typename Activation, int threads, int sharedSize, typename ActM, typename AccumType, typename Element, typename ElementC>
__host__ __forceinline__
auto gk_run(Element* const& a, Element* const& b,
    ElementC* const& c, ElementC* const& c_ref, ElementC* const& bias,
    AccumType* const& c_interim, AccumType* const& bias_interim,
    const int& M, const int& N, const int& K, const int& blocks,
    const float& rtol, const float& atol, matx::cudaExecutor& exec) {
    constexpr auto runs = 128;
    constexpr auto warmup = 32;
    gk<TileGEMM, Activation><<<blocks, threads, sharedSize, exec.getStream()>>>(a, b, c, bias, M, N, K);
    const auto [error_pct, ref_time_ms] = reference<warmup, runs, AccumType, ActM, MXE<Element>, MXE<ElementC>>(
        a, b, bias, bias_interim, c_ref, c_interim, c, M, N, K, rtol, atol, exec);
    // warmup
    for (int i = 0; i < warmup; ++i) {
        gk<TileGEMM, Activation><<<blocks, threads, sharedSize, exec.getStream()>>>(a, b, c, bias, M, N, K);
    }
    exec.sync();
    exec.start_timer();
    for (int i = 0; i < runs; ++i) {
        gk<TileGEMM, Activation><<<blocks, threads, sharedSize, exec.getStream()>>>(a, b, c, bias, M, N, K);
    }
    exec.stop_timer();
    exec.sync();
    CHECK_CUDA(cudaPeekAtLastError());
    const auto k_time_ms = exec.get_time_ms() / static_cast<float>(runs);
    return std::make_tuple(k_time_ms, error_pct, ref_time_ms);
}

template<int Arch, int bM, int bN, int bK, int pipeStages, typename AccumType, typename Element, typename ElementC>
__host__ __forceinline__
void driver(const int& M, const int& N, const int& K, const float& rtol, const float& atol, matx::cudaExecutor& exec) {
    Element* a = nullptr;
    Element* b = nullptr;
    ElementC* c = nullptr;
    ElementC* c_ref = nullptr;
    AccumType* c_interim = nullptr;
    AccumType* bias_interim = nullptr;
    ElementC* bias = nullptr;
    auto stream = exec.getStream();
    cudaMallocAsync(&a, M * K * sizeof(Element), stream);
    cudaMallocAsync(&b, N * K * sizeof(Element), stream);
    cudaMallocAsync(&c, M * N * sizeof(ElementC), stream);
    cudaMallocAsync(&c_ref, M * N * sizeof(ElementC), stream);
    cudaMallocAsync(&c_interim, M * N * sizeof(AccumType), stream);
    cudaMallocAsync(&bias_interim, N * sizeof(AccumType), stream);
    cudaMallocAsync(&bias, N * sizeof(ElementC), stream);

    using Act = cutlass::epilogue::thread::ReLU<AccumType>;
    using ActM = cutlass::epilogue::thread::ReLU<MXE<AccumType>>;
    constexpr int threads = flashmoe::tile::suggest_thread_count<bM, bN, bK, Arch, Element, AccumType>();
    using TileGEMM = flashmoe::tile::CollectiveMainloop<
            bM, bN, bK, Arch, Element, AccumType, threads, pipeStages
    >;
    auto kernel = gk<TileGEMM, Act, Element, ElementC>;
    int bps = 0;
    constexpr auto sharedSize = bK * pipeStages * (bM + bN) * sizeof(Element);
    CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedSize));
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kernel, threads, sharedSize));
    const int blocks = min((M / bM) * (N / bN), bps * NUM_SMS);
    std::random_device rd; // random seed provider
    constexpr auto min_v = -1.f;
    constexpr auto max_v = 1.f;
    randUniform<Arch>(a, M * K, rd(), min_v, max_v, exec.getStream());
    randUniform<Arch>(b, N * K, rd(), min_v, max_v, exec.getStream());
    randUniform<Arch>(bias, N, rd(), min_v, max_v, exec.getStream());
    const auto [k_ms, e_p, r_ms] = gk_run<TileGEMM, Act, threads, sharedSize, ActM>(a, b, c, c_ref, bias, c_interim,
        bias_interim, M, N, K, blocks, rtol, atol, exec);

    printf("%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %.1e, %.1e, %lf, %f, %f\n",
        M, N, K, bM, bN, bK, pipeStages, threads, bps, NUM_SMS, blocks, rtol, atol, e_p, k_ms, r_ms);

    CHECK_CUDA(cudaPeekAtLastError());
    cudaFreeAsync(a, stream);
    cudaFreeAsync(b, stream);
    cudaFreeAsync(c, stream);
    cudaFreeAsync(bias, stream);
    cudaFreeAsync(c_ref, stream);
    cudaFreeAsync(c_interim, stream);
    cudaFreeAsync(bias_interim, stream);
    cudaStreamSynchronize(stream);
}

// .gemmMNK <M> <N> <K> <rtol> <atol>
__host__ __forceinline__
void kickStart(const int argc, char** argv) {
    int M = 2048;
    int N = 14 * 1024;
    int K = 4096;
    float rtol = 2e-2f;
    float atol = 2e-3f;
    using Element = __half;
    using ElementC = Element;
    using AccumType = float;
    printf("M, N, K, bM, bN, bK, pipeStages, threads, blocks/SM, SMs, blocks, rtol, atol, error(%%), Kernel_Time(ms), "
           "Matx_Time(ms)\n");
    if (argc > 1) {
        M = std::stoi(argv[1]);
    }
    if (argc > 2) {
        N = std::stoi(argv[2]);
    }
    if (argc > 3) {
        K = std::stoi(argv[3]);
    }
    if (argc > 4) {
        rtol = std::stof(argv[4]);
    }
    if (argc > 5) {
        atol = std::stof(argv[5]);
    }
    // below minimizes instantiated templates
    constexpr int bK = sizeof(Element) == 8 ? 32 : 64;
    constexpr auto arch = FLASHMOE_ARCH;
    constexpr int pS = arch >= 800 ? 2 : 1;
    constexpr int bN = sizeof(Element) == 8 ? 64 : 64 * (4 / sizeof(Element));
    if (N < bN || N % bN != 0) {
        const auto errmsg = std::string("N >= and a multiple of ").append(std::to_string(bN));
        throw std::runtime_error(errmsg);
    }
    if (K < bK || K % bK != 0 ) {
        const auto errmsg = std::string("K >= and a multiple of ").append(std::to_string(bK));
        throw std::runtime_error(errmsg);
    }
    cudaSetDevice(0);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    matx::cudaExecutor exec{stream, true};
    switch (M) {
    case 1:
        if (K > bK) {
            driver<arch, 1, bN, bK, pS, AccumType, Element, ElementC>(M, N, K, rtol, atol, exec);
        }
        else {
            driver<arch, 1, bN, bK, 1, AccumType, Element, ElementC>(M, N, K, rtol, atol, exec);
        }
        break;
    case 2:
        if (K > bK) {
            driver<arch, 2, bN, bK, pS, AccumType, Element, ElementC>(M, N, K, rtol, atol, exec);
        }
        else {
            driver<arch, 2, bN, bK, 1, AccumType, Element, ElementC>(M, N, K, rtol, atol, exec);
        }
        break;
    case 4:
        if (K > bK) {
            driver<arch, 4, bN, bK, pS, AccumType, Element, ElementC>(M, N, K, rtol, atol, exec);
        }
        else {
            driver<arch, 4, bN, bK, 1, AccumType, Element, ElementC>(M, N, K, rtol, atol, exec);
        }
        break;
    case 8:
        if (K > bK) {
            driver<arch, 8, bN, bK, pS, AccumType, Element, ElementC>(M, N, K, rtol, atol, exec);
        }
        else {
            driver<arch, 8, bN, bK, 1, AccumType, Element, ElementC>(M, N, K, rtol, atol, exec);
        }
        break;
    case 16:
        if (K > bK) {
            driver<arch, 16, bN, bK, pS, AccumType, Element, ElementC>(M, N, K, rtol, atol, exec);
        }
        else {
            driver<arch, 16, bN, bK, 1, AccumType, Element, ElementC>(M, N, K, rtol, atol, exec);
        }
        break;
    case 32:
        if (K > bK) {
            driver<arch, 32, bN, bK, pS, AccumType, Element, ElementC>(M, N, K, rtol, atol, exec);
        }
        else {
            driver<arch, 32, bN, bK, 1, AccumType, Element, ElementC>(M, N, K, rtol, atol, exec);
        }
        break;
    case 64:
        if (K > bK) {
            driver<arch, 64, bN, bK, pS, AccumType, Element, ElementC>(M, N, K, rtol, atol, exec);
        }
        else {
            driver<arch, 64, bN, bK, 1, AccumType, Element, ElementC>(M, N, K, rtol, atol, exec);
        }
        break;
    default:
        if (M >= 128 && M % 128 == 0) {
            if (K > bK) {
                driver<arch, 128, bN, bK, pS, AccumType, Element, ElementC>(M, N, K, rtol, atol,exec);
            }
            else {
                driver<arch, 128, bN, bK, 1, AccumType, Element, ElementC>(M, N, K, rtol, atol,exec);
            }
        }
    }
    cudaStreamDestroy(stream);
}

int main(const int argc, char** argv) {
    kickStart(argc, argv);
}