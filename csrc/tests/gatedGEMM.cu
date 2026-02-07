//
// Created by osayamen on 2/2/26.
//
// unit test and benchmark for gated GEMM of gated MLP

#include <random>

#include <cutlass/epilogue/thread/activation.h>
#include <cublasdx.hpp>

#include "common.cuh"
#include "debug.cuh"
#include "../include/flashmoe/tile.cuh"

template<typename TileGEMM, typename Activation, typename ElementC, typename Element>
__device__ __forceinline__
void gemmMainloop(cuda::std::byte* __restrict__ const& workspace,
    const Element* __restrict__ const& a,
    const Element* __restrict__ const& b,
    const Element* __restrict__ const& bV,
    ElementC* __restrict__ const& c,
    const ElementC* __restrict__ const& bias,
    const ElementC* __restrict__ const& biasV,
    const typename TileGEMM::AccumType& swishAlpha,
    const typename TileGEMM::AccumType& swishBeta,
    const int& M, const int& N, const int& K, const int& tileIdx) {
    auto* __restrict__ gateCache = workspace + cutlass::round_up(cute::max(TileGEMM::SharedSizeC::value,
        TileGEMM::SharedSizeAB::value), TileGEMM::GeneralAlignment::value);
    using BLAS = TileGEMM::BLAS;
    auto accumulator = BLAS::suggest_accumulator();
    using BM = cute::Int<cublasdx::size_of<BLAS>::m>;
    using BN = cute::Int<cublasdx::size_of<BLAS>::n>;
    const auto tileCoord = flashmoe::tile::idx2Coord(M / BM{}, N / BN{}, tileIdx);
    // compute gate Tile
    constexpr TileGEMM tileMainloop{};
    tileMainloop(workspace, a, b, accumulator, M, N, K, tileCoord);
    __syncthreads();
    // gmem -> rmem: load bias
    const auto gD = flashmoe::tile::getBias<BM{}, BN{}>(bias, M, N, cute::select<0, 1>(tileCoord));
    auto d_frag = cublasdx::make_fragment_like<ElementC>(accumulator.get_results());
    cublasdx::copy_fragment<cublasdx::alignment_of<BLAS>::c>(gD, d_frag, accumulator);
    // Epilogue
    constexpr Activation act{}; // activation function like gelu or swish
    using AccumType = decltype(accumulator)::value_type;
    // ElementC -> accum type
    constexpr flashmoe::Converter<AccumType, ElementC> loadConv{};
    // accum type -> ElementC
    constexpr flashmoe::Converter<ElementC, AccumType> storeConv{};
    const auto c_frag = accumulator.get_results();
    constexpr int accum_size = cublasdx::size(c_frag);
    #pragma unroll
    for (int i = 0 ; i < accum_size; ++i) {
        const auto g = (c_frag(i) + loadConv(d_frag(i))) * swishBeta;
        d_frag(i) = storeConv(swishAlpha * act(g));
    }
    // rmem -> smem, cache gate results
    // holding in registers otherwise would blow up pressure
    auto sGate = cublasdx::make_tensor(reinterpret_cast<ElementC*>(gateCache), BLAS::suggest_layout_smem_c());
    cublasdx::copy_fragment<cublasdx::alignment_of<BLAS>::c>(d_frag, sGate, accumulator);
    // now, compute v tile
    tileMainloop(workspace, a, bV, accumulator, M, N, K, tileCoord);
    auto cv_frag = accumulator.get_results();
    const auto gV = flashmoe::tile::getBias<BM{}, BN{}>(biasV, M, N, cute::select<0, 1>(tileCoord));
    cublasdx::copy_fragment<cublasdx::alignment_of<BLAS>::c>(gV, d_frag, accumulator);
    #pragma unroll
    for (int i = 0; i < accum_size; ++i) {
        // x = (a @ bV) + biasV
        cv_frag(i) = cv_frag(i) + loadConv(d_frag(i));
    }
    // smem -> rmem, load g
    __syncthreads();
    cublasdx::copy_fragment<cublasdx::alignment_of<BLAS>::c>(sGate, d_frag, accumulator);
    #pragma unroll
    for (int i = 0; i < accum_size; ++i) {
        // y = x * (act(a @ b))
        d_frag(i) = storeConv(cv_frag(i) * loadConv(d_frag(i)));
    }
    auto gC = flashmoe::tile::getC<BM{}, BN{}, cublasdx::arrangement_of_v_c<BLAS>>(c, M, N,
        cute::select<0, 1>(tileCoord));
    // rmem -> smem
    auto sC = cublasdx::make_tensor(reinterpret_cast<ElementC*>(workspace), BLAS::suggest_layout_smem_c());
    __syncthreads();
    cublasdx::copy_fragment<cublasdx::alignment_of<BLAS>::c>(d_frag, sC, accumulator);
    __syncthreads();
    // smem -> gmem
    cublasdx::copy<BLAS, cublasdx::alignment_of<BLAS>::c>(sC, gC);
}

#define SC(T, v) static_cast<T>(v)
// Fused kernel for act(beta * ((a @ b) + bias)) x ((a @ bV) + biasV)
template<typename TileGEMM, typename Activation, typename Element, typename ElementC>
requires(cublasdx::is_blas_execution_v<typename TileGEMM::BLAS>)
__launch_bounds__(TileGEMM::BLAS::max_threads_per_block, 1)
__global__ void gk(const Element* __restrict__ a,
    const Element* __restrict__ b, const Element* __restrict__ bV,
    ElementC* __restrict__ c,
    const ElementC* __restrict__ bias, const ElementC* __restrict__ biasV,
    const __grid_constant__ typename TileGEMM::AccumType swishAlpha,
    const __grid_constant__ typename TileGEMM::AccumType swishBeta,
    const __grid_constant__ int M,
    const __grid_constant__ int N,
    const int __grid_constant__ K) {
    using BLAS = TileGEMM::BLAS;
    constexpr int bM = cublasdx::size_of<BLAS>::m;
    constexpr int bN = cublasdx::size_of<BLAS>::n;
    const int nTiles = (M / bM) * (N / bN);
    extern __shared__ __align__(TileGEMM::GeneralAlignment::value) cuda::std::byte gemmWorkspace[];
    // simple row-major tile scheduling,
    // TODO try threadblock swizzling
    for (int tileIdx = SC(int, blockIdx.x); tileIdx < nTiles; tileIdx += SC(int, gridDim.x)) {
        gemmMainloop<TileGEMM, Activation>(gemmWorkspace, a, b, bV, c, bias, biasV, swishAlpha, swishBeta, M, N, K, tileIdx);
    }
}

template<int warmup, int runs, typename AccumType, typename Activation, typename Element, typename ElementC>
__host__ __forceinline__
auto reference(void* const& a, void* const& b, void* const& bV,
    void* const& bias, void* const& biasV, void* const& bias_interim,
    void* const& c_ref, void* const& c_interim, void* const& c_final,
    void* const& c_ext, const AccumType& swishAlpha, const AccumType& swishBeta,
    const int& M, const int& N, const int& K, const float& rtol, const float& atol,
    matx::cudaExecutor& exec) {
    auto* mx_a = static_cast<Element*>(a);
    auto* mx_b = static_cast<Element*>(b);
    auto* mx_bv = static_cast<Element*>(bV);
    auto* mx_bias = static_cast<ElementC*>(bias);
    auto* mx_bias_interim = static_cast<ElementC*>(bias_interim);
    auto* mx_bias_v = static_cast<ElementC*>(biasV);
    auto* mx_c_ref = static_cast<ElementC*>(c_ref);
    auto* mx_c_final = static_cast<ElementC*>(c_final);
    auto* mx_c_interim = static_cast<AccumType*>(c_interim);
    auto* mx_c_ext = static_cast<ElementC*>(c_ext);

    auto tA = matx::make_tensor<Element>(mx_a, {M, K});
    auto tB = matx::make_tensor<Element>(mx_b, {N, K});
    auto tBV = matx::make_tensor<Element>(mx_bv, {N, K});
    auto tC = matx::make_tensor<ElementC>(mx_c_ref, {M, N});
    auto tC_interim = matx::make_tensor<AccumType>(mx_c_interim, {M, N});
    auto tC_final = matx::make_tensor<ElementC>(mx_c_final, {M, N});
    auto tCx = matx::make_tensor<ElementC>(mx_c_ext, {M, N});
    auto tBias = matx::make_tensor<ElementC>(mx_bias, {N});
    auto tBias_interim = matx::make_tensor<ElementC>(mx_bias_interim, {N});
    auto tBiasV = matx::make_tensor<ElementC>(mx_bias_v, {N});

    auto swiOut = matx::make_tensor<AccumType>({M, N});
    auto tC_stash = matx::make_tensor<AccumType>({M, N});

    // act(beta * ((a @ b) + bias)) x ((a @ bV) + biasV), b is in row-major -> as PyTorch does
    (tC_interim = matx::matmul(tA, tB.PermuteMatrix())).run(exec);
    (tBias_interim = matx::apply(Converter<AccumType, ElementC>{}, tBias)).run(exec);
    (swiOut = matx::apply(Activation{}, swishBeta * (tC_interim + tBias_interim))).run(exec);
    (tC = matx::apply(Converter<ElementC, AccumType>{}, swishAlpha * swiOut)).run(exec);
    // do v GEMM
    (tC_interim = matx::matmul(tA, tBV.PermuteMatrix())).run(exec);
    (tBias_interim = matx::apply(Converter<AccumType, ElementC>{}, tBiasV)).run(exec);
    (tC_stash = matx::apply(Converter<AccumType, Element>{}, tC)).run(exec);
    (tC_final = matx::apply(Converter<ElementC, AccumType>{}, (tC_interim + tBias_interim) * tC_stash)).run(exec);
    exec.sync();
    matx::cudaExecutor exec1{exec.getStream()};
    auto num_matches = matx::make_tensor<long int>({});
    (num_matches = matx::sum(matx::isclose(tCx, tC_final, rtol, atol))).run(exec1);
    exec1.sync();
    // calculate error percentage
    const auto ep =  (1.0 - (static_cast<double>(num_matches()) / static_cast<double>(M*N))) * 100;
    // warmup
    for (int i = 0; i < warmup; ++i) {
        (tC = swishAlpha * matx::apply(Activation{}, swishBeta * (matx::matmul(tA, tB.PermuteMatrix()) + tBias)) *
            (matx::matmul(tA, tBV.PermuteMatrix()) + tBiasV)).run(exec);
    }
    exec.sync();
    exec.start_timer();
    for (int i = 0; i < runs; ++i) {
        (tC = swishAlpha * matx::apply(Activation{}, swishBeta * (matx::matmul(tA, tB.PermuteMatrix()) + tBias)) *
            (matx::matmul(tA, tBV.PermuteMatrix()) + tBiasV)).run(exec);
    }
    exec.stop_timer();
    exec.sync();
    CHECK_CUDA(cudaPeekAtLastError());
    return std::make_tuple(ep, exec.get_time_ms() / static_cast<float>(runs));
}

template<typename TileGEMM, typename Activation, int threads, int sharedSize, typename ActM,
typename AccumType, typename Element, typename ElementC>
__host__ __forceinline__
auto gk_run(Element* const& a, Element* const& b, Element* const& bV,
    ElementC* const& c, ElementC* const& c_ref, ElementC* const& bias, ElementC* const& biasV,
    AccumType* const& bias_interim,
    AccumType* const& c_interim, ElementC* const& c_final, const AccumType& swishAlpha, const AccumType& swishBeta,
    const int& M, const int& N, const int& K, const int& blocks,
    const float& rtol, const float& atol, matx::cudaExecutor& exec) {
    constexpr auto runs = 128;
    constexpr auto warmup = 128;
    gk<TileGEMM, Activation><<<blocks, threads, sharedSize, exec.getStream()>>>(a, b, bV, c, bias, biasV,  swishAlpha, swishBeta, M, N, K);
    const auto [error_pct, ref_time_ms] = reference<warmup, runs, AccumType, ActM, MXE<Element>, MXE<ElementC>>(
        a, b, bV, bias, biasV, bias_interim, c_ref, c_interim, c_final,c, swishAlpha, swishBeta,
        M, N, K, rtol, atol, exec);
    // warmup
    for (int i = 0; i < warmup; ++i) {
        gk<TileGEMM, Activation><<<blocks, threads, sharedSize, exec.getStream()>>>(a, b, bV, c, bias, biasV,  swishAlpha, swishBeta, M, N, K);
    }
    exec.sync();
    exec.start_timer();
    for (int i = 0; i < runs; ++i) {
        gk<TileGEMM, Activation><<<blocks, threads, sharedSize, exec.getStream()>>>(a, b, bV, c, bias, biasV, swishAlpha, swishBeta, M, N, K);
    }
    exec.stop_timer();
    exec.sync();
    CHECK_CUDA(cudaPeekAtLastError());
    const auto k_time_ms = exec.get_time_ms() / static_cast<float>(runs);
    return std::make_tuple(k_time_ms, error_pct, ref_time_ms);
}

template<int bM, int bN, int bK, int pipeStages, typename AccumType, typename Element, typename ElementC>
__host__ __forceinline__
void driver(const int& M, const int& N, const int& K, const float& rtol, const float& atol, matx::cudaExecutor& exec) {
    Element* a = nullptr;
    Element* b = nullptr;
    Element* bV = nullptr;
    ElementC* c = nullptr;
    ElementC* c_ref = nullptr;
    AccumType* c_interim = nullptr;
    ElementC* c_final = nullptr;
    ElementC* bias = nullptr;
    ElementC* biasV = nullptr;
    AccumType* bias_interim = nullptr;
    auto stream = exec.getStream();
    CHECK_CUDA(cudaMallocAsync(&a, sizeof(Element)* M * static_cast<size_t>(K), stream));
    CHECK_CUDA(cudaMallocAsync(&b, sizeof(Element) * N * K, stream));
    CHECK_CUDA(cudaMallocAsync(&bV, sizeof(Element) * N * K , stream));
    CHECK_CUDA(cudaMallocAsync(&c, sizeof(ElementC) * M * N, stream));
    CHECK_CUDA(cudaMallocAsync(&c_ref, sizeof(ElementC)* M * N, stream));
    CHECK_CUDA(cudaMallocAsync(&c_final, sizeof(ElementC) * M * N, stream));
    CHECK_CUDA(cudaMallocAsync(&c_interim, sizeof(AccumType) * M * N, stream));
    CHECK_CUDA(cudaMallocAsync(&bias, N * sizeof(ElementC), stream));
    CHECK_CUDA(cudaMallocAsync(&biasV, N * sizeof(ElementC), stream));
    CHECK_CUDA(cudaMallocAsync(&bias_interim, N * sizeof(AccumType), stream));

    using Act = cutlass::epilogue::thread::SiLu<AccumType>;
    using ActM = cutlass::epilogue::thread::SiLu<MXE<AccumType>>;
    constexpr int threads = flashmoe::tile::suggest_thread_count<bM, bN, bK, FLASHMOE_ARCH, Element, AccumType>();
    using TileGEMM = flashmoe::tile::CollectiveMainloop<
            bM, bN, bK, FLASHMOE_ARCH, Element, AccumType, threads, pipeStages
    >;
    auto kernel = gk<TileGEMM, Act, Element, ElementC>;
    int bps = 0;
    constexpr auto totalSharedSize = cutlass::round_up(cute::max(TileGEMM::SharedSizeC::value,
        TileGEMM::SharedSizeAB::value), TileGEMM::GeneralAlignment::value) + TileGEMM::SharedSizeC::value;
    CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, totalSharedSize));
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kernel, threads, totalSharedSize));
    int maxSharedMemory = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&maxSharedMemory,cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    if (totalSharedSize > maxSharedMemory) {
        const auto errmsg = std::string("Required shared memory ").append(std::to_string(totalSharedSize))
        .append(" exceeds hardware limits: ").append(std::to_string(maxSharedMemory)).append(" Reduce tile shapes or input sizes.");
        throw std::runtime_error(errmsg);
    }
    const int blocks = min((M / bM) * (N / bN), bps * NUM_SMS);
    std::random_device rd; // random seed provider
    constexpr auto min_v = -1.f;
    constexpr auto max_v = 1.f;
    randUniform<FLASHMOE_ARCH>(a, static_cast<size_t>(M) * K, rd(), min_v, max_v, exec.getStream());
    randUniform<FLASHMOE_ARCH>(b, static_cast<size_t>(N) * K, rd(), min_v, max_v, exec.getStream());
    randUniform<FLASHMOE_ARCH>(bV, static_cast<size_t>(N) * K, rd(), min_v, max_v, exec.getStream());
    randUniform<FLASHMOE_ARCH>(bias, N, rd(), min_v, max_v, exec.getStream());
    randUniform<FLASHMOE_ARCH>(biasV, N, rd(), min_v, max_v, exec.getStream());
    const auto swishAlpha = static_cast<AccumType>(random_float(min_v, max_v));
    const auto swishBeta = static_cast<AccumType>(random_float(min_v, max_v));
    const auto [k_ms, e_p, r_ms] = gk_run<TileGEMM, Act, threads, totalSharedSize, ActM>(a, b, bV, c, c_ref, bias, biasV,
        bias_interim, c_interim, c_final,swishAlpha, swishBeta, M, N, K, blocks, rtol, atol, exec);

    printf("%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %.1e, %.1e, %lf, %f, %f\n",
        M, N, K, bM, bN, bK, pipeStages, threads, bps, NUM_SMS, blocks, rtol, atol, e_p, k_ms, r_ms);

    CHECK_CUDA(cudaPeekAtLastError());
    cudaFreeAsync(a, stream);
    cudaFreeAsync(b, stream);
    cudaFreeAsync(bV, stream);
    cudaFreeAsync(c, stream);
    cudaFreeAsync(c_ref, stream);
    cudaFreeAsync(c_final, stream);
    cudaFreeAsync(bias, stream);
    cudaFreeAsync(biasV, stream);
    cudaFreeAsync(c_interim, stream);
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
    using AccumType = float;
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
            driver<2, 2, 2, 1, AccumType, Element, ElementC>(i, i, i, rtol, atol, exec);
            break;
        case 4:
            driver<4, 4, 4, 1, AccumType, Element, ElementC>(i, i, i, rtol, atol, exec);
            break;
        case 8:
            driver<8, 8, 8, 1, AccumType, Element, ElementC>(i, i, i, rtol, atol, exec);
            break;
        case 16:
            driver<16, 16, 16, 1, AccumType, Element, ElementC>(i, i, i, rtol, atol, exec);
            break;
        case 32:
            driver<32, 32, 32, 1, AccumType, Element, ElementC>(i, i, i, rtol, atol, exec);
            break;
        case 64:
            driver<64, 64, 64, 1, AccumType, Element, ElementC>(i, i, i, rtol, atol, exec);
            break;
        default:
            {
                //constexpr int pS = FLASHMOE_ARCH >= 800 ? 2 : 1;
                constexpr int pS = 2;
                constexpr int bK = 32;
                if (i >= 128 && i <= 2048) {
                    driver<128, 64, bK, pS, AccumType, Element, ElementC>(i, i, i, rtol, atol,exec);
                }
                else if (i > 2048) {
                    driver<128, cute::max(64, 64 * (4 / sizeof(Element))), bK, pS, AccumType, Element, ElementC>(i, i, i, rtol, atol,exec);
                }
            }
        }
    }
    cudaStreamDestroy(stream);
}

int main(const int argc, char** argv) {
    kickStart(argc, argv);
}