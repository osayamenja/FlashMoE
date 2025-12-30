//
// Created by osayamen on 12/28/25.
//
// Benchmark and unit tests for the fused gate
#include <random>

#include "common.cuh"
#include "../include/flashmoe/gate.cuh"

template<typename RE, typename REC>
__host__ __forceinline__
auto reference(RE* const& tokens, RE* const& gateWeights,
    const int& S, const int& H, const int& E, const int& k, const int& EC,
    int* const& _tokenIds, matx::index_t* const& _tokenIds_ref,
    int* const& _eCounts, matx::index_t* const& _eCounts_ref, matx::index_t* const& _topK,
    REC* const& routing, REC* const& routing_ref,
    const float& rtol, const float& atol, matx::cudaExecutor& exec) {
    using Element = MXE<RE>;
    using ElementC = MXE<REC>;
    auto tA = matx::make_tensor<Element>(reinterpret_cast<Element*>(tokens), {S, H});
    auto tB = matx::make_tensor<Element>(reinterpret_cast<Element*>(gateWeights), {H, E});
    auto tC = matx::make_tensor<ElementC>(reinterpret_cast<ElementC*>(routing_ref), {S, E});
    auto tCx = matx::make_tensor<ElementC>(reinterpret_cast<ElementC*>(routing), {S, E});

    auto tokenIds = matx::make_tensor<matx::index_t>(_tokenIds_ref, {E, S});
    auto eCounts = matx::make_tensor<matx::index_t>(_eCounts_ref, {E});
    auto tokenIds_x = matx::make_tensor<int>(_tokenIds, {E, EC});
    auto eCounts_x = matx::make_tensor<int>(_eCounts, {E});

    auto gemm_n_matches = matx::make_tensor<long int>({});
    auto ec_matches = matx::make_tensor<int>({});

    // GEMM + Softmax
    (tC = matx::softmax(matx::matmul(tA, tB.PermuteMatrix()), {1})).run(exec);
    // check error for GEMM+Softmax
    matx::cudaExecutor exec1{exec.getStream()};
    (gemm_n_matches = matx::sum(matx::isclose(tCx, tC, rtol, atol))).run(exec1);
    auto topK = matx::make_tensor<matx::index_t>(_topK, {S, E});
    // select topK
    // use external probabilities to verify token indices
    (topK = matx::argsort(tCx, matx::SORT_DIR_DESC)).run(exec);
    // Gather tokenIds
    for (int i = 0; i < E; ++i) {
        // get ith row
        auto tKRow = topK.PermuteMatrix().Slice<1>({i, 0}, {matx::matxDropDim, matx::matxEnd});
        // get ith row
        auto tIdsRow = tokenIds.Slice<1>({i, 0}, {matx::matxDropDim, matx::matxEnd});
        auto eC = eCounts.Slice<0>({i}, {matx::matxDropDim});
        (matx::mtie(tIdsRow, eC) = matx::find_idx(tKRow, matx::LT<matx::index_t>{k})).run(exec);
        // check if it matches
        auto tIdsRow_x = tokenIds_x.Slice<1>({i, 0}, {matx::matxDropDim, matx::matxEnd});
        //auto st_x = matx::make_tensor<>()
    }
    // check eC
    (ec_matches = matx::sum(eCounts_x == eCounts)).run(exec);
    CHECK_CUDA(cudaPeekAtLastError());
    exec.sync();
    // calculate error percentage
    const auto ep_gs =  (1.0 - (static_cast<double>(gemm_n_matches()) /
        static_cast<double>(tC.TotalSize()))) * 100;
    const auto ep_ec = (1.0 - (static_cast<double>(ec_matches()) /
        static_cast<double>(tC.TotalSize()))) * 100;

    // benchmark results

    return std::make_tuple(0.f, 0.f);
}

template<
        typename TileShape,
        int Arch,
        int threads,
        flashmoe::GateReductionLevel grl = flashmoe::GateReductionLevel::singleBlock,
        flashmoe::SoftMaxOptimizationLevel sro = flashmoe::SoftMaxOptimizationLevel::none,
        flashmoe::gate::InsideFusedKernel ifk = flashmoe::gate::InsideFusedKernel::yes,
        typename MMA_C = float,
        typename Element,
        typename ElementR
    >
__launch_bounds__(threads, 1)
__global__ void gateKernel(const Element* __restrict__ tokens,
        const Element* __restrict__ _gateWeights,
        ElementR* __restrict__ _routing,
        flashmoe::TPS* __restrict__ tokenIds,
        int* __restrict__ expertCounts,
        const __grid_constant__ int S,
        const __grid_constant__ int H,
        const __grid_constant__ int E,
        const __grid_constant__ int k,
        const __grid_constant__ int EC,
        flashmoe::SoftmaxStatePacked* __restrict__ rSp,
        flashmoe::RingTopKPayload* __restrict__ rTp
        ) {
        flashmoe::gate::forward<TileShape, Arch, threads, grl, sro, ifk, MMA_C>(tokens, _gateWeights,
            _routing, tokenIds, expertCounts, S, H, E, k, EC, rSp, rTp);
}

template<int threads, typename Kernel, typename Element, typename ElementC>
__host__ __forceinline__
auto gk_run(Kernel& kernel,
    Element* const& tokens, Element* const& gateWeights,
    const int& S, const int& H, const int& E, const int& k, const int& EC, const int& blocks,
    const float& rtol, const float& atol, matx::cudaExecutor& exec,
    ElementC* const& routing, ElementC* const& routing_ref,
    int* const& tokenIds, matx::index_t* const& tokenIds_ref,
    int* const& eCounts, matx::index_t* const& eCounts_ref, matx::index_t* const& topK,
    flashmoe::SoftmaxStatePacked* const& rSp, flashmoe::RingTopKPayload* const& rTp) {
    constexpr auto runs = 128;
    constexpr auto warmup = 32;
    kernel<<<blocks, threads, 0, exec.getStream()>>>(tokens, gateWeights, routing, tokenIds,
        eCounts, S, H, E, k, EC, rSp, rTp);
    const auto [error_pct, ref_time_ms] = reference();
    for (int i = 0; i < warmup; ++i) {
        kernel<<<blocks, threads, 0, exec.getStream()>>>(tokens, gateWeights, routing, tokenIds,
            eCounts, S, H, E, k, EC, rSp, rTp);
    }
    exec.sync();
    exec.start_timer();
    for (int i = 0; i < runs; ++i) {
        kernel<<<blocks, threads, 0, exec.getStream()>>>(tokens, gateWeights, routing, tokenIds,
            eCounts, S, H, E, k, EC, rSp, rTp);
    }
    exec.stop_timer();
    CHECK_CUDA(cudaPeekAtLastError());
    const auto k_time_ms = exec.get_time_ms() / static_cast<float>(runs);
    return std::make_tuple(k_time_ms, error_pct, ref_time_ms);
}

template<
    int Arch,
    int bM, int bN, int bK, int pipeStages,
    flashmoe::GateReductionLevel grl = flashmoe::GateReductionLevel::singleBlock,
    flashmoe::SoftMaxOptimizationLevel sro = flashmoe::SoftMaxOptimizationLevel::none,
    flashmoe::gate::InsideFusedKernel ifk = flashmoe::gate::InsideFusedKernel::no,
    typename Element, typename ElementC
>
__host__ __forceinline__
void driver(const int& S, const int& E, const int& H, const int& k, const float& rtol, const float& atol, matx::cudaExecutor& exec,
    const bool dropTokens = true) {
    const int M = S;
    const int N = E;
    const int K = H;
    const int eCap = dropTokens ? cute::ceil_div(S, E) * k : S;
    Element* a = nullptr; // token matrix
    Element* b = nullptr; // weights
    ElementC* routing = nullptr; // routing
    ElementC* routing_ref = nullptr;
    int* tokenIds = nullptr;
    matx::index_t* tokenIds_ref = nullptr;
    int* eCounts = nullptr;
    matx::index_t* eCounts_ref = nullptr;
    matx::index_t* topK = nullptr;
    flashmoe::SoftmaxStatePacked* rSp = nullptr;
    flashmoe::RingTopKPayload* rTp = nullptr;

    auto stream = exec.getStream();
    cudaMallocAsync(&a, M * K * sizeof(Element), stream);
    cudaMallocAsync(&b, K * N * sizeof(Element), stream);
    cudaMallocAsync(&routing, M * N * sizeof(ElementC), stream);
    cudaMallocAsync(&routing_ref, M * N * sizeof(ElementC), stream);
    cudaMallocAsync(&tokenIds, E * eCap * sizeof(int), stream);
    cudaMallocAsync(&tokenIds_ref, E * S * sizeof(matx::index_t), stream);
    cudaMallocAsync(&eCounts, M * N * sizeof(int), stream);
    cudaMemsetAsync(eCounts, 0, M * N * sizeof(int), stream);
    cudaMallocAsync(&eCounts_ref, M * N * sizeof(matx::index_t), stream);
    cudaMallocAsync(&topK, M * N * sizeof(matx::index_t), stream);
    if (E > bN) {
        const auto sspSize = S * (E / bN);
        cudaMallocAsync(&rSp, sspSize * sizeof(flashmoe::SoftmaxStatePacked), stream);
        cudaMemsetAsync(rSp, 0, sspSize * sizeof(flashmoe::SoftmaxStatePacked), stream);
        const auto rtpSize = 2 * S * (E / bN);
        cudaMallocAsync(&rTp, M * N * sizeof(matx::index_t), stream);
        cudaMemsetAsync(rTp, 0, rtpSize * sizeof(flashmoe::RingTopKPayload), stream);
    }

    using AccumType = float;
    using TileShape = cute::Shape<cute::Int<bM>, cute::Int<bN>, cute::Int<bK>, cute::Int<pipeStages>>;
    constexpr int threads = flashmoe::tile::suggest_thread_count<bM, bN, bK, Arch, Element, AccumType>();
    auto kernel = gateKernel<TileShape, Arch, threads, grl, sro, ifk, AccumType, Element, ElementC>;
    int bps = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kernel, threads, 0);
    const int blocks = min((M / bM) * (N / bN), bps * NUM_SMS);
    constexpr auto min_v = -1.f;
    constexpr auto max_v = 1.f;
    std::random_device rd;
    // fill a
    randUniform<Arch, /*add jitter*/true>(a, M * K, rd(), min_v, max_v, exec.getStream());
    // fill b
    randUniform<Arch, true>(b, K * N, rd(), min_v, max_v, exec.getStream());

    printf("%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %.1e, %.1e, %lf, %f, %f\n",
        M, N, K, bM, bN, bK, pipeStages, threads, bps, NUM_SMS, blocks, rtol, atol, e_p, k_ms, r_ms);
}

enum class BenchType {
    sweepExperts = 0,
    sweepSeqEmb = 1,
};
//./gtb <bench_type(0 or 1)> <E> <rtol> <atol>
template<BenchType b = BenchType::sweepExperts>
__host__ __forceinline__
void kickStart(const int argc, char** argv) {
    static_assert(b == BenchType::sweepExperts);
    // we have to fix these to minimize instantiated templates
    constexpr auto S = 128;
    constexpr auto H = 2048;
    using Element = __half;
    using ElementC = float;
    float rtol = 2e-2f;
    float atol = 2e-3f;
    int E = 2;
    int E_max = 256;

    printf("S, E, H, bM, bN, bK, pipeStages, threads, blocks/SM, SMs, blocks, rtol, atol, error(%%), Kernel_Time(ms), "
           "Matx_Time(ms)\n");
    if (argc > 2) {
        E = std::stoi(argv[2]);
    }
    if (argc > 3) {
        E_max = std::stoi(argv[3]);
    }
    if (argc > 4) {
        rtol = std::stof(argv[4]);
    }
    if (argc > 5) {
        atol = std::stof(argv[5]);
    }
    cudaSetDevice(0);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    matx::cudaExecutor exec{stream, true};
    // tiling heuristics
    constexpr int bM = cute::min(S, 128);
    constexpr int kF = 32 * (4 / sizeof(Element));
    constexpr int bK = cute::min(H, kF);
    constexpr int pS = H >= bK * 2 ? 2 : 1;
    for (int i = E; i <= E_max; i *= 2) {
        switch (i) {
        case 8:
            driver<bM, 8, bK, pS, Element, ElementC>(S, i, H, rtol, atol, exec);
            break;
        case 16:
            driver<bM, 16, bK, pS, Element, ElementC>(S, i, i, rtol, atol, exec);
            break;
        case 32:
            driver<bM, 32, bK, pS, Element, ElementC>(S, i, H, rtol, atol, exec);
            break;
        case 64:
            driver<bM, 64, bK, pS, Element, ElementC>(S, i, H, rtol, atol, exec);
            break;
        default:
            {
                if (i >= 128 && i <= 2048) {
                    driver<bM, 64, bK, pS, Element, ElementC>(S, i, H, rtol, atol,exec);
                }
                else if (i > 2048) {
                    driver<bM, 64 * (4 / sizeof(Element)), bK, pS, Element, ElementC>(S, i, H, rtol, atol,exec);
                }
            }

        }
    }
    cudaStreamDestroy(stream);
}

template<>
__host__ __forceinline__
void kickStart<BenchType::sweepSeqEmb>(const int argc, char** argv) {
    // fixed to minimize template instantiations
    constexpr int E = 8;
    int MK = 8;
    int MK_max = 8192;
    float rtol = 2e-2f;
    float atol = 2e-3f;
    using Element = __half;
    using ElementC = Element;
    printf("S, E, H, bM, bN, bK, pipeStages, threads, blocks/SM, SMs, blocks, rtol, atol, error(%%), Kernel_Time(ms), "
           "Matx_Time(ms)\n");
    if (argc > 2) {
        MK = std::stoi(argv[2]);
    }
    if (argc > 3) {
        MK_max = std::stoi(argv[3]);
    }
    if (argc > 4) {
        rtol = std::stof(argv[4]);
    }
    if (argc > 5) {
        atol = std::stof(argv[5]);
    }
    cudaSetDevice(0);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    matx::cudaExecutor exec{stream, true};
    // tiling heuristics
    constexpr int bN = cute::min(E, 64);
    for (int i = MK; i <= MK_max; i *= 2) {
        switch (i) {
        case 8:
            driver<8, bN, 8, 1, Element, ElementC>(i, E, i, rtol, atol, exec);
            break;
        case 16:
            driver<16, bN, 16, 1, Element, ElementC>(i, E, i, rtol, atol, exec);
            break;
        case 32:
            driver<32, bN, 32, 1, Element, ElementC>(i, E, i, rtol, atol, exec);
            break;
        case 64:
            driver<64, bN, 64, 1, Element, ElementC>(i, i, E, rtol, atol, exec);
            break;
        default:
            {
                if (i >= 128 && i <= 2048) {
                    driver<128, bN, 32, 2, Element, ElementC>(i, E, i, rtol, atol,exec);
                }
                else if (i > 2048) {
                    driver<128, bN * (4 / sizeof(Element)), 32, 2, Element, ElementC>(i, E, i, rtol, atol,exec);
                }
            }

        }
    }
    cudaStreamDestroy(stream);
}
int main(const int argc, char** argv) {
    int bench_type = 0;
    printf("S, E, H, bM, bN, bK, pipeStages, threads, blocks/SM, SMs, blocks, rtol, atol, error(%%), "
           "Kernel_Time(ms), Matx_Time(ms)\n");
    if (argc > 1) {
        bench_type = atoi(argv[1]);
    }
    if (bench_type == 0) {
        kickStart<BenchType::sweepExperts>(argc, argv);
    }
    else {
        kickStart<BenchType::sweepExperts>(argc, argv);
    }
    kickStart(argc, argv);
}