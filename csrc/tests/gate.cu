//
// Created by osayamen on 12/28/25.
//
// Benchmark and unit tests for the fused gate
#include <random>
#include <tuple>
#include <vector>
#include <cstdio>
#include <string>
#include <nvtx3/nvtx3.hpp>

#include "common.cuh"
#include "debug.cuh"
#include "../include/flashmoe/gate.cuh"

struct IndexSanitizer {
    const matx::index_t y;
    explicit IndexSanitizer(const matx::index_t& _y): y(_y){}
    __host__ __device__
    auto operator()(const matx::index_t& x) const {
        return x / y;
    }
};

struct SplitFunctor {
    __host__ __device__
    auto operator()(const flashmoe::TPS& t) const {
        return t.tokenIdx;
    }
};

struct __align__(16) GateArgs {
    void* tokens;
    void* gateWeights;
    void* routing;
    void* routing_ref;
    flashmoe::TPS* tokenIds_packed;
    uint* tokenIds;
    matx::index_t* tokenIds_ref;
    int* eCounts;
    int* eCGuards;
    matx::index_t* eCounts_ref;
    matx::index_t* topK;
    flashmoe::SoftmaxStatePacked* rSp;
    flashmoe::RingTopKPayload* rTp;
    int S;
    int H;
    int E;
    int k;
    int EC;
    float rtol;
    float atol;
};

template<
        typename TileShape,
        int Arch,
        int threads,
        flashmoe::GateReductionLevel grl = flashmoe::GateReductionLevel::singleBlock,
        flashmoe::SoftMaxOptimizationLevel sro = flashmoe::SoftMaxOptimizationLevel::none,
        typename AccumType = float,
        typename Element,
        typename ElementR
    >
__launch_bounds__(threads, 1)
__global__ void gateKernel(const Element* __restrict__ tokens,
        const Element* __restrict__ _gateWeights,
        ElementR* __restrict__ _routing,
        flashmoe::TPS* __restrict__ tokenIds,
        int* __restrict__ expertCounts,
        int* __restrict__ eCGuards,
        const __grid_constant__ int S,
        const __grid_constant__ int H,
        const __grid_constant__ int E,
        const __grid_constant__ int k,
        const __grid_constant__ int EC,
        flashmoe::SoftmaxStatePacked* __restrict__ rSp,
        flashmoe::RingTopKPayload* __restrict__ rTp
        ) {
    constexpr int bM = cute::get<0>(TileShape{});
    constexpr int bN = cute::get<1>(TileShape{});
    constexpr int bK = cute::get<2>(TileShape{});
    constexpr int pipeStages = cute::get<3>(TileShape{});
    using TileGEMM = flashmoe::tile::CollectiveMainloop<
            bM, bN, bK, Arch, Element, AccumType, threads, pipeStages
        >;
    extern __shared__ __align__(TileGEMM::GeneralAlignment::value) cuda::std::byte gateWorkspace[];
    const auto roundEC = cute::ceil_div(EC, bM) * bM;
    flashmoe::gate::forward<TileGEMM, grl, sro>(gateWorkspace, tokens, _gateWeights,
        _routing, tokenIds, expertCounts,
        S, H, E, k, EC, roundEC, static_cast<int>(gridDim.x), eCGuards, rSp, rTp);
}

template<int warmup, int runs, typename RE, typename REC>
__host__ __forceinline__
auto reference(matx::cudaExecutor& exec, const GateArgs& gArgs, cudaEvent_t start, cudaEvent_t stop) {
    using Element = MXE<RE>;
    using ElementC = MXE<REC>;
    auto tA = matx::make_tensor<Element>(static_cast<Element*>(gArgs.tokens),
        {gArgs.S, gArgs.H});
    auto tB = matx::make_tensor<Element>(static_cast<Element*>(gArgs.gateWeights),
        {gArgs.E, gArgs.H});
    auto tC = matx::make_tensor<ElementC>(static_cast<ElementC*>(gArgs.routing_ref),
        {gArgs.S, gArgs.E});
    auto tCx = matx::make_tensor<ElementC>(static_cast<ElementC*>(gArgs.routing),
        {gArgs.S, gArgs.E});

    auto tokenIds = matx::make_tensor<matx::index_t>(gArgs.tokenIds_ref,
        {gArgs.E, gArgs.S});
    auto eCounts = matx::make_tensor<matx::index_t>(gArgs.eCounts_ref,
        {gArgs.E});
    auto tokenIds_packed = matx::make_tensor<flashmoe::TPS>(gArgs.tokenIds_packed,
        {gArgs.E, gArgs.EC});
    auto tokenIds_x = matx::make_tensor<uint>(gArgs.tokenIds, {gArgs.E, gArgs.EC});
    (tokenIds_x = matx::apply(SplitFunctor{}, tokenIds_packed)).run(exec);
    auto eCounts_x = matx::make_tensor<int>(gArgs.eCounts, {gArgs.E});

    // dynamically allocated managed memory
    auto gemm_n_matches = matx::make_tensor<long int>({});
    auto ec_matches = matx::make_tensor<int>({gArgs.E});
    auto s_ec_matches = matx::make_tensor<int>({});
    auto tIds_matches = matx::make_tensor<int>({gArgs.E});
    auto s_tIds_matches = matx::make_tensor<int>({});
    auto st_x = matx::make_tensor<uint>({gArgs.EC});
    auto st = matx::make_tensor<matx::index_t>({gArgs.EC});
    (tIds_matches = matx::zeros<int>(tIds_matches.Shape())).run(exec);

    // GEMM + Softmax
    (tC = matx::softmax(matx::matmul(tA, tB.PermuteMatrix()), {1})).run(exec);
    // check error for GEMM+Softmax
    matx::cudaExecutor exec1{exec.getStream()};
    (gemm_n_matches = matx::sum(matx::isclose(tCx, tC, gArgs.rtol, gArgs.atol))).run(exec1);
    auto sIndices = matx::make_tensor<matx::index_t>(gArgs.topK, {gArgs.S, gArgs.E});
    // select topK
    // we use external probabilities to verify token indices which avoids
    // unpredictable side-effects from any numerical discrepancy between tCx and tC.
    // Specifically, if tC is used instead, given a failed check,
    // we would not be able to tell if it is due to the indices being generated
    // incorrectly by the fused kernel or due to the floating point disparity in comparison to tCx.
    // That's why the check is two-phased. The first check verifies the floating-point correctness
    // of the GEMM+Softmax while the other verifies the indices generation logic _independently_.
    (sIndices = matx::argsort(tCx, matx::SORT_DIR_DESC)).run(exec);
    auto topK_idx = sIndices.Slice<2>({0, 0}, {matx::matxEnd, gArgs.k});
    // Gather tokenIds
    for (int i = 0; i < gArgs.E; ++i) {
        auto tIdx_row = tokenIds.Slice<1>({i, 0}, {matx::matxDropDim, matx::matxEnd});
        auto eC = eCounts.Slice<0>({i}, {matx::matxDropDim});
        (matx::mtie(tIdx_row, eC) = matx::find_idx(topK_idx, matx::EQ<matx::index_t>{i})).run(exec);
        // converts indices from linearized [0, S*K) to [0, S).
        // This is necessary when using matx::find_idx
        (tIdx_row = matx::apply(IndexSanitizer{gArgs.k}, tIdx_row)).run(exec);
    }
    // check eC
    (ec_matches = (eCounts_x == eCounts)).run(exec);
    (s_ec_matches = matx::sum(ec_matches)).run(exec);
    std::vector<matx::index_t> hec(eCounts.TotalSize());
    cudaMemcpyAsync(hec.data(), gArgs.eCounts_ref, sizeof(matx::index_t) * eCounts.TotalSize(),
        cudaMemcpyDeviceToHost, exec.getStream());
    exec.sync();
    // check tokenIds
    matx::index_t totalRoutedTokens = 0;
    for (int i = 0; i < gArgs.E; ++i) {
        if (ec_matches(i)) {
            // expert count matches
            // now check token indices
            const auto eCount = hec[i];
            totalRoutedTokens += eCount;
            auto tIdsRow_x = tokenIds_x.Slice<1>({i, 0}, {matx::matxDropDim, eCount});
            auto st_x_s = st_x.Slice<1>({0}, {eCount});
            (st_x_s = matx::sort(tIdsRow_x, matx::SORT_DIR_DESC)).run(exec);
            // check if it matches
            auto tIdsRow = tokenIds.Slice<1>({i, 0}, {matx::matxDropDim, eCount});
            auto st_s = st.Slice<1>({0}, {eCount});
            (st_s = matx::sort(tIdsRow, matx::SORT_DIR_DESC)).run(exec);
            auto tIm_r = tIds_matches.Slice<0>({i}, {matx::matxDropDim});
            // compare sorted indices arrays
            (tIm_r = matx::sum(st_x_s == st_s)).run(exec);
        }
    }
    (s_tIds_matches = matx::sum(tIds_matches)).run(exec);
    CHECK_CUDA(cudaPeekAtLastError());
    exec.sync();
    // calculate error percentage
    const auto ep_gs =  (1.0 - (static_cast<double>(gemm_n_matches()) /
        static_cast<double>(tC.TotalSize()))) * 100;
    const auto ep_ec = (1.0 - (static_cast<double>(s_ec_matches()) /
        static_cast<double>(eCounts.TotalSize()))) * 100;
    const auto ep_tIds = (1.0 - (static_cast<double>(s_tIds_matches()) /
        static_cast<double>(totalRoutedTokens))) * 100;

    auto gate_via_matx = [&]() {
        nvtx3::scoped_range matxRange{"Gate"};
        // GEMM + Softmax
        (tC = matx::softmax(matx::matmul(tA, tB.PermuteMatrix()), {1})).run(exec);
        (sIndices = matx::argsort(tCx, matx::SORT_DIR_DESC)).run(exec);
        // Gather tokenIds
        for (int i = 0; i < gArgs.E; ++i) {
            auto tIdx_row = tokenIds.Slice<1>({i, 0}, {matx::matxDropDim, matx::matxEnd});
            auto eC = eCounts.Slice<0>({i}, {matx::matxDropDim});
            (matx::mtie(tIdx_row, eC) = matx::find_idx(topK_idx, matx::EQ<matx::index_t>{i})).run(exec);
            (tIdx_row = matx::apply(IndexSanitizer{gArgs.k}, tIdx_row)).run(exec);
        }
    };
    for (int i = 0; i < warmup; ++i) {
        gate_via_matx();
    }
    exec.sync();
    cudaEventRecord(start, exec.getStream());
    for (int i = 0; i < runs; ++i) {
        gate_via_matx();
    }
    cudaEventRecord(stop, exec.getStream());
    CHECK_CUDA(cudaEventSynchronize(stop));
    float m_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&m_ms, start, stop));
    const float m_time_ms = m_ms / static_cast<float>(runs);

    // benchmark results
    return std::make_tuple(m_time_ms, ep_gs, ep_ec, ep_tIds);
}

template<int Arch, int sharedSize,
    typename TileShape,
    flashmoe::GateReductionLevel grl = flashmoe::GateReductionLevel::singleBlock,
    flashmoe::SoftMaxOptimizationLevel sro = flashmoe::SoftMaxOptimizationLevel::none,
    int threads, typename AccumType, typename Element, typename ElementC
>
__host__ __forceinline__
auto gk_run(matx::cudaExecutor& exec, const GateArgs& gArgs, const int& blocks, const int& checkCorrectness) {
    constexpr auto runs = 128;
    constexpr auto warmup = 32;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    auto kernel = [&]() {
        gateKernel<TileShape, Arch, threads, grl, sro, AccumType>
        <<<blocks, threads, sharedSize, exec.getStream()>>>(
            static_cast<Element*>(gArgs.tokens),
            static_cast<Element*>(gArgs.gateWeights),
            static_cast<ElementC*>(gArgs.routing),
            gArgs.tokenIds_packed,
            gArgs.eCounts, gArgs.eCGuards,
            gArgs.S, gArgs.H, gArgs.E, gArgs.k, gArgs.EC, gArgs.rSp, gArgs.rTp);
    };
    kernel();
    auto ref_result = std::make_tuple(0.f, -0.0, -0.0, -0.0);
    if (checkCorrectness) {
        ref_result = reference<warmup, runs, Element, ElementC>(exec, gArgs, start, stop);
    }
    for (int i = 0; i < warmup; ++i) {
        kernel();
    }
    exec.sync();
    cudaEventRecord(start, exec.getStream());
    for (int i = 0; i < runs; ++i) {
        kernel();
    }
    cudaEventRecord(stop, exec.getStream());
    float k_ms = 0;
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&k_ms, start, stop));
    CHECK_CUDA(cudaPeekAtLastError());
    const auto k_time_ms = k_ms / static_cast<float>(runs);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return std::make_tuple(k_time_ms, ref_result);
}

template<
    int Arch,
    int bM, int bN, int bK, int pipeStages,
    flashmoe::GateReductionLevel grl = flashmoe::GateReductionLevel::singleBlock,
    flashmoe::SoftMaxOptimizationLevel sro = flashmoe::SoftMaxOptimizationLevel::none,
    typename Element, typename ElementC
>
__host__ __forceinline__
void driver(const int& S, const int& E, const int& H, const int& k, const float& rtol, const float& atol,
    const int& checkCorrectness, matx::cudaExecutor& exec) {
    nvtx3::scoped_range driverRange{std::string("S: ")
        .append(std::to_string(S)).append(", E: ")
        .append(std::to_string(E)).append(", H: ")
        .append(std::to_string(H).append(", topK: ")
        .append(std::to_string(k)))};
    const int M = S;
    const int N = E;
    const int K = H;
    const int eCap = S; // no dropping for the test
    Element* tokens = nullptr; // token matrix
    Element* gateWeights = nullptr; // weights
    ElementC* routing = nullptr; // routing
    ElementC* routing_ref = nullptr;
    flashmoe::TPS* tokenIds = nullptr; // TPS is a packed struct of float and int
    uint* tokenIds_idx = nullptr; // TPS is a packed struct of float and int
    matx::index_t* tokenIds_ref = nullptr;
    int* eCounts = nullptr;
    int* eCGuards = nullptr;
    matx::index_t* eCounts_ref = nullptr;
    matx::index_t* topK = nullptr;
    flashmoe::SoftmaxStatePacked* rSp = nullptr;
    flashmoe::RingTopKPayload* rTp = nullptr;

    auto stream = exec.getStream();
    cudaMallocAsync(&eCounts, E * sizeof(int), stream);
    cudaMallocAsync(&eCGuards, E * sizeof(int), stream);
    cudaMemsetAsync(eCGuards, flashmoe::STALE_AS_BYTE, E * sizeof(int), stream);
    cudaMallocAsync(&tokens, M * K * sizeof(Element), stream);
    cudaMallocAsync(&gateWeights, K * N * sizeof(Element), stream);
    cudaMallocAsync(&routing, M * N * sizeof(ElementC), stream);
    cudaMallocAsync(&tokenIds, E * eCap * sizeof(flashmoe::TPS), stream);
    if (checkCorrectness) {
        // correctness checking consumes a lot of memory
        cudaMallocAsync(&routing_ref, M * N * sizeof(ElementC), stream);
        cudaMallocAsync(&tokenIds_idx, E * eCap * sizeof(uint), stream);
        cudaMallocAsync(&tokenIds_ref, E * S * sizeof(matx::index_t), stream);
        cudaMallocAsync(&topK, S * E * sizeof(matx::index_t), stream);
        cudaMallocAsync(&eCounts_ref, E * sizeof(matx::index_t), stream);
    }
    if (E > bN) {
        const auto sspSize = S * cute::ceil_div(E, bN) * sizeof(flashmoe::SoftmaxStatePacked);
        cudaMallocAsync(&rSp, sspSize, stream);
        // only need to reset this once, the kernel internally resets after usage
        cudaMemsetAsync(rSp, 0, sspSize, stream);
        const auto rtpSize = 2 * S * cute::ceil_div(E, bN) * sizeof(flashmoe::RingTopKPayload);
        cudaMallocAsync(&rTp, rtpSize, stream);
        // only need to reset this once, the kernel internally resets after usage
        cudaMemsetAsync(rTp, 0, rtpSize, stream);
    }

    using AccumType = float;
    using TileShape = cute::Shape<cute::Int<bM>, cute::Int<bN>, cute::Int<bK>, cute::Int<pipeStages>>;
    constexpr int threads = flashmoe::tile::suggest_thread_count<bM, bN, bK, Arch, Element, AccumType>();
    auto kernel = gateKernel<TileShape, Arch, threads, grl, sro, AccumType, Element, ElementC>;
    int bps = 0;
    constexpr auto sharedSize = cute::max(bK * pipeStages * (bM + bN) * sizeof(Element),
        bM * bN * sizeof(AccumType));
    CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedSize));
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kernel, threads, sharedSize));
    const int blocks = cute::min(cute::ceil_div(M, bM) * cute::ceil_div(N, bN),
        bps * NUM_SMS); // use ceil_div to avoid zero
    if (E > blocks * bN) {
        throw std::invalid_argument("E is too big!");
    }
    constexpr auto min_v = -1.f;
    constexpr auto max_v = 1.f;
    std::random_device rd;
    // fill a
    randUniform<Arch, /*add jitter*/true>(tokens, M * K, rd(), min_v,
        max_v, exec.getStream());
    // fill b
    randUniform<Arch, true>(gateWeights, K * N, rd(), min_v, max_v, exec.getStream());
    const GateArgs gArgs{
        tokens,
        gateWeights,
        routing,
        routing_ref,
        tokenIds,
        tokenIds_idx,
        tokenIds_ref, eCounts, eCGuards, eCounts_ref, topK, rSp, rTp,
        S, H, E, k, eCap,
        rtol, atol
    };
    // returns [kernel_ms, [m_time_ms, ep_gs, ep_ec, ep_tIds]]
    const auto results = gk_run<Arch, sharedSize, TileShape, grl, sro, threads, AccumType, Element, ElementC>
    (exec, gArgs, blocks, checkCorrectness);
    const float kernel_ms = std::get<0>(results);
    const auto r_tuple = std::get<1>(results);
    const float matx_ms = std::get<0>(r_tuple);
    const double ep_gs = std::get<1>(r_tuple);
    const double ep_ec = std::get<2>(r_tuple);
    const double ep_tIds = std::get<3>(r_tuple);

    printf("%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %.1e, %.1e, %lf, %lf, %lf, %f, %f\n",
        M, N, K, bM, bN, bK, pipeStages, threads, bps, NUM_SMS,
        blocks, rtol, atol, ep_gs, ep_ec, ep_tIds, kernel_ms, matx_ms);

    cudaFreeAsync(tokens, stream);
    cudaFreeAsync(gateWeights, stream);
    cudaFreeAsync(routing, stream);
    cudaFreeAsync(tokenIds, stream);
    cudaFreeAsync(eCounts, stream);
    cudaFreeAsync(eCGuards, stream);
    if (checkCorrectness) {
        cudaFreeAsync(routing_ref, stream);
        cudaFreeAsync(tokenIds_idx, stream);
        cudaFreeAsync(tokenIds_ref, stream);
        cudaFreeAsync(topK, stream);
        cudaFreeAsync(eCounts_ref, stream);
    }
    if (E > bN) {
        cudaFreeAsync(rSp, stream);
        cudaFreeAsync(rTp, stream);
    }
    exec.sync();
}

//./gtb <E> <E_max> <k> <checkCorrectness> <rtol> <atol>
// Sequence length and H are fixed
__host__ __forceinline__
void kickStart(const int argc, char** argv) {
    // we have to fix S and H to minimize instantiated templates as tile shapes are dependent on them
    constexpr auto S = 2048;
    constexpr auto H = 4096;
    using Element = __half;
    using ElementC = float;
    float rtol = 2e-2f;
    float atol = 2e-3f;
    int E = 8;
    int E_max = 256;
    int k = 8;
    int checkCorrectness = 1;
    constexpr int Arch = FLASHMOE_ARCH;
    printf("S, E, H, bM, bN, bK, pipeStages, threads, blocks/SM, SMs, "
           "blocks, rtol, atol, error_gemm_softmax(%%), error_expert_counts(%%), error_tokenIds(%%), "
           "Kernel_Time(ms), Matx_Time(ms)\n");
    if (argc > 1) {
        E = std::stoi(argv[1]);
    }
    if (argc > 2) {
        E_max = std::stoi(argv[2]);
    }
    if (argc > 3) {
        k = std::stoi(argv[3]);
    }
    if (argc > 4) {
        checkCorrectness = std::stoi(argv[4]);
    }
    if (argc > 5) {
        rtol = std::stof(argv[5]);
    }
    if (argc > 6) {
        atol = std::stof(argv[6]);
    }
    if (k > E) {
        throw std::invalid_argument("k must be at most number of experts");
    }
    cudaSetDevice(0);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    matx::cudaExecutor exec{stream};
    constexpr auto sro = flashmoe::SoftMaxOptimizationLevel::none;
    // tiling for A100 (not tuned)
    constexpr int bM = cute::min(S, 64);
    constexpr int bK = cute::min(H, 128);
    constexpr int pS = H >= bK * 2 ? (FLASHMOE_ARCH >= 800 ? 2 : 1) : 1;
    for (int i = E; i <= E_max; i *= 2) {
        switch (i) {
        case 2:
            {
                constexpr int bN = 2;
                driver<Arch, bM, bN, bK, pS, flashmoe::GateReductionLevel::singleBlock, sro,
                        Element, ElementC>(S, i, H, k, rtol, atol, checkCorrectness, exec);
            }
            break;
        case 4:
            {
                constexpr int bN = 4;
                driver<Arch, bM, bN, bK, pS, flashmoe::GateReductionLevel::singleBlock, sro,
                        Element, ElementC>(S, i, H, k, rtol, atol, checkCorrectness, exec);
            }
            break;
        case 8:
            {
                constexpr int bN = 8;
                driver<Arch, bM, bN, bK, pS, flashmoe::GateReductionLevel::singleBlock, sro,
                        Element, ElementC>(S, i, H, k, rtol, atol, checkCorrectness, exec);
            }
            break;
        case 16:
            {
                constexpr int bN = 16;
                driver<Arch, bM, bN, bK, pS, flashmoe::GateReductionLevel::singleBlock, sro,
                        Element, ElementC>(S, i, H, k, rtol, atol, checkCorrectness, exec);
            }
            break;
        case 32:
            {
                constexpr int bN = 32;
                driver<Arch, bM, bN, bK, pS, flashmoe::GateReductionLevel::singleBlock, sro,
                        Element, ElementC>(S, i, H, k, rtol, atol, checkCorrectness, exec);
            }
            break;
        default:
            {
                if (i > 32) {
                    constexpr int bN = 32;
                    driver<Arch, bM, bN, bK, pS, flashmoe::GateReductionLevel::multiBlock, sro,
                            Element, ElementC>(S, i, H, k, rtol, atol, checkCorrectness, exec);
                }
            }
        }
    }
    cudaStreamDestroy(stream);
}

int main(const int argc, char** argv) {
    kickStart(argc, argv);
}