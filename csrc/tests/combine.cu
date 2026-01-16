//
// Created by osayamen on 1/14/26.
//
// unit tests for combine
#include <array>
#include <vector>
#include <tuple>
#include <random>
#include <cassert>
#include <stdexcept>

#include "common.cuh"
#include "debug.cuh"
#include "../include/flashmoe/combine.cuh"

// Kernel Under Test (KUT)
// note this is not an optimal implementation!
template<int Arch, int bM, int bN, int threads, flashmoe::CombineMode c, typename Element>
__launch_bounds__(threads, 1)
__global__ void combineKernel(const __grid_constant__ int EC,
    const __grid_constant__ int S,
    const __grid_constant__ int E,
    const __grid_constant__ int H,
    const __grid_constant__ int k,
    const int* __restrict__ expertCounts, // [E]
    const Element* __restrict__ tokens, // [E, EC, H]
    Element* __restrict__ output, // [S, H]
    const flashmoe::TPS* __restrict__ tokenIndices // [E, EC]
    ) {
    constexpr int Alignment = flashmoe::ElementAlignment<Element, bN>;
    __shared__ __align__(Alignment) cuda::std::byte workspace [bM * bN * sizeof(Element)];
    const auto tilesM = EC / bM;
    const auto tilesN = H / bN;
    const auto tilesPerExpert = tilesM * tilesN;
    const auto numTiles = E * tilesM * tilesN;
    const auto tokTensor = cute::make_tensor(cute::make_gmem_ptr(tokens),
        cute::make_layout(cute::make_shape(E, EC, H), cute::LayoutRight{}));
    for (int globalIdx = blockIdx.x; globalIdx < numTiles; globalIdx += gridDim.x) {
        const auto expertIdx = globalIdx / tilesPerExpert;
        const auto tileIdx = globalIdx % tilesPerExpert;
        const auto coord = flashmoe::tile::idx2Coord(tilesM, tilesN, tileIdx);
        const auto tileCoord = cute::make_coord(cute::_0{}, cute::get<1>(coord));
        const auto tileM = cute::get<0>(coord);
        const auto expertCount  = expertCounts[expertIdx];
        const auto actualTiles = cute::ceil_div(expertCount, bM) * tilesN;
        if (tileIdx < actualTiles) {
            const auto numFullMTiles = expertCount / bM;
            auto* __restrict__ tP = &tokTensor(expertIdx, tileM * bM, 0);
            const auto tileSize = tileM < numFullMTiles ? bM : (expertCount - numFullMTiles * bM);
            flashmoe::combine<bM, bN, Arch, threads, c>(EC, S, E, H, k, workspace,
                tokenIndices, output, tP, tileM * bM,
                expertIdx, tileSize, tileCoord);
        }
    }
}

// A single CTA reads each token and accumulates into the output buffer
// Note this is not an optimal implementation
template<typename AccumType, typename Element>
__global__ void combineReference(const __grid_constant__ int E, const __grid_constant__ int S,
    const __grid_constant__ int H, const __grid_constant__ int EC, const __grid_constant__ int topK,
    const Element* __restrict__ tokens, // [E, EC, H]
    const flashmoe::TPS* __restrict__ tokenIds, //[E, EC] metadata
    const int* __restrict__ expertCounts, // [E]
    Element* __restrict__ result, // [S, H]
    float* __restrict__ oracleResult // [S, H]
    ) {
    static_assert(cuda::std::is_same_v<AccumType, Element> || cuda::std::is_same_v<AccumType, float>);
    if (blockIdx.x > 0) {
        return;
    }
    const auto tIds = cute::make_tensor(cute::make_gmem_ptr(tokenIds),
        cute::make_layout(cute::make_shape(E, EC), cute::LayoutRight{}));
    const auto tokTensor = cute::make_tensor(cute::make_gmem_ptr(tokens),
        cute::make_layout(cute::make_shape(E, EC, H), cute::LayoutRight{}));
    auto resultTensor = cute::make_tensor(cute::make_gmem_ptr(result),
        cute::make_layout(cute::make_shape(S, H), cute::LayoutRight{}));
    auto oracleT = cute::make_tensor(cute::make_gmem_ptr(oracleResult),
        cute::make_layout(cute::make_shape(S, H), cute::LayoutRight{}));
    for (int i = 0; i < E; ++i) {
        const auto expertCount = expertCounts[i];
        for (int j = 0; j < expertCount; ++j) {
            const auto tokenId = tIds(i, j);
            if (topK == 1) {
                for (auto k = threadIdx.x; k < H; k += blockDim.x) {
                    resultTensor(tokenId.tokenIdx, k) = tokTensor(i, j, k);
                }
            }
            else {
                constexpr flashmoe::Converter<float, Element> loadOp{};
                // Kernel Under Test (KUT)
                // below is to closely emulate how the KUT does accumulation which is in `Element`.
                constexpr flashmoe::Converter<AccumType, Element> convOp1{};
                constexpr flashmoe::Converter<AccumType, float> convOp2{};
                constexpr flashmoe::Converter<Element, AccumType> storeOp{};
                for (auto k = threadIdx.x; k < H; k += blockDim.x) {
                    // read token value and convert to float.
                    const auto v = loadOp(tokTensor(i, j, k));
                    const float scaledV = v * tokenId.probability;
                    oracleT(tokenId.tokenIdx, k) += scaledV;
                    // read current result value and convert to AccumType
                    const auto rv = resultTensor(tokenId.tokenIdx, k);
                    const auto c = convOp1(rv);
                    // multiply in float -> the KUT multiplies in float as well
                    // accumulate scaled value in either float or Element
                    const AccumType res = c + convOp2(scaledV);
                    // store result
                    resultTensor(tokenId.tokenIdx, k) = storeOp(res);
                }
            }
        }
    }
}

__host__ __forceinline__
auto getEC(const int& S) {
    return S; // no dropping
}
enum class WeightDistribution {
  Uniform01,      // U(0,1]
  Exponential1,   // Exp(lambda=1)
  LogNormal       // logN(mu=0, sigma=1)
};

// Below is LLM-generated
// Generates a routing assignment with NO token dropping:
// - EC == S
// - each token appears exactly k times globally
// - within each expert row, each token appears at most once (guaranteed by construction)
// - expertCounts[e] <= EC == S always
//
// Returns: (expertCounts[E], tokenIds[E*EC] in row-major [e, j])
template<WeightDistribution Dist>
__host__ __forceinline__
float sample_positive_weight(std::mt19937& rng) {
  float w = 0.0f;
  if constexpr (Dist == WeightDistribution::Uniform01) {
    std::uniform_real_distribution<float> uni01(0.0f, 1.0f);
    w = uni01(rng);
  } else if constexpr (Dist == WeightDistribution::Exponential1) {
    std::exponential_distribution<float> exp1(1.0f);
    w = exp1(rng);
  } else if constexpr (Dist == WeightDistribution::LogNormal) {
    std::lognormal_distribution<float> logn(0.0f, 1.0f);
    w = logn(rng);
  } else {
    static_assert(Dist == WeightDistribution::Uniform01 ||
                  Dist == WeightDistribution::Exponential1 ||
                  Dist == WeightDistribution::LogNormal,
                  "Unsupported distribution");
  }
  if (w == 0.0f) w = 1e-7f;
  return w;
}

std::vector<int>
__host__ __forceinline__
choose_topk_experts_from_scores(std::mt19937& rng, const int& E, const int& k) {
  assert(E > 0 && k > 0 && k <= E);

  std::uniform_real_distribution<float> score_dist(0.0f, 1.0f);
  std::vector<std::pair<float,int>> scored;
  scored.reserve(static_cast<size_t>(E));
  for (int e = 0; e < E; ++e) scored.emplace_back(score_dist(rng), e);

  auto nth = scored.begin() + (E - k);
  std::ranges::nth_element(
      scored.begin(), nth, scored.end(),
      [](auto const& a, auto const& b) { return a.first < b.first; });

  std::vector<int> experts;
  experts.reserve(static_cast<size_t>(k));
  for (int r = 0; r < k; ++r) experts.push_back(scored[E - k + r].second);
  return experts;
}

// Dynamic S, E, k; EC is assumed equal to S (no dropping).
// Returns: (expertCounts[E], tokenIds[E*EC] in row-major [e, j])
template<WeightDistribution Dist = WeightDistribution::Uniform01>
std::tuple<std::vector<int>, std::vector<flashmoe::TPS>>
generate_token_ids_and_expert_counts(const int& S, const int& E, const int& k) {
  assert(S > 0 && E > 0 && k > 0);
  assert(k <= E && "Need k <= E to pick k unique experts per token");
  const int EC = S;

  std::random_device rd;
  std::mt19937 rng(rd());

  std::vector<int> expertCounts(static_cast<size_t>(E), 0);
  std::vector<int> writePos(static_cast<size_t>(E), 0);
  std::vector<int> seen(static_cast<size_t>(S), 0);

  std::vector<flashmoe::TPS> tokenIds(
      static_cast<size_t>(E) * static_cast<size_t>(EC),
      flashmoe::TPS{0, 0.0f});

  // For each token, select k unique experts, generate k weights, normalize, append.
  for (uint t = 0; t < S; ++t) {
    const std::vector<int> experts = choose_topk_experts_from_scores(rng, E, k);

    // Raw weights
    std::vector<float> w(static_cast<size_t>(k), 0.0f);
    float sumW = 0.0f;
    for (int r = 0; r < k; ++r) {
      w[static_cast<size_t>(r)] = sample_positive_weight<Dist>(rng);
      sumW += w[static_cast<size_t>(r)];
    }
    const float invSum = 1.0f / sumW;

    // Append to experts
    for (int r = 0; r < k; ++r) {
      const int e = experts[static_cast<size_t>(r)];
      const int j = writePos[static_cast<size_t>(e)]++;

      assert(j < EC && "expert row overflow (unexpected with EC==S)");
      tokenIds[static_cast<size_t>(e) * static_cast<size_t>(EC) +
               static_cast<size_t>(j)] =
          flashmoe::TPS{t, w[static_cast<size_t>(r)] * invSum};

      expertCounts[static_cast<size_t>(e)] = writePos[static_cast<size_t>(e)];
      seen[static_cast<size_t>(t)] += 1;
    }
  }

  // Pad remaining entries (optional)
  for (int e = 0; e < E; ++e) {
    for (int j = expertCounts[static_cast<size_t>(e)]; j < EC; ++j) {
      tokenIds[static_cast<size_t>(e) * static_cast<size_t>(EC) +
               static_cast<size_t>(j)] = flashmoe::TPS{0, 0.0f};
    }
  }

  // Sanity checks
  {
    long long total = 0;
    for (int e = 0; e < E; ++e) {
      assert(expertCounts[static_cast<size_t>(e)] <= EC);
      total += expertCounts[static_cast<size_t>(e)];
    }
    assert(total == 1LL * S * k);
    for (int t = 0; t < S; ++t) assert(seen[static_cast<size_t>(t)] == k);
  }

  return {std::move(expertCounts), std::move(tokenIds)};
}

__host__ __forceinline__
void printMetadata(const std::vector<int>& counts, const std::vector<flashmoe::TPS>& ids,
    const int& EC, const int& E) {
    const auto t0 = cute::make_tensor(counts.data(), cute::make_layout(
        cute::make_shape(E, 1), cute::LayoutRight{}));
    print_tensor(t0);
    void* p = std::malloc(ids.size() * sizeof(uint));
    auto* indices = static_cast<uint*>(p);
    std::ranges::transform(ids.begin(), ids.end(), indices, [](const flashmoe::TPS t) {
        return t.tokenIdx;
    });
    const auto t1 = cute::make_tensor(indices, cute::make_layout(
        cute::make_shape(E, EC), cute::LayoutRight{}));
    print_tensor(t1);
    auto* scales = static_cast<float*>(p);
    std::ranges::transform(ids.begin(), ids.end(), scales, [](const flashmoe::TPS t) {
        return t.probability;
    });
    const auto t2 = cute::make_tensor(scales, cute::make_layout(
        cute::make_shape(E, EC), cute::LayoutRight{}));
    print_tensor(t2);
    std::free(p);
}

template<int Arch, int H, int bM, int bN, typename Element>
__host__ __forceinline__
void kickStart(const int& S, const int& E, const int& k, const float& rtol, const float& atol) {
    const auto EC = getEC(S);
    const auto [counts, indices] = generate_token_ids_and_expert_counts(S, E, k);
    cudaSetDevice(0);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    Element* tokens = nullptr;
    flashmoe::TPS* tIds = nullptr;
    int* expertCounts = nullptr;
    Element* kut_result = nullptr;
    Element* ref_result = nullptr;
    float* oracleResult = nullptr;

    cudaMallocAsync(&tokens, sizeof(Element) * E * EC * H, stream);
    cudaMallocAsync(&tIds, sizeof(flashmoe::TPS) * E * EC, stream);
    cudaMallocAsync(&expertCounts, sizeof(int) * E, stream);
    cudaMallocAsync(&ref_result, sizeof(Element) * S * H, stream);
    cudaMemsetAsync(ref_result, 0, sizeof(Element) * S * H, stream);

    std::random_device rd;
    cudaMallocAsync(&kut_result, sizeof(Element) * S * H, stream);
    if (k > 1) {
        // needed since we accumulate into the buffer
        cudaMemsetAsync(kut_result, 0, sizeof(Element) * S * H, stream);
        cudaMallocAsync(&oracleResult, sizeof(float) * S * H, stream);
        cudaMemsetAsync(oracleResult, 0, sizeof(float) * S * H, stream);
    }
    CHECK_CUDA(cudaPeekAtLastError());
    // fill token array
    randUniform<Arch>(tokens, E * EC * H, rd(), -1.0f, 1.0f, stream);
    // copy expert counts
    cudaMemcpyAsync(expertCounts, counts.data(), sizeof(int) * E, cudaMemcpyHostToDevice, stream);
    // copy indices data structure
    cudaMemcpyAsync(tIds, indices.data(), sizeof(flashmoe::TPS) * E * EC, cudaMemcpyHostToDevice, stream);
    // call reference kernel
    using AccumType = Element;
    // 32 <= threads <= 1024
    constexpr int threads = cute::max(cute::min(H, 1024), 32);
    combineReference<AccumType><<<1, threads, 0, stream>>>(E, S, H, EC, k, tokens, tIds, expertCounts, ref_result, oracleResult);
    // call KUT
    constexpr int cThreads = cute::max(cute::min(bM, 128), 32);
    int bps = 0;
    int blocks = 0;
    if (k > 1) {
        auto kernel = combineKernel<Arch, bM, bN, cThreads, flashmoe::CombineMode::plural, Element>;
        CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kernel, cThreads, 0));
        blocks = cute::min(((E * EC) / bM) * (H / bN), bps * NUM_SMS);
        combineKernel<Arch, bM, bN, cThreads, flashmoe::CombineMode::plural>
        <<<blocks, cThreads, 0, stream>>>(EC, S, E, H, k,
            expertCounts, tokens, kut_result, tIds);
    }
    else {
        auto kernel = combineKernel<Arch, bM, bN, cThreads, flashmoe::CombineMode::single, Element>;
        CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kernel, cThreads, 0));
        blocks = cute::min(((E * EC) / bM) * (H / bN), bps * NUM_SMS);
        combineKernel<Arch, bM, bN, cThreads, flashmoe::CombineMode::single>
        <<<blocks, cThreads, 0, stream>>>(EC, S, E, H, k,
            expertCounts, tokens, kut_result, tIds);
    }
    // compare and report error
    using MatXType = MXE<Element>;
    auto tKUT = matx::make_tensor<MatXType>(reinterpret_cast<MatXType*>(kut_result), {S, H});
    auto tRef = matx::make_tensor<MatXType>(reinterpret_cast<MatXType*>(ref_result), {S, H});
    matx::cudaExecutor exec{stream};
    auto num_matches_ref = matx::make_tensor<long int>({});
    auto num_matches_oracle = matx::make_tensor<long int>({});
    (num_matches_ref = matx::sum(matx::isclose(tKUT, tRef, rtol, atol))).run(exec);
    if (k > 1) {
        auto tOracle = matx::make_tensor<float>(oracleResult, {S, H});
        (num_matches_oracle = matx::sum(matx::isclose(tKUT, tOracle, rtol, atol))).run(exec);
    }
    exec.sync();
    const auto error_ref = (1.0 - (static_cast<double>(num_matches_ref()) / static_cast<double>(tKUT.TotalSize())))*100.0;
    double error_oracle = -0.0;
    if (k > 1) {
        error_oracle = (1.0 - (static_cast<double>(num_matches_oracle()) / static_cast<double>(tKUT.TotalSize())))*100.0;
    }
    printf("%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %.1e, %.1e, %lf, %lf\n",
        S, H, E, k, bM, bN, cThreads, bps, NUM_SMS, blocks, rtol, atol, error_ref, error_oracle);
    CHECK_CUDA(cudaPeekAtLastError());
    cudaFreeAsync(tokens, stream);
    cudaFreeAsync(tIds, stream);
    cudaFreeAsync(expertCounts, stream);
    cudaFreeAsync(ref_result, stream);
    cudaFreeAsync(oracleResult, stream);
    cudaFreeAsync(kut_result, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    cudaStreamDestroy(stream);
}
// ./testCombine <S> <E> <topK> <atol> <rtol>
__host__ __forceinline__
void doTest(const int argc, char** argv) {
    int S = 8192;
    int E = 16;
    int k = 2;
    float rtol = 2e-2f;
    float atol = 2e-3f;
    if (argc > 1) {
        S = std::stoi(argv[1]);
    }
    if (argc > 2) {
        E = std::stoi(argv[2]);
    }
    if (argc > 3) {
        k = std::stoi(argv[3]);
    }
    if (argc > 4) {
        rtol = std::stof(argv[4]);
    }
    if (argc > 5) {
        atol = std::stof(argv[5]);
    }
    if (k > E) {
        throw std::runtime_error("k must be at most E");
    }
    if (!cutlass::ispow2(S)) {
        throw std::runtime_error("S must be a power of 2");
    }
    // fix H to minimize template instantiations.
    using Element = __half;
    constexpr int H = 4096;
    constexpr int bN = cute::min(H, 64 * (sizeof(Element) == 2 ? 2 : 1));
    static_assert(H % bN == 0);
    constexpr int Arch = FLASHMOE_ARCH; // compute capability
    printf("S,H,E,k,bM,bN,threads,blocks/SM,SMs,blocks,rtol,atol, error_ref(%%),error_oracle(%%)\n");
    switch (S) {
    case 1:
        kickStart<Arch, H, 1, bN, Element>(S, E, k, rtol, atol);
        break;
    case 2:
        kickStart<Arch, H, 2, bN, Element>(S, E, k, rtol, atol);
        break;
    case 4:
        kickStart<Arch, H, 4, bN, Element>(S, E, k, rtol, atol);
        break;
    case 8:
        kickStart<Arch, H, 8, bN, Element>(S, E, k, rtol, atol);
        break;
    case 16:
        kickStart<Arch, H, 16, bN, Element>(S, E, k, rtol, atol);
        break;
    case 32:
        kickStart<Arch, H, 32, bN, Element>(S, E, k, rtol, atol);
        break;
    case 64:
        kickStart<Arch, H, 64, bN, Element>(S, E, k, rtol, atol);
        break;
    default:
        if (S >= 128) {
            kickStart<Arch, H, 128, bN, Element>(S, E, k, rtol, atol);
        }
    }
}
int main(const int argc, char** argv) {
    doTest(argc, argv);
}