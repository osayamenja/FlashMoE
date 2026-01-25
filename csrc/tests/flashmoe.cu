/******************************************************************************
 * Copyright (c) 2026, Osayamen Jonathan Aimuyo.
 ******************************************************************************/
// E2E correctness test and benchmark of fused distributed MoE kernel
#include <random>
#include <stdexcept>
#include <tuple>

#include "../include/flashmoe/bootstrap.cuh"
#include "../include/flashmoe/gate.cuh"
#include "../include/flashmoe/moe.cuh"

#include "common.cuh"

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
  for (int tileIdx = blockIdx.x; tileIdx < nTiles; tileIdx += gridDim.x) {
    gemmMainloop<TileGEMM, Activation>(gemmWorkspace, a, b, c, bias, M, N, K, tileIdx);
  }
}

// note this is not an optimal implementation!
template<typename Element>
__global__ void gatherTokens(const flashmoe::TPS* __restrict__ tokenIds,
  const Element* __restrict__ src, Element* __restrict__ dst,
  const int* __restrict__ expertCounts, const __grid_constant__ uint EC, const __grid_constant__ uint S,
  const __grid_constant__ uint H) {
  // assert(n <= EC)
  const auto count = *expertCounts;
  const auto srcTensor = cute::make_tensor(cute::make_gmem_ptr(src),
    cute::make_layout(cute::make_shape(S, H), cute::LayoutRight{}));
  auto dstTensor = cute::make_tensor(cute::make_gmem_ptr(dst),
    cute::make_layout(cute::make_shape(EC, H), cute::LayoutRight{}));
  for (uint idx = blockIdx.x; idx < count; idx += gridDim.x) {
    const auto tIdx = tokenIds[idx].tokenIdx;
    for (uint i = threadIdx.x; i < H; i += blockDim.x) {
      dstTensor(idx, i) = srcTensor(tIdx, i);
    }
  }
}

struct SplitFunctor {
  __host__ __device__
  auto operator()(const flashmoe::TPS& t) const {
    return t.tokenIdx;
  }
};

template<typename Element>
__global__ void combineReference( const __grid_constant__ int S,
    const __grid_constant__ int H, const __grid_constant__ int EC, const __grid_constant__ int topK,
    const Element* __restrict__ tokens, // [EC, H]
    const flashmoe::TPS* __restrict__ tokenIds, //[EC] metadata
    const int* __restrict__ expertCounts, // [1]
    Element* __restrict__ result // [S, H]
    ) {
  const auto tIds = cute::make_tensor(cute::make_gmem_ptr(tokenIds),
      cute::make_layout(cute::make_shape(EC), cute::LayoutRight{}));
  const auto tokTensor = cute::make_tensor(cute::make_gmem_ptr(tokens),
      cute::make_layout(cute::make_shape(EC, H), cute::LayoutRight{}));
  auto resultTensor = cute::make_tensor(cute::make_gmem_ptr(result),
      cute::make_layout(cute::make_shape(S, H), cute::LayoutRight{}));
  const auto expertCount = *expertCounts;
  for (int j = blockIdx.x; j < expertCount; j += gridDim.x) {
    const auto tokenId = tIds(j);
    if (topK == 1) {
      for (auto k = threadIdx.x; k < H; k += blockDim.x) {
        resultTensor(tokenId.tokenIdx, k) = tokTensor(j, k);
      }
    }
    else {
      // a token is mapped to an expert at most once, so we can safely accumulate without atomics
      constexpr flashmoe::Converter<float, Element> loadOp{};
      constexpr flashmoe::Converter<Element, float> storeOp{};
      for (auto k = threadIdx.x; k < H; k += blockDim.x) {
        // read token value and convert to float.
        const auto v = loadOp(tokTensor(j, k));
        const float scaledV = v * tokenId.probability;
        // read current result value
        const auto rv = resultTensor(tokenId.tokenIdx, k);
        // store summed result
        resultTensor(tokenId.tokenIdx, k) = storeOp(scaledV) + rv;
      }
    }
  }
}
// single GPU E2E MoE
template<
  typename Config, int GEMM0SharedSize, int GEMM1SharedSize,
  typename Activation, typename AccumType, typename Element
>
__host__ __forceinline__
void reference(const flashmoe::TPS* __restrict__ const& tokenIds,
  Element* __restrict__ const& tokens,
  Element* __restrict__ const& ref_input,
  Element* __restrict__ const& expertUp, // [E, H, I]
  Element* __restrict__ const& expertDown, // [E, I, H]
  Element* __restrict__ const& biasUp, // [E, I]
  Element* __restrict__ const& biasDown, // [E, H]
  Element* __restrict__ const& ref_interim0,
  Element* __restrict__ const& ref_interim1,
  int* __restrict__ const& expertCounts,
  Element* __restrict__ const& ref_out,
  const uint& S, const uint& H, const uint& EC, const uint& E, const uint I, const int& topK,
  cudaStream_t stream) {
  constexpr int bM0 = cute::get<0>(typename Config::G0TS{});
  constexpr int bN0 = cute::get<1>(typename Config::G0TS{});
  constexpr int bK0 = cute::get<2>(typename Config::G0TS{});
  constexpr int pS0 = cute::get<3>(typename Config::G0TS{});

  constexpr int bM1 = cute::get<0>(typename Config::G1TS{});
  constexpr int bN1 = cute::get<1>(typename Config::G1TS{});
  constexpr int bK1 = cute::get<2>(typename Config::G1TS{});
  constexpr int pS1 = cute::get<3>(typename Config::G1TS{});
  static_assert(bM0 == bM1);
  if (EC % bM0 != 0) {
    throw std::runtime_error("EC should be a multiple of bM");
  }
  constexpr auto threads = Config::Threads::value;
  constexpr int arch = Config::Arch::value;
  using TileGEMM0 = flashmoe::tile::CollectiveMainloop<bM0, bN0, bK0, arch, Element, AccumType, threads, pS0>;
  using TileGEMM1 = flashmoe::tile::CollectiveMainloop<bM1, bN1, bK1, arch, Element, AccumType, threads, pS1>;

  auto kernel0 = gk<TileGEMM0, Activation, Element, Element>;
  // set shared memory
  CHECK_CUDA(cudaFuncSetAttribute(kernel0, cudaFuncAttributeMaxDynamicSharedMemorySize, GEMM0SharedSize));
  int bps = 0;
  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kernel0, threads, GEMM0SharedSize));
  const auto blocks0 = cute::min((EC / bM0) * (I / bN0), bps * NUM_SMS);
  auto kernel1 = gk<TileGEMM1, cublasdx::identity, Element, Element>;
  // set shared memory
  CHECK_CUDA(cudaFuncSetAttribute(kernel1, cudaFuncAttributeMaxDynamicSharedMemorySize, GEMM1SharedSize));
  // get blocks
  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kernel1, threads, GEMM1SharedSize));
  const auto blocks1 = cute::min((EC / bM0) * (H / bN1), bps * NUM_SMS);
  for (int i = 0; i < E; ++i) {
    // get the tokens routed to expert i
    auto* __restrict__ tIds = tokenIds + i * EC;
    gatherTokens<<<EC, threads, 0, stream>>>(tIds, tokens, ref_input, expertCounts + i, E, S, H);
    // now do GEMM0 + bias + act
    auto* __restrict__ expertU = expertUp + i * (static_cast<size_t>(H) * I);
    auto* __restrict__ biasU = biasUp + i * I;
    gk<TileGEMM0, Activation><<<blocks0, threads, GEMM0SharedSize, stream>>>
    (ref_input, expertU, ref_interim0, biasU, EC, I, H);
    // do GEMM 1 + bias
    auto* __restrict__ expertD = expertDown + i * (static_cast<size_t>(H) * I);
    auto* __restrict__ biasD = biasDown + i * H;
    gk<TileGEMM1, cublasdx::identity><<<blocks1, threads, GEMM1SharedSize, stream>>>
    (ref_interim0, expertD, ref_interim1, biasD, EC, H, I);
    // do combine
    combineReference<<<1, threads, 0, stream>>>(S, H, EC, topK, ref_interim1, tIds, expertCounts + i, ref_out);
  }
  /*{
    std::vector<Element> hr(S * H);
    cudaMemcpyAsync(hr.data(), ref_out, sizeof(Element) * S * H, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    const auto tr = cute::make_tensor(hr.data(), cute::make_layout(
      cute::make_shape(S, H), cute::LayoutRight{}));
    printf("Reference result\n");
    print_tensor(tr);
  }*/
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
// template<
//   int Arch,
//   int bM, int bN0, int bK0, int pSK0,
//   int bN1, int bK1, int bNGate, int pSK1,
//   bool isKG1,
//   flashmoe::Activation act,
//   flashmoe::Topology topo,
//   flashmoe::GateReductionLevel grl,
//   flashmoe::DropTokens dtk = flashmoe::DropTokens::no
// >
void kickstart(/*const uint S, const uint H, const uint I, const uint E, const uint k*/) {
  // debug mode for now
  constexpr int S = 16;
  constexpr int H = 16;
  constexpr int I = 16;
  constexpr int E = 2;
  constexpr int k = 1;
  constexpr int bNGate = cute::min(32, E);
  constexpr auto topo = flashmoe::Topology::NVLINK_ONLY;
  constexpr auto grl = E > bNGate ? flashmoe::GateReductionLevel::multiBlock :
  flashmoe::GateReductionLevel::singleBlock;
  constexpr auto sro = flashmoe::SoftMaxOptimizationLevel::highest;
  constexpr auto dtk = flashmoe::DropTokens::no;
  constexpr auto Arch = FLASHMOE_ARCH;
  constexpr auto act = flashmoe::Activation::relu;
  using Element = float;
  using AccumType = float;
  constexpr int bM = cute::min(S, 128);
  constexpr int bN0 = cute::min(I, 128);
  constexpr int bN1 = cute::min(H, 128);
  constexpr int bK0 = cute::min(H, 64);
  constexpr int bK1 = cute::min(I, 64);
  constexpr bool isKG1 = k > 1;
  constexpr int pSK0 = 1;
  constexpr int pSK1 = 1;

  nvshmem_init();
  const uint devId = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  CHECK_CUDA(cudaSetDevice(devId));
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  const auto world = nvshmem_n_pes();
  const auto epRank = nvshmem_my_pe();
  if (E % world != 0) {
    throw std::runtime_error("E should be a multiple of world");
  }
  const auto numLocalExperts = E / world;
  const auto tilesN1 = H / bN1;

  constexpr auto threadsGEMM0 = flashmoe::tile::suggest_thread_count<bM, bN0, bK0, Arch, Element, AccumType>();
  constexpr auto threadsGEMM1 = flashmoe::tile::suggest_thread_count<bM, bN1, bK1, Arch, Element, AccumType>();
  constexpr auto threads = cute::max(threadsGEMM0, threadsGEMM1);
  static_assert(dtk == flashmoe::DropTokens::no);
  const auto EC = dtk == flashmoe::DropTokens::no ? S : cute::ceil_div(S, E) * k;
  constexpr auto GEMM0Sz = cutlass::round_up(sizeof(Element) * bK0 * pSK0 * (bM + bN0), flashmoe::MAX_ALIGNMENT);
  constexpr auto GEMM1Sz = cutlass::round_up(sizeof(Element) * bK1 * pSK1 * (bM + bN1), flashmoe::MAX_ALIGNMENT);
  constexpr auto GateSz = cutlass::round_up(cute::max(sizeof(Element) * bK0 * pSK0 * (bM + bNGate),
    sizeof(flashmoe::gate::SoftType) * bM * bNGate), flashmoe::MAX_ALIGNMENT);
  const auto dispatchSz = E * (sizeof(flashmoe::PEL) + sizeof(int));
  const auto OSSz = flashmoe::os::getSharedSize<threads, bM>(world, numLocalExperts, E, EC, tilesN1);
  const auto taskSz = sizeof(flashmoe::Task) * tilesN1;
  const auto combineSz = cutlass::round_up(sizeof(Element) * bM * bN1, flashmoe::MAX_ALIGNMENT);
  const auto kernelSz = cute::max(GEMM0Sz, GEMM1Sz, dispatchSz, OSSz, taskSz, combineSz);
  int maxSharedMemory = 0;
  CHECK_CUDA(cudaDeviceGetAttribute(&maxSharedMemory,cudaDevAttrMaxSharedMemoryPerBlock, devId));
  if (kernelSz > maxSharedMemory) {
    throw std::runtime_error("Required shared memory exceeds hardware limits. Reduce tile shapes or input sizes.");
  }
  // [S, H] x [H, E] -> [S, E]
  using GateTile = cute::Shape<cute::Int<bM>, cute::Int<bNGate>, cute::Int<bK0>, cute::Int<pSK0>>;
  // [S, H] x [H, I] -> [S, I]
  using GEMM0Tile = cute::Shape<cute::Int<bM>, cute::Int<bN0>, cute::Int<bK0>, cute::Int<pSK0>>;
  // [S, I] x [I, H] -> [S, H]
  using GEMM1Tile = cute::Shape<cute::Int<bM>, cute::Int<bN1>, cute::Int<bK1>, cute::Int<pSK1>>;
  using Config = flashmoe::moe::MoEConfig<Element, Arch, threads, isKG1, dtk, GEMM0Tile, GEMM1Tile>;

  auto kernel = flashmoe::moe::forward<Config, act, topo>;
  CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kernelSz));
  int bps = 0;
  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kernel, threads, kernelSz));
  const auto dispatchBlocks = cute::ceil_div(128, cute::max(E, 4));
  const auto processorBlocks = (S /bM) * cute::max(I / bN0, H / bN1);
  const uint blocks = cute::max(cute::min(cute::max(processorBlocks, dispatchBlocks) + 1, bps * NUM_SMS), 2);

  int bps1 = 0;
  constexpr auto gateThreads = flashmoe::tile::suggest_thread_count<bM, bNGate, bK0, Arch, Element, AccumType>();
  auto gateKernel = flashmoe::gate::forwardKernel<GateTile, Arch, gateThreads, grl, sro,AccumType, Element, Element>;
  CHECK_CUDA(cudaFuncSetAttribute(gateKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, GateSz));
  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps1, gateKernel, gateThreads, GateSz));
  const auto gateBlocks = cute::min((S / bM) * (E / bNGate), bps1 * NUM_SMS);

  Element* tokens = nullptr;
  cudaMallocAsync(&tokens, sizeof(Element) * S * H, stream);
  std::vector<float> th(S*H);
  auto tokTensor = cute::make_tensor(th.data(),
    cute::make_layout(cute::make_shape(S, H), cute::LayoutRight{}));
  for (int i = 0; i < S; ++i) {
    for (int j = 0; j < H; ++j) {
      tokTensor(i, j) = static_cast<float>(i + j);
    }
  }
  cudaMemcpyAsync(tokens, th.data(), sizeof(Element) * th.size(), cudaMemcpyHostToDevice, stream);

  Element* gateWeights = nullptr;
  std::random_device rd;
  cudaMallocAsync(&gateWeights, sizeof(Element) * H * E, stream);
  randUniform<Arch>(gateWeights, H * E, rd(), -1.0f, 1.0f, stream);

  Element* expertUpWeights = nullptr;
  Element* expertDownWeights = nullptr;
  // allocate all expert weights since we need it for single-GPU correctness checks
  cudaMallocAsync(&expertUpWeights, E * sizeof(Element) * H * I, stream);
  cudaMallocAsync(&expertDownWeights, E * sizeof(Element) * I * H, stream);
  matx::cudaExecutor exec{stream};
  using MT = MXE<Element>;
  const auto eWz = static_cast<size_t>(H) * I;
  for (int i = 0; i < E; ++i) {
    auto euw = matx::make_tensor<MT>(reinterpret_cast<MT*>(expertUpWeights + i * eWz), {H, I});
    (euw = matx::eye<Element>({H, I})).run(exec);
    auto edw = matx::make_tensor<MT>(reinterpret_cast<MT*>(expertDownWeights + i * eWz), {I, H});
    (edw = matx::eye<Element>({I, H})).run(exec);
  }

  Element* biasUp = nullptr;
  cudaMallocAsync(&biasUp, sizeof(Element) * E * I, stream);
  auto bu = matx::make_tensor<MT>(reinterpret_cast<MT*>(biasUp), {numLocalExperts, I});
  (bu = matx::zeros<MT>({numLocalExperts, I})).run(exec);

  Element* biasDown = nullptr;
  cudaMallocAsync(&biasDown, sizeof(Element) * E * H, stream);
  auto bd = matx::make_tensor<MT>(reinterpret_cast<MT*>(biasDown), {numLocalExperts, H});
  (bd = matx::zeros<MT>({numLocalExperts, H})).run(exec);

  Element* gateOut = nullptr;
  cudaMallocAsync(&gateOut, sizeof(Element) * S * E, stream);

  int* expertCounts = nullptr;
  cudaMallocAsync(&expertCounts, sizeof(int) * E, stream);

  Element* moeOut = nullptr;
  cudaMallocAsync(&moeOut, sizeof(Element) * S * H, stream);

  Element* referenceInput;
  cudaMallocAsync(&referenceInput, sizeof(Element) * EC * H, stream);
  Element* referenceInterim0;
  cudaMallocAsync(&referenceInterim0, sizeof(Element) * EC * I, stream);
  Element* referenceInterim1;
  cudaMallocAsync(&referenceInterim1, sizeof(Element) * EC * H, stream);
  Element* referenceOut;
  cudaMallocAsync(&referenceOut, sizeof(Element) * S * H, stream);
  if (k > 1) {
    cudaMemsetAsync(referenceOut, 0, sizeof(Element) * S * H, stream);
  }

  // initialize
  flashmoe::MoEArgs args{
    sizeof(Element), S, H, I, EC, bM, bN0, bN1, bK0, bK1, threads,
    blocks, static_cast<uint16_t>(epRank), static_cast<uint16_t>(world),
    static_cast<uint16_t>(nvshmem_my_pe()), E, static_cast<uint16_t>(numLocalExperts)
  };
  // blocked partitioning
  // for 8 experts and 4 ranks
  // rank 0 gets [E0, E1], rank 1 gets [E2, E3] and so on
  std::vector<uint> expertToEpRank(E);
  for (int i = 0; i < world; ++i) {
    for (int j = 0; j < numLocalExperts; ++j) {
      expertToEpRank[i * numLocalExperts + j] = i;
    }
  }
  // world == ePworld for this instance
  std::vector<int> epRankToGlobalRank(world);
  for (int i = 0; i < world; ++i) {
    epRankToGlobalRank[i] = i;
  }
  const auto contexts = flashmoe::initialize(args, bNGate,
    expertToEpRank.data(), epRankToGlobalRank.data(), stream);
  auto moeContext = std::get<0>(contexts);
  auto gateCtx = std::get<1>(contexts);
  // call kernel as many times as needed
  auto* __restrict__ localExpertUpWeights = expertUpWeights + (static_cast<size_t>(epRank) * (H * I));
  auto* __restrict__ localExpertDownWeights = expertDownWeights + (static_cast<size_t>(epRank) * (I * H));
  auto* __restrict__ localBiasUp = biasUp + (static_cast<size_t>(epRank) * (I));
  auto* __restrict__ localBiasDown = biasDown + (static_cast<size_t>(epRank) * (H));

  const flashmoe::moe::KernelArgs kArgs{
    reinterpret_cast<const cuda::std::byte*>(tokens),
    reinterpret_cast<const cuda::std::byte*>(gateWeights),
    reinterpret_cast<const cuda::std::byte*>(localExpertUpWeights),
    reinterpret_cast<const cuda::std::byte*>(localBiasUp),
    reinterpret_cast<const cuda::std::byte*>(localExpertDownWeights),
    reinterpret_cast<const cuda::std::byte*>(localBiasDown),
    reinterpret_cast<cuda::std::byte*>(gateOut),
    expertCounts, reinterpret_cast<cuda::std::byte*>(moeOut),
    S, H, I, E, k, EC
  };

  // run gate to populate data structures
  flashmoe::gate::forwardKernel<GateTile, Arch, gateThreads, grl, sro, AccumType>
  <<<gateBlocks, gateThreads, GateSz, stream>>>(tokens, gateWeights, gateOut,
    expertCounts, S, H, E, k, EC, moeContext.tokenIndices, gateCtx.ecGuards, gateCtx.ssp, gateCtx.rtp);
  /*{
    std::vector<int> counts(E);
    std::vector<flashmoe::TPS> ids(EC * E);
    cudaMemcpyAsync(counts.data(), expertCounts, sizeof(int) * E, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(ids.data(), moeContext.tokenIndices, sizeof(flashmoe::TPS) * E * EC, cudaMemcpyDeviceToHost, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    printMetadata(counts, ids, EC, E);
  }*/
  flashmoe::moe::forwardHost<Config, topo, act>(kArgs, moeContext, kernelSz, stream);
  CHECK_CUDA(cudaPeekAtLastError());
  // check correctness
  using ActType = flashmoe::ActivationType<Element, act>::AT;
  reference<Config, GEMM0Sz, GEMM1Sz, ActType, AccumType>(moeContext.tokenIndices, tokens,
    referenceInput, expertUpWeights, expertDownWeights, biasUp, biasDown,
    referenceInterim0, referenceInterim1, expertCounts, referenceOut,
    S, H, EC, E, I, k, stream);
  CHECK_CUDA(cudaPeekAtLastError());
  // calculate error percentage
  constexpr auto rtol = 2e-2;
  constexpr auto atol = 2e-3;
  auto tC = matx::make_tensor<MT>(reinterpret_cast<MT*>(moeOut), {S, H});
  /*{
    std::vector<Element> hr(S * H);
    cudaMemcpyAsync(hr.data(), moeOut, sizeof(Element) * S * H, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    const auto tr = cute::make_tensor(hr.data(), cute::make_layout(
      cute::make_shape(S, H), cute::LayoutRight{}));
    printf("FlashMoE result\n");
    print_tensor(tr);
  }*/
  auto tRef = matx::make_tensor<MT>(reinterpret_cast<MT*>(referenceOut), {S, H});
  auto num_matches = matx::make_tensor<long int>({});
  (num_matches = matx::sum(matx::isclose(tC, tRef, rtol, atol))).run(exec);
  exec.sync();
  // calculate error percentage
  const auto ep =  (1.0 - (static_cast<double>(num_matches()) / static_cast<double>(tC.TotalSize()))) * 100;
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  // benchmark distributed moe fused kernel
  constexpr int warmup = 128;
  for (int i = 0; i < warmup; ++i) {
    flashmoe::moe::forwardHost<Config, topo, act>(kArgs, moeContext, kernelSz, stream);
  }
  constexpr int runs = 128;
  CHECK_CUDA(cudaStreamSynchronize(stream));
  cudaEventRecord(start, exec.getStream());
  for (int i = 0; i < runs; ++i) {
    flashmoe::moe::forwardHost<Config, topo, act>(kArgs, moeContext, kernelSz, stream);
  }
  cudaEventRecord(stop, exec.getStream());
  CHECK_CUDA(cudaEventSynchronize(stop));
  float m_ms = 0;
  CHECK_CUDA(cudaEventElapsedTime(&m_ms, start, stop));
  const float m_time_ms = m_ms / static_cast<float>(runs);
  if (nvshmem_my_pe() == 0) {
    printf("S, H, I, E, k, EC, bM, bN0, bK0, bN1, bK1, bNGate, threads, blocks/SM, SMs,blocks, rtol, atol, error(%%), "
         "Kernel_Time(ms)\n");
    printf("%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %.1e, %.1e, %f, %f\n",
      S, H, I, E, k, EC, bM, bN0, bK0, bN1, bK1, bNGate, threads, bps, NUM_SMS, blocks, rtol, atol, ep, m_time_ms);
  }

  // finalize
  flashmoe::finalize(moeContext, gateCtx, stream);
  cudaFreeAsync(tokens, stream);
  cudaFreeAsync(gateWeights, stream);
  cudaFreeAsync(expertUpWeights, stream);
  cudaFreeAsync(expertDownWeights, stream);
  cudaFreeAsync(biasUp, stream);
  cudaFreeAsync(biasDown, stream);
  cudaFreeAsync(gateOut, stream);
  cudaFreeAsync(expertCounts, stream);
  cudaFreeAsync(moeOut, stream);
  cudaFreeAsync(referenceInput, stream);
  cudaFreeAsync(referenceInterim0, stream);
  cudaFreeAsync(referenceInterim1, stream);
  cudaFreeAsync(referenceOut, stream);
  CHECK_CUDA(cudaPeekAtLastError());
  nvshmem_finalize();
  CHECK_CUDA(cudaStreamSynchronize(stream));
  cudaStreamDestroy(stream);
}

/*__host__ __forceinline__
void drive(const int argc, char** argv) {
  uint S = 16;
  uint H = 128;
  uint I = 128;
  uint k = 2;
  if (argc > 1) {
    S = std::stoi(argv[1]);
  }
  if (argc > 2) {
    H = std::stoi(argv[2]);
  }
  if (argc > 3) {
    I = std::stoi(argv[3]);
  }
  if (argc > 4) {
    k = std::stoi(argv[4]);
  }
  // below values are static to minimize instantiated templates.
  constexpr int E = 16;
  constexpr int bNGate = cute::min(32, E);
  constexpr auto topo = flashmoe::Topology::NVLINK_ONLY;
  constexpr auto grl = E > bNGate ? flashmoe::GateReductionLevel::multiBlock :
  flashmoe::GateReductionLevel::singleBlock;
  constexpr auto dtk = flashmoe::DropTokens::no;
  constexpr auto arch = FLASHMOE_ARCH;
  constexpr auto act = flashmoe::Activation::relu;
  if (k > E) {
    throw std::runtime_error("k must be <= E");
  }
  // to minimize instantiated templates
  if (I < 128) {
    throw std::runtime_error("I must be >= 128 for this bench");
  }
  if (H < 128) {
    throw std::runtime_error("H must be >= 128 for this bench");
  }

  switch (S) {
  case 1:
    break;
  case 2:
    break;
  case 4:
    break;
  case 8:
    break;
  case 16:
    break;
  case 32:
    break;
  case 64:
    break;
  default:
    break;
  }
}*/
//./testFlashMoE <S> <H> <I> <k>
int main(/*const int argc, char** argv*/) {
  kickstart();
}
