/******************************************************************************
 * Copyright (c) 2026, Osayamen Jonathan Aimuyo.
 ******************************************************************************/
// E2E correctness test and benchmark of fused distributed MoE kernel
#include <charconv>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>
#include <random> // for seeds
#include <stdexcept>

#include <mpi.h>

#include "../include/flashmoe/bootstrap.cuh"
#include "../include/flashmoe/moe.cuh"

#include "common.cuh"

template<typename TileGEMM, typename Activation, flashmoe::MLPMatmulType mt, typename ElementC, typename Element>
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
  if constexpr (mt == flashmoe::MLPMatmulType::gated) {
    __syncthreads();
    auto* __restrict__ gateCache = workspace + cutlass::round_up(cute::max(TileGEMM::SharedSizeC::value,
        TileGEMM::SharedSizeAB::value), TileGEMM::GeneralAlignment::value);
    cute::for_each(cute::make_int_sequence<accum_size>{}, [&](auto i) {
      const auto g = (c_frag(i) + loadConv(d_frag(i))) * swishBeta;
      d_frag(i) = storeConv(swishAlpha * act(g));
    });
    // rmem -> smem, cache gate results
    // holding in registers otherwise would blow up pressure
    auto sGate = cublasdx::make_tensor(reinterpret_cast<ElementC*>(gateCache), BLAS::suggest_layout_smem_c());
    cublasdx::copy_fragment<cublasdx::alignment_of<BLAS>::c>(d_frag, sGate, accumulator);
    // now, compute v tile
    tileMainloop(workspace, a, bV, accumulator, M, N, K, tileCoord);
    auto cv_frag = accumulator.get_results();
    const auto gV = flashmoe::tile::getBias<BM{}, BN{}>(biasV, M, N, cute::select<0, 1>(tileCoord));
    cublasdx::copy_fragment<cublasdx::alignment_of<BLAS>::c>(gV, d_frag, accumulator);
    cute::for_each(cute::make_int_sequence<accum_size>{}, [&](auto i) {
      // x = (a @ bV) + biasV
      cv_frag(i) = cv_frag(i) + loadConv(d_frag(i));
    });
    // smem -> rmem, load g
    __syncthreads();
    cublasdx::copy_fragment<cublasdx::alignment_of<BLAS>::c>(sGate, d_frag, accumulator);
    cute::for_each(cute::make_int_sequence<accum_size>{}, [&](auto i) {
      // y = x * (act(a @ b))
      d_frag(i) = storeConv(cv_frag(i) * loadConv(d_frag(i)));
    });
  }
  else {
    cute::for_each(cute::make_int_sequence<accum_size>{}, [&c_frag, &d_frag](auto i) {
      d_frag(i) = storeConv(act(c_frag(i) + loadConv(d_frag(i))));
    });
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

template<typename TileGEMM, typename Activation, flashmoe::MLPMatmulType mt, typename Element, typename ElementC>
requires(cublasdx::is_blas_execution_v<typename TileGEMM::BLAS>)
__launch_bounds__(TileGEMM::BLAS::max_threads_per_block, 1)
__global__ void gk(const Element* __restrict__ a, const Element* __restrict__ b,
  const Element* __restrict__ bV, ElementC* __restrict__ c, const ElementC* __restrict__ bias,
  const ElementC* __restrict__ biasV, const __grid_constant__ typename TileGEMM::AccumType swishAlpha,
  const __grid_constant__ typename TileGEMM::AccumType swishBeta,
  const __grid_constant__ int M, const __grid_constant__ int N,
  const int __grid_constant__ K) {
  using BLAS = TileGEMM::BLAS;
  constexpr int bM = cublasdx::size_of<BLAS>::m;
  constexpr int bN = cublasdx::size_of<BLAS>::n;
  const int nTiles = (M / bM) * (N / bN);
  extern __shared__ __align__(TileGEMM::GeneralAlignment::value) cuda::std::byte gemmWorkspace[];
  for (int tileIdx = blockIdx.x; tileIdx < nTiles; tileIdx += gridDim.x) {
    gemmMainloop<TileGEMM, Activation, mt>(gemmWorkspace, a, b, bV, c, bias, biasV, swishAlpha, swishBeta,
      M, N, K, tileIdx);
  }
}

// note this is not an optimal implementation!
template<typename Element>
__global__ void gatherTokens(const flashmoe::TPS* __restrict__ tokenIds,
  const Element* __restrict__ src, Element* __restrict__ dst,
  const __grid_constant__ uint roundEC,
  const __grid_constant__ int count,
  const __grid_constant__ uint S,
  const __grid_constant__ uint H) {
  const auto srcTensor = cute::make_tensor(cute::make_gmem_ptr(src),
    cute::make_layout(cute::make_shape(S, H), cute::LayoutRight{}));
  auto dstTensor = cute::make_tensor(cute::make_gmem_ptr(dst),
    cute::make_layout(cute::make_shape(roundEC, H), cute::LayoutRight{}));
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
    const __grid_constant__ int H, const __grid_constant__ int roundEC,
    const __grid_constant__ int expertCount,
    const __grid_constant__ int topK,
    const Element* __restrict__ tokens, // [roundEC, H]
    const flashmoe::TPS* __restrict__ tokenIds, //[roundEC] metadata
    Element* __restrict__ result // [S, H]
    ) {
  const auto tIds = cute::make_tensor(cute::make_gmem_ptr(tokenIds),
      cute::make_layout(cute::make_shape(roundEC), cute::LayoutRight{}));
  const auto tokTensor = cute::make_tensor(cute::make_gmem_ptr(tokens),
      cute::make_layout(cute::make_shape(roundEC, H), cute::LayoutRight{}));
  auto resultTensor = cute::make_tensor(cute::make_gmem_ptr(result),
      cute::make_layout(cute::make_shape(S, H), cute::LayoutRight{}));
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
  typename Config, flashmoe::MLPMatmulType mt, int GEMM0SharedSize, int GEMM1SharedSize,
  typename Activation, typename AccumType, typename Element
>
__host__ __forceinline__
void reference(const flashmoe::TPS* __restrict__ const& tokenIds,
  Element* __restrict__ const& tokens,
  Element* __restrict__ const& ref_input,
  Element* __restrict__ const& expertUp, // [E, H, I]
  Element* __restrict__ const& expertUpV, // [E, H, I]
  Element* __restrict__ const& expertDown, // [E, I, H]
  Element* __restrict__ const& biasUp, // [E, I]
  Element* __restrict__ const& biasUpV,
  Element* __restrict__ const& biasDown, // [E, H]
  Element* __restrict__ const& ref_interim0,
  Element* __restrict__ const& ref_interim1,
  int* __restrict__ const& expertCounts,
  Element* __restrict__ const& ref_out,
  const AccumType& swishAlpha, const AccumType& swishBeta,
  const size_t& S, const size_t& H, const size_t& EC, const size_t& E, const size_t& I, const size_t& topK,
  const int& num_sms,
  cudaStream_t stream) {
#if defined(FLASHMOE_NVTX) && FLASHMOE_NVTX
  const flashmoe::flashmoeRange range{"Reference"};
#endif
  constexpr int bM0 = cute::get<0>(typename Config::G0TS{});
  constexpr int bN0 = cute::get<1>(typename Config::G0TS{});
  constexpr int bK0 = cute::get<2>(typename Config::G0TS{});
  constexpr int pS0 = cute::get<3>(typename Config::G0TS{});

  constexpr int bM1 = cute::get<0>(typename Config::G1TS{});
  constexpr int bN1 = cute::get<1>(typename Config::G1TS{});
  constexpr int bK1 = cute::get<2>(typename Config::G1TS{});
  constexpr int pS1 = cute::get<3>(typename Config::G1TS{});
  static_assert(bM0 == bM1);
  const auto roundEC = cute::ceil_div(EC, bM0) * bM0;
  constexpr auto threads = Config::Threads::value;
  constexpr int arch = Config::Arch::value;
  using TileGEMM0 = flashmoe::tile::CollectiveMainloop<bM0, bN0, bK0, arch, Element, AccumType, threads, pS0>;
  using TileGEMM1 = flashmoe::tile::CollectiveMainloop<bM1, bN1, bK1, arch, Element, AccumType, threads, pS1>;

  auto kernel0 = gk<TileGEMM0, Activation, mt, Element, Element>;
  // set shared memory
  CHECK_CUDA(cudaFuncSetAttribute(kernel0, cudaFuncAttributeMaxDynamicSharedMemorySize, GEMM0SharedSize));
  int bps = 0;
  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kernel0, threads, GEMM0SharedSize));
  const auto blocks0 = cute::min((roundEC / bM0) * (I / bN0), bps * num_sms);
  auto kernel1 = gk<TileGEMM1, cublasdx::identity, flashmoe::MLPMatmulType::vanilla, Element, Element>;
  // set shared memory
  CHECK_CUDA(cudaFuncSetAttribute(kernel1, cudaFuncAttributeMaxDynamicSharedMemorySize, GEMM1SharedSize));
  // get blocks
  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kernel1, threads, GEMM1SharedSize));
  const auto blocks1 = cute::min((roundEC / bM0) * (H / bN1), bps * num_sms);
  std::vector<int> hCounts (E);
  cudaMemcpyAsync(hCounts.data(), expertCounts, sizeof(int) * E, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  constexpr auto nonAlpha = static_cast<AccumType>(1.f);
  constexpr auto nonBeta = static_cast<AccumType>(1.f);
  Element* nonV = nullptr;
  for (int i = 0; i < E; ++i) {
    const auto count = cute::min(hCounts[i], EC);
    if (count > 0) {
      // get the tokens routed to expert i
      auto* __restrict__ tIds = tokenIds + i * roundEC;
      gatherTokens<<<count, threads, 0, stream>>>(tIds, tokens, ref_input, roundEC, count, S, H);
      // now do GEMM0 + bias + act
      auto* __restrict__ expertU = expertUp + i * (static_cast<size_t>(H) * I);
      auto* __restrict__ expertUV = (mt == flashmoe::MLPMatmulType::gated ?
        expertUpV + i * (static_cast<size_t>(H) * I) : nullptr);
      auto* __restrict__ biasU = biasUp + i * I;
      auto* __restrict__ biasUV = (mt == flashmoe::MLPMatmulType::gated ? biasUpV + i * I : nullptr);
      gk<TileGEMM0, Activation, Config::MT::value><<<blocks0, threads, GEMM0SharedSize, stream>>>
      (ref_input, expertU, expertUV, ref_interim0, biasU, biasUV, swishAlpha, swishBeta,
        roundEC, I, H);
      // do GEMM 1 + bias
      auto* __restrict__ expertD = expertDown + i * (static_cast<size_t>(H) * I);
      auto* __restrict__ biasD = biasDown + i * H;
      gk<TileGEMM1, cublasdx::identity, flashmoe::MLPMatmulType::vanilla><<<blocks1, threads, GEMM1SharedSize, stream>>>
      (ref_interim0, expertD, nonV, ref_interim1, biasD, nonV,
        nonAlpha, nonBeta, roundEC, H, I);
      // do combine
      combineReference<<<count, threads, 0, stream>>>(S, H, roundEC, count, topK, ref_interim1, tIds, ref_out);
    }
  }
}

struct Seeds {
  using SeedType = uint64_t;
  SeedType expertUp;
  SeedType expertUpV;
  SeedType expertDown;
  SeedType biasUp;
  SeedType biasUpV;
  SeedType biasDown;
  SeedType alpha;
  SeedType beta;
};

template <typename Element>
constexpr const char* element_string() {
  static_assert(
    cuda::std::is_same_v<Element, __half> ||
    cuda::std::is_same_v<Element, __nv_bfloat16> ||
    cuda::std::is_same_v<Element, float> ||
    cuda::std::is_same_v<Element, double>,
    "Unsupported Element type"
  );
  if constexpr (cuda::std::is_same_v<Element, double>) return "fp64";
  else if constexpr (cuda::std::is_same_v<Element, float>) return "fp32";
  else if constexpr (cuda::std::is_same_v<Element, __half>) return "fp16";
  else return "bf16";
}
template<
  typename Element, typename AccumType,
  int Arch,
  int bM, int bN0, int bK0, int pSK0,
  int bN1, int bK1, int pSK1,
  flashmoe::CombineMode cm,
  flashmoe::Activation act,
  flashmoe::MLPMatmulType mt
>
void kickstart(const size_t& S, const size_t& H, const size_t& I, const size_t& E, const size_t& EC, const size_t& k,
  const uint warmup, const uint runs, const float& rtol, const float& atol) {

  nvshmem_init();
  const uint devId = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  CHECK_CUDA(cudaSetDevice(devId));
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  int archMajor = 0;
  CHECK_CUDA(cudaDeviceGetAttribute(&archMajor,cudaDevAttrComputeCapabilityMajor, devId));
  int archMinor = 0;
  CHECK_CUDA(cudaDeviceGetAttribute(&archMinor,cudaDevAttrComputeCapabilityMinor, devId));
  const auto actualArch = archMajor * 100 + archMinor * 10;
  if (Arch > actualArch) {
    throw std::invalid_argument("Invalid Arch");
  }
  int num_sms = 0;
  CHECK_CUDA(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, devId));

  const auto world = nvshmem_n_pes(); // this is not a requrement
  const auto epRank = nvshmem_my_pe(); // this is not a requirement
  if (epRank == 0) {
    printf("EP Rank, dtype, S, H, I, E, k, EC, bM, bN0, bK0, bN1, bK1, threads, blocks/SM, SMs,blocks, rtol, atol, "
           "error(%%), warmup, runs, workspace_bytes(MiB), "
           "FlashMoE_Time(ms)\n");
  }
  nvshmem_sync_all(); // this is needed to force initialization of NVSHMEM state.

  if (E % world != 0) {
    throw std::runtime_error("E should be a multiple of world");
  }
  if (k > E) {
    throw std::invalid_argument("k must be at most number of experts");
  }
  const auto numLocalExperts = E / world;
  const auto tilesN1 = H / bN1;
  const auto tilesK0 = H / bK0;
  const auto tilesK1 = I / bK1;
  if (pSK0 > tilesK0) {
    throw std::invalid_argument("pSK0 must be <= tilesK0");
  }
  if (pSK1 > tilesK1) {
    throw std::invalid_argument("pSK1 must be <= tilesK1");
  }

  constexpr auto threadsGEMM0 = flashmoe::tile::suggest_thread_count<bM, bN0, bK0, Arch, Element, AccumType>();
  constexpr auto threadsGEMM1 = flashmoe::tile::suggest_thread_count<bM, bN1, bK1, Arch, Element, AccumType>();
  constexpr auto threads = cute::max(threadsGEMM0, threadsGEMM1, 64);
  const auto roundEC = cute::ceil_div(EC, bM) * bM;
  constexpr auto GEMM0Sz = cutlass::round_up(cute::max(sizeof(Element) * bK0 * pSK0 * (bM + bN0),
    sizeof(Element) * bM * bN0) + (mt == flashmoe::MLPMatmulType::gated ? sizeof(Element) * bM * bN0 : 0),
    flashmoe::MAX_ALIGNMENT);
  constexpr auto GEMM1Sz = cutlass::round_up(cute::max(sizeof(Element) * bK1 * pSK1 * (bM + bN1),
    sizeof(Element) * bM * bN1), flashmoe::MAX_ALIGNMENT);
  const auto dispatchSz = E * (sizeof(flashmoe::PEL) + sizeof(int));
  const auto OSSz = flashmoe::os::getSharedSize<threads, bM>(world, numLocalExperts, E, EC, tilesN1);
  const auto taskSz = sizeof(flashmoe::Task) * tilesN1;
  constexpr auto combineSz = cutlass::round_up(sizeof(Element) * bM * bN1, flashmoe::MAX_ALIGNMENT);
  const auto kernelSz = cute::max(GEMM0Sz, GEMM1Sz, dispatchSz, OSSz, taskSz, combineSz);
  int maxSharedMemory = 0;
  CHECK_CUDA(cudaDeviceGetAttribute(&maxSharedMemory,cudaDevAttrMaxSharedMemoryPerBlockOptin, devId));
  if (kernelSz > maxSharedMemory) {
    const auto errmsg = std::string("Required shared memory ").append(std::to_string(kernelSz))
    .append(" exceeds hardware limits: ").append(std::to_string(maxSharedMemory)).append(" Reduce tile shapes or input sizes.");
    throw std::runtime_error(errmsg);
  }
  // [S, H] x [H, I] -> [S, I]
  using GEMM0Tile = cute::Shape<cute::Int<bM>, cute::Int<bN0>, cute::Int<bK0>, cute::Int<pSK0>>;
  // [S, I] x [I, H] -> [S, H]
  using GEMM1Tile = cute::Shape<cute::Int<bM>, cute::Int<bN1>, cute::Int<bK1>, cute::Int<pSK1>>;
  using Config = flashmoe::moe::MoEConfig<Element, Arch, threads, cm, mt, GEMM0Tile, GEMM1Tile>;

  int bps = 0;
  const auto kernelTopo = flashmoe::detectTopo();
  if (kernelTopo == flashmoe::Topology::NVLINK_ONLY) {
    auto kernel = flashmoe::moe::forward<Config, act, flashmoe::Topology::NVLINK_ONLY>;
    CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kernelSz));
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kernel, threads, kernelSz));
  }
  else {
    auto kernel = flashmoe::moe::forward<Config, act, flashmoe::Topology::MIXED>;
    CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kernelSz));
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kernel, threads, kernelSz));
  }
  const auto processorBlocks = ((S /bM) * ((I / bN0) + (H / bN1))) + ((S/bM) * (H / bN1));
  const auto dispatchBlocks = flashmoe::moe::dispatchSuperBlockSize(E) * E;
  const uint blocks = cute::max(cute::min(cute::max(processorBlocks, dispatchBlocks) + 1, bps * num_sms), 2);

  constexpr float minv = -1.0f;
  constexpr float maxv = 1.0f;
  std::random_device rd;
  Element* tokens = nullptr;
  CHECK_CUDA(cudaMallocAsync(&tokens, sizeof(Element) * S * H, stream));
  randUniform<Arch>(tokens, static_cast<size_t>(S) * H, rd(), minv, maxv, stream);

  Seeds seeds{};
  if (epRank == 0) {
    seeds.expertUp = rd();
    seeds.expertUpV = rd();
    seeds.expertDown = rd();
    seeds.biasUp = rd();
    seeds.biasUpV = rd();
    seeds.biasDown = rd();
    seeds.alpha = rd();
    seeds.beta = rd();
  }
  if (world > 1) {
    // Rank 0 will generate a random seed and propagate to every rank
    // the weights need to be uniform across all ranks as each rank will do an independent check
    // NVSHMEM init would have initialized MPI already
    int isMPIInitialized = 0;
    MPI_Initialized(&isMPIInitialized);
    if (!isMPIInitialized) {
      throw std::runtime_error("MPI is not initialized!");
    }
    constexpr int count = sizeof(Seeds) / sizeof(Seeds::SeedType);
    static_assert(cuda::std::is_same_v<Seeds::SeedType, uint64_t>);
    MPI_Bcast(&seeds, count, MPI_UINT64_T, 0, MPI_COMM_WORLD);
  }

  Element* expertUpWeights = nullptr;
  Element* expertUpV = nullptr;
  Element* expertDownWeights = nullptr;
  // allocate all expert weights since we need it for single-GPU correctness checks
  CHECK_CUDA(cudaMallocAsync(&expertUpWeights, sizeof(Element) * E * H * I, stream));
  randUniform<Arch>(expertUpWeights, static_cast<size_t>(E) * H * I, seeds.expertUp, minv, maxv, stream);
  if (mt == flashmoe::MLPMatmulType::gated) {
    CHECK_CUDA(cudaMallocAsync(&expertUpV, sizeof(Element) * E * H * I, stream));
    randUniform<Arch>(expertUpV, static_cast<size_t>(E) * H * I, seeds.expertUpV, minv, maxv, stream);
  }
  CHECK_CUDA(cudaMallocAsync(&expertDownWeights, sizeof(Element) * E * I * H, stream));
  randUniform<Arch>(expertDownWeights, static_cast<size_t>(E) * H * I, seeds.expertDown, minv, maxv, stream);
  using MT = MXE<Element>;

  Element* biasUp = nullptr;
  Element* biasUpV = nullptr;
  CHECK_CUDA(cudaMallocAsync(&biasUp, sizeof(Element) * E * I, stream));
  randUniform<Arch>(biasUp, static_cast<size_t>(E) * I, seeds.biasUp, minv, maxv, stream);
  if (mt == flashmoe::MLPMatmulType::gated) {
    CHECK_CUDA(cudaMallocAsync(&biasUpV, sizeof(Element) * E * I, stream));
    randUniform<Arch>(biasUpV, static_cast<size_t>(E) * I, seeds.biasUpV, minv, maxv, stream);
  }

  Element* biasDown = nullptr;
  CHECK_CUDA(cudaMallocAsync(&biasDown, sizeof(Element) * E * H, stream));
  randUniform<Arch>(biasDown, static_cast<size_t>(E) * H, seeds.biasDown, minv, maxv, stream);

  int* expertCounts = nullptr;
  CHECK_CUDA(cudaMallocAsync(&expertCounts, sizeof(int) * E, stream));

  Element* moeOut = nullptr;
  CHECK_CUDA(cudaMallocAsync(&moeOut, sizeof(Element) * S * H, stream));

  Element* referenceInput;
  CHECK_CUDA(cudaMallocAsync(&referenceInput, sizeof(Element) * roundEC * H, stream));
  Element* referenceInterim0;
  CHECK_CUDA(cudaMallocAsync(&referenceInterim0, sizeof(Element) * roundEC * I, stream));
  Element* referenceInterim1;
  CHECK_CUDA(cudaMallocAsync(&referenceInterim1, sizeof(Element) * roundEC * H, stream));
  Element* referenceOut;
  CHECK_CUDA(cudaMallocAsync(&referenceOut, sizeof(Element) * S * H, stream));
  if (k > 1) {
    CHECK_CUDA(cudaMemsetAsync(referenceOut, 0, sizeof(Element) * S * H, stream));
  }
  const auto swishAlpha = random_float(minv, maxv, seeds.alpha);
  const auto swishBeta = random_float(minv, maxv, seeds.beta);

  // initialize
  flashmoe::MoEArgs args{
    sizeof(Element), static_cast<uint>(S), static_cast<uint>(H), static_cast<uint>(I), EC,
    bM, bN0, bN1, bK0, bK1, threads,
    blocks, static_cast<uint16_t>(epRank), static_cast<uint16_t>(world),
    static_cast<uint16_t>(nvshmem_my_pe()), static_cast<uint16_t>(E),
    static_cast<uint16_t>(numLocalExperts), kernelTopo
  };
  const auto workspaceBytesMiB = static_cast<double>(flashmoe::getWorkspaceBytes(args)) / static_cast<double>(1024 * 1024);
  // blocked partitioning
  // for 8 experts and 4 ranks
  // rank 0 gets [E0, E1], rank 1 gets [E2, E3] and so on
  std::vector<uint> expertToEpRank(E);
  for (int i = 0; i < world; ++i) {
    for (int j = 0; j < numLocalExperts; ++j) {
      expertToEpRank[i * numLocalExperts + j] = i;
    }
  }
  std::vector<int> epRankToGlobalRank(world);
  for (int i = 0; i < world; ++i) {
    epRankToGlobalRank[i] = i;
  }
  auto moeContext = flashmoe::initialize(args, Arch,expertToEpRank.data(), epRankToGlobalRank.data(), stream);

  size_t expertBase = static_cast<size_t>(epRank) * numLocalExperts;
  auto* __restrict__ localExpertUpWeights = expertUpWeights + (expertBase * static_cast<size_t>(H) * I);
  auto* __restrict__ localExpertUpVWeights = mt == flashmoe::MLPMatmulType::gated ?
    expertUpV + (expertBase * static_cast<size_t>(H) * I) : nullptr;
  auto* __restrict__ localExpertDownWeights = expertDownWeights + (expertBase * static_cast<size_t>(I) * H);
  auto* __restrict__ localBiasUp = biasUp + (expertBase * I);
  auto* __restrict__ localBiasUpV = mt == flashmoe::MLPMatmulType::gated ? biasUpV + (expertBase * I) : nullptr;
  auto* __restrict__ localBiasDown = biasDown + (expertBase * H);

  constexpr auto isGated = mt == flashmoe::MLPMatmulType::gated;
  const flashmoe::moe::KernelArgs kArgs{
    reinterpret_cast<const cuda::std::byte*>(tokens),
    reinterpret_cast<const cuda::std::byte*>(localExpertUpWeights),
    reinterpret_cast<const cuda::std::byte*>(localExpertUpVWeights),
    reinterpret_cast<const cuda::std::byte*>(localBiasUp),
    reinterpret_cast<const cuda::std::byte*>(localBiasUpV),
    reinterpret_cast<const cuda::std::byte*>(localExpertDownWeights),
    reinterpret_cast<const cuda::std::byte*>(localBiasDown),
    expertCounts, reinterpret_cast<cuda::std::byte*>(moeOut),
    S, H, I, E, k, EC, Arch, mt, isGated ? swishAlpha : 1.f, isGated ? swishBeta : 1.f
  };

  const auto ids_counts = generate_token_ids_and_expert_counts(
    static_cast<int>(S), static_cast<int>(E), static_cast<int>(EC), roundEC, static_cast<int>(k), 0.f, 0.001f);
  auto counts = std::get<0>(ids_counts);
  auto indices = std::get<1>(ids_counts);
  using CT = decltype(counts)::value_type;
  using IT = decltype(indices)::value_type;
  CHECK_CUDA(cudaMemcpyAsync(expertCounts, counts.data(), sizeof(CT) * counts.size(),
    cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemcpyAsync(moeContext.tokenIndices, indices.data(), sizeof(IT) * indices.size(),
    cudaMemcpyHostToDevice, stream));
  auto flashMK = [&](const flashmoe::Topology topology, const uint& k_runs) {
    if (topology == flashmoe::Topology::NVLINK_ONLY) {
      for (int i = 0; i < k_runs; ++i) {
        flashmoe::moe::forwardHost<Config, flashmoe::Topology::NVLINK_ONLY, act>(kArgs, moeContext, kernelSz, stream);
      }
    }
    else {
      for (int i = 0; i < k_runs; ++i) {
        flashmoe::moe::forwardHost<Config, flashmoe::Topology::MIXED, act>(kArgs, moeContext, kernelSz, stream);
      }
    }
  };
  flashMK(moeContext.topo, 1);
  CHECK_CUDA(cudaPeekAtLastError());
  // check correctness
  using ActType = flashmoe::ActivationType<AccumType, act>::AT;
  reference<Config, mt, GEMM0Sz, GEMM1Sz, ActType, AccumType>(moeContext.tokenIndices, tokens,
    referenceInput, expertUpWeights, expertUpV, expertDownWeights, biasUp, biasUpV, biasDown,
    referenceInterim0, referenceInterim1, expertCounts, referenceOut,
    swishAlpha, swishBeta, S, H, EC, E, I, k, num_sms, stream);
  CHECK_CUDA(cudaPeekAtLastError());

  auto tC = matx::make_tensor<MT>(reinterpret_cast<MT*>(moeOut), {static_cast<matx::index_t>(S), static_cast<matx::index_t>(H)});
  auto tRef = matx::make_tensor<MT>(reinterpret_cast<MT*>(referenceOut), tC.Shape());
  auto num_matches = matx::make_tensor<long int>({});
  matx::cudaExecutor exec{stream};
  (num_matches = matx::sum(matx::isclose(tC, tRef, rtol, atol))).run(exec);
  exec.sync();
  // calculate error percentage
  const auto ep =  (1.0 - (static_cast<double>(num_matches()) / static_cast<double>(tC.TotalSize()))) * 100;
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  {
    // benchmark distributed moe fused kernel
    flashMK(moeContext.topo, warmup);
    CHECK_CUDA(cudaStreamSynchronize(stream));
#if defined(FLASHMOE_NVTX) && FLASHMOE_NVTX
    const flashmoe::flashmoeRange range{std::string("FlashMoE Bench, S: ").append(std::to_string(S))
        .append(", H: ").append(std::to_string(H))
        .append(", I: ").append(std::to_string(I))
        .append(", E: ").append(std::to_string(E))
        .append(". nLx: ").append(std::to_string(moeContext.nLx))
        .append(", k: ").append(std::to_string(k))};
#endif
    cudaEventRecord(start, stream);
    flashMK(moeContext.topo, runs);
    cudaEventRecord(stop, stream);
    CHECK_CUDA(cudaEventSynchronize(stop));
    float m_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&m_ms, start, stop));
    const float m_time_ms = m_ms / static_cast<float>(runs);
    constexpr auto es = element_string<Element>();
    printf("%d, %s, %lu, %lu, %lu, %lu, %lu, %lu, %d, %d, %d, %d, %d, %d, %d, %d, %d, %.1e, %.1e, %f, %d, %d, %.4lf, %f\n",
           epRank, es, S, H, I, E, k, EC, bM, bN0, bK0, bN1, bK1, threads, bps, num_sms, blocks, rtol, atol, ep,
           warmup, runs, workspaceBytesMiB, m_time_ms);
  }

  // finalize
  flashmoe::finalize(moeContext, stream);
  cudaFreeAsync(tokens, stream);
  cudaFreeAsync(expertUpWeights, stream);
  if constexpr (mt == flashmoe::MLPMatmulType::gated) {
    cudaFreeAsync(expertUpV, stream);
    cudaFreeAsync(biasUpV, stream);
  }
  cudaFreeAsync(expertDownWeights, stream);
  cudaFreeAsync(biasUp, stream);
  cudaFreeAsync(biasDown, stream);
  cudaFreeAsync(expertCounts, stream);
  cudaFreeAsync(moeOut, stream);
  cudaFreeAsync(referenceInput, stream);
  cudaFreeAsync(referenceInterim0, stream);
  cudaFreeAsync(referenceInterim1, stream);
  cudaFreeAsync(referenceOut, stream);
  CHECK_CUDA(cudaPeekAtLastError());
  CHECK_CUDA(cudaStreamSynchronize(stream));
  cudaEventDestroy(start);cudaEventDestroy(stop);
  nvshmem_finalize();
  cudaStreamDestroy(stream);
}

__host__ __forceinline__
size_t parse_integer(const char* s, const char* name, const uint lo = 1, const uint hi = std::numeric_limits<uint>::max()) {
  // Fast, non-allocating, rejects negatives automatically
  std::string_view sv{s};
  uint64_t v = 0;
  auto [ptr, ec] = std::from_chars(sv.data(), sv.data() + sv.size(), v);
  if (ec != std::errc{} || ptr != sv.data() + sv.size()) {
    throw std::invalid_argument(std::string(name) + " must be an integer");
  }
  if (v < lo || v > hi) {
    throw std::invalid_argument(std::string(name) + " out of range");
  }
  return v;
}

__host__ __forceinline__
float parse_f32(const char* s, const char* name) {
  try {
    size_t idx = 0;
    float v = std::stof(s, &idx);
    if (idx != std::string(s).size()) {
      throw std::invalid_argument("trailing chars");
    }
    return v;
  } catch (...) {
    throw std::invalid_argument(std::string(name) + " must be a float");
  }
}

//./testFlashMoE <S> <H> <I> <E> <k> <warmup> <runs> <rtol> <atol> <EC>
__host__
void drive(const int argc, char** argv) {
  size_t S = 8192;
  size_t H = 2048;
  size_t I = 2048;
  size_t E = 32;
  size_t k = 2;
  size_t EC = cute::ceil_div(S, E) * k;
  uint warmup = 128;
  uint runs = 128;
  float rtol = 8e-2;
  float atol = 8e-3;
  if (argc > 1) S = parse_integer(argv[1], "S", 1);
  if (argc > 2) H = parse_integer(argv[2], "H", 1);
  if (argc > 3) I = parse_integer(argv[3], "I", 1);
  if (argc > 4) E = parse_integer(argv[4], "E", 1);
  if (argc > 5) k = parse_integer(argv[5], "k", 1);
  if (argc > 6) warmup = static_cast<uint>(parse_integer(argv[6], "warmup", 0));
  if (argc > 7) runs = static_cast<uint>(parse_integer(argv[7], "runs", 1));
  if (argc > 8) rtol = parse_f32(argv[8], "rtol");
  if (argc > 9) atol = parse_f32(argv[9], "atol");
  if (argc > 10) {
    EC = parse_integer(argv[10], "EC", 1, S);
  }
  else {
    EC = cute::ceil_div(S, E) * k;
  }


  using Element = __nv_bfloat16;
  static_assert(cuda::std::is_same_v<Element, __half> ||
    cuda::std::is_same_v<Element, __nv_bfloat16> ||
    cuda::std::is_same_v<Element, float> ||
    cuda::std::is_same_v<Element, double>);

  using AccumType = cuda::std::conditional_t<cuda::std::is_same_v<Element, double>, double, float>;
  // below values are static to minimize instantiated templates.
  // some values are 700 (Volta), 750 (Turing), 800 (Ampere), 900 (Hopper), 1000 (Blackwell), 1100 (?), 1200 (?)
  static_assert(FLASHMOE_ARCH >= 700);
  constexpr auto arch = FLASHMOE_ARCH;
  constexpr auto act = flashmoe::Activation::silu;
  constexpr auto mt = flashmoe::MLPMatmulType::gated;
  if (k > E) {
    throw std::invalid_argument("k must be <= E");
  }
  // to minimize instantiated templates
  constexpr int bN0 = sizeof(Element) == 2 ? 128 : 64; // 144 for gpt-oss
  constexpr int bN1 = bN0;
  constexpr int bK0 = cuda::std::is_same_v<Element, double> ? 32 : mt == flashmoe::MLPMatmulType::vanilla ? 64 :
  arch >= 900 ? 64 : 32;
  constexpr int bK1 = bK0;
  if (H <= bK0 || H % bK0 != 0 || H < bN1 || H % bN1 != 0) {
    throw std::invalid_argument("H is invalid");
  }
  if (I <= bK1 || I % bK1 != 0 || I < bN0 || I % bN0 != 0) {
    throw std::invalid_argument("I is invalid");
  }
  constexpr auto pS = arch >= 800 ? 2 : 1;
  constexpr auto bM = cuda::std::is_same_v<Element, double> ? 64 : 128;
  static_assert(bM >= 1);
  if (S % bM != 0) {
    const auto errmsg = std::string("S must be a multiple of ").append(std::to_string(bM));
      throw std::invalid_argument(errmsg);
  }
  if (k > 1) {
    kickstart<Element, AccumType, arch, bM, bN0, bK0, pS, bN1, bK1, pS, flashmoe::CombineMode::plural, act, mt>
    (S, H, I, E, EC, k, warmup, runs, rtol, atol);
  }
  else {
    kickstart<Element, AccumType, arch, bM, bN0, bK0, pS, bN1, bK1, pS, flashmoe::CombineMode::single, act, mt>
    (S, H, I, E, EC, k, warmup, runs, rtol, atol);
  }
}
int main(const int argc, char** argv) {
  drive(argc, argv);
}
