/******************************************************************************
 * Copyright (c) 2026, Osayamen Jonathan Aimuyo.
 ******************************************************************************/
#include <random>
#include <tuple>

#include "../include/flashmoe/bootstrap.cuh"
#include "../include/flashmoe/moe.cuh"

#include "common.cuh"
#include "debug.cuh"
template<
  int Arch,
  int bM, int bN0, int bK0, int pSK0,
  int bN1, int bK1, int bNGate, int pSK1,
  bool isKG1, flashmoe::Activation act,
  flashmoe::Topology topo,
  flashmoe::GateReductionLevel grl,
  flashmoe::DropTokens dtk = flashmoe::DropTokens::no
>
void kickstart(const uint S, const uint H, const uint I, const uint E, const uint k) {
  nvshmem_init();
  const uint devId = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  CHECK_CUDA(cudaSetDevice(devId));
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  const auto world = nvshmem_n_pes();
  const auto numLocalExperts = E / world;
  const auto tilesN1 = H / bN1;

  using Element = float;
  using AccumType = float;
  constexpr auto threadsGEMM0 = flashmoe::tile::suggest_thread_count<bM, bN0, bK0, Arch, Element, AccumType>();
  constexpr auto threadsGEMM1 = flashmoe::tile::suggest_thread_count<bM, bN1, bK1, Arch, Element, AccumType>();
  constexpr auto threads = cute::max(threadsGEMM0, threadsGEMM1);
  static_assert(dtk == flashmoe::DropTokens::no);
  constexpr bool doGate = true;
  const auto EC = S; // no token dropping for tests
  constexpr auto GEMM0Sz = cutlass::round_up(sizeof(Element) * bK0 * pSK0 * (bM + bN0), flashmoe::MAX_ALIGNMENT);
  constexpr auto GEMM1Sz = cutlass::round_up(sizeof(Element) * bK1 * pSK1 * (bM + bN1), flashmoe::MAX_ALIGNMENT);
  constexpr auto GateSz = cutlass::round_up(cute::max(sizeof(Element) * bK0 * pSK0 * (bM + bNGate),
    sizeof(flashmoe::gate::SoftType) * bM * bNGate), flashmoe::MAX_ALIGNMENT);
  const auto dispatchSz = E * (sizeof(flashmoe::PEL) + sizeof(int));
  const auto OSSz = flashmoe::os::getSharedSize<threads, bM>(world, numLocalExperts, E, EC, tilesN1);
  constexpr auto kernelSz = cute::max(GEMM0Sz, GEMM1Sz, GateSz, dispatchSz, OSSz);
  int maxSharedMemory = 0;
  CHECK_CUDA(cudaDeviceGetAttribute(&maxSharedMemory,cudaDevAttrMaxSharedMemoryPerBlock, devId));
  if (kernelSz > maxSharedMemory) {
    throw std::runtime_error("Required shared memory exceeds hardware limits. Reduce tile shapes.");
  }
  // TODO
  using GateTile = cute::Shape<>;
  using GEMM0Tile = cute::Shape<>;
  using GEMM1Tile = cute::Shape<>;
  using Config = flashmoe::moe::MoEConfig<Arch, threads, isKG1, dtk, GateTile, GEMM0Tile, GEMM1Tile>;

  auto kernel = flashmoe::moe::forward<Config, act, topo, doGate, grl, Element>;
  CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kernelSz));
  int bps = 0;
  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kernel, threads, kernelSz));
  const auto blocks = cute::min(2, 4, bps * NUM_SMS);

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
  cudaMallocAsync(&expertUpWeights, sizeof(Element) * H * I, stream);
  cudaMallocAsync(&expertDownWeights, sizeof(Element) * I * H, stream);
  matx::cudaExecutor exec{stream};
  using MT = MXE<Element>;
  const auto eWz = static_cast<size_t>(H) * I;
  for (int i = 0; i < numLocalExperts; ++i) {
    auto euw = matx::make_tensor<MT>(expertUpWeights + i * eWz, {H, I});
    (euw = matx::eye<Element>({H, I})).run(exec);
    auto edw = matx::make_tensor<MT>(expertDownWeights + i * eWz, {I, H});
    (edw = matx::eye<Element>({I, H})).run(exec);
  }

  Element* biasUp = nullptr;
  cudaMallocAsync(&biasUp, sizeof(Element) * numLocalExperts * I, stream);
  auto bu = matx::make_tensor<MT>(expertUpWeights, {numLocalExperts, I});
  (bu = matx::zeros<MT>({numLocalExperts, I})).run(exec);

  Element* biasDown = nullptr;
  cudaMallocAsync(&biasDown, sizeof(Element) * numLocalExperts * H, stream);
  auto bd = matx::make_tensor<MT>(expertUpWeights, {numLocalExperts, H});
  (bd = matx::zeros<MT>({numLocalExperts, H})).run(exec);

  Element* gateOut = nullptr;
  cudaMallocAsync(&gateOut, sizeof(Element) * S * E, stream);

  int* expertCounts = nullptr;
  cudaMallocAsync(&expertCounts, sizeof(int) * E, stream);

  Element* moeOut = nullptr;
  cudaMallocAsync(&moeOut, sizeof(Element) * S * H, stream);

  // initialize
  // TODO
  flashmoe::MoEArgs args{};
  uint* expertToEpRank = nullptr;
  int* epRankToGlobalRank = nullptr;
  const auto contexts = flashmoe::initialize(args, bNGate,
    expertToEpRank, epRankToGlobalRank, stream);
  auto moeContext = std::get<0>(contexts);
  auto gateCtx = std::get<1>(contexts);
  // call kernel as many times as needed
  flashmoe::moe::forwardHost<Config, topo, doGate, act, grl>(S, H, I, E, k, EC,
    tokens, gateWeights, expertUpWeights, biasUp, expertDownWeights, biasDown, gateOut, expertCounts, moeOut,
    moeContext, gateCtx, blocks, kernelSz, stream);
  // check correctness TODO
  // finalize
  flashmoe::finalize(moeContext, gateCtx, stream);
  nvshmem_finalize();
  cudaStreamDestroy(stream);
}
int main() {

}
