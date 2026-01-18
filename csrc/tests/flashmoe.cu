/******************************************************************************
 * Copyright (c) 2024, Osayamen Jonathan Aimuyo.
 ******************************************************************************/
#include "../include/flashmoe/bootstrap.cuh"
#include "../include/flashmoe/moe.cuh"

#include "debug.cuh"
template<
  int Arch,
  int bM, int bN0, int bK0, int pSK0,
  int bN1, int bK1, int bNGate, int pSK1,
  bool isKG1,
  flashmoe::Topology topo,
  flashmoe::DropTokens dtk = flashmoe::DropTokens::no
>
void kickstart(const uint S, const uint H, const uint I, const uint E, const uint k) {
  nvshmem_init();
  const uint devId = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  CHECK_CUDA(cudaSetDevice(devId));

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
  const auto dispatchSz = 0;
  const auto OSSz = flashmoe::os::getSharedSize<threads, bM>(world, numLocalExperts, E, EC, tilesN1);
  constexpr auto kernelSz = cute::max(GEMM0Sz, GEMM1Sz, GateSz, dispatchSz, OSSz);
  int maxSharedMemory = 0;
  CHECK_CUDA(cudaDeviceGetAttribute(&maxSharedMemory,cudaDevAttrMaxSharedMemoryPerBlock, devId));
  if (kernelSz > maxSharedMemory) {
    throw std::runtime_error("Required shared memory exceeds hardware limits. Reduce tile shapes.");
  }

  Element* tokens = nullptr;
  std::vector<float> th(S*H);
  auto tokTensor = cute::make_tensor(th.data(),
    cute::make_layout(cute::make_shape(S, H), cute::LayoutRight{}));
  for (int i = 0; i < S; ++i) {
    for (int j = 0; j < H; ++j) {
      tokTensor(i, j) = static_cast<float>(i + j);
    }
  }
  Element* gateWeights = nullptr;
  Element* expertUpWeights = nullptr;
  Element* biasUp = nullptr;
  Element* expertDownWeights = nullptr;
  Element* biasDown = nullptr;
  Element* gateOut = nullptr;
  int* expertCounts = nullptr;
  Element* moeOut = nullptr;

  nvshmem_finalize();
}
int main() {

}
