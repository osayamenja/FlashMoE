//
// Created by osayamen on 12/22/25.
//

// place to experiment

#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <vector>
#include <nvshmem.h>

#include "../include/flashmoe/flashmoe.cuh"
#include "debug.cuh"
namespace py = pybind11;

constexpr int S = 128; // jit value
constexpr int H = 2048; // jit value
constexpr int I = 2048; // jit value
constexpr int Arch = 800; // jit value
constexpr int topK = 2; // jit value
constexpr auto topo = flashmoe::defineTopology<0>(); // jit value
constexpr auto mt = flashmoe::defineMLPType<0>(); // jit value
using Element = flashmoe::DataType<0>::Type; // jit value
constexpr auto act = flashmoe::defineAct<0>(); // jit value
using AccumType = cuda::std::conditional_t<cuda::std::is_same_v<Element, double>, double, float>;
constexpr auto cm = topK > 1 ? flashmoe::CombineMode::plural : flashmoe::CombineMode::single;

// tile shapes
constexpr auto bM = flashmoe::heuristics::getMoETileM<S, Arch>();
constexpr auto bK0 = flashmoe::heuristics::getTileK<H, Arch, mt, Element>();
constexpr auto bK1 = flashmoe::heuristics::getTileK<I, Arch, mt, Element>();
constexpr auto bN0 = flashmoe::heuristics::getTileN<I, Element>();
constexpr auto bN1 = flashmoe::heuristics::getTileN<H, Element>();
constexpr auto pSK0 = flashmoe::heuristics::getPipeStages<H, bK0, Arch>();
constexpr auto pSK1 = flashmoe::heuristics::getPipeStages<I, bK1, Arch>();

constexpr auto threadsGEMM0 = flashmoe::tile::suggest_thread_count<bM, bN0, bK0, Arch, Element, AccumType>();
constexpr auto threadsGEMM1 = flashmoe::tile::suggest_thread_count<bM, bN1, bK1, Arch, Element, AccumType>();
constexpr auto threads = cute::max(threadsGEMM0, threadsGEMM1, 64);

// [S, H] x [H, I] -> [S, I]
using GEMM0Tile = cute::Shape<cute::Int<bM>, cute::Int<bN0>, cute::Int<bK0>, cute::Int<pSK0>>;
// [S, I] x [I, H] -> [S, H]
using GEMM1Tile = cute::Shape<cute::Int<bM>, cute::Int<bN1>, cute::Int<bK1>, cute::Int<pSK1>>;
using Config = flashmoe::moe::MoEConfig<Element, Arch, threads, cm, mt, GEMM0Tile, GEMM1Tile>;

static py::capsule moe_initialize(const size_t& numExperts, const size_t& EC,
  const int& epWorld, const int& myPE, const int& epRank, const int& devId, const int& nLx,
  const std::vector<int>& expertToEpRank, const std::vector<int> &epRankToGlobalRank,
  const std::uintptr_t stream_ptr) {
  auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  auto kernel = flashmoe::moe::forward<Config, act, topo>;
  const auto smemSize = flashmoe::moe::kernelSMEM<Config>(numExperts, EC, epWorld, nLx, H / bN1);
  int maxSharedMemory = 0;
  CHECK_CUDA(cudaDeviceGetAttribute(&maxSharedMemory,cudaDevAttrMaxSharedMemoryPerBlockOptin, devId));
  if (smemSize > maxSharedMemory) {
    const auto errmsg = std::string("Required shared memory ").append(std::to_string(smemSize))
    .append(" exceeds hardware limits: ").append(std::to_string(maxSharedMemory)).append(" Reduce tile shapes or input sizes.");
    throw std::runtime_error(errmsg);
  }
  int numSMs = 0;
  CHECK_CUDA(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId));
  CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smemSize));
  int blocksPerSM = 0;
  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel, threads, smemSize));
  const auto blocks = flashmoe::moe::kernelBlocks<bM, bN0, bN1>(S, H, I, numExperts, topK, blocksPerSM, numSMs);
  const flashmoe::MoEArgs args{
    sizeof(Element),
    static_cast<uint>(S),
    static_cast<uint>(H),
    static_cast<uint>(I),
    EC,
    bM, bN0, bN1, bK0, bK1, threads,
    blocks, smemSize, static_cast<uint16_t>(epRank),
    static_cast<uint16_t>(epWorld), static_cast<uint16_t>(myPE),
    static_cast<uint16_t>(numExperts), static_cast<uint16_t>(nLx),
    topo
  };
  const auto moeContext = flashmoe::initialize(args, Arch,expertToEpRank.data(), epRankToGlobalRank.data(), stream);
  auto* heapCtx = new flashmoe::Context(moeContext);
  return py::capsule(heapCtx, "moe.Context");
}

static void moe_forward(const py::capsule& ctx_cap,
  const uint& S, const uint& H, const uint& I, const uint& E, const uint& k, const uint& EC,
  const std::uintptr_t& tokens,
  const std::uintptr_t& expertCounts,
  const std::uintptr_t localExpertUpWeights,
  const std::uintptr_t& localExpertUpVWeights,
  const std::uintptr_t localBiasUp,
  const std::uintptr_t& localBiasUpV,
  const std::uintptr_t localExpertDownWeights,
  const std::uintptr_t& localBiasDown,
  const std::uintptr_t& moeOut,
  const float& swishAlpha, const float& swishBeta,
  const std::uintptr_t& stream_ptr) {
  const auto* ctx = ctx_cap.get_pointer<flashmoe::Context>();
  auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  if (!ctx) {
    throw std::runtime_error("Invalid context");
  }
  constexpr auto isGated = mt == flashmoe::MLPMatmulType::gated;
  const flashmoe::moe::KernelArgs kArgs{
    reinterpret_cast<const cuda::std::byte*>(tokens),
    reinterpret_cast<const cuda::std::byte*>(localExpertUpWeights),
    reinterpret_cast<const cuda::std::byte*>(localExpertUpVWeights),
    reinterpret_cast<const cuda::std::byte*>(localBiasUp),
    reinterpret_cast<const cuda::std::byte*>(localBiasUpV),
    reinterpret_cast<const cuda::std::byte*>(localExpertDownWeights),
    reinterpret_cast<const cuda::std::byte*>(localBiasDown),
    reinterpret_cast<const int*>(expertCounts), reinterpret_cast<cuda::std::byte*>(moeOut),
    S, H, I, E, k, EC, Arch, mt, isGated ? swishAlpha : 1.f, isGated ? swishBeta : 1.f, false
  };

  flashmoe::moe::forwardHost<Config, topo, act>(kArgs, *ctx, stream);
}

void moe_finalize(const py::capsule& cap, const std::uintptr_t stream_ptr) {
  const auto* ctx = cap.get_pointer<flashmoe::Context>();
  auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  if (!ctx) return;

  flashmoe::finalize(*ctx, stream);
  delete ctx;
  PyCapsule_SetPointer(cap.ptr(), nullptr);
}

// const size_t& numExperts, const size_t& EC,
//   const int& epWorld, const int& myPE, const int& epRank, const int& devId, const int& nLx,
//   const std::vector<int>& expertToEpRank, const std::vector<int> &epRankToGlobalRank,
//   const std::uintptr_t stream_ptr
PYBIND11_MODULE($mod_name, m) {
  m.def("initialize", &moe_initialize,
    py::arg("numExperts"),
    py::arg("EC"),
    py::arg("epWorld"),
    py::arg("myPE"),
    py::arg("epRank"),
    py::arg("devId"),
    py::arg("nLx"),
    py::arg("expertToEpRank"),
    py::arg("epRankToGlobalRank"),
    py::arg("stream_ptr"));
  m.def("forward", &moe_forward,
    py::arg("ctx_cap"),
    py::arg("S"), py::arg("H"), py::arg("I"), py::arg("E"), py::arg("k"),
    py::arg("EC"),
    py::arg("tokens"), py::arg("expertCounts"), py::arg("localExpertUpWeights"),
    py::arg("stream_ptr"));
}

struct Foo {
  int a;
  int b;
};

int main() {
  constexpr Foo foo{6, 7};
  const auto p = new Foo(foo);
  p->a += 1;
  auto q = *p;
  printf("a is %d\n", q.a);
  delete p;
}
