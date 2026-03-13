//
// Created by osayamen on 12/22/25.
//

// place to experiment

#include <cstdint>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>

#include "../include/flashmoe/bootstrap.cuh"
#include "../include/flashmoe/gate.cuh"

#include <cstdio>
#if !defined(CHECK_CUDA)
#  define CHECK_CUDA(e)                                      \
do {                                                         \
    cudaError_t code = (e);                                  \
    if (code != cudaSuccess) {                               \
        fprintf(stderr, "<%s:%d> %s:\n    %s: %s\n",         \
            __FILE__, __LINE__, #e,                          \
            cudaGetErrorName(code),                          \
            cudaGetErrorString(code));                       \
        fflush(stderr);                                      \
        exit(1);                                             \
    }                                                        \
} while (0)
#endif
namespace py = pybind11;

constexpr int S = 128; // jit value
constexpr int H = 2048; // jit value
constexpr int E = 2048; // jit value
constexpr int topK = 2; // jit value
constexpr int EC = 1024; // jit value
constexpr int Arch = 800; // jit value
constexpr int returnLogits = 0; // jit value
constexpr auto rl = returnLogits ? flashmoe::gate::ReturnLogits::yes : flashmoe::gate::ReturnLogits::no;
using Element = flashmoe::DataType<0>::Type; // jit value
using AccumType = cuda::std::conditional_t<cuda::std::is_same_v<Element, double>, double, float>;


// tile shapes
constexpr auto bM = flashmoe::heuristics::getTileM<S, Arch>();
constexpr auto bK = flashmoe::heuristics::getGateTileK<H, Element>();
constexpr auto bN = flashmoe::heuristics::getGateTileN<E, flashmoe::gate::BLOCK_N_CAP>();
constexpr auto pSK = flashmoe::heuristics::getPipeStages<H, bK, Arch>();
constexpr auto grl = E > bN ? flashmoe::GateReductionLevel::multiBlock : flashmoe::GateReductionLevel::singleBlock;
constexpr auto sro = flashmoe::SoftMaxOptimizationLevel::highest;

constexpr auto threads = cute::max(flashmoe::tile::suggest_thread_count<bM, bN, bK, Arch, Element, AccumType>(), bM);
constexpr auto smemSize = flashmoe::gate::kernelSMEM<bM, bN, bK, pSK, Element>();
// [S, H] x [H, E] -> [S, E]
using GEMMTile = cute::Shape<cute::Int<bM>, cute::Int<bN>, cute::Int<bK>, cute::Int<pSK>>;

static std::uintptr_t gate_initialize(const int& devId, const std::uintptr_t stream_ptr) {
  auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  auto kernel = flashmoe::gate::forwardKernel<GEMMTile, Arch, threads, grl, sro, rl, AccumType, Element, Element>;

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
  const int gateBlocks = cute::min(cute::ceil_div(S, bM) * cute::ceil_div(E, bN), blocksPerSM * numSMs);
  if (E > gateBlocks * bN) {
    throw std::invalid_argument("E is too big!");
  }
  auto gateCtx = flashmoe::initializeGate(bN, E, S, stream);
  gateCtx.blocks = gateBlocks;
  auto* gCtx = new flashmoe::GateContext(gateCtx);
  return reinterpret_cast<std::uintptr_t>(gCtx);
}

static void gate_forward(const std::uintptr_t& raw_ctx,
  const std::uintptr_t& tokens_,
  const std::uintptr_t& weights_,
  const std::uintptr_t& routing_,
  const std::uintptr_t& expertCounts_,
  const std::uintptr_t& tokenIndices_,
  const std::uintptr_t& stream_ptr) {
  const auto* __restrict__ tokens = reinterpret_cast<const Element*>(tokens_);
  const auto* __restrict__ weights = reinterpret_cast<const Element*>(weights_);
  auto* __restrict__ routing = reinterpret_cast<Element*>(routing_);
  auto* __restrict__ expertCounts = reinterpret_cast<int*>(expertCounts_);
  auto* __restrict__ tokenIndices = reinterpret_cast<flashmoe::TPS*>(tokenIndices_);
  const auto* ctx = reinterpret_cast<flashmoe::GateContext*>(raw_ctx);
  auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  if (!ctx) {
    throw std::runtime_error("Invalid context");
  }
  flashmoe::gate::forwardKernel<GEMMTile, Arch, threads, grl, sro, rl, AccumType>
  <<<ctx->blocks, threads, smemSize, stream>>>(tokens, weights, routing, expertCounts, S, H, E, topK, EC,
    tokenIndices, ctx->ecGuards, ctx->ssp, ctx->rtp);
}

void gate_finalize(const std::uintptr_t& raw_ctx, const std::uintptr_t stream_ptr) {
  const auto* ctx = reinterpret_cast<flashmoe::GateContext*>(raw_ctx);
  auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  if (!ctx) return;
  flashmoe::finalizeGate(*ctx, stream);
  delete ctx;
}

PYBIND11_MODULE($mod_name, m) {
  m.def("initialize", &gate_initialize,
    py::arg("device_id"),
    py::arg("stream_ptr"));
  m.def("forward", &gate_forward,
    py::arg("raw_ctx"),
    py::arg("tokens"),
    py::arg("weights"),
    py::arg("routing"),
    py::arg("expert_counts"),
    py::arg("token_indices"),
    py::arg("stream_ptr"));
  m.def("finalize", &gate_finalize,
    py::arg("raw_ctx"),
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
