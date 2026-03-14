from string import Template

flashmoe_bindings = Template(r"""
#include <cstdint>
#include <stdexcept>

#include <cuda_runtime.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include <flashmoe/bootstrap.cuh>
#include <flashmoe/moe.cuh>

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

constexpr int S = $s; // jit value
constexpr int H = $h; // jit value
constexpr int I = $i; // jit value
constexpr int E = $e; // jit value
constexpr int EC = $ec; // jit value
constexpr int Arch = $arch; // jit value
constexpr int topK = $tk; // jit value
constexpr auto topo = flashmoe::defineTopology<$topo>(); // jit value
constexpr auto mt = flashmoe::defineMLPType<$mt>(); // jit value
using Element = flashmoe::DataType<$dt>::Type; // jit value
constexpr auto act = flashmoe::defineAct<$act>(); // jit value
using AccumType = cuda::std::conditional_t<cuda::std::is_same_v<Element, double>, double, float>;
constexpr auto cm = topK > 1 ? flashmoe::CombineMode::plural : flashmoe::CombineMode::single;

// tile shapes
constexpr auto bM = flashmoe::heuristics::getMoETileM<S, Arch>();
constexpr auto tkCap = cuda::std::is_same_v<Element, double> ? 32 : (mt == flashmoe::MLPMatmulType::vanilla ? 64 : (Arch >= 900 ? 64 : 32));
constexpr auto bK0 = flashmoe::heuristics::getTileK<H, tkCap>();
constexpr auto bK1 = flashmoe::heuristics::getTileK<I, tkCap>();
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

static std::uintptr_t moe_initialize(const size_t& numExperts, const size_t& EC,
  const int& epWorld, const int& myPE, const int& epRank, const int& devId, const int& nLx,
  const std::vector<int>& expertToEpRank, const std::vector<int> &epRankToGlobalRank,
  const std::uintptr_t& stream_ptr) {
  auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  if (expertToEpRank.size() != numExperts) {
    throw std::invalid_argument("expert map size should be == # of experts");
  }
  if (epRankToGlobalRank.size() != epWorld) {
    throw std::invalid_argument("rank map size should be == epWorld");
  }

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
  return reinterpret_cast<std::uintptr_t>(heapCtx);
}

static void moe_forward(const std::uintptr_t& raw_ctx,
  const std::uintptr_t& tokens,
  const std::uintptr_t& expertCounts,
  const std::uintptr_t& localExpertUpWeights,
  const std::uintptr_t& localExpertUpVWeights,
  const std::uintptr_t& localBiasUp,
  const std::uintptr_t& localBiasUpV,
  const std::uintptr_t& localExpertDownWeights,
  const std::uintptr_t& localBiasDown,
  const std::uintptr_t& moeOut,
  const float& swishAlpha, const float& swishBeta,
  const std::uintptr_t& stream_ptr) {
  const auto* ctx = reinterpret_cast<flashmoe::Context*>(raw_ctx);
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
    S, H, I, E, EC, Arch, mt, isGated ? swishAlpha : 1.f, isGated ? swishBeta : 1.f, false
  };

  flashmoe::moe::forwardHost<Config, topo, act>(kArgs, *ctx, stream);
}

static std::uintptr_t get_token_indices(const std::uintptr_t& raw_ctx) {
  return reinterpret_cast<std::uintptr_t>(reinterpret_cast<flashmoe::Context*>(raw_ctx)->tokenIndices);
}

static void moe_finalize(const std::uintptr_t& raw_ctx, const std::uintptr_t& stream_ptr) {
  const auto* ctx = reinterpret_cast<flashmoe::Context*>(raw_ctx);
  auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  if (!ctx) return;
  flashmoe::finalize(*ctx, stream);
  delete ctx;
}

PYBIND11_MODULE($mod_name, m) {
  m.def("initialize", &moe_initialize,
    py::arg("num_experts"),
    py::arg("expert_peer_capacity"),
    py::arg("ep_world"),
    py::arg("my_pe"),
    py::arg("ep_rank"),
    py::arg("local_rank"),
    py::arg("num_local_experts"),
    py::arg("expert_map"),
    py::arg("rank_map"),
    py::arg("stream_ptr"));
  m.def("forward", &moe_forward,
    py::arg("raw_ctx"),
    py::arg("tokens"), 
    py::arg("expert_counts"), 
    py::arg("local_expert_up"),
    py::arg("local_expert_up_v"), 
    py::arg("local_bias_up"), 
    py::arg("local_bias_up_v"),
    py::arg("local_expert_down"), 
    py::arg("local_bias_down"), 
    py::arg("moe_out"),
    py::arg("swish_alpha"), 
    py::arg("swish_beta"),
    py::arg("stream_ptr"));
  m.def("get_tIdx", &get_token_indices,
    py::arg("raw_ctx"));
  m.def("finalize", &moe_finalize,
    py::arg("raw_ctx"),
    py::arg("stream_ptr"));
}
""")

gate_bindings = Template(r"""
#include <cstdint>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>

#include <flashmoe/bootstrap.cuh>
#include <flashmoe/gate.cuh>

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

constexpr int S = $s; // jit value
constexpr int H = $h; // jit value
constexpr int E = $e; // jit value
constexpr int topK = $top_k; // jit value
constexpr int EC = $ec; // jit value
constexpr int Arch = $arch; // jit value
constexpr int returnLogits = $return_logits; // jit value
constexpr auto rl = returnLogits ? flashmoe::gate::ReturnLogits::yes : flashmoe::gate::ReturnLogits::no;
using Element = flashmoe::DataType<$dt>::Type; // jit value
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

static void gate_finalize(const std::uintptr_t& raw_ctx, const std::uintptr_t stream_ptr) {
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
""")