/******************************************************************************
 * Copyright (c) 2026, Osayamen Jonathan Aimuyo.
 ******************************************************************************/
//
// Created by osayamen on 7/13/24.
//

#ifndef FLASHMOE_COMPUTE_CUH
#define FLASHMOE_COMPUTE_CUH
#include <cuda/atomic>
#include <nvshmem.h>

#include "combine.cuh" //  combine function
#include "infra/constants.cuh"
#include "infra/packed.cuh"
#include "infra/signal.cuh"
#include "infra/structures.cuh"
#include "infra/task.cuh"
#include "infra/trace.cuh"
#include "tile.cuh" // cublasdx and tileGEMM
#include "tile_sm120_mxfp8.cuh" // block-scaled MXFP8 tile GEMM
#include "tile_sm120_fp4.cuh" // block-scaled MXFP4 / NVFP4 tile GEMM
#include "infra/tq.cuh"

namespace flashmoe::processor
{
  // GEMM1 of a task whose tokens stay on this device (the combine is fused into the epilogue, see fGET and
  // processor::start): the tile goes to the output rows of its tokens instead of the symmetric heap
  template <typename ElementC>
  struct LocalCombine {
    ElementC* moeOutput = nullptr; // [S, H]
    const TPS* tokenIds = nullptr; // [bM] of the tile, nullptr: the tile is not fused
    int S = 0;
    uint tileSize = 0;
  };

  // fused GEMM, epilogue and data transfer
  // fuseCombine == true (GEMM1): if lc.tokenIds != nullptr, the epilogue is the combine of the tile instead of the
  // store to c, which is then unused
  template <
    typename TileGEMM,
    typename Activation,
    bool fuseCombine = false,
    CombineMode cm = CombineMode::single,
    int threads = 0,
    typename Element,
    typename ElementC
  >
  __forceinline__ __device__
  void fGET(void* __restrict__ const& workspace,
            const Element* __restrict__ const& a,
            const Element* __restrict__ const& b,
            ElementC* __restrict__ const& c,
            const ElementC* __restrict__ const& bias,
            const int& M, const int& N, const int& K, const int& tileIdx,
            [[maybe_unused]] const LocalCombine<ElementC>& lc = {}) {
    using BLAS = TileGEMM::BLAS;
    auto accumulator = BLAS::suggest_accumulator();
    using BM = cute::Int<cublasdx::size_of<BLAS>::m>;
    using BN = cute::Int<cublasdx::size_of<BLAS>::n>;
    const auto tileCoord = tile::idx2Coord(M / BM{}, N / BN{}, tileIdx);
    // gmem -> rmem: prefetch the token indices and routing probabilities of the tile, before the mainloop
    static_assert(!fuseCombine || threads > 0);
    constexpr int bM = BM::value;
    constexpr int idsPerThread = fuseCombine ? (bM + threads - 1) / (threads > 0 ? threads : 1) : 1;
    TPS ids[idsPerThread] = {};
    if constexpr (fuseCombine) {
      if (lc.tokenIds != nullptr) {
        #pragma unroll
        for (int i = 0; i < idsPerThread; ++i) {
          const uint row = threadIdx.x + i * threads;
          if (row < lc.tileSize) {
            ids[i] = lc.tokenIds[row];
          }
        }
      }
    }
    // compute Tile
    constexpr TileGEMM tileMainloop{};
    tileMainloop(workspace, a, b, accumulator, M, N, K, tileCoord);
    FLASHMOE_TRACE_EVENT_T0(procMainDone, 0, 0);
    // gmem -> rmem: prefetch bias
    const auto gD = tile::getBias<BM{}, BN{}>(bias, M, N, cute::select<0, 1>(tileCoord));
    auto d_frag = cublasdx::make_fragment_like<ElementC>(accumulator.get_results());
    cublasdx::copy_fragment<cublasdx::alignment_of<BLAS>::c>(gD, d_frag, accumulator);
    // Epilogue
    constexpr Activation act{}; // activation function like relu, etc
    // ElementC -> accum type
    constexpr Converter<typename decltype(accumulator)::value_type, ElementC> loadConv{};
    // accum type -> ElementC
    constexpr Converter<ElementC, typename decltype(accumulator)::value_type> storeConv{};
    const auto c_frag = accumulator.get_results();
    constexpr int accum_size = cublasdx::size(c_frag);
    cute::for_each(cute::make_int_sequence<accum_size>{}, [&c_frag, &d_frag](auto i) {
      d_frag(i) = storeConv(act(c_frag(i) + loadConv(d_frag(i))));
    });
    auto gC = tile::getC<BM{}, BN{}, cublasdx::arrangement_of_v_c<BLAS>>(c, M, N, cute::select<0, 1>(tileCoord));
    // TODO For Optimizing the below copy, TMA may help
    // rmem -> smem
    auto sC = cublasdx::make_tensor(static_cast<ElementC*>(workspace), BLAS::suggest_layout_smem_c());
    __syncthreads();
    if constexpr (fuseCombine) {
      if (lc.tokenIds != nullptr) {
        // rmem -> smem as a row-major tile, which is the layout combineScatter reads
        constexpr int bN = BN::value;
        using RowMajor = cute::Layout<cute::Shape<cute::Int<bM>, cute::Int<bN>>, cute::Stride<cute::Int<bN>, cute::_1>>;
        auto sRow = cute::make_tensor(cute::make_smem_ptr(static_cast<ElementC*>(workspace)), RowMajor{});
        cublasdx::copy_fragment<cublasdx::alignment_of<BLAS>::c>(d_frag, sRow, accumulator);
        auto* __restrict__ stIds = combineIds<bM>();
        #pragma unroll
        for (int i = 0; i < idsPerThread; ++i) {
          const int row = static_cast<int>(threadIdx.x) + i * threads;
          if (row < bM) {
            stIds[row] = ids[i];
          }
        }
        __syncthreads();
        // smem -> HBM: scale by the probabilities and accumulate (plural) or copy (single) to the token rows
        combineScatter<bM, bN, TileGEMM::TileArch::value, threads, cm>(lc.S, N, workspace, stIds, lc.moeOutput,
          lc.tileSize, cute::make_coord(cute::_0{}, cute::get<1>(tileCoord)));
        return;
      }
    }
    cublasdx::copy_fragment<cublasdx::alignment_of<BLAS>::c>(d_frag, sC, accumulator);
    __syncthreads();
    // smem -> gmem
    cublasdx::copy<BLAS, cublasdx::alignment_of<BLAS>::c>(sC, gC);
  }

  template <
    typename TileGEMM,
    typename Activation,
    typename Element,
    typename ElementC
  >
  __forceinline__ __device__
  void fGET_gated(cuda::std::byte* __restrict__ const& workspace,
            const Element* __restrict__ const& a,
            const Element* __restrict__ const& b,
            const Element* __restrict__ const& bV,
            ElementC* __restrict__ const& c,
            const ElementC* __restrict__ const& bias,
            const ElementC* __restrict__ const& biasV,
            const typename TileGEMM::AccumType& swishAlpha,
            const typename TileGEMM::AccumType& swishBeta,
            const int& M, const int& N, const int& K, const int& tileIdx) {
    auto* __restrict__ gateCache = workspace + cutlass::round_up(cute::max(TileGEMM::SharedSizeC::value,
        TileGEMM::SharedSizeAB::value), TileGEMM::GeneralAlignment::value);
    using BLAS = TileGEMM::BLAS;
    auto accumulator = BLAS::suggest_accumulator();
    using BM = cute::Int<cublasdx::size_of<BLAS>::m>;
    using BN = cute::Int<cublasdx::size_of<BLAS>::n>;
    const auto tileCoord = tile::idx2Coord(M / BM{}, N / BN{}, tileIdx);
    // compute Tile
    constexpr TileGEMM tileMainloop{};
    tileMainloop(workspace, a, b, accumulator, M, N, K, tileCoord);
    __syncthreads();
    // gmem -> rmem: prefetch bias
    const auto gD = tile::getBias<BM{}, BN{}>(bias, M, N, cute::select<0, 1>(tileCoord));
    auto d_frag = cublasdx::make_fragment_like<ElementC>(accumulator.get_results());
    cublasdx::copy_fragment<cublasdx::alignment_of<BLAS>::c>(gD, d_frag, accumulator);
    // Epilogue
    constexpr Activation act{}; // activation function like silu, gelu, etc
    // ElementC -> accum type
    using AccumType = decltype(accumulator)::value_type;
    static_assert(cuda::std::is_same_v<AccumType, typename TileGEMM::AccumType>);
    constexpr Converter<AccumType, ElementC> loadConv{};
    // accum type -> ElementC
    constexpr Converter<ElementC, AccumType> storeConv{};
    const auto c_frag = accumulator.get_results();
    constexpr int accum_size = cublasdx::size(c_frag);
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
    FLASHMOE_TRACE_EVENT_T0(procMainDone, 0, 0);
    auto cv_frag = accumulator.get_results();
    const auto gV = tile::getBias<BM{}, BN{}>(biasV, M, N, cute::select<0, 1>(tileCoord));
    cublasdx::copy_fragment<cublasdx::alignment_of<BLAS>::c>(gV, d_frag, accumulator);
    cute::for_each(cute::make_int_sequence<accum_size>{}, [&](auto i) {
      // x = (a @ bV) + biasV
      cv_frag(i) = cv_frag(i) + loadConv(d_frag(i));
    });
    // smem -> rmem, load g tile
    __syncthreads();
    cublasdx::copy_fragment<cublasdx::alignment_of<BLAS>::c>(sGate, d_frag, accumulator);
    cute::for_each(cute::make_int_sequence<accum_size>{}, [&](auto i) {
      // y = x * (act(a @ b))
      d_frag(i) = storeConv(cv_frag(i) * loadConv(d_frag(i)));
    });
    auto gC = tile::getC<BM{}, BN{}, cublasdx::arrangement_of_v_c<BLAS>>(c, M, N, cute::select<0, 1>(tileCoord));
    // TODO For Optimizing the below copy, TMA may help
    // rmem -> smem
    auto sC = cublasdx::make_tensor(reinterpret_cast<ElementC*>(workspace), BLAS::suggest_layout_smem_c());
    __syncthreads();
    cublasdx::copy_fragment<cublasdx::alignment_of<BLAS>::c>(d_frag, sC, accumulator);
    __syncthreads();
    // smem -> gmem
    cublasdx::copy<BLAS, cublasdx::alignment_of<BLAS>::c>(sC, gC);
  }

  struct ProcessorArgs {
    unsigned int* const sQ = nullptr;
    uint64_t* const pDB = nullptr;
    unsigned int* const tQH = nullptr;
    Task* const tQ = nullptr;
    Task* const ptQ = nullptr;
    unsigned int* const tQS = nullptr;

    ProcessorArgs() = default;
    __device__ __forceinline__
    ProcessorArgs(unsigned int* const& sq,
                  TQSignal* const& pdb,
                  unsigned int* const& tqh,
                  Task* const& tq,
                  Task* const& ptq,
                  unsigned int* const& tqs) :
      sQ(sq), pDB(reinterpret_cast<uint64_t*>(pdb)), tQH(tqh), tQ(tq), ptQ(ptq), tQS(tqs) {
    }
  };

  template <
    PeerConnectivity p,
    int threads
  >
  __device__ __forceinline__
  void notifyNext(void* __restrict__ const& workspace, const Task& task, Task* __restrict__ const& tQ,
                  const uint& tasks, const uint& flagColStride,
                  const uint& offset, uint* __restrict__ const& tQH) {
    auto ingredients = task.ingredients;
    ingredients.taskType = TaskType::GEMM1;
    // registers -> shared memory
    static_assert(sizeof(Task) % sizeof(uint4) == 0 && alignof(Task) % sizeof(uint4) == 0);
    static_assert(cuda::std::is_trivially_copyable_v<Task>);
    using TVT = cutlass::AlignedArray<uint4, sizeof(Task) / sizeof(uint4), alignof(Task)>;
    auto* __restrict__ sTQ = static_cast<TVT*>(workspace);
    for (int i = threadIdx.x; i < tasks; i += threads) {
      const auto tvv = cuda::std::bit_cast<TVT>(
        Task{
          ingredients, task.cData[0], task.cData, task.rcData,
          task.flags + (p == PeerConnectivity::remote ? 0 : i * flagColStride), // col-major indexing
          task.syncIdx, offset + i
      });
      sTQ[i] = tvv;
    }
    __syncthreads();
    // shared memory to global memory
    constexpr int nRows = sizeof(Task) / sizeof(uint4);
    const uint numElems = tasks * nRows; // TODO <- store this in constant memory
    // project TaskQ as a [tasks, nRows] matrix of 128B elements
    auto gT = cute::make_tensor(cute::make_gmem_ptr(reinterpret_cast<uint4*>(tQ)),
                                cute::make_layout(cute::make_shape(tasks, nRows), cute::LayoutRight{}));
    auto sT = cute::make_tensor(cute::make_smem_ptr(static_cast<uint4*>(workspace)),
                                cute::make_layout(cute::make_shape(tasks, nRows), cute::LayoutRight{}));
    for (int i = threadIdx.x; i < numElems; i += threads) {
      const auto rowIdx = i / nRows;
      const auto colIdx = i % nRows;
      gT(rowIdx, colIdx) = sT(rowIdx, colIdx);
    }
    __syncthreads();
    if (!threadIdx.x) {
      cuda::atomic_ref<uint, cuda::thread_scope_device> tqh{*(tQH + task.syncIdx)};
      // notify scheduler
      cuda::std::ignore = tqh.fetch_add(tasks, cuda::memory_order_release);
    }
  }

  // Block-scaled recipes only: scale factors of the quantized expert weights and of GEMM0Staging, and for nvfp4 the
  // per-tensor scales, see moe::KernelArgs and Context
  struct QuantArgs {
    // sfVec is the number of K elements per scale factor: 32 (mxfp8, mxfp4) or 16 (nvfp4)
    const uint8_t* upSF = nullptr; // [num_local_experts, mx::sfBytes(I, H / sfVec)]
    const uint8_t* upVSF = nullptr; // [num_local_experts, mx::sfBytes(I, H / sfVec)]
    const uint8_t* downSF = nullptr; // [num_local_experts, mx::sfBytes(H, I / sfVec)]
    uint8_t* stagingSF = nullptr; // [world, num_local_experts, mx::sfBytes(roundEC, I / sfVec)]
    uint nLx = 0; // number of local experts
    // nvfp4 only: FP32 decode scales of the weights, one per local expert
    const float* upGS = nullptr; // [num_local_experts]
    const float* upVGS = nullptr; // [num_local_experts]
    const float* downGS = nullptr; // [num_local_experts]
    // nvfp4 only: static decode scale of the dispatched tokens and of the GEMM0 output
    float inputScale = 1.f;
    float interimScale = 1.f;
  };

  template <
    MLPMatmulType mt,
    Topology topo,
    int threads,
    CombineMode combineMode,
    typename TileGEMM0, // Descriptor for GEMM0
    typename TileGEMM1, // Descriptor for GEMM1
    typename Activation, // Activation function after GEMM0
    QuantRecipe recipe = QuantRecipe::none,
    typename Element,
    typename PBM
  >
  __device__ __forceinline__
  void start(cuda::std::byte* __restrict__ const& workspace, // shared memory
             const int& S, // sequence length
             const int& H, // token hidden dimension
             const int& I, // FFN intermediate size
             const uint& roundEC,
             const uint& flagColStride, // ecTilesM * E
             const uint& tilesN0, // I / bN0
             const uint& tilesN1, // H / bN1
             const Element* __restrict__ const& expertUpWeights, // [num_local_experts, H, I]
             const Element* __restrict__ const& expertUpVWeights, // [num_local_experts, H, I]
             const Element* __restrict__ const& biasUp, // [num_local_experts, I]
             const Element* __restrict__ const& biasUpV, // [num_local_experts, I]
             const typename TileGEMM0::AccumType& swishAlpha,
             const typename TileGEMM0::AccumType& swishBeta,
             const Element* __restrict__ const& expertDownWeights, // [num_local_experts, I, H]
             const Element* __restrict__ const& biasDown, // [num_local_experts, H]
             const TPS* __restrict__ const& tokenIndices, // [E, roundEC]
             Element* __restrict__ const& moeOutput,
             const PBM& producerBitMap,
             const uint8_t& stateNumber,
             const int& epRank, // expert parallel rank of this device
             const Heap& symHeap, const ProcessorArgs& pA, const QuantArgs& mxA = {}) {
    static_assert(sizeof(Task) % sizeof(uint4) == 0 && alignof(Task) % alignof(uint4) == 0);
    static_assert(cuda::std::is_trivially_copyable_v<Task>);
    __shared__ Task currentTask;
    __shared__ uint globalInterrupt;
    __shared__ uint enqueue;
    TQSignal tqs{0U, 0U};
    static_assert(sizeof(TQSignal) == sizeof(uint64_t) && alignof(TQSignal) == alignof(uint64_t));
    static_assert(sizeof(SignalPayload<PacketStage::last>) == sizeof(uint64_t) &&
      alignof(SignalPayload<PacketStage::last>) == alignof(uint64_t));
    if (!threadIdx.x) {
      globalInterrupt = 0U;
      enqueue = 0U;
    }
    __syncthreads();
    using SQT = cuda::std::underlying_type_t<flashmoe::ReadySignal>;
    constexpr auto taskWidth = sizeof(Task) / sizeof(uint4);
    const size_t expertWeightSize = static_cast<size_t>(I) * H;
    // bytes of one expert's weight matrix: E2M1 (mxfp4, nvfp4) packs two elements per byte
    [[maybe_unused]] const size_t expertWeightBytes = expertWeightSize * quantElementBits(recipe) / 8;
    static_assert(taskWidth > 0 && taskWidth < WARP_SIZE);
    while (!tqs.interrupt) {
      if (threadIdx.x / WARP_SIZE == 0) {
        if (threadIdx.x == 0) {
          cuda::atomic_ref<uint64_t, cuda::thread_scope_device> doorbell{*pA.pDB};
          FLASHMOE_TRACE_EVENT(procWaitBegin, 0, 0);
          // await new task from scheduler
          auto payload = cuda::std::bit_cast<TQSignal>(doorbell.load(cuda::memory_order_acquire));
          while (payload.signal == tqs.signal && payload.interrupt == 0) {
            payload = cuda::std::bit_cast<TQSignal>(doorbell.load(cuda::memory_order_acquire));
          }
          FLASHMOE_TRACE_EVENT(procDoorbell, payload.interrupt, payload.decodeSig());
          if (payload.interrupt) {
            // we need to clear out our mailbox to ensure it's ready for a subsequent epoch
            constexpr auto TQSZero = cuda::std::bit_cast<uint64_t>(TQSignal{0, 0});
            doorbell.store(TQSZero, cuda::memory_order_relaxed);
          }
          else {
            cuda::atomic_ref<SQT, cuda::thread_scope_device> sqd{*pA.sQ};
            // Eagerly indicate readiness for the next task
            sqd.store(flashmoe::ready, cuda::memory_order_relaxed);
          }
          globalInterrupt = payload.interrupt;
          // replace old payload with new one
          tqs = payload;
        }
        __syncwarp();
        auto payload = cuda::std::bit_cast<uint64_t>(tqs);
        // broadcast new payload from thread 0 to other threads in the warp
        payload = __shfl_sync(0xffffffff, payload, 0);
        tqs = cuda::std::bit_cast<TQSignal>(payload);
        const auto* __restrict__ gtQ = pA.tQ + tqs.decodeSig();
        if (!tqs.interrupt && threadIdx.x < taskWidth) {
          reinterpret_cast<uint4*>(&currentTask)[threadIdx.x] = reinterpret_cast<const uint4*>(gtQ)[threadIdx.x];
        }
      }
      __syncthreads();
      tqs.interrupt = globalInterrupt;
      if (!tqs.interrupt) {
        FLASHMOE_TRACE_EVENT_T0(procTaskLoaded,
          (static_cast<uint>(currentTask.getTaskType()) << 16) | currentTask.localExpertIdx(),
          currentTask.getTaskType() == TaskType::combine ? currentTask.combineTileIdx() : currentTask.tileIdx);
        // shared -> registers
        switch (const auto task = currentTask; task.getTaskType()) {
        case TaskType::GEMM0: {
          if constexpr (recipe == QuantRecipe::mxfp8) {
            // block-scaled GEMM0, the epilogue quantizes the result into the e4m3 A operand of GEMM1
            constexpr int bM = cute::get<0>(typename TileGEMM0::TileShape{});
            constexpr int bN = cute::get<1>(typename TileGEMM0::TileShape{});
            constexpr bool isGated = mt == MLPMatmulType::gated;
            using MXGEMM0 = tile::sm120::MXFP8GEMM0<bM, bN, threads, isGated>;
            const auto expert = task.localExpertIdx();
            const auto slot = static_cast<size_t>(task.localPeerIdx()) * mxA.nLx + expert;
            // A: dispatched e4m3 tokens with an H-byte row pitch and their scale factors
            const auto* __restrict__ aP = reinterpret_cast<const uint8_t*>(task.aData);
            const auto* __restrict__ sfaP = reinterpret_cast<const uint8_t*>(
              symHeap.advanceSF<0, 1>(task.localPeerIdx(), expert));
            // B: e4m3 weights, [I, H] (K-major) per expert
            const size_t sfUpBytes = mx::sfBytes(I, H / mx::BLOCK_SIZE);
            const auto* __restrict__ bP = reinterpret_cast<const uint8_t*>(expertUpWeights) +
              expertWeightSize * expert;
            const auto* __restrict__ sfbP = mxA.upSF + sfUpBytes * expert;
            // C: e4m3 staging with an I-byte row pitch and its scale factors
            auto* __restrict__ cP = reinterpret_cast<uint8_t*>(task.cData[0]);
            auto* __restrict__ sfcP = mxA.stagingSF + mx::sfBytes(roundEC, I / mx::BLOCK_SIZE) * slot;
            const auto* __restrict__ biasP = biasUp + static_cast<size_t>(I) * expert;
            const auto tileCoord = tile::idx2Coord(task.M() / bM, I / bN, task.tileIdx);
            constexpr MXGEMM0 mainloop{};
            auto acc = MXGEMM0::makeAccumulator();
            if constexpr (isGated) {
              const auto* __restrict__ bPv = reinterpret_cast<const uint8_t*>(expertUpVWeights) +
                expertWeightSize * expert;
              const auto* __restrict__ sfbPv = mxA.upVSF + sfUpBytes * expert;
              const auto* __restrict__ biasPv = biasUpV + static_cast<size_t>(I) * expert;
              auto acc2 = MXGEMM0::makeAccumulator();
              mainloop(workspace, aP, sfaP, bP, sfbP, bPv, sfbPv, acc, acc2, task.M(), I, H, tileCoord);
              FLASHMOE_TRACE_EVENT_T0(procMainDone, 0, 0);
              MXGEMM0::template epilogueGatedMXFP8<Activation>(acc, acc2, cP, sfcP, biasP, biasPv,
                swishAlpha, swishBeta, task.M(), I, tileCoord);
            }
            else {
              mainloop(workspace, aP, sfaP, bP, sfbP, acc, task.M(), I, H, tileCoord);
              FLASHMOE_TRACE_EVENT_T0(procMainDone, 0, 0);
              MXGEMM0::template epilogueMXFP8<Activation>(acc, cP, sfcP, biasP, task.M(), I, tileCoord);
            }
          }
          else if constexpr (isFP4(recipe)) {
            // block-scaled FP4 GEMM0, the epilogue quantizes the result into the E2M1 A operand of GEMM1
            constexpr int bM = cute::get<0>(typename TileGEMM0::TileShape{});
            constexpr int bN = cute::get<1>(typename TileGEMM0::TileShape{});
            constexpr bool isGated = mt == MLPMatmulType::gated;
            constexpr int sfVec = quantSFVecSize(recipe);
            using FPGEMM0 = tile::sm120::FP4GEMM0<recipe, bM, bN, threads, isGated>;
            const auto expert = task.localExpertIdx();
            const auto slot = static_cast<size_t>(task.localPeerIdx()) * mxA.nLx + expert;
            // A: dispatched E2M1 tokens with an H / 2-byte row pitch and their scale factors
            const auto* __restrict__ aP = reinterpret_cast<const uint8_t*>(task.aData);
            const auto* __restrict__ sfaP = reinterpret_cast<const uint8_t*>(
              symHeap.advanceSF<0, 1>(task.localPeerIdx(), expert));
            // B: E2M1 weights, [I, H / 2 bytes] (K-major) per expert
            const size_t sfUpBytes = mx::sfBytes(I, H / sfVec);
            const auto* __restrict__ bP = reinterpret_cast<const uint8_t*>(expertUpWeights) +
              expertWeightBytes * expert;
            const auto* __restrict__ sfbP = mxA.upSF + sfUpBytes * expert;
            // C: E2M1 staging with an I / 2-byte row pitch and its scale factors
            auto* __restrict__ cP = reinterpret_cast<uint8_t*>(task.cData[0]);
            auto* __restrict__ sfcP = mxA.stagingSF + mx::sfBytes(roundEC, I / sfVec) * slot;
            const auto* __restrict__ biasP = biasUp + static_cast<size_t>(I) * expert;
            // nvfp4: the accumulator is scaled by the decode scales of A and B, and the output by the encode scale
            // of the GEMM0 output tensor; mxfp4 has no tensor scales
            float alphaG = 1.f;
            float alphaU = 1.f;
            float st = 1.f;
            if constexpr (recipe == QuantRecipe::nvfp4) {
              alphaG = mxA.inputScale * mxA.upGS[expert];
              if constexpr (isGated) {
                alphaU = mxA.inputScale * mxA.upVGS[expert];
              }
              st = fp4::nvfp4EncodeScale(mxA.interimScale);
            }
            const auto tileCoord = tile::idx2Coord(task.M() / bM, I / bN, task.tileIdx);
            constexpr FPGEMM0 mainloop{};
            auto acc = FPGEMM0::makeAccumulator();
            if constexpr (isGated) {
              const auto* __restrict__ bPv = reinterpret_cast<const uint8_t*>(expertUpVWeights) +
                expertWeightBytes * expert;
              const auto* __restrict__ sfbPv = mxA.upVSF + sfUpBytes * expert;
              const auto* __restrict__ biasPv = biasUpV + static_cast<size_t>(I) * expert;
              auto acc2 = FPGEMM0::makeAccumulator();
              mainloop(workspace, aP, sfaP, bP, sfbP, bPv, sfbPv, acc, acc2, task.M(), I, H, tileCoord);
              FLASHMOE_TRACE_EVENT_T0(procMainDone, 0, 0);
              FPGEMM0::template epilogueGatedFP4<Activation>(acc, acc2, cP, sfcP, biasP, biasPv, alphaG, alphaU,
                swishAlpha, swishBeta, st, task.M(), I, tileCoord);
            }
            else {
              mainloop(workspace, aP, sfaP, bP, sfbP, acc, task.M(), I, H, tileCoord);
              FLASHMOE_TRACE_EVENT_T0(procMainDone, 0, 0);
              FPGEMM0::template epilogueFP4<Activation>(acc, cP, sfcP, biasP, alphaG, st, task.M(), I, tileCoord);
            }
          }
          else {
            const auto* aP = reinterpret_cast<const Element*>(task.aData);
            const auto* bP = expertUpWeights + expertWeightSize * task.localExpertIdx();
            auto* __restrict__ cP = reinterpret_cast<Element*>(task.cData[0]);
            const auto* __restrict__ biasP = biasUp + static_cast<size_t>(I) * task.localExpertIdx();
            if constexpr (mt == MLPMatmulType::vanilla) {
              fGET<TileGEMM0, Activation>(workspace, aP, bP, cP, biasP, task.M(), I, H, task.tileIdx);
            }
            else {
              const auto* bPv = expertUpVWeights + expertWeightSize * task.localExpertIdx();
              const auto* __restrict__ biasPv = biasUpV + static_cast<size_t>(I) * task.localExpertIdx();
              fGET_gated<TileGEMM0, Activation>(workspace, aP, bP, bPv, cP, biasP, biasPv,
                swishAlpha, swishBeta, task.M(), I, H, task.tileIdx);
            }
          }
          __syncthreads();
          if (!threadIdx.x) {
            cuda::atomic_ref<uint, cuda::thread_scope_device> tileSync{*(pA.tQS + task.syncIdx)};
            const auto isLast = tileSync.fetch_add(1, cuda::memory_order_acq_rel) + 1 == tilesN0;
            enqueue = isLast;
            if (isLast && (topo == Topology::NVLINK_ONLY || !task.isPeerRemote())) {
              // clear the counter as it would no longer be used in this epoch
              tileSync.store(0, cuda::memory_order_relaxed);
            }
          }
          __syncthreads();
          if (enqueue) {
            const auto offset = tilesN1 * (task.tileIdx / tilesN0);
            auto* __restrict__ tQ = pA.ptQ + (task.syncIdx * tilesN1);
            if (topo == Topology::MIXED && task.isPeerRemote()) {
              notifyNext<PeerConnectivity::remote, threads>(workspace, task, tQ, tilesN1, flagColStride,
                offset, pA.tQH);
            }
            else {
              notifyNext<PeerConnectivity::p2p, threads>(workspace, task, tQ, tilesN1, flagColStride,
                offset, pA.tQH);
            }
          }
          __syncthreads();
        }
        break;
        case TaskType::GEMM1: {
          // The tokens of a task of this device's own peer stay here: their GEMM1 epilogue is the combine, it writes
          // (single) or adds (plural) straight into moeOutput. So there is no write to the symmetric heap, no flag
          // (no system scope release) and no combine task, see os::start. The packets of any other peer take the
          // path below: bf16 tile to the heap of the peer, flag, combine task on the peer.
          // localPeerIdx is the EP rank of the peer (pe() is the NVSHMEM PE, a different number in general)
          constexpr bool fuseLocal = fusesLocalCombine<TileGEMM1::TileArch::value>();
          const bool isLocal = fuseLocal && static_cast<int>(task.localPeerIdx()) == epRank;
          constexpr int bM1 = cute::get<0>(typename TileGEMM1::TileShape{});
          // tokenIndices has shape [E, roundEC]: the tile rows are those of the combine task of the tile
          const auto* __restrict__ tokenIds = tokenIndices +
            (static_cast<size_t>(task.expertIdx()) * roundEC + static_cast<size_t>(task.tileIdx / tilesN1) * bM1);
          if constexpr (recipe == QuantRecipe::mxfp8) {
            // block-scaled GEMM1 with a bf16 output: the tile goes to the heap for a combine task or is combined here
            constexpr int bM = cute::get<0>(typename TileGEMM1::TileShape{});
            constexpr int bN = cute::get<1>(typename TileGEMM1::TileShape{});
            using MXGEMM1 = tile::sm120::MXFP8GEMM1<bM, bN, threads>;
            const auto expert = task.localExpertIdx();
            const auto slot = static_cast<size_t>(task.localPeerIdx()) * mxA.nLx + expert;
            // A: the e4m3 GEMM0 staging and its scale factors
            const auto* __restrict__ aP = reinterpret_cast<const uint8_t*>(task.aData);
            const auto* __restrict__ sfaP = mxA.stagingSF + mx::sfBytes(roundEC, I / mx::BLOCK_SIZE) * slot;
            // B: e4m3 weights, [H, I] (K-major) per expert
            const auto* __restrict__ bP = reinterpret_cast<const uint8_t*>(expertDownWeights) +
              expertWeightSize * expert;
            const auto* __restrict__ sfbP = mxA.downSF + mx::sfBytes(H, I / mx::BLOCK_SIZE) * expert;
            auto* __restrict__ cP = reinterpret_cast<Element*>(task.cData[1]);
            const auto* __restrict__ biasP = biasDown + static_cast<size_t>(H) * expert;
            const auto tileCoord = tile::idx2Coord(task.M() / bM, H / bN, task.tileIdx);
            constexpr MXGEMM1 mainloop{};
            auto acc = MXGEMM1::makeAccumulator();
            // local: token indices, probabilities and bias into registers, the loads overlap with the mainloop
            typename MXGEMM1::CombineRegs pf{};
            if (isLocal) {
              pf = MXGEMM1::prefetchCombine(tokenIds, biasP, task.tileSize(), tileCoord);
            }
            mainloop(workspace, aP, sfaP, bP, sfbP, acc, task.M(), H, I, tileCoord);
            FLASHMOE_TRACE_EVENT_T0(procMainDone, 0, 0);
            if (isLocal) {
              MXGEMM1::template epilogueCombine<combineMode == CombineMode::plural, cublasdx::identity>(acc,
                moeOutput, pf, task.tileSize(), H, tileCoord);
            }
            else {
              MXGEMM1::template epilogueBF16<cublasdx::identity>(acc, cP, biasP, task.M(), H, tileCoord);
            }
          }
          else if constexpr (isFP4(recipe)) {
            // block-scaled FP4 GEMM1 with a bf16 output: the tile goes to the heap for a combine task or is combined
            // here
            constexpr int bM = cute::get<0>(typename TileGEMM1::TileShape{});
            constexpr int bN = cute::get<1>(typename TileGEMM1::TileShape{});
            constexpr int sfVec = quantSFVecSize(recipe);
            using FPGEMM1 = tile::sm120::FP4GEMM1<recipe, bM, bN, threads>;
            const auto expert = task.localExpertIdx();
            const auto slot = static_cast<size_t>(task.localPeerIdx()) * mxA.nLx + expert;
            // A: the E2M1 GEMM0 staging and its scale factors
            const auto* __restrict__ aP = reinterpret_cast<const uint8_t*>(task.aData);
            const auto* __restrict__ sfaP = mxA.stagingSF + mx::sfBytes(roundEC, I / sfVec) * slot;
            // B: E2M1 weights, [H, I / 2 bytes] (K-major) per expert
            const auto* __restrict__ bP = reinterpret_cast<const uint8_t*>(expertDownWeights) +
              expertWeightBytes * expert;
            const auto* __restrict__ sfbP = mxA.downSF + mx::sfBytes(H, I / sfVec) * expert;
            auto* __restrict__ cP = reinterpret_cast<Element*>(task.cData[1]);
            const auto* __restrict__ biasP = biasDown + static_cast<size_t>(H) * expert;
            // nvfp4: decode scales of the GEMM0 output (A) and of the down weights (B)
            float alpha = 1.f;
            if constexpr (recipe == QuantRecipe::nvfp4) {
              alpha = mxA.interimScale * mxA.downGS[expert];
            }
            const auto tileCoord = tile::idx2Coord(task.M() / bM, H / bN, task.tileIdx);
            constexpr FPGEMM1 mainloop{};
            auto acc = FPGEMM1::makeAccumulator();
            // local: token indices, probabilities and bias into registers, the loads overlap with the mainloop
            typename FPGEMM1::CombineRegs pf{};
            if (isLocal) {
              pf = FPGEMM1::prefetchCombine(tokenIds, biasP, task.tileSize(), tileCoord);
            }
            mainloop(workspace, aP, sfaP, bP, sfbP, acc, task.M(), H, I, tileCoord);
            FLASHMOE_TRACE_EVENT_T0(procMainDone, 0, 0);
            if (isLocal) {
              FPGEMM1::template epilogueCombine<combineMode == CombineMode::plural, cublasdx::identity>(acc,
                moeOutput, pf, alpha, task.tileSize(), H, tileCoord);
            }
            else {
              FPGEMM1::template epilogueBF16<cublasdx::identity>(acc, cP, biasP, alpha, task.M(), H, tileCoord);
            }
          }
          else {
            const auto* aP = reinterpret_cast<const Element*>(task.aData);
            const auto* bP = expertDownWeights + expertWeightSize * task.localExpertIdx();
            auto* __restrict__ cP = reinterpret_cast<Element*>(task.cData[1]);
            const auto* __restrict__ biasP = biasDown + static_cast<size_t>(H) * task.localExpertIdx();
            // local: the epilogue is combineScatter of the tile, which is made of the token indices, probabilities
            // loaded before the mainloop
            fGET<TileGEMM1, cublasdx::identity, fuseLocal, combineMode, threads>(workspace, aP, bP, cP, biasP,
              task.M(), H, I, task.tileIdx,
              LocalCombine<Element>{moeOutput, isLocal ? tokenIds : nullptr, S, task.tileSize()});
          }
          if (isLocal) {
            // done: the tile has been combined into moeOutput. The next task of this block starts with a barrier,
            // which also orders this epilogue before whatever reuses the workspace
            break;
          }
          __syncthreads();
          if (!threadIdx.x) {
            constexpr int bM = cute::get<0>(typename TileGEMM1::TileShape{});
            const auto mCoord = task.tileIdx / tilesN1;
            const auto nCoord = task.tileIdx % tilesN1;
            if (topo == Topology::MIXED && task.isPeerRemote()) {
              // Remote; check if we need to do the transfer
              cuda::atomic_ref<uint, cuda::thread_scope_device> tileSync{*(pA.tQS + task.syncIdx)};
              const auto doTransfer = tileSync.fetch_add(1,cuda::memory_order_acq_rel) + 1 == (tilesN0 + tilesN1);
              if (doTransfer) {
                // read and flip the current bit
                const auto prevBit = producerBitMap(task.localPeerIdx(), task.localExpertIdx(), mCoord, 0);
                const auto producerBit = static_cast<uint8_t>(prevBit == 0 ? 1 : 0);
                // Pack payload into single signal word of 8 bytes
                const auto flagSignal = SignalPayload<PacketStage::last>{
                  mCoord,
                  task.tileSize(),
                  producerBit,
                  stateNumber,
                };
                const auto sigPayload = cuda::std::bit_cast<uint64_t>(flagSignal);
                // flip the bit
                producerBitMap(task.localPeerIdx(), task.localExpertIdx(), mCoord, 0) = producerBit;
                // clear the counter as it would no longer be used in this epoch
                tileSync.store(0, cuda::memory_order_relaxed);
                const auto symOffset = static_cast<size_t>(bM) * mCoord * H * sizeof(Element);
                nvshmem_putmem_signal_nbi(task.rcData + symOffset,
                                          task.cData[1] + symOffset,
                                          static_cast<size_t>(task.tileSize() * H) * sizeof(Element),
                                          task.flags,
                                          sigPayload,
                                          NVSHMEM_SIGNAL_SET,
                                          task.pe());
              }
            }
            else {
              // read and flip the current bit
              const auto prevBit = producerBitMap(task.localPeerIdx(), task.localExpertIdx(), mCoord, nCoord);
              const auto producerBit = static_cast<uint8_t>(prevBit == 0 ? 1 : 0);
              // Pack payload into single signal word of 8 bytes
              const auto flagSignal = SignalPayload<PacketStage::last>{
                mCoord,
                task.tileSize(),
                producerBit,
                stateNumber,
              };
              const auto sigPayload = cuda::std::bit_cast<uint64_t>(flagSignal);
              // flip the bit
              producerBitMap(task.localPeerIdx(), task.localExpertIdx(), mCoord, nCoord) = producerBit;
              // individual tile, no batching here,
              // note, this CTA alone did the transfer over NVlink
              cuda::atomic_ref<uint64_t, cuda::thread_scope_system> remoteFlags{*task.flags};
              remoteFlags.store(sigPayload, cuda::memory_order_release);
            }
          }
          __syncthreads();
        }
        break;
        case TaskType::combine: {
          const auto tileCoord = cute::make_coord(cute::_0{}, task.combineTileIdx());
          const auto* __restrict__ tokens = reinterpret_cast<Element*>(symHeap.advance<1, 1>(task.epRank(),
            task.localExpertIdx(), static_cast<int>(task.tokenBatchStart())));
          using Tiler = TileGEMM1::TileShape;
          constexpr int bM = cute::get<0>(Tiler{});
          constexpr int bN = cute::get<1>(Tiler{});
          constexpr int Arch = TileGEMM1::TileArch::value;
          // tIds has shape [E, roundEC] with a row-major layout
          const auto* __restrict__ tokenIds = tokenIndices + (task.expertIdx() * roundEC + task.tokenBatchStart());
          combine<bM, bN, Arch, threads, combineMode>(S, H, workspace, tokenIds, moeOutput, tokens, task.tileSize(),
          tileCoord);
        }
        break;
        }
        FLASHMOE_TRACE_EVENT_T0(procTaskDone,
          (static_cast<uint>(currentTask.getTaskType()) << 16) | currentTask.tileSize(),
          currentTask.getTaskType() == TaskType::combine ? currentTask.combineTileIdx() : currentTask.tileIdx);
      }
    }
  }
}
#endif //FLASHMOE_COMPUTE_CUH
