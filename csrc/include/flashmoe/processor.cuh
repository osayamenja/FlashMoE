/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Flashmoe Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 */
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
#include "infra/task.cuh"
#include "tile.cuh" // cublasdx and tileGEMM
#include "infra/tq.cuh"

namespace flashmoe
{
  enum class MLPMatmulType {
    gated,
    vanilla
  };
}
namespace flashmoe::processor
{
  // fused GEMM, epilogue and data transfer
  template <
    typename TileGEMM,
    typename Activation,
    typename Element,
    typename ElementC
  >
  __forceinline__ __device__
  void fGET(void* __restrict__ const& workspace,
            const Element* __restrict__ const& a,
            const Element* __restrict__ const& b,
            ElementC* __restrict__ const& c,
            const ElementC* __restrict__ const& bias,
            const int& M, const int& N, const int& K, const int& tileIdx) {
    using BLAS = TileGEMM::BLAS;
    auto accumulator = BLAS::suggest_accumulator();
    using BM = cute::Int<cublasdx::size_of<BLAS>::m>;
    using BN = cute::Int<cublasdx::size_of<BLAS>::n>;
    const auto tileCoord = tile::idx2Coord(M / BM{}, N / BN{}, tileIdx);
    // compute Tile
    constexpr TileGEMM tileMainloop{};
    tileMainloop(workspace, a, b, accumulator, M, N, K, tileCoord);
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
    // rmem -> smem
    auto sC = cublasdx::make_tensor(static_cast<ElementC*>(workspace), BLAS::suggest_layout_smem_c());
    __syncthreads();
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
    constexpr Converter<typename decltype(accumulator)::value_type, ElementC> loadConv{};
    // accum type -> ElementC
    constexpr Converter<ElementC, typename decltype(accumulator)::value_type> storeConv{};
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
    auto* __restrict__ sTQ = static_cast<Task*>(workspace);
    auto ingredients = task.ingredients;
    ingredients.taskType = TaskType::GEMM1;
    // registers -> shared memory
    for (int i = threadIdx.x; i < tasks; i += threads) {
      sTQ[i] = Task{
        ingredients, task.cData[0], task.cData, task.rcData,
        task.flags + (p == PeerConnectivity::remote ? 0 : i * flagColStride), // col-major indexing
        task.syncIdx, offset + i
      };
    }
    __syncthreads();
    // shared memory to global memory
    constexpr int nRows = sizeof(Task) / sizeof(uint4);
    const uint numElems = tasks * nRows;
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

  template <
    MLPMatmulType mt,
    Topology topo,
    int threads,
    CombineMode combineMode,
    typename TileGEMM0, // Descriptor for GEMM0
    typename TileGEMM1, // Descriptor for GEMM1
    typename Activation, // Activation function after GEMM0
    typename Element,
    typename PBM
  >
  __device__ __forceinline__
  void start(cuda::std::byte* __restrict__ const& workspace, // shared memory
             const int& S, // sequence length
             const int& H, // token hidden dimension
             const int& I, // FFN intermediate size
             const int& E, // total number of experts
             const int& roundEC,
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
             const Heap& symHeap, const ProcessorArgs& pA) {
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
    static_assert(taskWidth > 0 && taskWidth < WARP_SIZE);
    while (!tqs.interrupt) {
      if (threadIdx.x / WARP_SIZE == 0) {
        if (threadIdx.x == 0) {
          cuda::atomic_ref<uint64_t, cuda::thread_scope_device> doorbell{*pA.pDB};
          // await new task from scheduler
          auto payload = cuda::std::bit_cast<TQSignal>(doorbell.load(cuda::memory_order_acquire));
          while (payload.signal == tqs.signal && payload.interrupt == 0) {
            payload = cuda::std::bit_cast<TQSignal>(doorbell.load(cuda::memory_order_acquire));
          }
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
        // shared -> registers
        switch (const auto task = currentTask; task.getTaskType()) {
        case TaskType::GEMM0: {
          const auto* aP = reinterpret_cast<const Element*>(task.aData);
          const auto* bP = expertUpWeights + expertWeightSize * task.localExpertIdx();
          auto* __restrict__ cP = reinterpret_cast<Element*>(task.cData[0]);
          const auto* __restrict__ biasP = biasUp + I * task.localExpertIdx();
          if constexpr (mt == MLPMatmulType::vanilla) {
            fGET<TileGEMM0, Activation>(workspace, aP, bP, cP, biasP, task.M(), I, H, task.tileIdx);
          }
          else {
            const auto* bPv = expertUpVWeights + expertWeightSize * task.localExpertIdx();
            const auto* __restrict__ biasPv = biasUpV + I * task.localExpertIdx();
            fGET_gated<TileGEMM0, Activation>(workspace, aP, bP, bPv, cP, biasP, biasPv,
              swishAlpha, swishBeta, task.M(), I, H, task.tileIdx);
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
          const auto* aP = reinterpret_cast<const Element*>(task.aData);
          const auto* bP = expertDownWeights + expertWeightSize * task.localExpertIdx();
          auto* __restrict__ cP = reinterpret_cast<Element*>(task.cData[1]);
          const auto* __restrict__ biasP = biasDown + H * task.localExpertIdx();
          fGET<TileGEMM1, cublasdx::identity>(workspace, aP, bP, cP, biasP, task.M(), H, I, task.tileIdx);
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
                                          // Batched remote network transfer to avoid overwhelming the NIC
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
          const auto tIds = cute::make_tensor(cute::make_gmem_ptr(tokenIndices),
            cute::make_layout(cute::make_shape(E, roundEC), cute::LayoutRight{}));
          const auto* __restrict__ tokenIds = &tIds(task.expertIdx(), task.tokenBatchStart());
          combine<bM, bN, Arch, threads, combineMode>(S, H, workspace, tokenIds, moeOutput, tokens, task.tileSize(),
          tileCoord);
        }
        break;
        }
      }
    }
  }
}
#endif //FLASHMOE_COMPUTE_CUH
