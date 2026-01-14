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
#include "infra/packed.cuh"
#include "infra/signal.cuh"
#include "infra/task.cuh"
#include "tile.cuh" // cublasdx and tileGEMM
#include "infra/tq.cuh"
namespace flashmoe::processor{
    constexpr int WARP_SIZE = 32;
    // fused GEMM, epilogue and data transfer
    template<
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
        // gmem -> rmem: prefetch bias
        const auto gD = tile::getBias<BM{}, BN{}>(bias, M, N, cute::select<0, 1>(tileCoord));
        auto d_frag = cublasdx::make_fragment_like<ElementC>(accumulator.get_results());
        cublasdx::copy_fragment<cublasdx::alignment_of<BLAS>::c>(gD, d_frag, accumulator);
        // compute Tile
        constexpr TileGEMM tileMainloop{};
        tileMainloop(workspace, a, b, accumulator, M, N, K, tileCoord);
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
        auto gC = tile::getC<BM{}, BN{}, cublasdx::arrangement_of_v_c<BLAS>>(c, M, N,
            cute::select<0, 1>(tileCoord));
        // rmem -> gmem
        cublasdx::copy_fragment<cublasdx::alignment_of<BLAS>::c>(d_frag, gC, accumulator);
    }

    struct __align__(16) ProcessorArgs{
        // sensible sentinel values
        unsigned int* __restrict__ sQ = nullptr;
        uint64_t* __restrict__ pDB = nullptr;
        unsigned int* __restrict__ tQH = nullptr;
        Task* __restrict__ tQ = nullptr;
        Task* __restrict__ ptQ = nullptr;
        unsigned int* __restrict__ tQS = nullptr;

        ProcessorArgs() = default;
        __device__
        ProcessorArgs(unsigned int* const& _sQ,
            TQSignal* const& _pDB,
            unsigned int* const& _tQH,
            Task* const& _tQ,
            Task* const& _ptQ,
            unsigned int* const& _tQS) :
        sQ(_sQ), pDB(reinterpret_cast<uint64_t*>(_pDB)), tQH(_tQH), tQ(_tQ), ptQ(_ptQ), tQS(_tQS) {}
    };

    template<
        PeerConnectivity p,
        int threads
    >
    __device__ __forceinline__
    void notifyNext(void* __restrict__ const& workspace, const Task& task, Task* __restrict__ const& tQ,
        const uint& tasks, const uint& offset, uint* __restrict__ const& tQH) {
        auto* __restrict__ sTQ = static_cast<Task*>(workspace);
        // registers -> shared memory
        for (int i = threadIdx.x; i < tasks; i += threads) {
            sTQ[i] = Task{task.ingredients, task.cData[0], task.cData, task.rcData,
            task.flags + offset + (p == PeerConnectivity::p2p ? i : 0), task.syncIdx, offset + i};
        }
        __syncthreads();
        // shared memory to global memory
        static_assert(sizeof(Task) % sizeof(uint4) == 0 && alignof(Task) % alignof(uint4) == 0);
        constexpr int nRows = sizeof(Task) / sizeof(uint4);
        const int numElems = tasks * nRows;
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

    template<
        int threads,
        CombineMode combineMode,
        typename TileGEMM0, // Descriptor for GEMM0
        typename TileGEMM1, // Descriptor for GEMM1
        typename Activation, // Activation function after GEMM0
        typename Element
    >
    __device__ __forceinline__
    void start(void* __restrict__ const& workspace, // shared memory
        const int& S, // sequence length
        const int& H, // token hidden dimension
        const int& I, // FFN intermediate size
        const int& E, // total number of experts
        const int& k, // top k
        const uint& tilesN0, // I / bN0
        const uint& tilesN1, // H / bN1
        const Element* __restrict__ const& expertUpWeights, // [num_local_experts, H, I]
        const Element* __restrict__ const& biasUp, // [num_local_experts, I]
        const Element* __restrict__ const& expertDownWeights, // [num_local_experts, I, H]
        const Element* __restrict__ const& biasDown, // [num_local_experts, H]
        const TPS* __restrict__ const& tokenIds, // [E, roundEC]
        Element* __restrict__ const& moeOutput,
        const uint16_t& seqNumber,
        const Heap& symHeap,
        const Bookkeeping& bookkeeping){
        __shared__ Task currentTask;
        __shared__ uint globalInterrupt;
        __shared__ uint enqueue;
        // Register allocations
        Task task{};
        TQSignal tqs{0U, 0U};
        static_assert(sizeof(TQSignal) == sizeof(uint64_t) && alignof(TQSignal) == alignof(uint64_t));
        static_assert(sizeof(SignalPayload<PacketStage::last>) == sizeof(uint64_t) &&
            alignof(SignalPayload<PacketStage::last>) == alignof(uint64_t));
        const auto pA = ProcessorArgs{
            bookkeeping.sQ() + blockIdx.x,
            bookkeeping.pDB() + blockIdx.x,
            bookkeeping.tQH(),
            bookkeeping.tQ(),
            bookkeeping.ptQ(),
            bookkeeping.tSA()
        };
        if (!threadIdx.x) {
            atomicExch_block(&globalInterrupt, 0U);
            atomicExch_block(&enqueue, 0U);
        }
        __syncthreads();
        using SQT = cuda::std::underlying_type_t<flashmoe::ReadySignal>;
        constexpr auto taskWidth = sizeof(Task) / sizeof(uint4);
        const long int expertWeightSize = I * H;
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
                // The below is necessary as it ensures memory ordering
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
                task = currentTask;
                switch (task.getTaskType()) {
                    case TaskType::GEMM0: {
                        const auto* aP = reinterpret_cast<const Element*>(task.aData);
                        const auto* bP = expertUpWeights + expertWeightSize * task.localExpertIdx();
                        auto* __restrict__ cP = reinterpret_cast<Element*>(task.cData[0]);
                        const auto* __restrict__ biasP = biasUp + I * task.localExpertIdx();
                        fGET<TileGEMM0, Activation>(workspace, aP, bP, cP, biasP, task.M(), I, H,
                            task.tileIdx);
                        __syncthreads();
                        if (!threadIdx.x) {
                            cuda::atomic_ref<uint, cuda::thread_scope_device> tileSync{*(pA.tQS + task.syncIdx)};
                            const auto isLast = tileSync.fetch_add(1, cuda::memory_order_acq_rel) + 1 == tilesN0;
                            enqueue = isLast;
                            if (isLast && !task.isPeerRemote()) {
                                // clear the counter as it would no longer be used in this epoch
                                tileSync.store(0, cuda::memory_order_relaxed);
                            }
                        }
                        __syncthreads();
                        if (enqueue) {
                            const auto offset = tilesN1 * task.flagBatchIdx();
                            auto* __restrict__ tQ = pA.ptQ + (task.syncIdx * tilesN1);
                            if (!task.isPeerRemote()) {
                                notifyNext<PeerConnectivity::p2p,threads>(workspace, task, tQ, tilesN1, offset, pA.tQH);
                            }
                            else {
                                notifyNext<PeerConnectivity::remote,threads>(workspace, task, tQ, tilesN1, offset, pA.tQH);
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
                        fGET<TileGEMM1, cublasdx::identity>(workspace, aP, bP, cP, biasP, task.M(), H, I,
                            task.tileIdx);
                        __syncthreads();
                        if (!threadIdx.x) {
                            // Pack payload into single signal word of 8 bytes
                            const auto flagSignal = SignalPayload<PacketStage::last>{
                                task.flagBatchIdx(),
                                task.tileSize(),
                                seqNumber,
                            };
                            const auto sigPayload = cuda::std::bit_cast<uint64_t>(flagSignal);
                            if (task.isPeerRemote()) {
                                // Remote; check if we need to do the transfer
                                cuda::atomic_ref<uint, cuda::thread_scope_device> tileSync{*(pA.tQS + task.syncIdx)};
                                const auto doTransfer = tileSync.fetch_add(1,
                                    cuda::memory_order_acq_rel) + 1 == (tilesN0 + tilesN1);
                                if (doTransfer) {
                                    // clear the counter as it would no longer be used in this epoch
                                    tileSync.store(0, cuda::memory_order_relaxed);
                                    nvshmem_putmem_signal_nbi(task.rcData,
                                         task.cData[1],
                                         // Batched remote network transfer to avoid overwhelming the NIC
                                         task.tileSize() * H * sizeof(Element),
                                         task.flags,
                                         sigPayload,
                                         NVSHMEM_SIGNAL_SET,
                                         task.peerIdx());
                                }
                            }
                            else {
                                // individual tile, no batching here
                                // Already did the network transfer,
                                // so set signal only
                                cuda::atomic_ref<uint64_t, cuda::thread_scope_system> remoteFlags{*task.flags};
                                remoteFlags.store(sigPayload, cuda::memory_order_release);
                            }
                        }
                        __syncthreads();
                    }
                    break;
                    case TaskType::combine: {
                        combine<TileGEMM1, threads, combineMode>(symHeap, workspace, tokenIds, S, E, H,
                            moeOutput, task);
                    }
                    break;
                }
            }
        }
        // last block clears tguardGuar
    }
}
#endif //FLASHMOE_COMPUTE_CUH
