//
// Created by osayamen on 7/13/24.
//

#ifndef ARISTOS_COMPUTE_CUH
#define ARISTOS_COMPUTE_CUH

#include <cuda/std/type_traits>
#include <cutlass/array.h>
#include <cute/tensor.hpp>
#include <cutlass/epilogue/thread/activation.h>

#include <nvshmemx.h>
#include <nvshmem.h>
#include <host/nvshmemx_api.h>

#include "gemm.cuh"

namespace aristos::processor{
    // Fused GEMM, Epilogue and Data transfer
    template<
        typename BlockGEMM,
        TaskType t = TaskType::preGEMM,
        class FrgTensorD,
        class RegisterScratch
    >
    requires(t == TaskType::preGEMM || t == TaskType::postGEMM)
    __forceinline__ __device__
    void fGET(typename BlockGEMM::MatrixDType* __restrict__ workspace,
        FrgTensorD& accumulator,
        RegisterScratch& rScratch,
        const typename BlockGEMM::MatrixAType* __restrict__ inputs,
        const typename BlockGEMM::MatrixBType* __restrict__ weights,
        typename BlockGEMM::MatrixDType* __restrict__ output,
        const typename BlockGEMM::MatrixDType* __restrict__ bias,
        const unsigned int& M,
        const unsigned int& N,
        const unsigned int& K,
        const unsigned int& tileIdx) {
        static_assert(size(accumulator) % rScratch.size() == 0 && cutlass::detail::is_Array_v<RegisterScratch>);
        // Instantiate mainloop
        typename BlockGEMM::CollectiveMainloop mainLoop{};
        cute::clear(accumulator);

        // Row-major
        auto mA = make_tensor(cute::make_gmem_ptr(inputs),
            make_layout(cute::make_shape(M, K), cute::make_stride(K, 1)));
        // Row-major, transposed
        auto mB = make_tensor(cute::make_gmem_ptr(weights),
            make_layout(cute::make_shape(N, K), cute::make_stride(K, 1)));
        // Row-major
        auto mC = make_tensor(cute::make_gmem_ptr(output,
            make_layout(cute::make_shape(M, N), cute::make_stride(N, 1))));
        auto mD = make_tensor(cute::make_gmem_ptr(bias),
            make_layout(cute::make_shape(M, N), cute::make_stride(0, 1)));

        // M is padded, such that the below is correct
        const auto tilesM = M / cute::get<0>(BlockGEMM::BlockTiler{});
        // We assert the below prior to this point
        const auto tilesN = N / cute::get<1>(BlockGEMM::BlockTiler{});

        auto tileCoord = idx2crd(tileIdx, cute::Shape(tilesM, tilesN), cute::Stride(tilesN ,1));
        auto ctaCoord = make_coord(cute::get<0>(tileCoord), cute::get<1>(tileCoord), cute::_);
        auto gA = cute::local_tile(mA, typename BlockGEMM::BlockTiler{}, ctaCoord, cute::Step<cute::_1, cute::X,cute::_1>{});
        auto gB = cute::local_tile(mB, typename BlockGEMM::BlockTiler{}, ctaCoord, cute::Step< cute::X,cute::_1,cute::_1>{});
        auto gC = cute::local_tile(mC, typename BlockGEMM::BlockTiler{}, ctaCoord, cute::Step<cute::_1,cute::_1, cute::X>{});
        auto gD = cute::local_tile(mD, typename BlockGEMM::BlockTiler{}, ctaCoord, cute::Step<cute::_1,cute::_1, cute::X>{});

        auto k_tile_iter = cute::make_coord_iterator(size<2>(gA));
        int k_tile_count = size<2>(gA);

        using ElementD = typename BlockGEMM::MatrixDType;
        mainLoop(
            accumulator,
            gA,
            gB,
            accumulator,
            k_tile_iter, k_tile_count,
            cute::Underscore{},
            threadIdx.x,
            CAST_TO(char, workspace));
        /// There is a block-wide barrier at the end of the above ^

        // Epilogue
        typename BlockGEMM::MMA tiledMMA{};
        auto tCgC = tiledMMA.get_slice(threadIdx.x).partition_C(gC);
        auto tDgD = tiledMMA.get_slice(threadIdx.x).partition_C(gD);

        // Accounts for GEMMs that accumulate in types differing from input types,
        // given that the result may moonlight as the input for the succeeding GEMM.
        auto gCStoreOp = cutlass::NumericConverter<typename decltype(tCgC)::value_type,
                                                    typename decltype(accumulator)::value_type>{};
        auto gDLoadOp = cutlass::NumericConverter<typename decltype(accumulator)::value_type,
                                                    ElementD>{};

        // Assume elementwise operator
        typename BlockGEMM::FusedEpilogue epilogueOp{};
        constexpr auto trips = size(accumulator) / rScratch.size();
        constexpr auto elems = rScratch.size();

        // Prefetch from global to shared memory
        #pragma unroll
        for (int j = 0; j < elems; ++j) {
            workspace[threadIdx.x + j * THREADS] = tDgD(j);
        }

        #pragma unroll
        for (unsigned int i = 0; i < trips; ++i) {
            #pragma unroll
            for (unsigned int j = 0; j < elems; ++j) {
                rScratch[j] = workspace[threadIdx.x + j * THREADS];
                if (i + 1 < trips) {
                    // Eagerly start loads for the next batch, if needed
                    workspace[threadIdx.x + j * THREADS] = tDgD(j + i * elems);
                }
            }
            // Fused Bias Add and Activation Function on register fragment
            // Also fuses copy to GMEM, which is where things get interesting
            #pragma unroll
            for (int j = 0; j < elems; ++j) {
                tCgC(j + i * elems) = gCStoreOp(epilogueOp(accumulator(j + i * elems), gDLoadOp(rScratch[j])));
            }
        }

        __syncthreads();
        if (!threadIdx.x) {
            if constexpr (t == TaskType::preGEMM) {
                __threadfence();
            }
            else {
                // this could be a local or network transfer, thus requires a fence spanning both.
                __threadfence_system();
            }
        }
    }

    template<
        unsigned int processorCount,
        unsigned int Arch,
        typename ElementA,
        typename ElementB,
        typename ElementC = float,
        typename ElementD = ElementA,
        typename ActivationOp = cute::identity,
        typename ActivationOpX = cute::identity
    > requires(processorCount > 0 && Arch >= MIN_ARCH)
    __device__ __forceinline__
    void start(cuda::std::byte* __restrict__ const& workspace){
        assert(__isShared(workspace));
        __shared__ unsigned int signal;
        __shared__ Task currentTask;
        __shared__ unsigned int interrupt;
        const auto* __restrict__ cachedConfig = CAST_TO(Config, workspace);
        const auto* __restrict__ scState = CAST_TO(SchedulerConfig, workspace + sizeof(Config));
        auto* processorWorkspace = CAST_TO(ElementD, workspace + sizeof(Config) + sizeof(SchedulerConfig));
        if (!threadIdx.x) {
            *CAST_TO(Config, workspace) = moeConfig;
            *CAST_TO(SchedulerConfig, workspace + sizeof(Config)) = schedulerState;
        }
        using Operation = BlockMM<Arch, ElementA, ElementB, ElementC, ActivationOp>;
        using OperationX = BlockMM<Arch, ElementA, ElementB, ElementC, ActivationOpX>;
        auto accumulator = cute::partition_fragment_C(typename Operation::MMA{}, typename Operation::TilerOut{});
        constexpr auto elems = SHARED_SIZE / (THREADS * sizeof(ElementD));
        static_assert(cute::size(accumulator) % elems == 0);
        cutlass::AlignedArray<ElementC, elems> rScratch{};

        atomicExch_block(&interrupt, 0U);
        atomicExch_block(&signal, 0UL);
        __syncthreads();

        while (!interrupt) {
            if (!threadIdx.x) {
                // Indicate readiness
                scState->readyQ[atomicAdd(scState->readyQHead, 1U) % processorCount] = blockIdx.x;
                // Grabs next task
                auto nextTask = atomicLoad(scState->taskQSignals + blockIdx.x);
                while (nextTask == signal) {
                    nextTask = atomicLoad(scState->taskQSignals + blockIdx.x);
                }
                signal = nextTask;
                currentTask = scState->taskQ[signal - 1];
            }
            __syncthreads();
            switch (currentTask.taskType) {
                case TaskType::preGEMM: {
                    constexpr unsigned int preIndex = 0;
                    fGET<Operation, TaskType::preGEMM>(processorWorkspace, accumulator, rScratch,
                        CAST_TO(typename Operation::MatrixAType, currentTask.aData),
                        CAST_TO(typename Operation::MatrixBType, currentTask.bData[preIndex]),
                        CAST_TO(typename Operation::MatrixDType, currentTask.cData[preIndex]),
                        CAST_TO(typename Operation::MatrixDType, currentTask.dData[preIndex]),
                        currentTask.M,
                        cachedConfig->upProjection,
                        cachedConfig->embedDim,
                        currentTask.tileIdx);
                    if (!threadIdx.x &&
                        atomicAdd(scState->taskSync + currentTask.syncIdx, 1U) == cachedConfig->tilesN) {
                        const auto tasks = cachedConfig->tilesNx;
                        auto* tQHead = scState->taskQSignals + 1;
                        auto* tQ = scState->taskQ;
                        for (unsigned int i = 0; i < tasks; ++i) {
                            tQ[atomicAdd(tQHead, 1U)] = Task{
                                TaskType::postGEMM,
                                currentTask.cData[preIndex],
                                currentTask.bData,
                                currentTask.cData,
                                currentTask.dData,
                                currentTask.scale,
                                currentTask.syncIdx,
                                i,
                                currentTask.M,
                                currentTask.flagIdx,
                                currentTask.tileSize,
                                currentTask.peerIdx
                            };
                        }
                        __threadfence();
                        // notify scheduler
                        atomicAdd(scState->taskQSignals, tasks);
                    }
                }
                break;
                case TaskType::postGEMM: {
                    constexpr unsigned int postIndex = 0;
                    fGET<OperationX, TaskType::postGEMM>(processorWorkspace, accumulator, rScratch,
                        CAST_TO(typename Operation::MatrixAType, currentTask.aData),
                        CAST_TO(typename Operation::MatrixBType, currentTask.bData[postIndex]),
                        CAST_TO(typename Operation::MatrixDType, currentTask.cData[postIndex]),
                        CAST_TO(typename Operation::MatrixDType, currentTask.dData[postIndex]),
                        cachedConfig->embedDim,
                        cachedConfig->upProjection,
                        currentTask.tileIdx);
                    if (!threadIdx.x) {
                        if (atomicAdd(scState->taskSync + currentTask.syncIdx, 1U)
                            == cachedConfig->tilesN + cachedConfig->tilesNx) {
                            if (nvshmem_ptr(currentTask.cData[postIndex], currentTask.peerIdx) == nullptr) {
                                // Batch remote network transfer to avoid overwhelming the NIC
                                nvshmem_putmem_signal_nbi(currentTask.cData[postIndex], currentTask.cData[postIndex],
                                    cachedConfig->finalPacketSize<ElementA>(currentTask.tileSize),
                                    cachedConfig->flags + currentTask.flagIdx,
                                    constructSignal(PacketStage::final), NVSHMEM_SIGNAL_SET, currentTask.peerIdx);
                            }
                            else {
                                // Already did the network transfer in fGET, so set signal only
                                nvshmemx_signal_op(cachedConfig->flags + currentTask.flagIdx,
                                 constructSignal(PacketStage::final), NVSHMEM_SIGNAL_SET, currentTask.peerIdx);
                            }
                        }
                    }
                }
                break;
                case TaskType::GateScale: {
                    // Do scale
                    // TODO atomicAdd to close scale
                }
                break;
                case TaskType::Interrupt: {
                    if (!threadIdx.x) {
                        interrupt = 1U;
                    }
                    __syncthreads();
                }
            }
        }
    }
}
#endif //ARISTOS_COMPUTE_CUH
