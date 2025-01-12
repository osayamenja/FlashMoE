//
// Created by osayamen on 7/13/24.
//

#ifndef ARISTOS_COMPUTE_CUH
#define ARISTOS_COMPUTE_CUH

#include <cutlass/array.h>
#include <cute/tensor.hpp>

#include <nvshmemx.h>
#include <nvshmem.h>
#include <host/nvshmemx_api.h>

#include "../../arch.cuh"
#include "gemm.cuh"
#include "../packet.cuh"

namespace aristos::processor{
    enum class CombineMode {
        single,
        multithreaded
    };

    template<
        unsigned Arch,
        typename ElementCombine,
        CombineMode c = CombineMode::single
    > requires SupportedArch<Arch> && TensorValueType<ElementCombine>
    struct Combine {
        template<
            class Activations,
            class Registers,
            typename Element = typename Activations::value_type,
            typename RegisterScratch
        >
        requires(TensorValueType<Element> &&
            cute::is_tensor_v<Activations> && isRegisterV<Registers> && isRegisterV<RegisterScratch>)
        __device__ __forceinline__
        void operator()(Element* __restrict__ workspace,
            const TokenIdxTuple* __restrict__ tokenIndices,
            Registers& registers,
            RegisterScratch rScratch,
            const Element* __restrict__& inputs,
            Activations const& activations,
            const Element* __restrict__& scaleWeights,
            const unsigned int& M,
            const unsigned int& N,
            const unsigned int& tileIdx,
            const unsigned int& tileSize) const {
            using BlockTiler = cute::Shape<cute::Int<BLOCK_M>, cute::Int<BLOCK_N>>;
            constexpr BlockTiler tiler{};
            constexpr auto bM = cute::get<0>(tiler);
            constexpr auto bN = cute::get<1>(tiler);
            constexpr auto threads = THREADS;
            static_assert(!(cuda::std::is_same_v<Element, cute::float_e4m3_t> &&
                cuda::std::is_same_v<Element, cute::float_e4m3_t>), "fp8 atomic addition is not available, "
                                                                    "so no support for this operation yet");

            // Eagerly issue gmem read.
            // We only need the index
            auto [tokenIdx, combineWeight] = TokenIdxTuple{};
            auto scaleWeight = Element(0);
            if (threadIdx.x < tileSize) {
                auto [tokenIdx, combineWeight] = tokenIndices[threadIdx.x];
                scaleWeight = scaleWeights[threadIdx.x];
            }
            // Row-major
            const auto mA = make_tensor(cute::make_gmem_ptr(inputs),
                make_layout(cute::make_shape(M, N), cute::LayoutRight{}));

            const auto tilesM = M / bM;
            // We assert the below prior to this point
            const auto tilesN = N / bN;

            const auto tileCoord = idx2crd(tileIdx, cute::Shape(tilesM, tilesN), cute::Stride(tilesN ,1));
            const auto ctaCoord = cute::make_coord(cute::get<0>(tileCoord), cute::get<1>(tileCoord));
            const auto gA = cute::local_tile(mA, tiler, ctaCoord);
            constexpr auto elems = SHARED_SIZE / (threads * sizeof(Element));
            static_assert(bN % elems == 0);
            constexpr auto trips = bN / elems;
            static_assert(RegisterScratch::kElements == elems);

            // Transposed layout
            constexpr auto sCLay = cute::make_layout(cute::Shape<cute::Int<bM>, cute::Int<elems>>{}, cute::LayoutRight{});
            const auto sC = cute::make_tensor(cute::make_smem_ptr(workspace), sCLay);

            #pragma unroll
            for (uint i = 0; i < trips; ++i) {
                // global -> shared
                #pragma unroll
                for (uint j = 0; j < elems; ++j) {
                    const auto rIdx = j + threadIdx.x / elems * elems;
                    const auto cIdx =  threadIdx.x % elems + i * elems;
                    sC(rIdx, cIdx) = gA(rIdx, cIdx);
                }
                __syncthreads();
                #pragma unroll
                for (uint j = 0; j < elems; ++j) {
                    registers[j + i * elems] = sC(threadIdx.x, j);
                }
            }

            constexpr VAA<Arch, ElementCombine> vaa{};
            if (threadIdx.x < tileSize) {
                // do scale
                if constexpr (c == CombineMode::multithreaded) {
                    #pragma unroll
                    for (uint i = 0; i < bN; ++i) {
                        registers[i] *= __fdividef(scaleWeight, combineWeight);
                    }
                    vaa(&activations(tokenIdx, 0), registers);
                }
                else {
                    // vector copy from registers to global directly
                    constexpr auto vL = Registers::kElements * sizeof(Element) / sizeof(uint4);
                    auto* __restrict__ aP = CAST_TO(uint4, &activations(tokenIdx, 0));
                    const auto* __restrict__ rD = CAST_TO(uint4, registers.data());
                    #pragma unroll
                    for (uint i = 0; i < vL; ++i) {
                        aP[i] = rD[i];
                    }
                }
            }
        }
    };

    // Fused GEMM, Epilogue and data Transfer
    template<
        TaskType t = TaskType::preGEMM,
        typename BlockGEMM
    >
    struct FGT {
        static_assert(t == TaskType::preGEMM);
        template<class FrgTensorD, class RegisterScratch>
        __forceinline__ __device__
        void operator()(typename BlockGEMM::MatrixDType* __restrict__ workspace,
        FrgTensorD& accumulator,
        RegisterScratch& rScratch,
        const typename BlockGEMM::MatrixAType* __restrict__ inputs,
        const typename BlockGEMM::MatrixBType* __restrict__ weights,
        typename BlockGEMM::MatrixDType* __restrict__ output,
        const typename BlockGEMM::MatrixDType* __restrict__ bias,
        const unsigned int& M,
        const unsigned int& N,
        const unsigned int& K,
        const unsigned int& tileIdx) const {
            static_assert(size(accumulator) % rScratch.size() == 0 && cutlass::detail::is_Array_v<RegisterScratch>);
            // Instantiate mainloop
            typename BlockGEMM::CollectiveMainloop mainLoop{};
            cute::clear(accumulator);

            // Row-major
            const auto mA = make_tensor(cute::make_gmem_ptr(inputs),
                make_layout(cute::make_shape(M, K), cute::make_stride(K, 1)));
            // Row-major, transposed
            const auto mB = make_tensor(cute::make_gmem_ptr(weights),
                make_layout(cute::make_shape(N, K), cute::make_stride(K, 1)));
            // Row-major
            const auto mC = make_tensor(cute::make_gmem_ptr(output,
                make_layout(cute::make_shape(M, N), cute::make_stride(N, 1))));
            const auto mD = make_tensor(cute::make_gmem_ptr(bias),
                make_layout(cute::make_shape(M, N), cute::make_stride(0, 1)));

            // M is padded, such that the below is correct
            const auto tilesM = M / cute::get<0>(typename BlockGEMM::BlockTiler{});
            // We assert the below prior to this point
            const auto tilesN = N / cute::get<1>(typename BlockGEMM::BlockTiler{});

            const auto tileCoord = idx2crd(tileIdx, cute::Shape(tilesM, tilesN), cute::Stride(tilesN ,1));
            const auto ctaCoord = make_coord(cute::get<0>(tileCoord), cute::get<1>(tileCoord), cute::_);
            const auto gA = cute::local_tile(mA, typename BlockGEMM::BlockTiler{}, ctaCoord, cute::Step<cute::_1, cute::X,cute::_1>{});
            const auto gB = cute::local_tile(mB, typename BlockGEMM::BlockTiler{}, ctaCoord, cute::Step< cute::X,cute::_1,cute::_1>{});
            const auto gC = cute::local_tile(mC, typename BlockGEMM::BlockTiler{}, ctaCoord, cute::Step<cute::_1,cute::_1, cute::X>{});
            const auto gD = cute::local_tile(mD, typename BlockGEMM::BlockTiler{}, ctaCoord, cute::Step<cute::_1,cute::_1, cute::X>{});

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
            const auto tCgC = tiledMMA.get_slice(threadIdx.x).partition_C(gC);
            const auto tDgD = tiledMMA.get_slice(threadIdx.x).partition_C(gD);

            // Accounts for GEMMs that accumulate in types differing from input types,
            // given that the result may moonlight as the input for the succeeding GEMM.
            const auto gCStoreOp = cutlass::NumericConverter<typename decltype(tCgC)::value_type,
                                                        typename decltype(accumulator)::value_type>{};
            const auto gDLoadOp = cutlass::NumericConverter<typename decltype(accumulator)::value_type,
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
                        workspace[threadIdx.x + j * THREADS] = tDgD(j + (i + 1) * elems);
                    }
                }
                // Fused Bias Add and Activation Function on register fragment
                // Also fuses copy to GMEM.
                #pragma unroll
                for (int j = 0; j < elems; ++j) {
                    tCgC(j + i * elems) = gCStoreOp(epilogueOp(accumulator(j + i * elems), gDLoadOp(rScratch[j])));
                }
            }

            __syncthreads();
            if (!threadIdx.x) {
                __threadfence();
            }
        }
    };

    template<
        typename BlockGEMM
    >
    struct FGT<TaskType::postGEMM, BlockGEMM> {
        template<class FrgTensorD, class RegisterScratch>
        __forceinline__ __device__
        void operator()(typename BlockGEMM::MatrixDType* __restrict__ workspace,
        FrgTensorD& accumulator,
        RegisterScratch& rScratch,
        const typename BlockGEMM::MatrixAType* __restrict__& inputs,
        const typename BlockGEMM::MatrixBType* __restrict__& weights,
        typename BlockGEMM::MatrixDType* __restrict__& output,
        const typename BlockGEMM::MatrixDType* __restrict__& bias,
        const unsigned int& M,
        const unsigned int& N,
        const unsigned int& K,
        const unsigned int& tileIdx,
        const unsigned int& isRemote) const {
            static_assert(size(accumulator) % rScratch.size() == 0 && cutlass::detail::is_Array_v<RegisterScratch>);
            // Instantiate mainloop
            typename BlockGEMM::CollectiveMainloop mainLoop{};
            cute::clear(accumulator);

            // Row-major
            const auto mA = make_tensor(cute::make_gmem_ptr(inputs),
                make_layout(cute::make_shape(M, K), cute::make_stride(K, 1)));
            // Row-major, transposed
            const auto mB = make_tensor(cute::make_gmem_ptr(weights),
                make_layout(cute::make_shape(N, K), cute::make_stride(K, 1)));
            // Row-major
            const auto mC = make_tensor(cute::make_gmem_ptr(output,
                make_layout(cute::make_shape(M, N), cute::make_stride(N, 1))));
            const auto mD = make_tensor(cute::make_gmem_ptr(bias),
                make_layout(cute::make_shape(M, N), cute::make_stride(0, 1)));

            // M is padded, such that the below is correct
            const auto tilesM = M / cute::get<0>(typename BlockGEMM::BlockTiler{});
            // We assert the below prior to this point
            const auto tilesN = N / cute::get<1>(typename BlockGEMM::BlockTiler{});

            const auto tileCoord = idx2crd(tileIdx, cute::Shape(tilesM, tilesN), cute::Stride(tilesN ,1));
            const auto ctaCoord = make_coord(cute::get<0>(tileCoord), cute::get<1>(tileCoord), cute::_);
            const auto gA = cute::local_tile(mA, typename BlockGEMM::BlockTiler{}, ctaCoord, cute::Step<cute::_1, cute::X,cute::_1>{});
            const auto gB = cute::local_tile(mB, typename BlockGEMM::BlockTiler{}, ctaCoord, cute::Step< cute::X,cute::_1,cute::_1>{});
            const auto gC = cute::local_tile(mC, typename BlockGEMM::BlockTiler{}, ctaCoord, cute::Step<cute::_1,cute::_1, cute::X>{});
            const auto gD = cute::local_tile(mD, typename BlockGEMM::BlockTiler{}, ctaCoord, cute::Step<cute::_1,cute::_1, cute::X>{});

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
            constexpr typename BlockGEMM::MMA tiledMMA{};
            const auto tCgC = tiledMMA.get_slice(threadIdx.x).partition_C(gC);
            const auto tDgD = tiledMMA.get_slice(threadIdx.x).partition_C(gD);

            // Accounts for GEMMs that accumulate in types differing from input types,
            // given that the result may moonlight as the input for the succeeding GEMM.
            constexpr auto gCStoreOp = cutlass::NumericConverter<typename decltype(tCgC)::value_type,
                                                        typename decltype(accumulator)::value_type>{};
            constexpr auto gDLoadOp = cutlass::NumericConverter<typename decltype(accumulator)::value_type,
                                                        ElementD>{};

            // Assume elementwise operator
            constexpr typename BlockGEMM::FusedEpilogue epilogueOp{};
            constexpr auto trips = size(accumulator) / rScratch.size();
            constexpr auto elems = rScratch.size();

            #pragma unroll
            for (unsigned int i = 0; i < trips; ++i) {
                // Prefetch from global to shared memory
                #pragma unroll
                for (int j = 0; j < elems; ++j) {
                    workspace[threadIdx.x + j * THREADS] = tDgD(j);
                }

                #pragma unroll
                for (unsigned int j = 0; j < elems; ++j) {
                    rScratch[j] = workspace[threadIdx.x + j * THREADS];
                    if (i + 1 < trips) {
                        // Eagerly start loads for the next batch, if needed
                        workspace[threadIdx.x + j * THREADS] = tDgD(j + (i + 1) * elems);
                    }
                }

                #pragma unroll
                for (int j = 0; j < elems; ++j) {
                    tCgC(j + i * elems) = gCStoreOp(epilogueOp(accumulator(j + i * elems), gDLoadOp(rScratch[j])));
                }
            }

            __syncthreads();
            if (!threadIdx.x) {
                if (isRemote) {
                    __threadfence();
                }
                else {
                    // Below is expensive, so we only invoke when necessary
                    __threadfence_system();
                }
            }
        }
    };


    template<
        unsigned int processorCount,
        unsigned int Arch,
        CombineMode c,
        typename ElementA,
        typename ElementB,
        typename ElementC = float,
        typename ElementD = ElementA,
        typename ActivationOp = cute::identity,
        typename ActivationOpX = cute::identity
    > requires(processorCount > 0 && Arch >= MIN_ARCH)
    __device__ __forceinline__
    void start(cuda::std::byte* __restrict__ const& workspace, const uint16_t& _seqBit){
        assert(__isShared(workspace));
        static_assert(sizeof(SignalPayload<PacketStage::final>) == sizeof(uint64_t));
        __shared__ unsigned int signal;
        __shared__ Task currentTask;
        __shared__ Config cachedConfig;
        __shared__ SchedulerConfig scState;
        __shared__ unsigned int interrupt;
        auto rSeqBit = _seqBit;
        if (!threadIdx.x) {
            cachedConfig = moeConfig;
            scState = schedulerState;
            signal = 0U;
            interrupt = 0U;
            // Initially indicate this block's readiness
            atomicExch(scState.statusQ + blockIdx.x, ready);
        }
        using Operation = BlockMM<Arch, ElementA, ElementB, ElementC, ActivationOp>;
        using OperationX = BlockMM<Arch, ElementA, ElementB, ElementC, ActivationOpX>;
        auto accumulator = cute::partition_fragment_C(typename Operation::MMA{}, typename Operation::TilerOut{});
        constexpr auto elems = SHARED_SIZE / (THREADS * sizeof(ElementD));
        static_assert(cute::size(accumulator) % elems == 0);
        cutlass::AlignedArray<ElementC, elems> rScratch{};
        constexpr auto preGEMM = FGT<TaskType::preGEMM, Operation>{};
        constexpr auto postGEMM = FGT<TaskType::postGEMM, OperationX>{};
        constexpr auto combineOp = Combine<Arch, ElementA, c>{};
        __syncthreads();

        while (!interrupt) {
            if (!threadIdx.x) {
                auto* tQSignal = scState.taskSignal + blockIdx.x;
                // Grabs next task
                auto nextTask = atomicLoad(tQSignal);
                while (nextTask == signal) {
                    nextTask = atomicLoad(tQSignal);
                }
                signal = nextTask;
                // below ensures task read happens after signal reception
                __threadfence();
                currentTask = scState.taskQ[signal - 1];
            }
            __syncthreads();
            switch (currentTask.taskType) {
                case TaskType::preGEMM: {
                    // Eagerly indicate readiness for the next task
                    atomicExch(scState.statusQ + blockIdx.x, ready);
                    constexpr unsigned int preIndex = 0;
                    preGEMM(workspace, accumulator, rScratch,
                        CAST_TO(typename Operation::MatrixAType, currentTask.aData),
                        CAST_TO(typename Operation::MatrixBType, currentTask.bData[preIndex]),
                        CAST_TO(typename Operation::MatrixDType, currentTask.cData[preIndex]),
                        CAST_TO(typename Operation::MatrixDType, currentTask.dData[preIndex]),
                        currentTask.M,
                        cachedConfig.upProjection,
                        cachedConfig.embedDim,
                        currentTask.tileIdx);
                    if (!threadIdx.x &&
                        atomicAdd(scState.taskSync + currentTask.syncIdx, 1U) == cachedConfig.tilesN) {
                        const auto tasks = cachedConfig.tilesNx;
                        auto* tQ = scState.xTaskQ + currentTask.syncIdx;
                        for (unsigned int i = 0; i < tasks; ++i) {
                            tQ[i] = Task{
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
                                currentTask.peerIdx,
                                currentTask.batchIdx
                            };
                        }
                        __threadfence();
                        // notify scheduler
                        atomicAdd(scState.xTQHeads, tasks);
                    }
                }
                break;
                case TaskType::postGEMM: {
                    // Eagerly indicate readiness for the next task
                    atomicExch(scState.statusQ + blockIdx.x, ready);
                    constexpr unsigned int postIndex = 0;
                    postGEMM(workspace, accumulator, rScratch,
                        CAST_TO(typename Operation::MatrixAType, currentTask.aData),
                        CAST_TO(typename Operation::MatrixBType, currentTask.bData[postIndex]),
                        CAST_TO(typename Operation::MatrixDType, currentTask.cData[postIndex]),
                        CAST_TO(typename Operation::MatrixDType, currentTask.dData[postIndex]),
                        cachedConfig.embedDim,
                        cachedConfig.upProjection,
                        currentTask.tileIdx,
                        nvshmem_ptr(cachedConfig.sHeap, currentTask.peerIdx) == nullptr);
                    if (!threadIdx.x) {
                        uint64_t flagSignal = 0;
                        *CAST_TO(SignalPayload<PacketStage::final>, &flagSignal) = SignalPayload<PacketStage::final>{
                            currentTask.batchIdx,
                            rSeqBit,
                            currentTask.tileSize
                        };
                        if (atomicIncrement(scState.taskSync + currentTask.syncIdx)
                            == cachedConfig.tilesN + cachedConfig.tilesNx) {
                            if (nvshmem_ptr(currentTask.cData[postIndex], currentTask.peerIdx) == nullptr) {
                                // Batch remote network transfer to avoid overwhelming the NIC
                                nvshmem_putmem_signal_nbi(currentTask.cData[postIndex], currentTask.cData[postIndex],
                                    currentTask.tileSize * cachedConfig.embedDim * sizeof(ElementA),
                                    cachedConfig.flags + currentTask.flagIdx,
                                    flagSignal, NVSHMEM_SIGNAL_SET,
                                    currentTask.peerIdx);
                            }
                            else {
                                // send individual tile, no batching here
                                // Already did the network transfer in fGET, so set signal only
                                nvshmemx_signal_op(cachedConfig.flags + currentTask.flagIdx,
                                 flagSignal, NVSHMEM_SIGNAL_SET, currentTask.peerIdx);
                            }
                        }
                    }
                }
                break;
                case TaskType::combine: {
                    // Eagerly indicate readiness for the next task
                    atomicExch(scState.statusQ + blockIdx.x, ready);
                    combineOp(CAST_TO(ElementA, workspace), currentTask.aData, rScratch, currentTask.bData[0],
                        currentTask.cData[0], currentTask.M, cachedConfig.embedDim, currentTask.tileIdx,
                        currentTask.tileSize);
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
