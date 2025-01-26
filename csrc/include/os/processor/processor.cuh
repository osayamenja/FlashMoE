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
            class ScaleWeights,
            typename Element = typename Activations::value_type
        >
        requires(TensorValueType<Element> &&
            aristos::Tensor<Activations> && aristos::Tensor<ScaleWeights>)
        __device__ __forceinline__
        void operator()(Element* __restrict__ workspace,
            const TokenIdxTuple* __restrict__ tokenIndices,
            const Element* __restrict__& inputs,
            Activations const& activations,
            ScaleWeights const& scale,
            const unsigned int& M,
            const unsigned int& N,
            const unsigned int& tileIdx,
            const unsigned int& tileSize,
            const unsigned int& expertIdx) const {
            using BlockTiler = cute::Shape<cute::Int<BLOCK_M>, cute::Int<BLOCK_N>>;
            constexpr BlockTiler tiler{};
            constexpr auto bM = cute::get<0>(tiler);
            constexpr auto bN = cute::get<1>(tiler);
            constexpr auto threads = THREADS;
            cutlass::AlignedArray<Element, bN> registers{};
            // Eagerly issue gmem read.
            auto [tokenIdx, combineWeight] = TokenIdxTuple{};
            auto scaleWeight = Element(0);
            if (threadIdx.x < tileSize) {
                const auto t = tokenIndices[threadIdx.x];
                tokenIdx = t.first;
                combineWeight = t.second;
                scaleWeight = scale(tokenIdx, expertIdx) / combineWeight;
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
                        registers[i] *= scaleWeight;
                    }
                    vaa(&activations(tokenIdx, 0), registers);
                }
                else {
                    // vector copy from registers to global directly
                    constexpr auto vL = bN * sizeof(Element) / sizeof(uint4);
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
        __forceinline__ __device__
        void operator()(typename BlockGEMM::MatrixDType* __restrict__ workspace,
        const typename BlockGEMM::MatrixAType* __restrict__ inputs,
        const typename BlockGEMM::MatrixBType* __restrict__ weights,
        typename BlockGEMM::MatrixDType* __restrict__ output,
        const typename BlockGEMM::MatrixDType* __restrict__ bias,
        const unsigned int& M,
        const unsigned int& N,
        const unsigned int& K,
        const unsigned int& tileIdx) const {
            auto accumulator = cute::partition_fragment_C(typename BlockGEMM::MMA{}, typename BlockGEMM::TilerOut{});
            constexpr auto elems = SHARED_SIZE / (THREADS * sizeof(typename BlockGEMM::MatrixDType));
            static_assert(cute::size(accumulator) % elems == 0);
            cutlass::AlignedArray<typename BlockGEMM::MatrixDType, elems> rScratch{};
            constexpr auto bM = cute::get<0>(typename BlockGEMM::BlockTiler{});
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
            const auto tDgD = tiledMMA.get_slice(threadIdx.x).partition_C(gD);

            // Accounts for GEMMs that accumulate in types differing from input types,
            // given that the result may moonlight as the input for the succeeding GEMM.
            const auto gCStoreOp = cutlass::NumericConverter<typename decltype(gC)::value_type,
                                                        typename decltype(accumulator)::value_type>{};
            const auto gDLoadOp = cutlass::NumericConverter<typename decltype(accumulator)::value_type,
                                                        ElementD>{};

            // Assume elementwise operator
            typename BlockGEMM::FusedEpilogue epilogueOp{};
            constexpr auto trips = size(accumulator) / rScratch.size();

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
                #pragma unroll
                for (int j = 0; j < elems; ++j) {
                    accumulator(j + i * elems) = gCStoreOp(epilogueOp(accumulator(j + i * elems), gDLoadOp(rScratch[j])));
                }
            }

            __syncthreads();
            constexpr auto sCLay = cute::make_layout(cute::Shape<cute::Int<bM>, cute::Int<elems>>{},
                cute::LayoutRight{});
            const auto sC = cute::make_tensor(cute::make_smem_ptr(workspace), sCLay);
            const auto tCsC = tiledMMA.get_slice(threadIdx.x).partition_C(sC);
            const auto rIdx = threadIdx.x / elems * elems;
            const auto cIdx = threadIdx.x % elems;
            #pragma unroll
            for (unsigned int i = 0; i < trips; ++i) {
                #pragma unroll
                for (unsigned int j = 0; j < elems; ++j) {
                    tCsC(j) = gCStoreOp(accumulator(j + i * elems));
                }
                __syncthreads();
                #pragma unroll
                for (unsigned int j = 0; j < elems; ++j) {
                    gC(rIdx + j, cIdx + i * elems) = sC(rIdx + j, cIdx);
                }
            }
            __syncthreads();
        }
    };

    template<
        typename BlockGEMM
    >
    struct FGT<TaskType::postGEMM, BlockGEMM> {
        __forceinline__ __device__
        void operator()(typename BlockGEMM::MatrixDType* __restrict__ workspace,
        const typename BlockGEMM::MatrixAType* __restrict__& inputs,
        const typename BlockGEMM::MatrixBType* __restrict__& weights,
        typename BlockGEMM::MatrixDType* __restrict__& output,
        const typename BlockGEMM::MatrixDType* __restrict__& bias,
        const unsigned int& M,
        const unsigned int& N,
        const unsigned int& K,
        const unsigned int& tileIdx,
        const unsigned int& isRemote) const {
            auto accumulator = cute::partition_fragment_C(typename BlockGEMM::MMA{}, typename BlockGEMM::TilerOut{});
            constexpr auto elems = SHARED_SIZE / (THREADS * sizeof(typename BlockGEMM::MatrixDType));
            static_assert(cute::size(accumulator) % elems == 0);
            cutlass::AlignedArray<typename BlockGEMM::MatrixDType, elems> rScratch{};
            constexpr auto bM = cute::get<0>(typename BlockGEMM::BlockTiler{});
            static_assert(size(accumulator) % rScratch.size() == 0);
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
            const auto tDgD = tiledMMA.get_slice(threadIdx.x).partition_C(gD);

            constexpr auto gCStoreOp = cutlass::NumericConverter<typename decltype(gC)::value_type,
                                                        typename decltype(accumulator)::value_type>{};
            constexpr auto gDLoadOp = cutlass::NumericConverter<typename decltype(accumulator)::value_type,
                                                        ElementD>{};

            // Assume elementwise operator
            constexpr typename BlockGEMM::FusedEpilogue epilogueOp{};
            constexpr auto trips = size(accumulator) / rScratch.size();

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

                #pragma unroll
                for (int j = 0; j < elems; ++j) {
                    accumulator(j + i * elems) = epilogueOp(accumulator(j + i * elems), gDLoadOp(rScratch[j]));
                }
            }

            __syncthreads();
            constexpr auto sCLay = cute::make_layout(cute::Shape<cute::Int<bM>, cute::Int<elems>>{},
                cute::LayoutRight{});
            const auto sC = cute::make_tensor(cute::make_smem_ptr(workspace), sCLay);
            const auto tCsC = tiledMMA.get_slice(threadIdx.x).partition_C(sC);
            const auto rIdx = threadIdx.x / elems * elems;
            const auto cIdx = threadIdx.x % elems;
            #pragma unroll
            for (unsigned int i = 0; i < trips; ++i) {
                #pragma unroll
                for (unsigned int j = 0; j < elems; ++j) {
                    tCsC(j) = gCStoreOp(accumulator(j + i * elems));
                }
                __syncthreads();
                #pragma unroll
                for (unsigned int j = 0; j < elems; ++j) {
                    gC(rIdx + j, cIdx + i * elems) = sC(rIdx + j, cIdx);
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


    struct ProcessorArgs{
        // sensible sentinel values
        cuda::std::byte* __restrict__ sHeap = nullptr;
        unsigned int* __restrict__ sQ = nullptr;
        unsigned int* __restrict__ pDB = nullptr;
        unsigned int* __restrict__ tQH = nullptr;
        flagsType* __restrict__ flags = nullptr;
        Task* __restrict__ gtQ = nullptr;
        Task* __restrict__ tQ = nullptr;
        unsigned int* __restrict__ tQS = nullptr;
        unsigned int pd = 0U;
        unsigned int tokenSize = 0U;
        unsigned int tN = 0U;
        unsigned int tNx = 0U;

        ProcessorArgs() = default;
        ProcessorArgs(cuda::std::byte* const& _sHeap,
            unsigned int* const& _sQ,
            unsigned int* const& _pDB,
            unsigned int* const& _tQH,
            flagsType* const& _flags,
            Task* const& _gtQ,
            Task* const& _tQ,
            unsigned int* const& _tQS,
            unsigned int const& _pd,
            unsigned int const& _tokenSize,
            unsigned int const& _tN,
            unsigned int const& _tNx) :
        sHeap(_sHeap), sQ(_sQ), pDB(_pDB), tQH(_tQH), flags(_flags), gtQ(_gtQ), tQ(_tQ), tQS(_tQS), pd(_pd),
        tokenSize(_tokenSize), tN(_tN), tNx(_tNx) {}
    };

    template<
        unsigned int processorCount,
        unsigned int Arch,
        CombineMode c,
        typename ElementA,
        typename ElementB,
        typename ElementC = float,
        typename ActivationOp = cute::identity,
        typename ActivationOpX = cute::identity
    > requires(processorCount > 0 && Arch >= MIN_ARCH)
    __device__ __forceinline__
    void start(cuda::std::byte* __restrict__ const& workspace, const uint16_t& _seqBit){
        assert(__isShared(workspace));
        static_assert(sizeof(SignalPayload<PacketStage::last>) == sizeof(uint64_t));
        __shared__ Task currentTask;
        __shared__ ProcessorArgs pA;
        __shared__ uint enqueue;
        unsigned int signal = 0U;
        unsigned int interrupt = 0U;
        const auto rSeqBit = _seqBit;
        if (!threadIdx.x) {
            pA = ProcessorArgs{
                moeConfig.sHeap,
                bookkeeping.sQ() + blockIdx.x,
                bookkeeping.pDB() + blockIdx.x,
                bookkeeping.tQH() + SUBSCRIBERS,
                moeConfig.flags,
                bookkeeping.tQ(),
                bookkeeping.tQ() + bookkeeping.tPs * SUBSCRIBERS, // should be the external Q
                bookkeeping.tQS(),
                bookkeeping.px,
                moeConfig.embedDim,
                bookkeeping.tN,
                Config::tiles<BLOCK_N>(bookkeeping.pd)
            };
            // Initially indicate this block's readiness
            atomicExch(pA.sQ, ready);
        }
        atomicExch(&enqueue, 0U);
        using Operation = BlockMM<Arch, ElementA, ElementB, ElementC, ActivationOp>;
        using OperationX = BlockMM<Arch, ElementA, ElementB, ElementC, ActivationOpX>;
        constexpr auto preGEMM = FGT<TaskType::preGEMM, Operation>{};
        constexpr auto postGEMM = FGT<TaskType::postGEMM, OperationX>{};
        constexpr auto combineOp = Combine<Arch, ElementA, c>{};
        __syncthreads();

        while (!interrupt) {
            if (!threadIdx.x) {
                auto* __restrict__ tQSignal = pA.pDB;
                const auto* __restrict__ gtQ = pA.gtQ;
                // Grabs next task
                auto nextTask = atomicLoad(tQSignal);
                while (nextTask == signal) {
                    nextTask = atomicLoad(tQSignal);
                }
                signal = nextTask;
                // below ensures task read happens after signal reception
                __threadfence();
                currentTask = gtQ[signal - 1];
            }
            __syncthreads();
            switch (currentTask.taskType) {
                case TaskType::preGEMM: {
                    // Eagerly indicate readiness for the next task
                    atomicExch(pA.sQ, ready);
                    constexpr unsigned int preIndex = 0;
                    preGEMM(workspace,
                        CAST_TO(typename Operation::MatrixAType, currentTask.aData),
                        CAST_TO(typename Operation::MatrixBType, currentTask.bData[preIndex]),
                        CAST_TO(typename Operation::MatrixDType, currentTask.cData[preIndex]),
                        CAST_TO(typename Operation::MatrixDType, currentTask.dData[preIndex]),
                        currentTask.M,
                        pA.pd,
                        pA.tokenSize,
                        currentTask.tileIdx);
                    if (!threadIdx.x) {
                        __threadfence();
                        enqueue = atomicAdd(pA.tQS + currentTask.syncIdx, 1U) == pA.tN;
                    }
                    __syncthreads();
                    if (enqueue) {
                        const auto tasks = pA.tNx;
                        auto* tQ = pA.tQ + currentTask.syncIdx;
                        auto nextTask = Task {
                            TaskType::postGEMM,
                            currentTask.cData[preIndex],
                            currentTask.bData,
                            currentTask.cData,
                            currentTask.dData,
                            currentTask.syncIdx,
                            0,
                            currentTask.M,
                            currentTask.flagIdx,
                            currentTask.tileSize,
                            currentTask.peerIdx,
                            currentTask.batchIdx
                        };
                        for (unsigned int i = threadIdx.x; i < tasks; i += THREADS) {
                            nextTask.tileIdx = i;
                            tQ[i] = nextTask;
                        }
                        __syncthreads();
                        if (!threadIdx.x) {
                            __threadfence();
                            // notify scheduler
                            atomicAdd(pA.tQH, tasks);
                        }
                    }
                }
                break;
                case TaskType::postGEMM: {
                    // Eagerly indicate readiness for the next task
                    atomicExch(pA.sQ, ready);
                    constexpr unsigned int postIndex = 0;
                    postGEMM(workspace,
                        CAST_TO(typename Operation::MatrixAType, currentTask.aData),
                        CAST_TO(typename Operation::MatrixBType, currentTask.bData[postIndex]),
                        CAST_TO(typename Operation::MatrixDType, currentTask.cData[postIndex]),
                        CAST_TO(typename Operation::MatrixDType, currentTask.dData[postIndex]),
                        pA.tokenSize,
                        pA.pd,
                        currentTask.tileIdx,
                        nvshmem_ptr(pA.sHeap, currentTask.peerIdx) == nullptr);
                    if (!threadIdx.x) {
                        // Pack payload into single signal word
                        auto flagSignal = SignalPayload<PacketStage::last>{
                            currentTask.batchIdx,
                            rSeqBit,
                            currentTask.tileSize
                        };
                        if (atomicIncrement(pA.tQS + currentTask.syncIdx)
                            == pA.tN + pA.tNx) {
                            if (nvshmem_ptr(currentTask.cData[postIndex], currentTask.peerIdx) == nullptr) {
                                // Batch remote network transfer to avoid overwhelming the NIC
                                nvshmem_putmem_signal_nbi(currentTask.cData[postIndex], currentTask.cData[postIndex],
                                    currentTask.tileSize * pA.tokenSize * sizeof(ElementA),
                                    pA.flags + currentTask.flagIdx,
                                    *CAST_TO(uint64_t, &flagSignal), NVSHMEM_SIGNAL_SET,
                                    currentTask.peerIdx);
                            }
                            else {
                                // send individual tile, no batching here
                                // Already did the network transfer in fGET, so set signal only
                                nvshmemx_signal_op(pA.flags + currentTask.flagIdx,
                                 *CAST_TO(uint64_t, &flagSignal), NVSHMEM_SIGNAL_SET, currentTask.peerIdx);
                            }
                        }
                    }
                }
                break;
                case TaskType::combine: {
                    // Eagerly indicate readiness for the next task
                    atomicExch(pA.sQ, ready);
                    combineOp(CAST_TO(ElementA, workspace), currentTask.aData, currentTask.bData[0],
                        currentTask.cData[0], currentTask.M, pA.tokenSize, currentTask.tileIdx,
                        currentTask.tileSize, currentTask.expertIdx);
                }
                break;
                case TaskType::Interrupt: {
                    interrupt = 1U;
                }
            }
        }
    }
}
#endif //ARISTOS_COMPUTE_CUH
