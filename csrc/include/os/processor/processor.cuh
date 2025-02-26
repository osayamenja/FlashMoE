//
// Created by osayamen on 7/13/24.
//

#ifndef ARISTOS_COMPUTE_CUH
#define ARISTOS_COMPUTE_CUH

#include <cutlass/array.h>
#include <cute/tensor.hpp>
#include <nvshmem.h>

#include "gemm.cuh"

namespace aristos::processor{
    template<
        typename GPUType,
        CombineMode c = CombineMode::single
    >
    struct Combine {
        template<
            class ScaleWeights,
            typename Element,
            unsigned int Arch = GPUType::arch::value,
            unsigned int elems = GPUType::rScratch::value
        >
        requires(TensorValueType<Element> &&
            aristos::isMatrix<ScaleWeights> &&
            cuda::std::is_same_v<typename ScaleWeights::value_type, Element>)
        __device__ __forceinline__
        void operator()(Element* __restrict__ const& workspace,
            const TokenIdxTuple* __restrict__ const& tokenIndices,
            const Element* __restrict__ const& inputs,
            Element* __restrict__ const& moeOutput,
            ScaleWeights const& scale,
            const unsigned int& M,
            const unsigned int& N,
            const unsigned int& tileIdx,
            const uint16_t& tileSize,
            const unsigned int& expertIdx) const {
            using BlockTiler = cute::Shape<cute::Int<BLOCK_M>, cute::Int<BLOCK_N>>;
            constexpr BlockTiler tiler{};
            constexpr auto bM = cute::get<0>(tiler);
            constexpr auto bN = cute::get<1>(tiler);
            cutlass::AlignedArray<Element, bN> registers{};
            // Eagerly issue gmem read.
            auto [tokenIdx, combineWeight] = TokenIdxTuple{};
            auto scaleWeight = Element(0);
            constexpr auto mTe = cutlass::NumericConverter<Element, TokenIdxTuple::second_type>{};
            if (threadIdx.x < tileSize) {
                const auto t = tokenIndices[threadIdx.x];
                tokenIdx = t.first;
                combineWeight = t.second;
                scaleWeight = scale(tokenIdx, expertIdx) / mTe(combineWeight);
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

            constexpr VAA<Arch, Element> vaa{};
            if (threadIdx.x < tileSize) {
                // do scale
                if constexpr (c == CombineMode::multithreaded) {
                    #pragma unroll
                    for (uint i = 0; i < bN; ++i) {
                        registers[i] = registers[i] * scaleWeight;
                    }
                    vaa(moeOutput + tokenIdx * N, registers);
                }
                else {
                    // vector copy from registers to global directly
                    constexpr auto vL = bN * sizeof(Element) / sizeof(uint4);
                    auto* __restrict__ aP = CAST_TO(uint4, moeOutput + tokenIdx * N);
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
        TaskType t,
        typename BlockGEMM,
        typename GPUType,
        unsigned int elems = GPUType::rScratch::value,
        unsigned int threads = GPUType::OS::threads::value
    >
    struct FGT {
        static_assert(t == TaskType::preGEMM);
        __forceinline__ __device__
        void operator()(typename BlockGEMM::MatrixDType* __restrict__ const& workspace,
        const typename BlockGEMM::MatrixAType* __restrict__ const& inputs,
        const typename BlockGEMM::MatrixBType* __restrict__ const& weights,
        typename BlockGEMM::MatrixDType* __restrict__ const&output,
        const typename BlockGEMM::MatrixDType* __restrict__ const& bias,
        const unsigned int& M,
        const unsigned int& N,
        const unsigned int& K,
        const unsigned int& tileIdx) const {
            auto accumulator = cute::partition_fragment_C(typename BlockGEMM::MMA{}, typename BlockGEMM::TilerOut{});
            static_assert(cute::size(accumulator) % elems == 0);
            cutlass::AlignedArray<typename BlockGEMM::MatrixDType, elems> rScratch{};
            constexpr auto bM = cute::get<0>(typename BlockGEMM::BlockTiler{});
            constexpr auto bN = cute::get<1>(typename BlockGEMM::BlockTiler{});
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
            const auto mC = make_tensor(cute::make_gmem_ptr(output),
                make_layout(cute::make_shape(M, N), cute::make_stride(N, 1)));
            const auto mD = make_tensor(cute::make_gmem_ptr(bias),
                make_layout(cute::make_shape(M, N), cute::make_stride(0, 1)));

            // M is padded, such that the below is correct
            const auto tilesM = M / bM;
            // We assert the below prior to this point
            const auto tilesN = N / bN;

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

            constexpr auto gCStoreOp = cutlass::NumericConverter<typename decltype(gC)::value_type,
                                                        typename decltype(accumulator)::value_type>{};
            constexpr auto gDLoadOp = cutlass::NumericConverter<typename decltype(accumulator)::value_type,
                                                        ElementD>{};

            // Assume elementwise operator
            typename BlockGEMM::FusedEpilogue epilogueOp{};
            constexpr auto trips = size(accumulator) / rScratch.size();

            // Prefetch from global to shared memory
            #pragma unroll
            for (int j = 0; j < elems; ++j) {
                workspace[threadIdx.x + j * threads] = tDgD(j);
            }

            #pragma unroll
            for (unsigned int i = 0; i < trips; ++i) {
                #pragma unroll
                for (unsigned int j = 0; j < elems; ++j) {
                    rScratch[j] = workspace[threadIdx.x + j * threads];
                    if (i + 1 < trips) {
                        // Eagerly start loads for the next batch, if needed
                        workspace[threadIdx.x + j * threads] = tDgD(j + (i + 1) * elems);
                    }
                }
                // Fused Bias Add and Activation Function on register fragment
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
        }
    };

    template<
        typename BlockGEMM,
        typename GPUType
    >
    struct FGT<TaskType::postGEMM, BlockGEMM, GPUType> {
        template<
            unsigned int elems = GPUType::rScratch::value,
            unsigned int threads = GPUType::OS::threads::value
        >
        __forceinline__ __device__
        void operator()(typename BlockGEMM::MatrixDType* __restrict__ const& workspace,
        const typename BlockGEMM::MatrixAType* __restrict__ const& inputs,
        const typename BlockGEMM::MatrixBType* __restrict__ const& weights,
        typename BlockGEMM::MatrixDType* __restrict__ const& output,
        const typename BlockGEMM::MatrixDType* __restrict__ const& bias,
        const unsigned int& M,
        const unsigned int& N,
        const unsigned int& K,
        const unsigned int& tileIdx) const {
            auto accumulator = cute::partition_fragment_C(typename BlockGEMM::MMA{}, typename BlockGEMM::TilerOut{});
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
            const auto mC = make_tensor(cute::make_gmem_ptr(output),
                make_layout(cute::make_shape(M, N), cute::make_stride(N, 1)));
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
                workspace[threadIdx.x + j * threads] = tDgD(j);
            }

            #pragma unroll
            for (unsigned int i = 0; i < trips; ++i) {
                #pragma unroll
                for (unsigned int j = 0; j < elems; ++j) {
                    rScratch[j] = workspace[threadIdx.x + j * threads];
                    if (i + 1 < trips) {
                        // Eagerly start loads for the next batch, if needed
                        workspace[threadIdx.x + j * threads] = tDgD(j + (i + 1) * elems);
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
        }
    };

    struct ProcessorArgs{
        // sensible sentinel values
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
        __device__
        ProcessorArgs(unsigned int* const& _sQ,
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
        sQ(_sQ), pDB(_pDB), tQH(_tQH), flags(_flags), gtQ(_gtQ), tQ(_tQ), tQS(_tQS), pd(_pd),
        tokenSize(_tokenSize), tN(_tN), tNx(_tNx) {}
    };

    template<
        typename GPUType,
        CombineMode c,
        typename ActivationOp = cute::identity,
        typename ActivationOpX = cute::identity,
        typename ElementC = float,
        typename ElementA,
        typename ElementB = ElementA,
        typename ScaleWeights
    >
    __device__ __forceinline__
    void start(cuda::std::byte* const& workspace,
        ScaleWeights const& sW, const uint16_t& _seqBit){
        static_assert(sizeof(SignalPayload<PacketStage::last>) == sizeof(unsigned long long int)
            && alignof(SignalPayload<PacketStage::last>) == alignof(unsigned long long int));
        __shared__ Task currentTask;
        __shared__ ProcessorArgs pA;
        unsigned int signal = 0U;
        uint16_t interrupt = 0U;
        const auto rSeqBit = _seqBit;
        if (!threadIdx.x) {
            pA = ProcessorArgs{
                bookkeeping.sQ() + blockIdx.x,
                bookkeeping.pDB() + blockIdx.x,
                bookkeeping.tQH(),
                bookkeeping.flags,
                bookkeeping.tQ(),
                bookkeeping.tQ() + bookkeeping.tPs * SUBSCRIBERS, // should be the external Q
                bookkeeping.tQS(),
                bookkeeping.pd,
                bookkeeping.ed,
                bookkeeping.tN,
                Bookkeeping::tiles<BLOCK_N>(bookkeeping.pd)
            };
        }
        using Operation = BlockMM<GPUType, ElementA, ElementB, ElementC, ActivationOp>;
        using OperationX = BlockMM<GPUType, ElementA, ElementB, ElementC, ActivationOpX>;
        constexpr auto preGEMM = FGT<TaskType::preGEMM, Operation, GPUType>{};
        constexpr auto postGEMM = FGT<TaskType::postGEMM, OperationX, GPUType>{};
        constexpr auto combineOp = Combine<GPUType, c>{};
        __syncthreads();

        while (!interrupt) {
            if (!threadIdx.x) {
                auto* tQSignal = pA.pDB;
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
                if (currentTask.taskType != TaskType::Interrupt) {
                    // Eagerly indicate readiness for the next task
                    atomicExch(pA.sQ, ready);
                }
            }
            __syncthreads();
            switch (currentTask.taskType) {
                case TaskType::preGEMM: {
                    constexpr unsigned int preIndex = 0;
                    preGEMM(CAST_TO(typename Operation::MatrixDType, workspace),
                        CONST_CAST_TO(typename Operation::MatrixAType, currentTask.aData),
                        CONST_CAST_TO(typename Operation::MatrixBType, currentTask.bData[preIndex]),
                        CAST_TO(typename Operation::MatrixDType, currentTask.cData[preIndex]),
                        CONST_CAST_TO(typename Operation::MatrixDType, currentTask.dData[preIndex]),
                        currentTask.M,
                        pA.pd,
                        pA.tokenSize,
                        currentTask.tileIdx);
                    __syncthreads();
                    // use warp 0
                    if (constexpr auto wS = 32; threadIdx.x / wS == 0) {
                        uint enqueue = 0U;
                        if (!threadIdx.x) {
                            __threadfence();
                            enqueue = atomicAdd(pA.tQS + currentTask.syncIdx, 1U) == pA.tN;
                        }
                        // Broadcast from t0 to everyone else in the warp
                        enqueue = __shfl_sync(0xffffffff, enqueue, 0);
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
                                currentTask.batchIdx,
                                currentTask.isPeerRemote,
                            };
                            for (unsigned int i = threadIdx.x; i < tasks; i += wS) {
                                nextTask.tileIdx = i;
                                tQ[i] = nextTask;
                            }
                            __syncwarp();
                            if (!threadIdx.x) {
                                __threadfence();
                                // notify scheduler
                                atomicAdd(pA.tQH, tasks);
                            }
                        }
                    }
                }
                break;
                case TaskType::postGEMM: {
                    constexpr unsigned int postIndex = 0;
                    postGEMM(CAST_TO(typename Operation::MatrixDType, workspace),
                        CONST_CAST_TO(typename Operation::MatrixAType, currentTask.aData),
                        CONST_CAST_TO(typename Operation::MatrixBType, currentTask.bData[postIndex]),
                        CAST_TO(typename Operation::MatrixDType, currentTask.cData[postIndex]),
                        CONST_CAST_TO(typename Operation::MatrixDType, currentTask.dData[postIndex]),
                        currentTask.M,
                        pA.tokenSize,
                        pA.pd,
                        currentTask.tileIdx);
                    __syncthreads();
                    if (!threadIdx.x) {
                        // Pack payload into single signal word
                        auto flagSignal = SignalPayload<PacketStage::last>{
                            currentTask.batchIdx,
                            rSeqBit,
                            currentTask.tileSize
                        };
                        if (currentTask.isPeerRemote) {
                            // Remote; check if we need to do the transfer
                            __threadfence();
                            if (atomicIncrement(pA.tQS + currentTask.syncIdx) == pA.tN + pA.tNx) {
                                // Batch remote network transfer to avoid overwhelming the NIC
                                nvshmem_putmem_signal_nbi(currentTask.cData[postIndex], currentTask.cData[postIndex],
                                    currentTask.tileSize * pA.tokenSize * sizeof(ElementA),
                                    pA.flags + currentTask.flagIdx,
                                    *CAST_TO(uint64_t, &flagSignal), NVSHMEM_SIGNAL_SET,
                                    currentTask.peerIdx);
                            }
                        }
                        else {
                            // Below is expensive, so we only invoke when necessary
                            __threadfence_system();
                            // send individual tile, no batching here
                            // Already did the network transfer in fGET, so set signal only
                            nvshmemx_signal_op(pA.flags + currentTask.flagIdx,
                             *CAST_TO(uint64_t, &flagSignal), NVSHMEM_SIGNAL_SET,
                             currentTask.peerIdx);
                        }
                    }
                }
                break;
                case TaskType::combine: {
                    combineOp(CAST_TO(typename Operation::MatrixDType, workspace),
                        CONST_CAST_TO(TokenIdxTuple, currentTask.aData),
                        CONST_CAST_TO(typename Operation::MatrixAType, currentTask.bData[0]),
                        CAST_TO(typename Operation::MatrixDType, currentTask.cData[0]),
                        sW,
                        currentTask.M,
                        pA.tokenSize,
                        currentTask.tileIdx,
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
