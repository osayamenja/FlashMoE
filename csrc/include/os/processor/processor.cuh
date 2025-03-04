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
        CombineMode c = CombineMode::single,
        unsigned int N = ACC::H::value,
        class ScaleWeights,
        typename Element,
        unsigned int Arch = ACC::PeakHardware::arch::value,
        unsigned int elems = ACC::PeakHardware::rScratch::value
    >
    requires(TensorValueType<Element> &&
            aristos::isMatrix<ScaleWeights> &&
            cuda::std::is_same_v<typename ScaleWeights::value_type, Element>)
    __device__ __forceinline__
    void combine(Element* __restrict__ const& workspace,
            const TokenIdxTuple* __restrict__ const& tokenIndices,
            const Element* __restrict__ const& inputs,
            Element* __restrict__ const& moeOutput,
            ScaleWeights const& scale,
            const unsigned int& M,
            const unsigned int& tileIdx,
            const uint16_t& tileSize,
            const uint16_t& expertIdx) {
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

    // fused GEMM, epilogue and data transfer, with static M, N and K
    template<
        typename BlockGEMM,
        unsigned int M = ACC::S::value,
        unsigned int N = ACC::H::value,
        unsigned int K = ACC::P::value,
        unsigned int elems = ACC::PeakHardware::rScratch::value,
        unsigned int threads = ACC::PeakHardware::OS::threads::value
    >
    __forceinline__ __device__
    void sfGET(typename BlockGEMM::MatrixDType* __restrict__ const& workspace,
        const typename BlockGEMM::MatrixAType* __restrict__ const& inputs,
        const typename BlockGEMM::MatrixBType* __restrict__ const& weights,
        typename BlockGEMM::MatrixDType* __restrict__ const& output,
        const typename BlockGEMM::MatrixDType* __restrict__ const& bias,
        const unsigned int& tileIdx) {
        auto accumulator = cute::partition_fragment_C(typename BlockGEMM::MMA{}, typename BlockGEMM::TilerOut{});
        static_assert(cute::size(accumulator) % elems == 0);
        cutlass::AlignedArray<typename BlockGEMM::MatrixDType, elems> rScratch{};
        constexpr auto bM = cute::get<0>(typename BlockGEMM::BlockTiler{});
        constexpr auto bN = cute::get<1>(typename BlockGEMM::BlockTiler{});
        constexpr auto bK = cute::get<2>(typename BlockGEMM::BlockTiler{});
        // Instantiate mainloop
        typename BlockGEMM::CollectiveMainloop mainLoop{};
        cute::clear(accumulator);

        // Row-major
        const auto mA = make_tensor(cute::make_gmem_ptr(inputs),
        cute::Layout<cute::Shape<cute::Int<M>, cute::Int<K>>,
            cute::Stride<cute::Int<K>, cute::_1>>{});
        // Row-major, transposed
        const auto mB = make_tensor(cute::make_gmem_ptr(weights),
        cute::Layout<cute::Shape<cute::Int<N>, cute::Int<K>>,
            cute::Stride<cute::Int<K>, cute::_1>>{});
        // Row-major
        const auto mC = make_tensor(cute::make_gmem_ptr(output),
        cute::Layout<cute::Shape<cute::Int<M>, cute::Int<N>>,
            cute::Stride<cute::Int<N>, cute::_1>>{});
        const auto mD = make_tensor(cute::make_gmem_ptr(bias),
        cute::Layout<cute::Shape<cute::Int<M>, cute::Int<N>>,
            cute::Stride<cute::_0, cute::_1>>{});

        // M is padded, such that the below is correct
        constexpr auto tilesM = M / bM;
        // We assert the below prior to this point
        constexpr auto tilesN = N / bN;
        constexpr auto tilesK = K / bK;

        const auto tileCoord = idx2crd(tileIdx, cute::Shape<cute::Int<tilesM>, cute::Int<tilesN>>{},
            cute::Stride<cute::Int<tilesN>, cute::_1>{});
        const auto ctaCoord = make_coord(cute::get<0>(tileCoord), cute::get<1>(tileCoord), cute::_);
        const auto gA = cute::local_tile(mA, typename BlockGEMM::BlockTiler{}, ctaCoord, cute::Step<cute::_1, cute::X,cute::_1>{});
        const auto gB = cute::local_tile(mB, typename BlockGEMM::BlockTiler{}, ctaCoord, cute::Step< cute::X,cute::_1,cute::_1>{});
        const auto gC = cute::local_tile(mC, typename BlockGEMM::BlockTiler{}, ctaCoord, cute::Step<cute::_1,cute::_1, cute::X>{});
        const auto gD = cute::local_tile(mD, typename BlockGEMM::BlockTiler{}, ctaCoord, cute::Step<cute::_1,cute::_1, cute::X>{});

        auto k_tile_iter = cute::make_coord_iterator(tilesK);

        using ElementD = typename BlockGEMM::MatrixDType;
        mainLoop(
            accumulator,
            gA,
            gB,
            accumulator,
            k_tile_iter, tilesK,
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

    // fused GEMM, epilogue and data transfer, with dynamic M and static N and K
    template<
        typename BlockGEMM,
        unsigned int N = ACC::H::value,
        unsigned int K = ACC::P::value,
        unsigned int elems = ACC::PeakHardware::rScratch::value,
        unsigned int threads = ACC::PeakHardware::OS::threads::value
    >
    __forceinline__ __device__
    void fGET(typename BlockGEMM::MatrixDType* __restrict__ const& workspace,
        const typename BlockGEMM::MatrixAType* __restrict__ const& inputs,
        const typename BlockGEMM::MatrixBType* __restrict__ const& weights,
        typename BlockGEMM::MatrixDType* __restrict__ const& output,
        const typename BlockGEMM::MatrixDType* __restrict__ const& bias,
        const unsigned int& M,
        const unsigned int& tileIdx) {
        auto accumulator = cute::partition_fragment_C(typename BlockGEMM::MMA{}, typename BlockGEMM::TilerOut{});
        static_assert(cute::size(accumulator) % elems == 0);
        cutlass::AlignedArray<typename BlockGEMM::MatrixDType, elems> rScratch{};
        constexpr auto bM = cute::get<0>(typename BlockGEMM::BlockTiler{});
        constexpr auto bN = cute::get<1>(typename BlockGEMM::BlockTiler{});
        constexpr auto bK = cute::get<2>(typename BlockGEMM::BlockTiler{});
        // Instantiate mainloop
        typename BlockGEMM::CollectiveMainloop mainLoop{};
        cute::clear(accumulator);

        // Row-major
        const auto mA = make_tensor(cute::make_gmem_ptr(inputs),
            make_layout(cute::make_shape(M, K), cute::Stride<cute::Int<K>, cute::_1>{}));
        // Row-major, transposed
        const auto mB = make_tensor(cute::make_gmem_ptr(weights),
            cute::Layout<cute::Shape<cute::Int<N>, cute::Int<K>>,
                cute::Stride<cute::Int<K>, cute::_1>>{});
        // Row-major
        const auto mC = make_tensor(cute::make_gmem_ptr(output),
            make_layout(cute::make_shape(M, N), cute::Stride<cute::Int<N>, cute::_1>{}));
        const auto mD = make_tensor(cute::make_gmem_ptr(bias),
            make_layout(cute::make_shape(M, N), cute::Stride<cute::_0, cute::_1>{}));

        // M is padded, such that the below is correct
        const auto tilesM = M / bM;
        // We assert the below prior to this point
        constexpr auto tilesN = N / bN;
        constexpr auto tilesK = K / bK;

        const auto tileCoord = idx2crd(tileIdx, cute::Shape(tilesM, tilesN),
            cute::Stride<cute::Int<tilesN>, cute::_1>{});
        const auto ctaCoord = make_coord(cute::get<0>(tileCoord), cute::get<1>(tileCoord), cute::_);
        const auto gA = cute::local_tile(mA, typename BlockGEMM::BlockTiler{}, ctaCoord, cute::Step<cute::_1, cute::X,cute::_1>{});
        const auto gB = cute::local_tile(mB, typename BlockGEMM::BlockTiler{}, ctaCoord, cute::Step< cute::X,cute::_1,cute::_1>{});
        const auto gC = cute::local_tile(mC, typename BlockGEMM::BlockTiler{}, ctaCoord, cute::Step<cute::_1,cute::_1, cute::X>{});
        const auto gD = cute::local_tile(mD, typename BlockGEMM::BlockTiler{}, ctaCoord, cute::Step<cute::_1,cute::_1, cute::X>{});

        auto k_tile_iter = cute::make_coord_iterator(tilesK);

        using ElementD = typename BlockGEMM::MatrixDType;
        mainLoop(
            accumulator,
            gA,
            gB,
            accumulator,
            k_tile_iter, tilesK,
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

    struct ProcessorArgs{
        // sensible sentinel values
        unsigned int* __restrict__ sQ = nullptr;
        TQSignal* __restrict__ pDB = nullptr;
        unsigned int* __restrict__ tQH = nullptr;
        flagsType* __restrict__ flags = nullptr;
        Task* __restrict__ gtQ = nullptr;
        Task* __restrict__ tQ = nullptr;
        unsigned int* __restrict__ tQS = nullptr;

        ProcessorArgs() = default;
        __device__
        ProcessorArgs(unsigned int* const& _sQ,
            TQSignal* const& _pDB,
            unsigned int* const& _tQH,
            flagsType* const& _flags,
            Task* const& _gtQ,
            Task* const& _tQ,
            unsigned int* const& _tQS) :
        sQ(_sQ), pDB(_pDB), tQH(_tQH), flags(_flags), gtQ(_gtQ), tQ(_tQ), tQS(_tQS) {}
    };

    template<
        typename ScaleWeights
    >
    __device__ __forceinline__
    void start(cuda::std::byte* const& workspace,
        ScaleWeights const& sW, const uint16_t& _seqBit){
        using Element = ACC::Element;
        static_assert(sizeof(SignalPayload<PacketStage::last>) == sizeof(flagsType)
            && alignof(SignalPayload<PacketStage::last>) == alignof(flagsType));

        __shared__ Task currentTask;
        __shared__ ProcessorArgs pA;
        __shared__ uint globalInterrupt;
        __shared__ void* cachedFlags;

        // Register allocations
        uint16_t interrupt = 0U;
        const auto rSeqBit = _seqBit;
        Task rCurrentTask{};
        TQSignal tqs{};

        if (!threadIdx.x) {
            globalInterrupt = 0U;
            pA = ProcessorArgs{
                bookkeeping.sQ() + blockIdx.x,
                bookkeeping.pDB() + blockIdx.x,
                bookkeeping.tQH(),
                bookkeeping.flags,
                bookkeeping.tQ(),
                bookkeeping.tQ() + bookkeeping.tPs * SUBSCRIBERS, // should be the external Q
                bookkeeping.tSA()
            };
        }

        using PreGEMM = BlockMM<ACC::ActivationOp, Element>;
        using PostGEMM = BlockMM<ACC::ActivationOpX, Element>;
        constexpr uint H = ACC::H::value;
        constexpr auto bN = cute::get<1>(typename PreGEMM::BlockTiler{});
        constexpr auto tN = ACC::TN::value;
        constexpr auto tNx = ACC::TNx::value;
        __syncthreads();

        while (!interrupt) {
            if (!threadIdx.x) {
                auto* __restrict__ tQSignal = pA.pDB;
                const auto* __restrict__ gtQ = pA.gtQ;
                // Grabs next task
                awaitNotification(tQSignal, &tqs, tqs.signal);
                globalInterrupt = tqs.interrupt;
                if (!tqs.interrupt) {
                    // below ensures task read happens after signal reception
                    __threadfence();
                    // global -> shared
                    currentTask = gtQ[tqs.signal - 1];
                    // Eagerly indicate readiness for the next task
                    atomicExch(pA.sQ, ready);
                }
            }
            __syncthreads();
            interrupt = globalInterrupt;
            // if received interrupt, there is nothing to do next
            if (!interrupt) {
                // shared -> registers
                rCurrentTask = currentTask;
                switch (rCurrentTask.taskType) {
                    case TaskType::preGEMM: {
                        constexpr unsigned int preIndex = 0;
                        fGET<PreGEMM>(CAST_TO(typename PreGEMM::MatrixDType, workspace),
                            CONST_CAST_TO(typename PreGEMM::MatrixAType, rCurrentTask.aData),
                            CONST_CAST_TO(typename PreGEMM::MatrixBType, rCurrentTask.bData[preIndex]),
                            CAST_TO(typename PreGEMM::MatrixDType, rCurrentTask.cData[preIndex]),
                            CONST_CAST_TO(typename PreGEMM::MatrixDType, rCurrentTask.dData[preIndex]),
                            rCurrentTask.M,
                            rCurrentTask.tileIdx);
                        __syncthreads();
                        // use warp 0
                        if (constexpr auto wS = 32; threadIdx.x / wS == 0) {
                            uint enqueue = 0U;
                            if (!threadIdx.x) {
                                __threadfence();
                                enqueue = atomicAdd(pA.tQS + rCurrentTask.syncIdx, 1U) == tN;
                            }
                            // Broadcast from t0 to everyone else in the warp
                            enqueue = __shfl_sync(0xffffffff, enqueue, 0);
                            if (enqueue) {
                                auto* __restrict__ tQ = pA.tQ + rCurrentTask.syncIdx;
                                auto* __restrict__ tQH = pA.tQH;
                                auto nextTask = Task {
                                    TaskType::postGEMM,
                                    rCurrentTask.cData[preIndex],
                                    rCurrentTask.bData,
                                    rCurrentTask.cData,
                                    rCurrentTask.dData,
                                    rCurrentTask.syncIdx,
                                    0,
                                    rCurrentTask.M,
                                    rCurrentTask.flagIdx,
                                    rCurrentTask.tileSize,
                                    rCurrentTask.peerIdx,
                                    rCurrentTask.batchIdx,
                                    rCurrentTask.isPeerRemote,
                                };
                                #pragma unroll
                                for (unsigned int i = threadIdx.x; i < tNx; i += wS) {
                                    nextTask.tileIdx = i;
                                    tQ[i] = nextTask;
                                }
                                __syncwarp();
                                if (!threadIdx.x) {
                                    __threadfence();
                                    // notify scheduler
                                    atomicAdd(tQH + rCurrentTask.syncIdx, tNx);
                                }
                            }
                        }
                    }
                    break;
                    case TaskType::postGEMM: {
                        if (!threadIdx.x) {
                            // below is a ldg read so initiate it early
                            cachedFlags = nvshmem_ptr(pA.flags, rCurrentTask.peerIdx);
                        }
                        constexpr unsigned int postIndex = 0;
                        fGET<PostGEMM>(CAST_TO(typename PostGEMM::MatrixDType, workspace),
                            CONST_CAST_TO(typename PostGEMM::MatrixAType, rCurrentTask.aData),
                            CONST_CAST_TO(typename PostGEMM::MatrixBType, rCurrentTask.bData[postIndex]),
                            CAST_TO(typename PostGEMM::MatrixDType, rCurrentTask.cData[postIndex]),
                            CONST_CAST_TO(typename PostGEMM::MatrixDType, rCurrentTask.dData[postIndex]),
                            rCurrentTask.M,
                            currentTask.tileIdx);
                        __syncthreads();
                        if (!threadIdx.x) {
                            // Pack payload into single signal word of 8 bytes
                            const auto flagSignal = SignalPayload<PacketStage::last>{
                                rCurrentTask.batchIdx,
                                rCurrentTask.tileSize,
                                rSeqBit,
                            };
                            if (rCurrentTask.isPeerRemote) {
                                // Remote; check if we need to do the transfer
                                __threadfence();
                                if (atomicIncrement(pA.tQS + rCurrentTask.syncIdx) == tN + tNx) {
                                    // Batched remote network transfer to avoid overwhelming the NIC
                                    nvshmem_putmem_signal_nbi(rCurrentTask.cData[postIndex], rCurrentTask.cData[postIndex],
                                        rCurrentTask.tileSize * H * sizeof(Element),
                                        pA.flags + rCurrentTask.flagIdx,
                                        *CONST_CAST_TO(flagsType, &flagSignal), NVSHMEM_SIGNAL_SET,
                                        rCurrentTask.peerIdx);
                                }
                            }
                            else {
                                // individual tile, no batching here
                                // Already did the network transfer,
                                // so set signal only
                                __threadfence_system();
                                atomicExch_system(CAST_TO(ull_t, cachedFlags) + rCurrentTask.flagIdx,
                                    *CONST_CAST_TO(ull_t, &flagSignal));
                            }
                        }
                    }
                    break;
                    case TaskType::combine: {
                        combine<ACC::CM::value>(CAST_TO(typename PostGEMM::MatrixDType, workspace),
                            CONST_CAST_TO(TokenIdxTuple, rCurrentTask.aData),
                            CONST_CAST_TO(typename PostGEMM::MatrixAType, rCurrentTask.bData[0]),
                            CAST_TO(typename PostGEMM::MatrixDType, rCurrentTask.cData[0]),
                            sW,
                            rCurrentTask.M,
                            rCurrentTask.tileIdx,
                            rCurrentTask.tileSize, rCurrentTask.expertIdx);
                    }
                    break;
                }
            }
        }
    }
}
#endif //ARISTOS_COMPUTE_CUH
