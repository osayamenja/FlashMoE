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
    enum class ReleaseType {
        stable,
        experimental
    };
    template<
        ReleaseType r = ReleaseType::stable,
        CombineMode c = CombineMode::single,
        unsigned int gM = BLOCK_M,
        unsigned int M = ACC::S::value,
        unsigned int N = ACC::H::value,
        class ScaleWeights,
        typename Element,
        unsigned int Arch = ACC::PeakHardware::arch::value,
        unsigned int elems = ACC::PeakHardware::rScratch::value,
        unsigned int threads = ACC::PeakHardware::OS::threads::value,
        unsigned int sharedSize = ACC::PeakHardware::sharedMemory::value + ACC::PeakHardware::spare::value
    >
    requires(TensorValueType<Element> &&
            elems % WARP_SIZE == 0 && // guarantees warp convergence
            aristos::isMatrix<ScaleWeights> &&
            cuda::std::is_same_v<typename ScaleWeights::value_type, Element>)
    __device__ __forceinline__
    void combine(Element* __restrict__ const& workspace,
            const TPS* __restrict__ const& tokenIndices,
            const Element* __restrict__ const& inputs,
            Element* __restrict__ const& moeOutput,
            ScaleWeights const& scale,
            const unsigned int& tileIdx,
            const uint16_t& tileSize,
            const uint16_t& expertIdx) {
        using BlockTiler = cute::Shape<cute::Int<BLOCK_M>, cute::Int<BLOCK_N>>;
        constexpr BlockTiler tiler{};
        constexpr auto bM = cute::get<0>(tiler);
        constexpr auto bN = cute::get<1>(tiler);
        cutlass::AlignedArray<Element, bN> registers{};
        constexpr auto mTe = cutlass::NumericConverter<Element, mp_t>{};
        constexpr auto eTm = cutlass::NumericConverter<mp_t, Element>{};
        // Row-major
        const auto mA = make_tensor(cute::make_gmem_ptr(inputs),
            cute::Layout<cute::Shape<cute::Int<gM>, cute::Int<N>>,
                cute::Stride<cute::Int<N>, cute::_1>>{});
        const auto mC = make_tensor(cute::make_gmem_ptr(moeOutput),
            cute::Layout<cute::Shape<cute::Int<M>, cute::Int<N>>,
                cute::Stride<cute::Int<N>, cute::_1>>{});

        // We assert the below prior to this point
        static_assert(gM % bM == 0);
        constexpr auto tilesM = gM / bM;
        constexpr auto tilesN = N / bN;

        const auto tileCoord = idx2crd(tileIdx,
            cute::Shape<cute::Int<tilesM>, cute::Int<tilesN>>{},
                cute::Stride<cute::Int<tilesN>, cute::_1>{});
        const auto ctaCoord = cute::make_coord(cute::get<0>(tileCoord), cute::get<1>(tileCoord));
        const auto gA = cute::local_tile(mA, tiler, ctaCoord);

        const auto tileCoordOut = idx2crd(tileIdx,
            cute::Shape<cute::_1, cute::Int<tilesN>>{},
                cute::Stride<cute::Int<tilesN>, cute::_1>{});
        const auto gC = cute::local_tile(mC,
            cute::Shape<cute::Int<M>, cute::Int<bN>>{},
                cute::make_coord(cute::get<0>(tileCoordOut),
                    cute::get<1>(tileCoordOut)));
        static_assert(bN % elems == 0);
        constexpr auto trips = bN / elems;
        // Transposed layout
        constexpr auto sCLay = make_layout(cute::Shape<cute::Int<bM>, cute::Int<elems>>{});
        const auto sC = cute::make_tensor(cute::make_smem_ptr(workspace), sCLay);
        // ensures we have enough shared memory
        static_assert(sizeof(Element) * bM * (elems + 1) + sizeof(TPS) * bM <= sharedSize);
        static_assert(bM % elems == 0);
        if constexpr (r == ReleaseType::experimental) {
            cutlass::AlignedArray<TPS, elems> tIds{};
            auto* __restrict__ sTPS = CAST_TO(TPS, workspace + bM * elems);
            #pragma unroll
            for (uint i = threadIdx.x; i < bM; i += threads) {
                sTPS[i] = tokenIndices[i];
            }
            __syncthreads();
            // slice and dice token indices
            constexpr auto phases = bM / elems;
            // all threads in a phase read the same token index
            const auto phaseIdx = threadIdx.x / elems;
            // Eagerly prefetch inputs to registers
            #pragma unroll
            for (uint i = 0; i < trips; ++i) {
                // global -> shared
                #pragma unroll
                for (uint j = 0; j < elems; ++j) {
                    const auto rIdx = phaseIdx + j * phases;
                    const auto cIdx =  threadIdx.x % elems + i * elems;
                    sC(threadIdx.x, j) = gA(rIdx, cIdx);
                }
                #pragma unroll
                for (uint j = 0; j < elems; ++j) {
                    registers[j + i * elems] = sC(threadIdx.x, j);
                }
            }
            #pragma unroll
            for (uint i = 0; i < elems; ++i) {
                tIds[i] = sTPS[phaseIdx + i * phases];
            }

            if constexpr (c == CombineMode::multithreaded) {
                using CDxT = typename ToCDx<Element>::T;
                constexpr auto cTCx = cutlass::NumericConverter<CDxT, mp_t>{};
                // prefetch scale to shared memory
                auto* __restrict__ sScale = CAST_TO(Element, sTPS + bM);
                #pragma unroll
                for (uint i = threadIdx.x; i < bM; i += threads) {
                    sScale[i] = scale(sTPS[i], expertIdx);
                }
                __syncthreads();
                cutlass::AlignedArray<Element, elems> scaleRegs{};
                // fetch scale
                #pragma unroll
                for (uint i = 0; i < elems; ++i) {
                    scaleRegs[i] = sScale[phaseIdx + i * phases];
                }
                // apply scale
                #pragma unroll
                for (uint i = 0; i < elems; ++i) {
                    tIds[i].probability =__fdividef(eTm(scaleRegs[i]), tIds[i].probability);
                }

                #pragma unroll
                for (uint i = 0; i < trips; ++i) {
                    // do atomic addition
                    if (tileSize < gM) {
                        #pragma unroll
                        for (uint j = 0; j < elems; ++j) {
                            const auto cIdx = threadIdx.x % elems + i * elems;
                            if (phaseIdx + j * phases < tileSize) {
                                atomicAdd(CAST_TO(CDxT, &gC(tIds[j].tokenIdx, cIdx)),
                                    cTCx(tIds[j].probability * eTm(registers[j + i * elems])));
                            }
                        }
                    }
                    else {
                        #pragma unroll
                        for (uint j = 0; j < elems; ++j) {
                            const auto cIdx = threadIdx.x % elems + i * elems;
                            atomicAdd(CAST_TO(CDxT, &gC(tIds[j].tokenIdx, cIdx)),
                                cTCx(tIds[j].probability * eTm(registers[j + i * elems])));
                        }
                    }
                }
            }
            else {
                // vector copy from registers to global directly and call it a day
                #pragma unroll
                for (uint i = 0; i < trips; ++i) {
                    #pragma unroll
                    for (uint j = 0; j < elems; ++j) {
                        // registers -> shared
                        sC(threadIdx.x, j) = registers[j + i * elems];
                    }
                    // coalesced shared -> global
                    if (tileSize < gM) {
                        #pragma unroll
                        for (uint j = 0; j < elems; ++j) {
                            const auto cIdx = threadIdx.x % elems + i * elems;
                            const auto tId = tIds[j].tokenIdx;
                            // predicated writes
                            if (phaseIdx + j * phases < tileSize) {
                                gC(tId, cIdx) = sC(threadIdx.x, j);
                            }
                        }
                    }
                    else {
                        #pragma unroll
                        for (uint j = 0; j < elems; ++j) {
                            const auto cIdx = threadIdx.x % elems + i * elems;
                            gC(tIds[j].tokenIdx, cIdx) = sC(threadIdx.x, j);
                        }
                    }
                }
            }
        }
        else {
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
                const auto [tokenIdx, combineWeight] = tokenIndices[threadIdx.x];
                const auto scaleWeight = scale(tokenIdx, expertIdx) / mTe(combineWeight);
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
        const auto gA = cute::local_tile(mA, typename BlockGEMM::BlockTiler{}, ctaCoord,
            cute::Step<cute::_1, cute::X,cute::_1>{});
        const auto gB = cute::local_tile(mB, typename BlockGEMM::BlockTiler{}, ctaCoord,
            cute::Step< cute::X,cute::_1,cute::_1>{});
        const auto gC = cute::local_tile(mC, typename BlockGEMM::BlockTiler{}, ctaCoord,
            cute::Step<cute::_1,cute::_1, cute::X>{});
        const auto gD = cute::local_tile(mD, typename BlockGEMM::BlockTiler{}, ctaCoord,
            cute::Step<cute::_1,cute::_1, cute::X>{});

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
        unsigned int N,
        unsigned int K,
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
        const auto gA = cute::local_tile(mA, typename BlockGEMM::BlockTiler{}, ctaCoord,
            cute::Step<cute::_1, cute::X,cute::_1>{});
        const auto gB = cute::local_tile(mB, typename BlockGEMM::BlockTiler{}, ctaCoord,
            cute::Step< cute::X,cute::_1,cute::_1>{});
        const auto gC = cute::local_tile(mC, typename BlockGEMM::BlockTiler{}, ctaCoord,
            cute::Step<cute::_1,cute::_1, cute::X>{});
        const auto gD = cute::local_tile(mD, typename BlockGEMM::BlockTiler{}, ctaCoord,
            cute::Step<cute::_1,cute::_1, cute::X>{});

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

    struct __align__(16) ProcessorArgs{
        // sensible sentinel values
        unsigned int* __restrict__ sQ = nullptr;
        TQSignal* __restrict__ pDB = nullptr;
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
        sQ(_sQ), pDB(_pDB), tQH(_tQH), tQ(_tQ), ptQ(_ptQ), tQS(_tQS) {}
    };

    template<
        PeerConnectivity p,
        unsigned int tasks = ACC::TNx::value
    >
    __device__ __forceinline__
    void notifyNext(const Task& rCurrentTask, uint* __restrict__ const& workspace, const ProcessorArgs& pA) {
        static_assert(sizeof(Task) == 128);
        constexpr auto eS = sizeof(Task) / sizeof(uint);
        static_assert(eS == WARP_SIZE);
        constexpr auto threads = ACC::PeakHardware::OS::threads::value;
        constexpr auto sharedSize = ACC::sharedSize::value;
        static_assert(sharedSize % sizeof(Task) == 0);
        static_assert(sharedSize / sizeof(Task) >= threads);
        constexpr auto capacity = threads;
        constexpr auto trips = tasks / capacity;
        static_assert(threads % eS == 0);
        static_assert(capacity % threads == 0);
        constexpr auto elems = capacity * eS / threads;
        constexpr unsigned int preIndex = 0;

        const auto fO = ACC::TNx::value * (rCurrentTask.tileIdx / ACC::TN::value);
        auto* __restrict__ tQ = pA.ptQ + rCurrentTask.syncIdx * ACC::TNx::value;
        auto* __restrict__ tQH = pA.tQH;
        const auto cIdx = threadIdx.x % eS;
        const auto rIdx = threadIdx.x / eS * eS;
        // prep memory-view tensors
        const auto sTQ = make_tensor(cute::make_smem_ptr(workspace),
            cute::Layout<cute::Shape<cute::Int<threads>, cute::Int<eS>>,
                cute::Stride<cute::Int<eS>, cute::_1>>{});
        const auto gTQ = make_tensor(cute::make_gmem_ptr(tQ),
            cute::Layout<cute::Shape<cute::Int<tasks>, cute::Int<eS>>,
                cute::Stride<cute::Int<eS>, cute::_1>>{});
        // copy from registers to shared memory using swizzle
        #pragma unroll
        for (uint i = 0; i < trips; ++i) {
            // each thread does a copy from registers to shared memory
            const auto tileIdx = threadIdx.x + i * capacity;
            const auto nextTask = Task {
                TaskType::postGEMM,
                rCurrentTask.cData[preIndex],
                rCurrentTask.bData,
                rCurrentTask.cData,
                rCurrentTask.dData,
                rCurrentTask.rcData,
                rCurrentTask.flags + fO + (p == PeerConnectivity::p2p ? tileIdx : 0),
                rCurrentTask.syncIdx,
                tileIdx,
                rCurrentTask.M,
                rCurrentTask.tileSize,
                rCurrentTask.peerIdx,
                rCurrentTask.batchIdx,
                rCurrentTask.isPeerRemote,
            };
            // Directive to the compiler to reinterpret the Task structure as a stream of 4-byte blocks
            const auto* __restrict__ uT = CONST_CAST_TO(uint, &nextTask);
            #pragma unroll
            for (uint j = 0; j < eS; ++j) {
                // temporal shift of indices to eliminate bank conflicts
                const auto swizzleIdx = (j + threadIdx.x) % eS;
                sTQ(threadIdx.x, swizzleIdx) = uT[j];
            }
            __syncthreads();
            // now copy from shared memory to global memory
            #pragma unroll
            for (uint j = 0; j < elems; ++j) {
                gTQ(rIdx + (j + i * capacity), cIdx) = sTQ(rIdx + j, cIdx);
            }
        }
        if constexpr (constexpr auto residue = tasks - trips * capacity; residue) {
            if (threadIdx.x < residue) {
                const auto tileIdx = threadIdx.x + trips * threads;
                const auto nextTask = Task {
                    TaskType::postGEMM,
                    rCurrentTask.cData[preIndex],
                    rCurrentTask.bData,
                    rCurrentTask.cData,
                    rCurrentTask.dData,
                    rCurrentTask.rcData,
                    rCurrentTask.flags + fO + (p == PeerConnectivity::p2p ? tileIdx : 0),
                    rCurrentTask.syncIdx,
                    tileIdx,
                    rCurrentTask.M,
                    rCurrentTask.tileSize,
                    rCurrentTask.peerIdx,
                    rCurrentTask.batchIdx,
                    rCurrentTask.isPeerRemote,
                };
                // Directive to the compiler to reinterpret the Task structure as a stream of 4-byte blocks
                const auto* __restrict__ uT = CONST_CAST_TO(uint, &nextTask);
                #pragma unroll
                for (uint j = 0; j < eS; ++j) {
                    // temporal shift of indices to eliminate bank conflicts
                    const auto swizzleIdx = (j + threadIdx.x) % eS;
                    sTQ(threadIdx.x, swizzleIdx) = uT[j];
                }
            }
            __syncthreads();
            constexpr auto stride = threads / eS;
            const auto pIdx = threadIdx.x / eS;
            // now copy from shared memory to global memory by multiplexing each row across available warps
            #pragma unroll
            for (uint j = pIdx; j < residue; j += stride) {
                gTQ(rIdx + (j + trips * capacity), cIdx) = sTQ(rIdx + j, cIdx);
            }
        }

        __syncthreads();
        if (!threadIdx.x) {
            __threadfence();
            // notify scheduler
            atomicAdd(tQH + rCurrentTask.syncIdx, tasks);
        }
    }

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
        static_assert(sizeof(Task) == 128);
        __shared__ uint globalInterrupt;
        __shared__ uint enqueue;

        // Register allocations
        const auto rSeqBit = _seqBit;
        Task rCurrentTask{};
        TQSignal tqs{0U, 0U};
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
        using PreGEMM = BlockMM<ACC::ActivationOp, Element>;
        using PostGEMM = BlockMM<ACC::ActivationOpX, Element>;
        constexpr uint H = ACC::H::value;
        constexpr auto tN = ACC::TN::value;
        constexpr auto tNx = ACC::TNx::value;
        __syncthreads();
        while (!tqs.interrupt) {
            if (constexpr auto wS = 32; threadIdx.x / wS == 0) {
                if (!threadIdx.x) {
                    auto* __restrict__ tQSignal = pA.pDB;
                    // Grabs next task
                    awaitNotification(tQSignal, &tqs, tqs.signal);
                    __threadfence();
                    // Eagerly indicate readiness for the next task as the above fence allows us to do so correctly
                    globalInterrupt = tqs.interrupt;
                    atomicExch(pA.sQ, ready);
                }
                // The below is necessary as it guarantees memory ordering
                __syncwarp();
                auto* __restrict__ tqsP = CAST_TO(ull_t, &tqs);
                *tqsP = __shfl_sync(0xffffffff, *tqsP, 0);
                const auto* __restrict__ gtQ = pA.tQ + tqs.decodeSig();
                if (!tqs.interrupt) {
                    // coalesced copy from global to shared memory
                    CAST_TO(uint, &currentTask)[threadIdx.x] = __ldg(CONST_CAST_TO(uint, gtQ) + threadIdx.x);
                }
            }
            __syncthreads();
            tqs.interrupt = globalInterrupt;
            // if we received an interrupt, there is nothing to do next
            if (!tqs.interrupt) {
                // shared -> registers
                rCurrentTask = currentTask;
                switch (rCurrentTask.taskType) {
                    case TaskType::preGEMM: {
                        constexpr unsigned int preIndex = 0;
                        fGET<PreGEMM, ACC::P::value, ACC::H::value>(
                            CAST_TO(typename PreGEMM::MatrixDType, workspace),
                            CONST_CAST_TO(typename PreGEMM::MatrixAType, rCurrentTask.aData),
                            CONST_CAST_TO(typename PreGEMM::MatrixBType, rCurrentTask.bData[preIndex]),
                            CAST_TO(typename PreGEMM::MatrixDType, rCurrentTask.cData[preIndex]),
                            CONST_CAST_TO(typename PreGEMM::MatrixDType, rCurrentTask.dData[preIndex]),
                            rCurrentTask.M,
                            rCurrentTask.tileIdx);
                        __syncthreads();
                        if (!threadIdx.x) {
                            __threadfence();
                            enqueue = atomicAdd(pA.tQS + rCurrentTask.syncIdx, 1U) + 1 == tN;
                        }
                        __syncthreads();
                        if (enqueue) {
                            if (!rCurrentTask.isPeerRemote)[[likely]] {
                                notifyNext<PeerConnectivity::p2p>(rCurrentTask, CAST_TO(uint, workspace), pA);
                            }
                            else {
                                notifyNext<PeerConnectivity::remote>(rCurrentTask, CAST_TO(uint, workspace), pA);
                            }
                        }
                    }
                    break;
                    case TaskType::postGEMM: {
                        constexpr unsigned int postIndex = 1;
                        fGET<PostGEMM, ACC::H::value, ACC::P::value>(
                            CAST_TO(typename PostGEMM::MatrixDType, workspace),
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
                                if (atomicIncrement(pA.tQS + rCurrentTask.syncIdx) + 1 == tN + tNx) {
                                    nvshmem_putmem_signal_nbi(rCurrentTask.rcData,
                                        rCurrentTask.cData[postIndex],
                                        // Batched remote network transfer to avoid overwhelming the NIC
                                        rCurrentTask.tileSize * H * sizeof(Element),
                                        rCurrentTask.flags,
                                        *CONST_CAST_TO(flagsType, &flagSignal), NVSHMEM_SIGNAL_SET,
                                        rCurrentTask.peerIdx);
                                }
                            }
                            else {
                                // individual tile, no batching here
                                // Already did the network transfer,
                                // so set signal only
                                __threadfence_system();
                                atomicExch_system(CAST_TO(ull_t, rCurrentTask.flags),
                                    *CONST_CAST_TO(ull_t, &flagSignal));
                            }
                        }
                    }
                    break;
                    case TaskType::combine: {
                        constexpr unsigned int combineIndex = 0;
                        combine<ReleaseType::experimental, ACC::CM::value>(
                            CAST_TO(typename PostGEMM::MatrixDType, workspace),
                            CONST_CAST_TO(TPS, rCurrentTask.aData),
                            CONST_CAST_TO(typename PostGEMM::MatrixAType, rCurrentTask.bData[combineIndex]),
                            CAST_TO(typename PostGEMM::MatrixDType, rCurrentTask.cData[combineIndex]),
                            sW,
                            rCurrentTask.tileIdx,
                            rCurrentTask.tileSize,
                            rCurrentTask.expertIdx);
                    }
                    break;
                }
            }
        }
    }
}
#endif //ARISTOS_COMPUTE_CUH
