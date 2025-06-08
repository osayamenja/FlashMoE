/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Kleos Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 */

//
// Created by oja7 on 11/25/24.
//

#ifndef GATE_CUH
#define GATE_CUH

#include <cub/cub.cuh>
#include <cuda/std/array>

#include "../os/processor/gemm.cuh"
#include "../types.cuh"
#include "../atomics.cuh"

namespace kleos::gate {
    template<
        GateReductionLevel g = GateReductionLevel::singleBlock,
        JobType j = JobType::inference
    >
    struct GateArgs {
        static_assert(g == GateReductionLevel::singleBlock && j == JobType::inference);
        TPS* __restrict__ tP;
        BookType* __restrict__ eC;
        __device__
        GateArgs(TPS* const& _tP,
            BookType* const& _eC,
            mp_t* const&,
            mp_t* const&,
            RingSoftmaxPayload* const&,
            RingTopKPayload* const&) : tP(_tP), eC(_eC) {}
    };
    template<>
    struct GateArgs<GateReductionLevel::singleBlock, JobType::training> {
        TPS* __restrict__ tP;
        BookType* __restrict__ eC;
        mp_t* __restrict__ gMeC;
        mp_t* __restrict__ gML;
        __device__
        GateArgs(TPS* const& _tP,
            BookType* const& _eC,
            mp_t* const& _gMeC,
            mp_t* const& _gML,
            RingSoftmaxPayload* const&,
            RingTopKPayload* const&) :
        tP(_tP), eC(_eC), gMeC(_gMeC), gML(_gML) {}
    };
    template<>
    struct GateArgs<GateReductionLevel::multiBlock, JobType::inference> {
        TPS* __restrict__ tP;
        BookType* __restrict__ eC;
        RingSoftmaxPayload* __restrict__ bRsP;
        RingTopKPayload* __restrict__ rTp;
        __device__
        GateArgs(TPS* const& _tP,
            BookType* const& _eC,
            mp_t* const&,
            mp_t* const&,
            RingSoftmaxPayload* const& _bRsP,
            RingTopKPayload* const& _rTp) :
        tP(_tP), eC(_eC), bRsP(_bRsP), rTp(_rTp) {}
    };
    template<>
    struct GateArgs<GateReductionLevel::multiBlock, JobType::training> {
        TPS* __restrict__ tP;
        BookType* __restrict__ eC;
        mp_t* __restrict__ gMeC;
        mp_t* __restrict__ gML;
        RingSoftmaxPayload* __restrict__ bRsP;
        RingTopKPayload* __restrict__ rTp;
        __device__
        GateArgs(TPS* const& _tP,
            BookType* const& _eC,
            mp_t* const& _gMeC,
            mp_t* const& _gML,
            RingSoftmaxPayload* const& _bRsP,
            RingTopKPayload* const& _rTp) :
        tP(_tP), eC(_eC), gMeC(_gMeC), gML(_gML), bRsP(_bRsP), rTp(_rTp) {}
    };
    /// Fused GEMM, softmax, topKMask, and loss, assuming blocks >= tiles.N and no bias.
    /// Supporting the latter is trivial; the former requires a completely new algorithm
    template<
        GateReductionLevel g,
        typename BlockGEMM
    >
    struct FusedGate {
        static_assert(g == GateReductionLevel::multiBlock);
        template<
            typename MatrixA,
            typename MatrixB,
            typename MatrixC,
            typename GArg,
            typename ElementC,
            typename Element = typename MatrixA::value_type,
            unsigned int elems = ACC::STE::value,
            unsigned int sharedSize = ACC::PeakHardware::sharedMemory::value
        >
        __device__ __forceinline__
        void operator()(
            const MatrixA& activations,
            const MatrixB& weights, MatrixC const& routing,
            const unsigned int& tileIdx,
            GArg const& gArg,
            ElementC* __restrict__ gateScratch) {
            constexpr auto M = ACC::S::value;
            constexpr auto E = ACC::E::value;
            constexpr auto H = ACC::H::value;
            constexpr auto jT = ACC::JT::value;

            static_assert(cuda::std::is_same_v<ElementC, mp_t>);
            typename BlockGEMM::CollectiveMainloop mainLoop{};
            auto accumulator = cute::partition_fragment_C(typename BlockGEMM::MMA{}, typename BlockGEMM::TilerOut{});
            cute::clear(accumulator);
            constexpr auto bM = cute::get<0>(typename BlockGEMM::BlockTiler{});
            constexpr auto bN = cute::get<1>(typename BlockGEMM::BlockTiler{});
            constexpr auto bK = cute::get<2>(typename BlockGEMM::BlockTiler{});
            constexpr auto threads = BlockGEMM::Threads::value;

            constexpr auto tilesM = M / bM;
            // padded to fill bN
            constexpr auto tilesN = cute::ceil_div(E, bN);
            static_assert(ACC::PeakHardware::blocks::value >= tilesN);
            constexpr auto tilesK = H / bK;

            const auto tileCoord = idx2crd(tileIdx,
                cute::Shape<cute::Int<tilesM>, cute::Int<tilesN>>{},
                    cute::Stride<cute::Int<tilesN>, cute::_1>{});
            const auto tokenIds = make_tensor(cute::make_gmem_ptr(gArg.tP),
                cute::Layout<cute::Shape<cute::Int<ACC::E::value>, cute::Int<ACC::pEC::value>>,
                    cute::Stride<cute::Int<ACC::pEC::value>, cute::_1>>{});
            const auto ctaCoord = make_coord(cute::get<0>(tileCoord), cute::get<1>(tileCoord), cute::_);
            const auto gA = cute::local_tile(activations, typename BlockGEMM::BlockTiler{}, ctaCoord, cute::Step<cute::_1, cute::X,cute::_1>{});
            const auto gB = cute::local_tile(weights, typename BlockGEMM::BlockTiler{}, ctaCoord, cute::Step< cute::X,cute::_1,cute::_1>{});
            const auto gC = cute::local_tile(routing, typename BlockGEMM::BlockTiler{}, ctaCoord, cute::Step<cute::_1,cute::_1, cute::X>{});

            /// Pointers for flags needed in epilogue
            /// col-major indexing to facilitate coalescing
            constexpr uint16_t phases = 2U;
            const auto myTileOffset = bM * (cute::get<0>(tileCoord) + cute::get<1>(tileCoord) * tilesM) + threadIdx.x;
            const auto nextTileOffset = bM * (cute::get<0>(tileCoord) +
                (cute::get<1>(tileCoord) + 1 == tilesN ? 0 : cute::get<1>(tileCoord) + 1) * tilesM) + threadIdx.x;

            // Block Ring SoftMax pointers
            auto* __restrict__ brsMailbox = gArg.bRsP + myTileOffset;
            auto* __restrict__ brsXMailbox = gArg.bRsP + nextTileOffset;

            constexpr cutlass::NumericConverter<cute::half_t, ElementC> quantize{};
            constexpr cutlass::NumericConverter<ElementC, cute::half_t> deQuantize{};
            RingSoftmaxPayload rSp{};

            // Block Ring top k pointers
            const auto myTileOffsetP = bM * (cute::get<0>(tileCoord) + phases * cute::get<1>(tileCoord) * tilesM) +
                threadIdx.x;
            const auto nextTileOffsetP = bM * (cute::get<0>(tileCoord) +
                phases * (cute::get<1>(tileCoord) + 1 == tilesN ? 0 : cute::get<1>(tileCoord) + 1) * tilesM) +
                    threadIdx.x;
            auto* __restrict__ tkMailbox = gArg.rTp + myTileOffsetP;
            auto* __restrict__ tkXMailbox = gArg.rTp + nextTileOffsetP;
            RingTopKPayload rTp{};

            const auto k_tile_iter = cute::make_coord_iterator(tilesK);

            mainLoop(
                accumulator,
                gA,
                gB,
                accumulator,
                k_tile_iter, tilesK,
                cute::Underscore{},
                threadIdx.x,
                static_cast<char*>(static_cast<void*>(gateScratch)));
            __syncthreads();

            /// Epilogue
            static_assert(size(accumulator) % elems == 0);
            static_assert(elems % 32 == 0);
            constexpr auto trips = size(accumulator) / elems;

            // Transposed layout in shared memory to minimize bank conflicts
            constexpr auto sCLay = cute::make_layout(cute::Shape<cute::Int<bM>, cute::Int<elems>>{});
            auto sC = cute::make_tensor(cute::make_smem_ptr(CAST_TO(ElementC, gateScratch)), sCLay);
            typename BlockGEMM::MMA tiledMMA{};
            auto tCsC = tiledMMA.get_slice(threadIdx.x).partition_C(sC);
            constexpr auto gCStoreOp = cutlass::NumericConverter<typename decltype(gC)::value_type,
                                                        typename decltype(accumulator)::value_type>{};
            constexpr auto gCLoadOp = cutlass::NumericConverter<typename decltype(accumulator)::value_type,
                                                        typename decltype(gC)::value_type>{};

            // Transpose thread data to a blocked arrangement
            #pragma unroll
            for (unsigned int i = 0; i < trips; ++i) {
                #pragma unroll
                for (unsigned int j = 0; j < elems; ++j) {
                    tCsC(j) = accumulator(j + i * elems);
                }
                // Necessary to ensure THREADSxElems half-tile is ready as values are scattered across threads
                __syncthreads();

                // Prefetch to registers
                #pragma unroll
                for (unsigned int j = 0; j < elems; ++j) {
                    accumulator(j + i * elems) = sC(threadIdx.x, j);
                }
            }

            // Handle padding before softmax
            if constexpr (E % bN != 0) {
                if (cute::get<1>(tileCoord) + 1 == tilesN) {
                    #pragma unroll
                    for (uint i = E - E / bN * bN; i < bN; ++i) {
                        accumulator(i) = -cuda::std::numeric_limits<ElementC>::infinity();
                    }
                }
            }

            /// Below needed for assigning -infinity
            /// See https://stackoverflow.com/a/20016972
            static_assert(cuda::std::numeric_limits<ElementC>::is_iec559, "IEEE 754 required");
            static_assert(cuda::std::numeric_limits<ElementC>::has_infinity);
            // using notation from https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf
            auto dI = ElementC(0);
            auto mI = -cuda::std::numeric_limits<ElementC>::infinity();
            // Begin Block-Ring softmax
            constexpr uint16_t fB = 1U;
            constexpr uint16_t sB = 1U;
            if (cute::get<1>(tileCoord) > 0) {
                awaitPayload(brsMailbox, &rSp, fB);
                // We quantize dI from mp_t to half, and this yields no loss in precision.
                // We leave as an exercise to the reader to determine why this conversion is lossless.
                // Hint: N <= UINT16_MAX ≈ FP16_MAX
                dI = deQuantize(rSp.dI);
                mI = rSp.mI;
            }

            /// Reduce
            #pragma unroll
            for (uint i = 0; i < bN; ++i) {
                const auto pM = mI;
                mI = max(mI, accumulator(i));
                dI = fmaf(dI, __expf(pM - mI),__expf(accumulator(i) - mI));
            }

            if (cute::get<1>(tileCoord) + 1 < tilesN) {
                const auto sP = RingSoftmaxPayload{mI, quantize(dI), fB};
                signalPayload(brsXMailbox, &sP);
                awaitPayload(brsMailbox, &rSp, sB);
                dI = deQuantize(rSp.dI);
                mI = rSp.mI;
            }
            else {
                // Ring ends with me, let's unblock everyone else
                const auto sP = RingSoftmaxPayload{mI, quantize(dI), sB};
                #pragma unroll
                for (uint j = 0; j < tilesN - 1; ++j) {
                    signalPayload(brsXMailbox + bM * j * tilesM, &sP);
                }
            }

            #pragma unroll
            for (uint i = 0; i < bN; ++i) {
                accumulator(i) = __fdividef(__expf(accumulator(i) - mI), dI);
            }

            // Online softmax is complete
            // Begin loss computation and global token ordering construction
            if constexpr (jT == JobType::training) {
                constexpr uint S = ACC::S::value;
                constexpr auto wS = 32U; // warpSize
                ElementC cache[bN / wS]; // |cache| == 2
                using BlockReduce = cub::BlockReduce<ElementC, threads>;
                auto* __restrict__ cS = CAST_TO(typename BlockReduce::TempStorage, gateScratch);
                // Prior to reusing shared memory
                __syncthreads();
                // Reduce down columns with bespoke collective, completes in about 8.2 𝜇s
                #pragma unroll
                for (uint i = 0; i < bN; ++i) {
                    auto colAgg = BlockReduce(cS[i]).Sum(accumulator(i));
                    // thread0 only has the aggregate, which it broadcasts to all threads in its warp
                    colAgg = __shfl_sync(0xffffffff, colAgg , 0);
                    // Each thread owns bN / warpSize elements in striped arrangement.
                    // We duplicate this value layout across all warps in the block, but only use the first warp's values.
                    cache[i / wS] = threadIdx.x % wS == i % wS? colAgg : cache[i / wS];
                }
                if (threadIdx.x < wS) {
                    // Only the first warp aggregates atomically, as other warps have garbage values
                    #pragma unroll
                    for (uint i = 0; i < bN / wS; ++i) {
                        const auto eIdx = bN * cute::get<1>(tileCoord) + (threadIdx.x + i * wS);
                        atomicAdd(gArg.gML + eIdx, __fdividef(cache[i], S));
                    }
                }
            }

            // Now do online top-k mask
            // Prep shared memory view tensors
            static_assert(sharedSize >= 16 * 1024);
            using TKT = cuda::std::conditional_t<sharedSize < threads * bN, uint16_t, uint>;
            using CSL = cute::Layout<cute::Shape<cute::Int<bN>, cute::Int<threads>>>;
            const auto topK = cute::make_tensor(cute::make_smem_ptr(CAST_TO(TKT, gateScratch)), CSL{})
                (cute::_, threadIdx.x);
            cuda::std::array<uint, bN> rTopK{};
            __syncthreads();
            #pragma unroll
            for (uint i = 0; i < bN; ++i) {
                topK[i] = 0U;
            }

            auto sV = -cuda::std::numeric_limits<ElementC>::infinity();
            uint16_t sIdx = 0U;
            auto mCw = ElementC(0);
            auto lSV = sV;
            auto lSIdx = sIdx;
            bool shouldSweep = true;

            for (uint16_t i = 0; i < ACC::TK::value; ++i) {
                const uint16_t batonPrefix = phases * (i / 2U); // needed as we alternate between two buffers
                const uint16_t bPf = batonPrefix + 1U;
                const uint16_t bPs = batonPrefix + 2U;
                const auto flagPrefix = i % phases * bM * tilesM;
                // Sentinel that applies to the most westwards peer, as they initiate the proposal per round
                sV = -cuda::std::numeric_limits<ElementC>::infinity();
                if (shouldSweep) {
                    #pragma unroll
                    for(uint j = 0; j < bN; ++j) {
                        rTopK[j] = topK[j];
                    }
                    #pragma unroll
                    for (uint j = 0; j < bN; ++j) {
                        // local maximum
                        if (!rTopK[j] && accumulator(j) > lSV) {
                            lSIdx = cute::get<1>(tileCoord) * bN + j;
                            lSV = accumulator(j);
                        }
                        // proposal
                        if (!rTopK[j] && accumulator(j) > sV) {
                            sIdx = cute::get<1>(tileCoord) * bN + j;
                            sV = accumulator(j);
                        }
                    }
                    shouldSweep = false;
                }
                if (cute::get<1>(tileCoord) > 0) {
                    awaitPayload(tkMailbox + flagPrefix, &rTp, bPf);
                    sV = rTp.sV;
                    sIdx = rTp.sIdx;
                    if (lSV > sV) {
                        //we either relay the received values or propagate our proposal
                        sV = lSV;
                        sIdx = lSIdx;
                    }
                }

                // Every tile except the most eastwards
                if (cute::get<1>(tileCoord) + 1 < tilesN) {
                    // propagate our proposal
                    // Now we pass our proposal through the ring
                    const auto sP = RingTopKPayload{sV, sIdx, bPf};
                    signalPayload(tkXMailbox + flagPrefix, &sP);
                    // Now we await the results to return
                    awaitPayload(tkMailbox + flagPrefix, &rTp, bPs);
                    sV = rTp.sV;
                    sIdx = rTp.sIdx;
                }
                else {
                    // Phase 0 ends with me, let's unblock everyone else in one go
                    const auto sP = RingTopKPayload{sV, sIdx, bPs};
                    auto* __restrict__ mailboxes = tkXMailbox;
                    #pragma unroll
                    for (uint j = 0; j < tilesN - 1; ++j) {
                        signalPayload(mailboxes + flagPrefix, &sP);
                        mailboxes += phases * bM * tilesM;
                    }
                }

                if (sIdx / bN == cute::get<1>(tileCoord)) {
                    // Our proposal won in this round!
                    topK[sIdx % bN] = 1U;
                    // We need to sweep in the next round
                    shouldSweep = true;
                    lSV = -cuda::std::numeric_limits<ElementC>::infinity();
                }
                mCw += sV;
            }

            // prefetch topK to registers, one last time :)
            #pragma unroll
            for(uint j = 0; j < bN; ++j) {
                rTopK[j] = topK[j];
            }

            // Copy results to global memory
            __syncthreads();
            constexpr auto sCLayR = cute::make_layout(cute::Shape<cute::Int<bM>, cute::Int<elems>>{},
            cute::LayoutRight{});
            const auto sCR = cute::make_tensor(cute::make_smem_ptr(CAST_TO(Element, gateScratch)), sCLayR);
            static_assert(elems % WARP_SIZE == 0 && size(accumulator) % elems == 0);
            #pragma unroll
            for (uint i = 0; i < trips; ++i) {
                #pragma unroll
                for (uint j = 0; j < elems; ++j) {
                    const auto swizzleIdx = (j + threadIdx.x) % elems;
                    sCR(threadIdx.x, swizzleIdx) = gCStoreOp(accumulator(j + i * elems));
                }
                __syncthreads();
                const auto rIdx = threadIdx.x / elems * elems;
                const auto cIdx = threadIdx.x % elems;
                #pragma unroll
                for (uint j = 0; j < elems; ++j) {
                    const auto swIdx =  (j + threadIdx.x) % elems;
                    gC(rIdx + j, cIdx + i * elems) = sCR(rIdx + j, swIdx);
                }
            }

            // Prior to reusing shared memory
            __syncthreads();
            using BlockScan = cub::BlockScan<uint, threads>;
            auto* __restrict__ scanTempStorage = CAST_TO(typename BlockScan::TempStorage, gateScratch);
            auto* __restrict__ startIndices = CAST_TO(uint, scanTempStorage + bN);
            // Ensures we can safely use without any concern for overflow
            static_assert(bM <= cuda::std::numeric_limits<uint>::max());

            constexpr auto syncLimit = sharedSize / 1024;
            static_assert(sizeof(typename BlockScan::TempStorage) * syncLimit + sizeof(uint) * bN <= sharedSize);
            uint cachedSelected = 0U;
            cuda::std::array<uint, bN> myIndices{};
            // scan down the column
            #pragma unroll
            for (uint i = 0; i < bN; ++i) {
                uint selected = 0U;
                BlockScan(scanTempStorage[i % syncLimit]).InclusiveSum(rTopK[i], myIndices[i], selected);
                cachedSelected = threadIdx.x == i ? selected : cachedSelected;
                if (i > 0 && i % syncLimit == 0) {
                    __syncthreads();
                }
            }

            if (threadIdx.x < bN) {
                startIndices[threadIdx.x] = atomicAdd(gArg.eC + (bN * cute::get<1>(tileCoord) + threadIdx.x),
                    cachedSelected);
                if constexpr (jT == JobType::training) {
                    constexpr uint S = ACC::S::value;
                    atomicAdd(gArg.gMec + (bN * cute::get<1>(tileCoord) + threadIdx.x),
                        __fdividef(static_cast<ElementC>(cachedSelected),
                    static_cast<ElementC>(S)));
                }
            }
            __syncthreads();
            #pragma unroll
            for (uint i = 0; i < bN; ++i) {
                myIndices[i] = startIndices[i] + myIndices[i] - 1;
            }
            constexpr auto EC = ACC::EC::value;
            #pragma unroll
            for (uint i = 0; i < bN; ++i) {
                if (rTopK[i] && myIndices[i] < EC) {
                    const auto expertIdx = bN * cute::get<1>(tileCoord) + i;
                    tokenIds(expertIdx, myIndices[i]) = TPS{bM * cute::get<0>(tileCoord) + threadIdx.x, mCw};
                }
            }
        }
    };

    // Special, nice case where N <= BLOCK_N
    template<
        typename BlockGEMM
    >
    struct FusedGate<GateReductionLevel::singleBlock, BlockGEMM> {
        template<
            typename MatrixA,
            typename MatrixB,
            typename MatrixC,
            typename GArg,
            typename ElementC,
            typename Element = typename MatrixA::value_type,
            unsigned int elems = ACC::STE::value,
            unsigned int sharedSize = ACC::PeakHardware::sharedMemory::value
        >
        __device__ __forceinline__
        void operator()(const MatrixA& activations,
            const MatrixB& weights, MatrixC const& routing,
            const unsigned int& tileIdx,
            GArg const& gArg,
            ElementC* __restrict__ const& gateScratch) {
            // Matrix dimensions
            constexpr auto M = ACC::S::value;
            constexpr auto K = ACC::H::value;
            constexpr auto k = ACC::TK::value;
            constexpr auto jT = ACC::JT::value;
            static_assert(cuda::std::is_same_v<ElementC, float> && cuda::std::is_same_v<ElementC, mp_t>);
            typename BlockGEMM::CollectiveMainloop mainLoop{};
            auto accumulator = cute::partition_fragment_C(typename BlockGEMM::MMA{}, typename BlockGEMM::TilerOut{});
            cute::clear(accumulator);
            constexpr auto bM = cute::get<0>(typename BlockGEMM::BlockTiler{});
            constexpr auto bN = cute::get<1>(typename BlockGEMM::BlockTiler{});
            constexpr auto bK = cute::get<2>(typename BlockGEMM::BlockTiler{});
            static_assert(ACC::E::value <= bN);
            static_assert(cute::size(accumulator) == bN);
            constexpr auto threads = BlockGEMM::Threads::value;
            const auto tokenIds = make_tensor(cute::make_gmem_ptr(gArg.tP),
                cute::Layout<cute::Shape<cute::Int<ACC::E::value>, cute::Int<ACC::pEC::value>>,
                    cute::Stride<cute::Int<ACC::pEC::value>, cute::_1>>{});

            constexpr auto tilesM = M / bM;
            constexpr auto tilesN = 1U;
            constexpr auto tilesK = K / bK;

            const auto tileCoord = idx2crd(tileIdx,
                cute::Shape<cute::Int<tilesM>, cute::Int<tilesN>>{},
                    cute::Stride<cute::Int<tilesN>, cute::_1>{});
            const auto ctaCoord = make_coord(cute::get<0>(tileCoord), cute::get<1>(tileCoord), cute::_);
            const auto gA = cute::local_tile(activations, typename BlockGEMM::BlockTiler{}, ctaCoord, cute::Step<cute::_1, cute::X,cute::_1>{});
            const auto gB = cute::local_tile(weights, typename BlockGEMM::BlockTiler{}, ctaCoord, cute::Step< cute::X,cute::_1,cute::_1>{});
            const auto gC = cute::local_tile(routing, typename BlockGEMM::BlockTiler{}, ctaCoord, cute::Step<cute::_1,cute::_1, cute::X>{});

            static_assert(cuda::std::numeric_limits<ElementC>::is_iec559, "IEEE 754 required");
            static_assert(cuda::std::numeric_limits<ElementC>::has_infinity);
            auto k_tile_iter = cute::make_coord_iterator(tilesK);

            mainLoop(
                accumulator,
                gA,
                gB,
                accumulator,
                k_tile_iter, tilesK,
                cute::Underscore{},
                threadIdx.x,
                CAST_TO(char, gateScratch));
            __syncthreads();

            /// Epilogue
            static_assert(bN % elems == 0);
            constexpr auto trips = bN / elems;

            // Transposed layout in shared memory to minimize bank conflicts
            using sCLay = cute::Layout<cute::Shape<cute::Int<bM>, cute::Int<elems>>>;
            const auto sC = cute::make_tensor(cute::make_smem_ptr(CAST_TO(ElementC, gateScratch)), sCLay{});
            constexpr auto sCLayR = cute::make_layout(cute::Shape<cute::Int<bM>, cute::Int<elems>>{},
            cute::LayoutRight{});
            const auto sCR = cute::make_tensor(cute::make_smem_ptr(CAST_TO(Element, gateScratch)), sCLayR);
            static_assert(size(accumulator) == bN);
            typename BlockGEMM::MMA tiledMMA{};
            const auto tCsC = tiledMMA.get_slice(threadIdx.x).partition_C(sC);
            constexpr auto abN = cute::min(bN, ACC::E::value);
            constexpr auto gCStoreOp = cutlass::NumericConverter<typename decltype(gC)::value_type,
                                                        typename decltype(accumulator)::value_type>{};
            constexpr auto gCLoadOp = cutlass::NumericConverter<typename decltype(accumulator)::value_type,
                                                        typename decltype(gC)::value_type>{};

            // Begin softmax
            // using notation from https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf
            auto dI = ElementC(0);
            auto mI = -cuda::std::numeric_limits<ElementC>::infinity();
            // Transpose thread data to a blocked arrangement
            #pragma unroll
            for (unsigned int i = 0; i < trips; ++i) {
                #pragma unroll
                for (unsigned int j = 0; j < elems; ++j) {
                    tCsC(j) = accumulator(j + i * elems);
                }
                __syncthreads();
                #pragma unroll
                for (unsigned int j = 0; j < elems; ++j) {
                    accumulator(j + i * elems) = sC(threadIdx.x, j);
                }
            }

            /// Softmax Reduction
            #pragma unroll
            for (uint i = 0; i < abN; ++i) {
                const auto pM = mI;
                mI = max(mI, accumulator(i));
                dI = fmaf(dI, __expf(pM - mI),__expf(accumulator(i) - mI));
            }
            #pragma unroll
            for (uint i = 0; i < abN; ++i) {
                accumulator(i) = __fdividef(__expf(accumulator(i) - mI), dI);
            }

            /// Eagerly Copy results to global memory
            // before using shared memory
            __syncthreads();
            static_assert(elems % WARP_SIZE == 0 && size(accumulator) % elems == 0);
            #pragma unroll
            for (uint i = 0; i < trips; ++i) {
                #pragma unroll
                for (uint j = 0; j < elems; ++j) {
                    const auto swizzleIdx = (j + threadIdx.x) % elems;
                    sCR(threadIdx.x, swizzleIdx) = gCStoreOp(accumulator(j + i * elems));
                }
                __syncthreads();
                const auto rIdx = threadIdx.x / elems * elems;
                const auto cIdx = threadIdx.x % elems;
                #pragma unroll
                for (uint j = 0; j < elems; ++j) {
                    const auto swIdx =  (j + threadIdx.x) % elems;
                    gC(rIdx + j, cIdx + i * elems) = sCR(rIdx + j, swIdx);
                }
            }

            // Begin loss computation and global token ordering construction
            if constexpr (jT == JobType::training) {
                constexpr auto wS = 32U; // warpSize
                constexpr auto S = ACC::S::value;
                auto* __restrict__ gML = gArg.gML;
                static_assert(bN % wS == 0);
                ElementC cache[bN / wS];
                using BlockReduce = cub::BlockReduce<ElementC, threads>;
                auto* __restrict__ cS = CAST_TO(typename BlockReduce::TempStorage, gateScratch);
                // Prior to reusing shared memory
                __syncthreads();
                // Reduce down columns with bespoke collective, completes in about 8.2 𝜇s
                #pragma unroll
                for (uint i = 0; i < bN; ++i) {
                    auto colAgg = BlockReduce(cS[i]).Sum(accumulator(i));
                    // thread0 only has the aggregate, which it broadcasts to all threads in its warp
                    colAgg = __shfl_sync(0xffffffff, colAgg , 0);
                    // Each thread owns bN / warpSize elements in striped arrangement.
                    // We duplicate this value layout across all warps in the block, but only use the first warp's values.
                    cache[i / wS] = threadIdx.x % wS == i % wS? colAgg : cache[i / wS];
                }
                if (threadIdx.x < wS) {
                    // Only the first warp aggregates atomically, as other warps have garbage values
                    #pragma unroll
                    for (uint i = 0; i < bN / wS; ++i) {
                        atomicAdd(gML + (threadIdx.x + i * wS), __fdividef(cache[i], S));
                    }
                }
            }

            // sum of the combine weights per token
            auto mCw = ElementC(0);
            // Now do online top-k mask
            // Prep shared memory view tensors
            static_assert(sharedSize >= 16 * 1024);
            using TKT = cuda::std::conditional_t<sharedSize < threads * bN, uint16_t, uint>;
            using CSL = cute::Layout<cute::Shape<cute::Int<abN>, cute::Int<threads>>>;
            const auto topK = cute::make_tensor(cute::make_smem_ptr(CAST_TO(TKT, gateScratch)), CSL{})
                (cute::_, threadIdx.x);
            cutlass::AlignedArray<uint, abN> rTopK{};
            // Prior to reusing shared memory
            __syncthreads();
            #pragma unroll
            for (uint i = 0; i < abN; ++i) {
                topK[i] = 0U;
            }

            for (uint i = 0; i < k; ++i) {
                auto sV = -cuda::std::numeric_limits<ElementC>::infinity();
                uint sIdx = 0U;
                #pragma unroll
                for(uint j = 0; j < abN; ++j) {
                    rTopK[j] = topK[j];
                }
                #pragma unroll
                for (uint j = 0; j < abN; ++j) {
                    if (accumulator(j) > sV && !rTopK[j]) {
                        sIdx = j;
                        sV = accumulator(j);
                    }
                }
                topK[sIdx] = 1U;
                mCw += sV;
            }
            // prefetch topK to registers, one last time :)
            #pragma unroll
            for(uint j = 0; j < abN; ++j) {
                rTopK[j] = topK[j];
            }
            // needed for reusing shared memory
            __syncthreads();
            using BlockScan = cub::BlockScan<uint, threads>;
            auto* __restrict__ startIndices = CAST_TO(uint, gateScratch);
            auto* __restrict__ scanTempStorage = CAST_TO(typename BlockScan::TempStorage, startIndices + bN);
            static_assert(bM <= cuda::std::numeric_limits<uint>::max());

            uint cachedSelected = 0U;
            cuda::std::array<uint, abN> myIndices{};
            constexpr auto syncLimit = sharedSize / 1024;
            static_assert(sizeof(typename BlockScan::TempStorage) * syncLimit + sizeof(uint) * bN <= sharedSize);
            // scan down the column
            #pragma unroll
            for (uint i = 0; i < abN; ++i) {
                uint selected = 0U;
                BlockScan(scanTempStorage[i % syncLimit]).InclusiveSum(rTopK[i], myIndices[i], selected);
                cachedSelected = threadIdx.x == i ? selected : cachedSelected;
                if (i > 0 && i % syncLimit == 0) {
                    __syncthreads();
                }
            }

            if (threadIdx.x < abN) {
                startIndices[threadIdx.x] = atomicAdd(gArg.eC + threadIdx.x, cachedSelected);
                if constexpr (jT == JobType::training) {
                    constexpr auto S = ACC::S::value;
                    auto* __restrict__ gMeC = gArg.gMeC;
                    atomicAdd(gMeC + threadIdx.x, __fdividef(static_cast<ElementC>(cachedSelected),
                    static_cast<ElementC>(S)));
                }
            }
            __syncthreads();
            #pragma unroll
            for (uint i = 0; i < abN; ++i) {
                myIndices[i] = startIndices[i] + myIndices[i] - 1;
            }
            constexpr auto EC = ACC::EC::value;
            #pragma unroll
            for (uint i = 0; i < abN; ++i) {
                if (rTopK[i] && myIndices[i] < EC) {
                    tokenIds(i, myIndices[i]) = TPS{bM * cute::get<0>(tileCoord) + threadIdx.x, mCw};
                }
            }
        }
    };

    template<
        typename ElementC,
        typename MatrixA,
        typename MatrixB,
        typename MatrixC
    >
    __device__ __forceinline__
    void forward(const MatrixA& activations,
        const MatrixB& weights,
        MatrixC const& routing,
        ElementC* __restrict__ const& scratch){
        constexpr auto g = ACC::GRL::value;
        constexpr auto jT = ACC::JT::value;
        const auto gArg = GateArgs<g, jT> {
                bookkeeping.tP(),
                bookkeeping.eC(),
                bookkeeping.gMeC(),
                bookkeeping.gML(),
                bookkeeping.bRsP(),
                bookkeeping.rTp()
        };
        using GPUType = ACC::PeakHardware;
        // ALL SMs execute this function
        constexpr auto blocks = GPUType::blocks::value;
        static_assert(cuda::std::is_same_v<mp_t, ElementC>);
        using ElementA = typename MatrixA::value_type;
        using ElementB = typename MatrixB::value_type;
        using Operation = BlockMM<cute::identity, ElementA, ElementB, ElementC>;
        using ctaTiler = typename Operation::BlockTiler; // (BLK_M, BLK_N, BLK_K)
        constexpr auto threads = Operation::Threads::value;
        constexpr auto bM = cute::get<0>(ctaTiler{});
        constexpr auto bN = cute::get<1>(ctaTiler{});
        FusedGate<g, Operation> fusedGate{};

        constexpr auto nT = ACC::TM::value * ACC::TPX::value;
        for (unsigned int i = blockIdx.x; i < nT; i += blocks) {
            fusedGate(activations, weights, routing, i, gArg, scratch);
        }
        // Needed prior to packet dispatch
        gridBarrier();

        if constexpr (jT == JobType::training) {
            auto* __restrict__ gML = bookkeeping.gML();
            auto* __restrict__ gMeC = bookkeeping.gMeC();
            // Compute Gate loss
            auto* __restrict__ gL = bookkeeping.gL();
            for (unsigned int i = threads * blockIdx.x + threadIdx.x; i < ACC::E::value; i+= threads * blocks) {
                const auto me = gML[i];
                const auto ce = gMeC[i];
                atomicAdd(gL, __fdividef(me * ce, static_cast<mp_t>(ACC::E::value)));
            }
        }

        if constexpr (ACC::GRL::value == GateReductionLevel::multiBlock) {
            // asynchronously wipe flags clean for the next iteration
            auto* __restrict__ bRsP = bookkeeping.bRsP();
            constexpr auto rSlt = bookkeeping.rSlt();
            auto* __restrict__ rTp = bookkeeping.rTp();
            constexpr auto rTlt = bookkeeping.rTlt();
            const auto idx = threads * blockIdx.x + threadIdx.x;
            for (unsigned int i = idx; i < rSlt; i += threads * blocks) {
                bRsP[i] = RingSoftmaxPayload{};
            }
            for (unsigned int i = idx; i < rTlt; i += threads * blocks) {
                rTp[i] = RingTopKPayload{};
            }
        }
    }
}
#endif //GATE_CUH
