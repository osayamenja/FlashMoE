/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Flashmoe Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 */

//
// Created by oja7 on 11/25/24.
//

#ifndef GATE_CUH
#define GATE_CUH

#include <cub/cub.cuh>
#include <cuda/std/array>

#include "tile.cuh"
#include "types.cuh"
#include "atomics.cuh"

namespace flashmoe
{
    enum class SoftMaxReductionOrder {
        // slower (maybe?) but closest to the actual result of online softmax
        fixed,
        // eliminates bank conflicts but may numerically drift
        // from online softmax results due to non-associativity of fp addition
        unordered,
    };
    enum class InsideFusedKernel {
        // If inside fused kernel, we would need to limit the resource consumption of the router
        // to minimize its influence on determining occupancy.
        yes,
        no
    };
}
namespace flashmoe::gate {
    template<int threads>
    using BlockScan = cub::BlockScan<int, threads, cub::BLOCK_SCAN_WARP_SCANS>;
    using UPPER_SHARED_MEM = cute::Int<32 * 1024>;
    using SoftType = float;
    /// Fused GEMM, softmax, topKMask, and loss, assuming blocks >= tiles.N and no bias.
    template<
        GateReductionLevel g,
        typename BlockGEMM,
        int threads,
        SoftMaxReductionOrder sro
    >
    struct GateMainloop {
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
            // Begin global token ordering construction

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
        typename TileGEMM,
        int threads, // blockDim.x
        SoftMaxReductionOrder sro
    >
    struct GateMainloop<GateReductionLevel::singleBlock, TileGEMM, threads, sro> {
        template<
            typename Element,
            typename ElementC
        >
        __device__ __forceinline__
        void operator()(void* __restrict__ const& gateScratch,
            const Element* __restrict__ const& activations,
            const Element* __restrict__ const& weights,
            const ElementC* __restrict__ const& routing,
            const unsigned int& tileIdx,
            int* __restrict__ const& _tokenIds,
            int* __restrict__ const& expertCounts,
            const int& S, const int& H, const int& E,
            const int& k,const int& expertCap) {
            constexpr TileGEMM tileMainloop{};
            auto accumulator = TileGEMM::BLAS::suggest_accumulator();
            constexpr auto bM = cute::get<0>(typename TileGEMM::TileShape{});
            constexpr auto bN = cute::get<1>(typename TileGEMM::TileShape{});
            constexpr auto bK = cute::get<2>(typename TileGEMM::TileShape{});
            const auto tokenIds = make_tensor(cute::make_gmem_ptr(_tokenIds),
                cute::make_layout(cute::make_shape(E, expertCap), cute::LayoutRight{}));
            const auto tilesM = S / bM;
            constexpr auto tilesN = 1U;
            const auto tilesK = H / bK;
            const auto tileCoord = tile::idx2Coord(tilesM, tilesN, tileIdx);
            const auto gC = tile::getC<bM, bN, TileGEMM::CArr::value>(routing, S, E,
                cute::select<0, 1>(tileCoord));
            static_assert(cuda::std::numeric_limits<ElementC>::has_infinity);
            // compute tile
            tileMainloop(gateScratch, activations, weights, accumulator, S, E, H, tileCoord);
            __syncthreads();

            /// Epilogue -> softmax + topk + routing table construction
            // Begin online softmax
            // using notation from https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf
            // do all computation in float for stability
            auto dI = static_cast<SoftType>(0.f);
            auto mI = -cuda::std::numeric_limits<SoftType>::infinity();
            // assumes row-major as we would be performing the softmax on the row
            using AccumType = decltype(accumulator)::value_type;
            auto sC = cublasdx::make_tensor(
                cute::make_smem_ptr(static_cast<AccumType*>(gateScratch)),
                cute::Layout<cute::Shape<cute::Int<bM>, cute::Int<bN>>,
                cute::Stride<cute::Int<bN>, cute::_1>>{});
            // vectorized smem layout
            using VTD = VectorTypeDescriptor<AccumType>;
            using VT = VTD::VectorType;
            constexpr int vectorWidth = VTD::VectorWidth::value;
            constexpr int vbN = bN / VTD::VectorWidth::value;
            // assumes row-major
            auto vsC = cublasdx::make_tensor(
                cute::make_smem_ptr(static_cast<VT*>(gateScratch)),
                cute::Layout<cute::Shape<cute::Int<bM>, cute::Int<vbN>>,
                cute::Stride<cute::Int<vbN>, cute::_1>>{});
            const auto c_frag = accumulator.get_results();
            constexpr Converter<SoftType, AccumType> softLoad{};
            constexpr Converter<ElementC, AccumType> softStore{};
            // rmem -> smem
            cublasdx::copy_fragment<TileGEMM::CAlign::value>(c_frag, sC, accumulator);
            __syncthreads();
            // TODO loosen this
            static_assert(threads >= bM);
            SoftType reginald[bN];
            if (threadIdx.x < bM) {
                #pragma unroll
                for (int i = 0; i < vbN; ++i) {
                    // smem -> rmem in blocked format
                    // each thread owns a row
                    const int swizzleIdx = sro == SoftMaxReductionOrder::fixed ? i : (i + threadIdx.x) % vbN;
                    const auto v = vsC(threadIdx.x, swizzleIdx);
                    // unpack
                    #pragma unroll
                    for (int j = 0; j < vectorWidth; ++j) {
                        reginald[j + i * vectorWidth] = softLoad(v[j]);
                    }
                }
                /// Softmax Reduction
                #pragma unroll
                for (uint j = 0; j < bN; ++j) {
                    const auto pM = mI;
                    mI = max(mI, reginald[j]);
                    dI = fmaf(dI, __expf(pM - mI),__expf(reginald[j] - mI));
                }
                #pragma unroll
                for (uint j = 0; j < bN; ++j) {
                    reginald[j] = __fdividef(__expf(reginald[j] - mI), dI);
                }
                #pragma unroll
                for (int i = 0; i < vbN; ++i) {
                    const int swizzleIdx = sro == SoftMaxReductionOrder::fixed ? i : (i + threadIdx.x) % vbN;
                    auto v = VT{};
                    // pack
                    // ideally the compiler would not emit any data movement instructions here
                    #pragma unroll
                    for (int j = 0; j < vectorWidth; ++j) {
                        v[j] = softStore(reginald[j + i * vectorWidth]);
                    }
                    // rmem -> smem
                    vsC(threadIdx.x, swizzleIdx) = v;
                }
            }
            __syncthreads();
            // smem -> gmem
            cublasdx::copy<TileGEMM::BLAS, TileGEMM::CAlign>(sC, gC);
            int rTK[bN];
            #pragma unroll
            for (int i = 0; i < bN; ++i) {
                rTK[i] = 0;
            }
            // sum of the combine weights per token
            auto mCw = static_cast<SoftType>(0.f);
            // Now do online top-k mask
            if (threadIdx.x < bM) {
                auto sTK = cute::make_tensor(cute::make_smem_ptr(static_cast<uint16_t*>(gateScratch),
                    cute::Layout<cute::Shape<cute::Int<bM>, cute::Int<bN>>>{}))
                (threadIdx.x, cute::_);
                #pragma unroll
                for (int i = 0; i < bN; ++i) {
                    sTK(i) = 0;
                }
                for (int i = 0; i < k; ++i) {
                    auto sV = -cuda::std::numeric_limits<ElementC>::infinity();
                    int sIdx = 0;
                    #pragma unroll
                    for(int j = 0; j < bN; ++j) {
                        rTK[j] = static_cast<int>(sTK[j]);
                    }
                    #pragma unroll
                    for (int j = 0; j < bN; ++j) {
                        if (reginald[j] > sV && !rTK[j]) {
                            sIdx = j;
                            sV = reginald[j];
                        }
                    }
                    sTK[sIdx] = 1;
                    mCw += sV;
                }
            }
            // needed for reusing shared memory
            __syncthreads();
            __shared__ int startIndices[bN];
            auto* __restrict__ scanTempStorage = CAST_TO(BlockScan<threads>::TempStorage, gateScratch);
            int cachedSelected = 0;
            int myIndices[bN];
            // scan down the column
            #pragma unroll
            for (uint i = 0; i < bN; ++i) {
                int selected = 0;
                BlockScan(scanTempStorage[i]).InclusiveSum(rTK[i], myIndices[i], selected);
                cachedSelected = threadIdx.x == i ? selected : cachedSelected;
            }
            if (threadIdx.x < bN) {
                startIndices[threadIdx.x] = atomicAdd(expertCounts + threadIdx.x, cachedSelected);
            }
            __syncthreads();
            #pragma unroll
            for (uint i = 0; i < bN; ++i) {
                myIndices[i] = startIndices[i] + myIndices[i] - 1;
            }
            #pragma unroll
            for (uint i = 0; i < bN; ++i) {
                if (rTK[i] && myIndices[i] < expertCap) {
                    tokenIds(i, myIndices[i]) = TPS{bM * cute::get<0>(tileCoord) + threadIdx.x, mCw};
                }
            }
        }
    };

    template<
        typename TileShape,
        int Arch,
        int threads,
        GateReductionLevel grl = GateReductionLevel::singleBlock,
        SoftMaxReductionOrder sro = SoftMaxReductionOrder::fixed,
        InsideFusedKernel ifk = InsideFusedKernel::yes,
        typename Element,
        typename ElementR
    >
    __device__ __forceinline__
    void forward(const Element* __restrict__ const& _activations,
        const Element* __restrict__ const& _gateWeights,
        const ElementR* __restrict__ const& _routing,
        int* __restrict__ const& tokenIds,
        int* __restrict__ const& expertCounts,
        const int& S, const int& H, const int& E, const int& k, const int& EC,
        RingSoftmaxPayload* __restrict__ const& scratch0 = nullptr, // only needed for grl == multiblock
        RingTopKPayload* __restrict__ const& scratch1 = nullptr // only needed for grl == multiblock
        ){
        // row major
        const auto activations = make_tensor(
            cute::make_gmem_ptr(_activations),
                cute::make_layout(cute::make_shape(S, H), cute::LayoutRight{}));
        const auto gateWeights = make_tensor(cute::make_gmem_ptr(_gateWeights),
            cute::make_layout(cute::make_shape(H, E), cute::LayoutRight{}));
        const auto routing = make_tensor(cute::make_gmem_ptr(_routing),
            cute::make_layout(cute::make_shape(S, E), cute::LayoutRight{}));
        const int blocks = gridDim.x;
        // assert(blocks >= E / bN)
        using MMA_C = float;
        constexpr int bM = cute::get<0>(TileShape{});
        constexpr int bN = cute::get<1>(TileShape{});
        constexpr int bK = cute::get<2>(TileShape{});
        constexpr int pipeStages = cute::get<3>(TileShape{});
        using TileGEMM = tile::CollectiveMainloop<
            bM, bN, bK, Arch, Element, MMA_C, threads, pipeStages
        >;
        constexpr auto sharedSize = cute::max(
            TileGEMM::SharedSize,
            bM * bN * sizeof(TileGEMM::AccumType),
            sizeof(BlockScan<threads>::TempStorage) * bN);
        __shared__ __align__(cute::max(TileGEMM::GeneralAlignment,
            alignof(BlockScan<threads>::TempStorage))) cuda::std::byte workspace[sharedSize];
        if constexpr (ifk == InsideFusedKernel::yes) {
            static_assert(bN * sizeof(SoftType) <= 256, "Reduce bN to reduce Router's register pressure");
            static_assert(sharedSize <= UPPER_SHARED_MEM::value, "Shared memory is too high for the Router");
        }
        GateMainloop<grl, TileGEMM, threads, sro> gateMainLoop{};
        constexpr auto nT = S / bM * (E / bN);
        for (unsigned int i = blockIdx.x; i < nT; i += blocks) {
            if constexpr (grl == GateReductionLevel::singleBlock) {
                gateMainLoop(workspace, activations, gateWeights, routing, i,
                    tokenIds, expertCounts, S, H, E, k, EC);
            }
            else {
                gateMainLoop(activations, gateWeights, routing, i, bookkeeping.tP(),
                    bookkeeping.eC());
            }

        }
    }
}
#endif //GATE_CUH
