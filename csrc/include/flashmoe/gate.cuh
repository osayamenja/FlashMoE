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

#include "infra/atomics.cuh"
#include "infra/packed.cuh"
#include "tile.cuh"
#include "infra/vt.cuh"

namespace flashmoe
{
    enum class SoftMaxOptimizationLevel {
        none,
        // use fast exponential and float division
        highest,
    };

    enum class GateReductionLevel {
        singleBlock,
        multiBlock
    };
}
namespace flashmoe::gate {
    template<SoftMaxOptimizationLevel level = SoftMaxOptimizationLevel::none>
    __device__ __forceinline__
    float fexp(const float& x) {
        static_assert(level == SoftMaxOptimizationLevel::none);
        return expf(x);
    }
    template<>
    __device__ __forceinline__
    float fexp<SoftMaxOptimizationLevel::highest>(const float& x) {
        return __expf(x);
    }

    template<SoftMaxOptimizationLevel level = SoftMaxOptimizationLevel::none>
    __device__ __forceinline__
    float fdiv(const float& x, const float& y) {
        static_assert(level == SoftMaxOptimizationLevel::none);
        return fdividef(x, y);
    }
    template<>
    __device__ __forceinline__
    float fdiv<SoftMaxOptimizationLevel::highest>(const float& x, const float& y) {
        return __fdividef(x, y);
    }
    template<int threads>
    using BlockScan = cub::BlockScan<int, threads, cub::BLOCK_SCAN_WARP_SCANS>;
    using SoftType = float;
    /// Fused GEMM, softmax, topKMask, assuming blocks >= tiles.N
    template<
        GateReductionLevel g,
        typename TileGEMM,
        SoftMaxOptimizationLevel sro
    >
    struct GateMainloop {
        static_assert(g == GateReductionLevel::multiBlock);
        template<
            typename Element,
            typename ElementC
        >
        __device__ __forceinline__
        void operator()(void* __restrict__ const& gateScratch,
            const Element* __restrict__ const& tokens,
            const Element* __restrict__ const& weights,
            ElementC* __restrict__ routing, const int& tileIdx,
            TPS* __restrict__ const& _tokenIds, int* __restrict__ const& expertCounts,
            int* __restrict__ const& eCGuards,
            const int& S, const int& H, const int& E, const int& k, const int& expertCap, const int& roundEC,
            SoftmaxStatePacked* __restrict__ const& _rSp,
            RingTopKPayload* __restrict__ const& _rTp) {
            constexpr int threads = TileGEMM::Threads::value;
            constexpr TileGEMM tileMainloop{};
            auto accumulator = TileGEMM::BLAS::suggest_accumulator();
            constexpr auto bM = cute::get<0>(typename TileGEMM::TileShape{});
            constexpr auto bN = cute::get<1>(typename TileGEMM::TileShape{});
            constexpr auto bK = cute::get<2>(typename TileGEMM::TileShape{});
            const auto tokenIds = make_tensor(cute::make_gmem_ptr(_tokenIds),
                cute::make_layout(cute::make_shape(E, roundEC), cute::LayoutRight{}));
            // assert(S % bM == 0)
            const auto tilesM = S / bM;
            // assert(E % bN == 0)
            const auto tilesN = E / bN;
            // assert(H % bK == 0)
            const auto tilesK = H / bK;
            const auto tileCoord = tile::idx2Coord(tilesM, tilesN, tileIdx);
            static_assert(cuda::std::numeric_limits<SoftType>::has_infinity);
            // compute tile
            tileMainloop(gateScratch, tokens, weights, accumulator, S, E, H, tileCoord);
            __syncthreads();

            /// Pointers for flags needed in epilogue
            /// col-major indexing to facilitate coalescing
            constexpr int phases = 2;
            const auto myTileOffset = bM * (cute::get<0>(tileCoord) + cute::get<1>(tileCoord) * tilesM) + threadIdx.x;
            const auto nextTileOffset = bM * (cute::get<0>(tileCoord) +
                (cute::get<1>(tileCoord) + 1 == tilesN ? 0 : cute::get<1>(tileCoord) + 1) * tilesM) + threadIdx.x;

            // Block Ring SoftMax pointers
            auto* __restrict__ brsMailbox = _rSp + myTileOffset;
            auto* __restrict__ brsXMailbox = _rSp + nextTileOffset;

            // Block Ring top k pointers
            const auto myTileOffsetP = bM * (cute::get<0>(tileCoord) + phases * cute::get<1>(tileCoord) * tilesM) +
                threadIdx.x;
            const auto nextTileOffsetP = bM * (cute::get<0>(tileCoord) +
                phases * (cute::get<1>(tileCoord) + 1 == tilesN ? 0 : cute::get<1>(tileCoord) + 1) * tilesM) +
                    threadIdx.x;
            auto* __restrict__ tkMailbox = _rTp + myTileOffsetP;
            auto* __restrict__ tkXMailbox = _rTp + nextTileOffsetP;

            /// Epilogue -> RingSoftmax + topk + routing table construction
            // assumes row-major as we would be performing the softmax on the row
            using AccumType = decltype(accumulator)::value_type;
            auto sC = cublasdx::make_tensor(
                cute::make_smem_ptr(static_cast<AccumType*>(gateScratch)),
                cute::Layout<cute::Shape<cute::Int<bM>, cute::Int<bN>>,
                cute::Stride<cute::Int<bN>, cute::_1>>{});
            // vectorized smem layout
            using VTD = VectorTypeDescriptor<AccumType, ElementAlignment<AccumType, bN>>;
            using VT = VTD::VectorType;
            constexpr int vectorWidth = VTD::VectorWidth::value;
            static_assert(bN % vectorWidth == 0);
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
            // TODO relax this
            static_assert(threads >= bM);
            SoftType reginald[bN];
            // using notation from https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf
            auto dI = static_cast<SoftType>(0.f);
            /// Below needed for assigning -infinity
            /// See https://stackoverflow.com/a/20016972
            static_assert(cuda::std::numeric_limits<SoftType>::is_iec559, "IEEE 754 required");
            static_assert(cuda::std::numeric_limits<SoftType>::has_infinity);
            auto mI = -cuda::std::numeric_limits<SoftType>::infinity();
            // Begin Ring softmax
            if (threads == bM || threadIdx.x < bM) {
                #pragma unroll
                for (int i = 0; i < vbN; ++i) {
                    // smem -> rmem in blocked format
                    // each thread owns a row
                    const auto v = vsC(threadIdx.x, i);
                    // unpack
                    #pragma unroll
                    for (int j = 0; j < vectorWidth; ++j) {
                        reginald[j + i * vectorWidth] = softLoad(v[j]);
                    }
                }
                if (cute::get<1>(tileCoord) > 0) {
                    // await payload
                    auto payload = cuda::std::bit_cast<SoftmaxStatePacked>(
                        atomicExch(reinterpret_cast<ull_t*>(brsMailbox), 0UL));
                    while (!has_payload_arrived(payload)) {
                        payload = cuda::std::bit_cast<SoftmaxStatePacked>(
                            atomicExch(reinterpret_cast<ull_t*>(brsMailbox), 0UL));
                    }
                    unpack_state(payload, mI, dI);
                }
                /// Softmax Reduction
                #pragma unroll
                for (int j = 0; j < bN; ++j) {
                    const auto pM = mI;
                    mI = max(mI, reginald[j]);
                    dI = fmaf(dI, fexp<sro>(pM - mI),fexp<sro>(reginald[j] - mI));
                }
                if (cute::get<1>(tileCoord) + 1 < tilesN) {
                    const auto sP = pack_state(mI, dI);
                    // signal payload
                    atomicExch(reinterpret_cast<ull_t*>(brsXMailbox), cuda::std::bit_cast<ull_t>(sP));
                    // await payload
                    auto* __restrict__ receiveBox = reinterpret_cast<ull_t*>(brsMailbox);
                    auto payload = cuda::std::bit_cast<SoftmaxStatePacked>(atomicExch(receiveBox, 0UL));
                    while (!has_payload_arrived(payload)) {
                        payload = cuda::std::bit_cast<SoftmaxStatePacked>(atomicExch(receiveBox, 0UL));
                    }
                    unpack_state(payload, mI, dI);
                }
                else {
                    // Ring ends with me, let's unblock everyone else
                    const auto sP = pack_state(mI, dI);
                    for (int j = 0; j < tilesN - 1; ++j) {
                        // signal payload
                        atomicExch(reinterpret_cast<ull_t*>(brsXMailbox + bM * j * tilesM),
                            cuda::std::bit_cast<ull_t>(sP));
                    }
                }
                #pragma unroll
                for (int j = 0; j < bN; ++j) {
                    reginald[j] = fdiv<sro>(fexp<sro>(reginald[j] - mI), dI);
                }
                #pragma unroll
                for (int i = 0; i < vbN; ++i) {
                    auto v = VT{};
                    // pack
                    // ideally the compiler would not emit any MOV instructions here
                    #pragma unroll
                    for (int j = 0; j < vectorWidth; ++j) {
                        v[j] = softStore(reginald[j + i * vectorWidth]);
                    }
                    // rmem -> smem
                    vsC(threadIdx.x, i) = v;
                }
            }
            __syncthreads();
            // Ring Softmax is complete
            // smem -> gmem
            auto gC = tile::getC<bM, bN, TileGEMM::CArr::value>(routing, S, E,
                cute::select<0, 1>(tileCoord));
            const auto tsC = cublasdx::make_tensor(
                static_cast<ElementC*>(gateScratch), TileGEMM::BLAS::get_layout_smem_c());
            static_assert(cute::is_compatible<decltype(gC.layout()), decltype(tsC.layout())>::value);
            cublasdx::copy<TileGEMM::BLAS, ElementAlignment<ElementC, bN>>(tsC, gC);
            int rTK[bN];
            #pragma unroll
            for (int i = 0; i < bN; ++i) {
                rTK[i] = 0;
            }
            // sum of the combine weights per token
            auto mCw = static_cast<SoftType>(0.f);
            auto sV = -cuda::std::numeric_limits<SoftType>::infinity();
            uint32_t sIdx = 0;
            auto lSV = sV;
            auto lSIdx = sIdx;
            // Now do online top-k mask
            if (threads == bM || threadIdx.x < bM) {
                bool shouldSweep = true;
                for (int i = 0; i < k; ++i) {
                    // needed as we alternate between two buffers
                    const auto flagPrefix = i % phases * bM * tilesM;
                    if (shouldSweep) {
                        #pragma unroll
                        for (int j = 0; j < bN; ++j) {
                            // find local maximum
                            if (!rTK[j] && reginald[j] > lSV) {
                                lSIdx = cute::get<1>(tileCoord) * bN + j;
                                lSV = reginald[j];
                            }
                        }
                        shouldSweep = false;
                    }
                    // Strictly applies to the most westwards peer, as they initiate the proposal per round
                    sV = lSV;
                    sIdx = lSIdx;
                    if (cute::get<1>(tileCoord) > 0) {
                        // await payload
                        auto* __restrict__ receiveBox = reinterpret_cast<ull_t*>(tkMailbox + flagPrefix);
                        auto payload = cuda::std::bit_cast<RingTopKPayload>(atomicExch(receiveBox, 0UL));
                        while (!has_payload_arrived(payload)) {
                            payload = cuda::std::bit_cast<RingTopKPayload>(atomicExch(receiveBox, 0UL));
                        }
                        unpack_tk_payload(payload, sV, sIdx);
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
                        atomicExch(reinterpret_cast<ull_t*>(tkXMailbox + flagPrefix),
                            cuda::std::bit_cast<ull_t>(pack_tk_payload(sV, sIdx)));
                        // Now we await the final results to return
                        auto* __restrict__ receiveBox = reinterpret_cast<ull_t*>(tkMailbox + flagPrefix);
                        auto payload = cuda::std::bit_cast<RingTopKPayload>(atomicExch(receiveBox, 0UL));
                        while (!has_payload_arrived(payload)) {
                            payload = cuda::std::bit_cast<RingTopKPayload>(atomicExch(receiveBox, 0UL));
                        }
                        unpack_tk_payload(payload, sV, sIdx);
                    }
                    else {
                        // Phase 0 ends with me, let's unblock everyone else in one go
                        const auto payload = cuda::std::bit_cast<ull_t>(pack_tk_payload(sV, sIdx));
                        auto* __restrict__ mailboxes = tkXMailbox;
                        for (int j = 0; j < tilesN - 1; ++j) {
                            atomicExch(reinterpret_cast<ull_t*>(mailboxes + flagPrefix), payload);
                            mailboxes += phases * bM * tilesM;
                        }
                    }
                    if (sIdx / bN == cute::get<1>(tileCoord)) {
                        // Our proposal won in this round!
                        // let's update the mask
                        const int sIdxIntra = sIdx % bN;
                        #pragma unroll
                        for (int j = 0; j < bN; ++j) {
                            if (j == sIdxIntra) {
                                rTK[j] = 1;
                            }
                        }
                        // We need to sweep in the next round
                        shouldSweep = true;
                        lSV = -cuda::std::numeric_limits<SoftType>::infinity();
                    }
                    mCw += sV;
                }
            }
            // finalize routing table construction
            __shared__ int startIndices[bN];
            using BTS = BlockScan<threads>::TempStorage;
            __shared__ __align__(alignof(BTS)) BTS scanTempStorage[bN];
            int myIndices[bN];
            // scan down the column
            constexpr int sl = cute::ceil_div(bN, threads); // typically 1
            int stash[sl];
            #pragma unroll
            for (int i = 0; i < sl; ++i) {
                stash[i] = 0;
            }
            #pragma unroll
            for (int i = 0; i < bN; ++i) {
                int selected = 0;
                BlockScan<threads>(scanTempStorage[i]).InclusiveSum(rTK[i], myIndices[i], selected);
                stash[i / threads] = threadIdx.x == (i % threads) ? selected : stash[i / threads];
            }
            #pragma unroll
            for (int i = 0; i < sl; ++i) {
                const int idx = threadIdx.x + i * threads;
                if (idx < bN) {
                    const int expertIdx = bN * cute::get<1>(tileCoord) + idx;
                    startIndices[idx] = guardedAtomicAdd(eCGuards + expertIdx,
                    expertCounts + expertIdx, stash[i], tilesM);
                }
            }
            __syncthreads();
            #pragma unroll
            for (int i = 0; i < bN; ++i) {
                myIndices[i] = startIndices[i] + myIndices[i] - 1;
            }
            #pragma unroll
            for (int i = 0; i < bN; ++i) {
                if (rTK[i] && myIndices[i] < expertCap) {
                    const auto expertIdx = bN * cute::get<1>(tileCoord) + i;
                    tokenIds(expertIdx, myIndices[i]) = TPS{
                        bM * cute::get<0>(tileCoord) + threadIdx.x,
                        fdividef(reginald[i], mCw)};
                }
            }
        }
    };

    // Special, nice case where N == BLOCK_N
    template<
        typename TileGEMM,
        SoftMaxOptimizationLevel sro
    >
    struct GateMainloop<GateReductionLevel::singleBlock, TileGEMM, sro> {
        template<
            typename Element,
            typename ElementC
        >
        __device__ __forceinline__
        void operator()(void* __restrict__ const& gateScratch,
            const Element* __restrict__ const& tokens,
            const Element* __restrict__ const& weights,
            ElementC* __restrict__ const& routing,
            const int& tileIdx,
            TPS* __restrict__ const& _tokenIds,
            int* __restrict__ const& expertCounts,
            int* __restrict__ const& eCGuards,
            const int& S, const int& H, const int& E,
            const int& k,const int& expertCap, const int& roundEC) {
            constexpr int threads = TileGEMM::Threads::value;
            constexpr TileGEMM tileMainloop{};
            auto accumulator = TileGEMM::BLAS::suggest_accumulator();
            constexpr auto bM = cute::get<0>(typename TileGEMM::TileShape{});
            constexpr auto bN = cute::get<1>(typename TileGEMM::TileShape{});
            constexpr auto bK = cute::get<2>(typename TileGEMM::TileShape{});
            const auto tokenIds = make_tensor(cute::make_gmem_ptr(_tokenIds),
                cute::make_layout(cute::make_shape(E, roundEC), cute::LayoutRight{}));
            const auto tilesM = S / bM;
            constexpr int tilesN = 1;
            const auto tilesK = H / bK;
            const auto tileCoord = tile::idx2Coord(tilesM, tilesN, tileIdx);
            static_assert(cuda::std::numeric_limits<ElementC>::has_infinity);
            // compute tile
            tileMainloop(gateScratch, tokens, weights, accumulator, S, E, H, tileCoord);
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
            using VTD = VectorTypeDescriptor<AccumType, ElementAlignment<AccumType, bN>>;
            using VT = VTD::VectorType;
            constexpr int vectorWidth = VTD::VectorWidth::value;
            static_assert(bN % vectorWidth == 0);
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
            // TODO relax this
            static_assert(threads >= bM);
            SoftType reginald[bN];
            if (threads == bM || threadIdx.x < bM) {
                #pragma unroll
                for (int i = 0; i < vbN; ++i) {
                    // smem -> rmem in blocked format
                    // each thread owns a row
                    const auto v = vsC(threadIdx.x, i);
                    // unpack
                    #pragma unroll
                    for (int j = 0; j < vectorWidth; ++j) {
                        reginald[j + i * vectorWidth] = softLoad(v[j]);
                    }
                }
                /// Softmax Reduction
                #pragma unroll
                for (int j = 0; j < bN; ++j) {
                    const auto pM = mI;
                    mI = max(mI, reginald[j]);
                    dI = fmaf(dI, fexp<sro>(pM - mI),fexp<sro>(reginald[j] - mI));
                }
                #pragma unroll
                for (int j = 0; j < bN; ++j) {
                    reginald[j] = fdiv<sro>(fexp<sro>(reginald[j] - mI), dI);
                }
                #pragma unroll
                for (int i = 0; i < vbN; ++i) {
                    auto v = VT{};
                    // pack
                    // ideally the compiler would not emit any MOV instructions here
                    #pragma unroll
                    for (int j = 0; j < vectorWidth; ++j) {
                        v[j] = softStore(reginald[j + i * vectorWidth]);
                    }
                    // rmem -> smem
                    vsC(threadIdx.x, i) = v;
                }
            }
            __syncthreads();
            // smem -> gmem
            auto gC = tile::getC<bM, bN, TileGEMM::CArr::value>(routing, S, E,
                cute::select<0, 1>(tileCoord));
            const auto tsC = cublasdx::make_tensor(
                static_cast<ElementC*>(gateScratch), TileGEMM::BLAS::get_layout_smem_c());
            static_assert(cute::is_compatible<decltype(gC.layout()), decltype(tsC.layout())>::value);
            cublasdx::copy<TileGEMM::BLAS, ElementAlignment<ElementC, bN>>(tsC, gC);
            int rTK[bN];
            #pragma unroll
            for (int i = 0; i < bN; ++i) {
                rTK[i] = 0;
            }
            // sum of the combine weights per token
            auto mCw = static_cast<SoftType>(0.f);
            // Now do online top-k mask
            if (threads == bM || threadIdx.x < bM) {
                for (int i = 0; i < k; ++i) {
                    auto sV = -cuda::std::numeric_limits<SoftType>::infinity();
                    int sIdx = 0;
                    #pragma unroll
                    for (int j = 0; j < bN; ++j) {
                        if (reginald[j] > sV && !rTK[j]) {
                            sIdx = j;
                            sV = reginald[j];
                        }
                    }
                    #pragma unroll
                    for(int j = 0; j < bN; ++j) {
                        if (j == sIdx) {
                            rTK[j] = 1; // set the bit
                        }
                    }
                    mCw += sV;
                }
            }
            __shared__ int startIndices[bN];
            using BTS = BlockScan<threads>::TempStorage; // :)
            __shared__ __align__(alignof(BTS)) BTS scanTempStorage[bN];
            int myIndices[bN];
            // scan down the column
            constexpr int sl = cute::ceil_div(bN, threads); // typically 1
            int stash[sl];
            #pragma unroll
            for (int i = 0; i < sl; ++i) {
                stash[i] = 0;
            }
            #pragma unroll
            for (int i = 0; i < bN; ++i) {
                int selected = 0;
                BlockScan<threads>(scanTempStorage[i]).InclusiveSum(rTK[i], myIndices[i], selected);
                stash[i / threads] = threadIdx.x == (i % threads) ? selected : stash[i / threads];
            }
            #pragma unroll
            for (int i = 0; i < sl; ++i) {
                const int expertIdx = threadIdx.x + i * threads;
                if (expertIdx < bN) {
                    startIndices[expertIdx] = guardedAtomicAdd(eCGuards + expertIdx,
                    expertCounts + expertIdx, stash[i], tilesM);
                }
            }
            __syncthreads();
            #pragma unroll
            for (int i = 0; i < bN; ++i) {
                myIndices[i] = startIndices[i] + myIndices[i] - 1;
            }
            #pragma unroll
            for (int i = 0; i < bN; ++i) {
                if (rTK[i] && myIndices[i] < expertCap) {
                    tokenIds(i, myIndices[i]) = TPS{
                        bM * cute::get<0>(tileCoord) + threadIdx.x,
                        fdividef(reginald[i], mCw)};
                }
            }
        }
    };

    template<
        typename TileGEMM,
        GateReductionLevel grl = GateReductionLevel::singleBlock,
        SoftMaxOptimizationLevel sro = SoftMaxOptimizationLevel::none,
        typename Element,
        typename ElementR
    >
    __device__ __forceinline__
    void forward(void* __restrict__ const& workspace,
        const Element* __restrict__ const& tokens,
        const Element* __restrict__ const& _gateWeights,
        ElementR* __restrict__ const& _routing,
        TPS* __restrict__ const& tokenIds,
        int* __restrict__ const& expertCounts,
        int* __restrict__ eCGuards,
        const int& S, const int& H, const int& E, const int& k, const int& EC, const int& roundEC,
        const int& blocks,
        SoftmaxStatePacked* __restrict__ const& rSp = nullptr, // only needed for grl == multiblock
        RingTopKPayload* __restrict__ const& rTp = nullptr // only needed for grl == multiblock
        ){
        // assert(blocks >= E / bN)
        using TileShape = TileGEMM::TileShape;
        constexpr int bM = cute::get<0>(TileShape{});
        constexpr int bN = cute::get<1>(TileShape{});

        GateMainloop<grl, TileGEMM, sro> gateMainLoop{};
        const auto nT = S / bM * (E / bN);
        for (int i = static_cast<int>(blockIdx.x); i < nT; i += blocks) {
            if constexpr (grl == GateReductionLevel::singleBlock) {
                gateMainLoop(workspace, tokens, _gateWeights, _routing, i,
                    tokenIds, expertCounts, eCGuards, S, H, E, k, EC, roundEC);
            }
            else {
                gateMainLoop(workspace, tokens, _gateWeights, _routing, i,
                    tokenIds, expertCounts, eCGuards, S, H, E, k, EC, roundEC, rSp, rTp);
            }
        }
    }
}
#endif //GATE_CUH
