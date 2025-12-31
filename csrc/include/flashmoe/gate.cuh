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

#include "tile.cuh"
#include "atomics.cuh"

namespace flashmoe
{
    enum class SoftMaxOptimizationLevel {
        // slower (maybe?) but closest to the actual result of online softmax
        none,
        // eliminates bank conflicts but introduces permuted fp addition
        // use fast exponential and float division
        highest,
    };
}
namespace flashmoe::gate {
    enum class InsideFusedKernel {
        // If inside fused kernel, we would need to limit the resource consumption of the router
        // to minimize its influence on occupancy.
        yes,
        no
    };
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
    using UPPER_SHARED_MEM = cute::Int<32 * 1024>;
    using SoftType = float;
    /// Fused GEMM, softmax, topKMask, and loss, assuming blocks >= tiles.N and no bias.
    template<
        GateReductionLevel g,
        typename TileGEMM,
        int threads,
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
            const int& S, const int& H, const int& E, const int& k, const int& expertCap,
            SoftmaxStatePacked* __restrict__ const& _rSp,
            RingTopKPayload* __restrict__ const& _rTp) {
            constexpr TileGEMM tileMainloop{};
            auto accumulator = TileGEMM::BLAS::suggest_accumulator();
            constexpr auto bM = cute::get<0>(typename TileGEMM::TileShape{});
            constexpr auto bN = cute::get<1>(typename TileGEMM::TileShape{});
            constexpr auto bK = cute::get<2>(typename TileGEMM::TileShape{});
            const auto tokenIds = make_tensor(cute::make_gmem_ptr(_tokenIds),
                cute::make_layout(cute::make_shape(E, expertCap), cute::LayoutRight{}));
            // assert(S % bM == 0)
            const auto tilesM = S / bM;
            // assert(E % bN == 0)
            const auto tilesN = E / bN;
            // assert(H % bK == 0)
            const auto tilesK = H / bK;
            const auto tileCoord = tile::idx2Coord(tilesM, tilesN, tileIdx);
            const auto gC = tile::getC<bM, bN, TileGEMM::CArr::value>(routing, S, E,
                cute::select<0, 1>(tileCoord));
            static_assert(cuda::std::numeric_limits<ElementC>::has_infinity);
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
            RingTopKPayload rTp{};

            /// Epilogue -> RingSoftmax + topk + routing table construction
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
                    const int swizzleIdx = sro == SoftMaxOptimizationLevel::none ? i : (i + threadIdx.x) % vbN;
                    const auto v = vsC(threadIdx.x, swizzleIdx);
                    // unpack
                    #pragma unroll
                    for (int j = 0; j < vectorWidth; ++j) {
                        reginald[j + i * vectorWidth] = softLoad(v[j]);
                    }
                }
                if (cute::get<1>(tileCoord) > 0) {
                    // await payload
                    auto payload = reinterpret_cast<SoftmaxStatePacked>(
                        atomicExch(reinterpret_cast<ull_t*>(brsMailbox), 0L));
                    while (!has_payload_arrived(payload)) {
                        payload = reinterpret_cast<SoftmaxStatePacked>(
                            atomicExch(reinterpret_cast<ull_t*>(brsMailbox), 0L));
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
                    signalPayload(brsXMailbox, &sP);
                    // await payload
                    auto payload = reinterpret_cast<SoftmaxStatePacked>(
                        atomicExch(reinterpret_cast<ull_t*>(brsMailbox), 0L));
                    while (!has_payload_arrived(payload)) {
                        payload = reinterpret_cast<SoftmaxStatePacked>(
                            atomicExch(reinterpret_cast<ull_t*>(brsMailbox), 0L));
                    }
                    unpack_state(payload, mI, dI);
                }
                else {
                    // Ring ends with me, let's unblock everyone else
                    const auto sP = pack_state(mI, dI);
                    for (int j = 0; j < tilesN - 1; ++j) {
                        signalPayload(brsXMailbox + bM * j * tilesM, &sP);
                    }
                }
                #pragma unroll
                for (int j = 0; j < bN; ++j) {
                    reginald[j] = fdiv<sro>(fexp<sro>(reginald[j] - mI), dI);
                }
                #pragma unroll
                for (int i = 0; i < vbN; ++i) {
                    const int swizzleIdx = sro == SoftMaxOptimizationLevel::none ? i : (i + threadIdx.x) % vbN;
                    auto v = VT{};
                    // pack
                    // ideally the compiler would not emit any MOV instructions here
                    #pragma unroll
                    for (int j = 0; j < vectorWidth; ++j) {
                        v[j] = softStore(reginald[j + i * vectorWidth]);
                    }
                    // rmem -> smem
                    vsC(threadIdx.x, swizzleIdx) = v;
                }
            }
            __syncthreads();
            // Ring Softmax is complete
            // smem -> gmem
            cublasdx::copy<TileGEMM::BLAS, TileGEMM::CAlign>(sC, gC);
            int rTK[bN];
            #pragma unroll
            for (int i = 0; i < bN; ++i) {
                rTK[i] = 0;
            }
            // sum of the combine weights per token
            auto mCw = static_cast<SoftType>(0.f);
            auto sV = -cuda::std::numeric_limits<SoftType>::infinity();
            int sIdx = 0;
            auto lSV = sV;
            auto lSIdx = sIdx;
            // Now do online top-k mask
            if (threads == bM || threadIdx.x < bM) {
                bool shouldSweep = true;
                for (int i = 0; i < k; ++i) {
                    const int batonPrefix = phases * (i / 2); // needed as we alternate between two buffers
                    const int bPf = batonPrefix + 1;
                    const int bPs = batonPrefix + 2;
                    const auto flagPrefix = i % phases * bM * tilesM;
                    // Sentinel that applies to the most westwards peer, as they initiate the proposal per round
                    sV = -cuda::std::numeric_limits<SoftType>::infinity();
                    if (shouldSweep) {
                        #pragma unroll
                        for (int j = 0; j < bN; ++j) {
                            // local maximum
                            if (!rTK[j] && reginald[j] > lSV) {
                                lSIdx = cute::get<1>(tileCoord) * bN + j;
                                lSV = reginald[j];
                            }
                            // proposal
                            if (!rTK[j] && reginald[j] > sV) {
                                sIdx = cute::get<1>(tileCoord) * bN + j;
                                sV = reginald[j];
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
                        const auto sP = RingTopKPayload{sV,
                            static_cast<uint16_t>(sIdx),
                            static_cast<uint16_t>(bPf)};
                        signalPayload(tkXMailbox + flagPrefix, &sP);
                        // Now we await the results to return
                        awaitPayload(tkMailbox + flagPrefix, &rTp, bPs);
                        sV = rTp.sV;
                        sIdx = rTp.sIdx;
                    }
                    else {
                        // Phase 0 ends with me, let's unblock everyone else in one go
                        const auto sP = RingTopKPayload{sV,
                            static_cast<uint16_t>(sIdx),
                            static_cast<uint16_t>(bPs)};
                        auto* __restrict__ mailboxes = tkXMailbox;
                        for (int j = 0; j < tilesN - 1; ++j) {
                            signalPayload(mailboxes + flagPrefix, &sP);
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
                        lSV = -cuda::std::numeric_limits<ElementC>::infinity();
                    }
                    mCw += sV;
                }
            }
            // finalize routing table construction
            __shared__ int startIndices[bN];
            using BTS = BlockScan<threads>::TempStorage;
            __shared__ __align__(alignof(BTS)) BTS scanTempStorage[bN];
            int cachedSelected = 0;
            int myIndices[bN];
            // scan down the column
            #pragma unroll
            for (int i = 0; i < bN; ++i) {
                int selected = 0;
                BlockScan(scanTempStorage[i]).InclusiveSum(rTK[i], myIndices[i], selected);
                cachedSelected = threadIdx.x == i ? selected : cachedSelected;
            }
            if (threadIdx.x < bN) {
                startIndices[threadIdx.x] = atomicAdd(expertCounts +
                    (bN * cute::get<1>(tileCoord) + threadIdx.x), cachedSelected);
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
                    tokenIds(expertIdx, myIndices[i]) = TPS{bM * cute::get<0>(tileCoord) + threadIdx.x, mCw};
                }
            }
        }
    };

    // Special, nice case where N == BLOCK_N
    template<
        typename TileGEMM,
        int threads, // blockDim.x
        SoftMaxOptimizationLevel sro
    >
    struct GateMainloop<GateReductionLevel::singleBlock, TileGEMM, threads, sro> {
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
            if (threads == bM || threadIdx.x < bM) {
                #pragma unroll
                for (int i = 0; i < vbN; ++i) {
                    // smem -> rmem in blocked format
                    // each thread owns a row
                    const int swizzleIdx = sro == SoftMaxOptimizationLevel::none ? i : (i + threadIdx.x) % vbN;
                    const auto v = vsC(threadIdx.x, swizzleIdx);
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
                    const int swizzleIdx = sro == SoftMaxOptimizationLevel::none ? i : (i + threadIdx.x) % vbN;
                    auto v = VT{};
                    // pack
                    // ideally the compiler would not emit any MOV instructions here
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
            int cachedSelected = 0;
            int myIndices[bN];
            // scan down the column
            #pragma unroll
            for (int i = 0; i < bN; ++i) {
                int selected = 0;
                BlockScan(scanTempStorage[i]).InclusiveSum(rTK[i], myIndices[i], selected);
                cachedSelected = threadIdx.x == i ? selected : cachedSelected;
            }
            if (threadIdx.x < bN) {
                startIndices[threadIdx.x] = atomicAdd(expertCounts + threadIdx.x, cachedSelected);
            }
            __syncthreads();
            #pragma unroll
            for (int i = 0; i < bN; ++i) {
                myIndices[i] = startIndices[i] + myIndices[i] - 1;
            }
            #pragma unroll
            for (int i = 0; i < bN; ++i) {
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
        SoftMaxOptimizationLevel sro = SoftMaxOptimizationLevel::none,
        InsideFusedKernel ifk = InsideFusedKernel::yes,
        typename MMA_C = float,
        typename Element,
        typename ElementR
    >
    __device__ __forceinline__
    void forward(const Element* __restrict__ const& tokens,
        const Element* __restrict__ const& _gateWeights,
        ElementR* __restrict__ const& _routing,
        TPS* __restrict__ const& tokenIds,
        int* __restrict__ const& expertCounts,
        const int& S, const int& H, const int& E, const int& k, const int& EC,
        SoftmaxStatePacked* __restrict__ const& rSp = nullptr, // only needed for grl == multiblock
        RingTopKPayload* __restrict__ const& rTp = nullptr // only needed for grl == multiblock
        ){
        const int blocks = gridDim.x;
        // assert(blocks >= E / bN)
        constexpr int bM = cute::get<0>(TileShape{});
        constexpr int bN = cute::get<1>(TileShape{});
        constexpr int bK = cute::get<2>(TileShape{});
        constexpr int pipeStages = cute::get<3>(TileShape{});
        using TileGEMM = tile::CollectiveMainloop<
            bM, bN, bK, Arch, Element, MMA_C, threads, pipeStages
        >;
        constexpr auto sharedSize = cute::max(TileGEMM::SharedSize::value, bM * bN * sizeof(TileGEMM::AccumType));
        __shared__ __align__(TileGEMM::GeneralAlignment::value) cuda::std::byte workspace[sharedSize];
        if constexpr (ifk == InsideFusedKernel::yes) {
            static_assert(bN * sizeof(SoftType) <= 256, "Reduce bN to reduce Router's register pressure");
            static_assert(sharedSize <= UPPER_SHARED_MEM::value, "Shared memory is too high for the Router");
        }
        GateMainloop<grl, TileGEMM, threads, sro> gateMainLoop{};
        const auto nT = S / bM * (E / bN);
        for (int i = blockIdx.x; i < nT; i += blocks) {
            if constexpr (grl == GateReductionLevel::singleBlock) {
                gateMainLoop(workspace, tokens, _gateWeights, _routing, i,
                    tokenIds, expertCounts, S, H, E, k, EC);
            }
            else {
                gateMainLoop(workspace, tokens, _gateWeights, _routing, i,
                    tokenIds, expertCounts, S, H, E, k, EC, rSp, rTp);
            }
            if constexpr (ifk == InsideFusedKernel::yes) {
                cublasdx::copy_wait();
            }
        }
    }
}
#endif //GATE_CUH
