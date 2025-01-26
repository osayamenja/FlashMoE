//
// Created by oja7 on 11/25/24.
//

#ifndef GATE_CUH
#define GATE_CUH

#include <cub/cub.cuh>

#include "../os/processor/gemm.cuh"
#include "../types.cuh"
#include "../atomics.cuh"

namespace aristos::gate {
    struct GateArgs {
        const uint sl;
        const uint nx;
        TokenIdxTuple* tP;
        BookType* eC;
        mp_t* gMeC;
        mp_t* gML;
        RingSoftmaxPayload* bRsP;
        RingTopKPayload* rTp;
        GateArgs(uint const& _sl, uint const& _nx,
            TokenIdxTuple* const& _tP,
            BookType* const& _eC, mp_t* const& _gMeC, mp_t* const& _gML,
            RingSoftmaxPayload* const& _bRsP, RingTopKPayload* const& _rTp) :
        sl(_sl), nx(_nx), tP(_tP), eC(_eC), gMeC(_gMeC), gML(_gML), bRsP(_bRsP), rTp(_rTp) {}
    };
    /// Fused GEMM, softmax, topKMask, and loss, assuming blocks >= tiles.N and no bias.
    /// Supporting the latter is trivial; the former requires a completely new algorithm
    template<
        GateReductionLevel g = GateReductionLevel::multiBlock,
        typename BlockGEMM
    >
    struct FusedGate {
        static_assert(g == GateReductionLevel::multiBlock);
        template<
            class FrgTensorD,
            typename MatrixA,
            typename MatrixB,
            typename MatrixC,
            typename ElementC
        >
        __device__ __forceinline__
        void operator()(
            const MatrixA& activations,
            const MatrixB& weights, MatrixC& routing,
            FrgTensorD& accumulator,
            const unsigned int& tileIdx,
            GateArgs const& gArg,
            ElementC* __restrict__ gateScratch,
            const unsigned int& k) {
            static_assert(cuda::std::is_same_v<ElementC, mp_t> &&
                cuda::std::is_same_v<typename MatrixC::value_type, ElementC>);
            typename BlockGEMM::CollectiveMainloop mainLoop{};
            cute::clear(accumulator);
            constexpr auto bM = cute::get<0>(typename BlockGEMM::BlockTiler{});
            constexpr auto bN = cute::get<0>(typename BlockGEMM::BlockTiler{});
            constexpr auto threads = BlockGEMM::GEMM::block_dim.x;

            // padded to fill bM
            const auto tilesM = cute::ceil_div(cute::get<0>(routing.shape()), bM);
            // padded to fill bN
            const auto tilesN = cute::ceil_div(cute::get<1>(routing.shape()), bN);

            const auto tileCoord = idx2crd(tileIdx, cute::Shape(tilesM, tilesN), cute::Stride(tilesN, 1));
            const auto ctaCoord = make_coord(cute::get<0>(tileCoord), cute::get<1>(tileCoord), cute::_);
            const auto gA = cute::local_tile(activations, typename BlockGEMM::BlockTiler{}, ctaCoord, cute::Step<cute::_1, cute::X,cute::_1>{});
            const auto gB = cute::local_tile(weights, typename BlockGEMM::BlockTiler{}, ctaCoord, cute::Step< cute::X,cute::_1,cute::_1>{});
            const auto gC = cute::local_tile(routing, typename BlockGEMM::BlockTiler{}, ctaCoord, cute::Step<cute::_1,cute::_1, cute::X>{});

            /// Pointers for flags needed in epilogue
            /// col-major indexing to facilitate coalescing
            const auto myTileOffset = bM * (cute::get<0>(tileCoord) + cute::get<1>(tileCoord) * tilesM) + threadIdx.x;
            const auto nextTileOffset = bM * (cute::get<0>(tileCoord) +
                (cute::get<1>(tileCoord) + 1 == tilesN ? 0 : cute::get<1>(tileCoord) + 1) * tilesM) + threadIdx.x;

            // cache
            const auto nx = gArg.nx;
            auto* __restrict__ tP = gArg.tP();
            auto* __restrict__ eC = gArg.eC();
            auto* __restrict__ gMeC = gArg.gMeC();
            auto* __restrict__ gML = gArg.gML();

            // Block Ring SoftMax pointers
            auto* __restrict__ brsMailbox = gArg.bRsP() + myTileOffset;
            auto* __restrict__ brsXMailbox = gArg.bRsP() + nextTileOffset;

            cutlass::NumericConverter<cute::half_t, ElementC> quantize{};
            cutlass::NumericConverter<ElementC, cute::half_t> deQuantize{};
            RingSoftmaxPayload rSp{};

            // Block Ring top k pointers
            auto* __restrict__ tkMailbox = gArg.rTp() + myTileOffset;
            auto* __restrict__ tkXMailbox = gArg.rTp() + nextTileOffset;
            RingTopKPayload rTp{};

            auto k_tile_iter = cute::make_coord_iterator(size<2>(gA));
            int k_tile_count = size<2>(gA);

            mainLoop(
                accumulator,
                gA,
                gB,
                accumulator,
                k_tile_iter, k_tile_count,
                cute::Underscore{},
                threadIdx.x,
                static_cast<char*>(static_cast<void*>(gateScratch)));
            __syncthreads();

            /// Epilogue
            static_assert(SHARED_SIZE % (threads * sizeof(ElementC) == 0));
            constexpr auto elems = SHARED_SIZE / (threads * sizeof(ElementC));
            static_assert(size(accumulator) % elems == 0);
            static_assert(elems % 32 == 0);
            constexpr auto trips = size(accumulator) / elems;

            // Transposed layout in shared memory to minimize bank conflicts
            constexpr auto sCLay = cute::make_layout(cute::Shape<cute::Int<bM>, cute::Int<elems>>{});
            auto sC = cute::make_tensor(cute::make_smem_ptr(gateScratch), sCLay);
            typename BlockGEMM::MMA tiledMMA{};
            auto tCsC = tiledMMA.get_slice(threadIdx.x).partition_C(sC);

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
            if (cute::get<1>(tileCoord) + 1 == tilesN) {
                #pragma unroll
                for (uint i = 0; i < bN; ++i) {
                    accumulator(i) = i < nx ? accumulator(i) : -cuda::std::numeric_limits<ElementC>::infinity();
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
            if (cute::get<1>(tileCoord) > 0) {
                awaitPayload(brsMailbox, rSp, 1U);
                // We quantize dI from mp_t to half, and this yields no loss in precision.
                // We leave as an exercise to the reader to determine why this conversion is lossless.
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
                const auto sP = RingSoftmaxPayload{mI, quantize(dI), 1U};
                signal(brsXMailbox, sP);
                awaitPayload(brsMailbox, rSp, 2U);
                dI = deQuantize(rSp.dI);
                mI = rSp.mI;
            }
            else {
                // Ring ends with me, let's unblock everyone else
                auto sP = RingSoftmaxPayload{mI, quantize(dI), 2U};
                for (uint j = 0; j < tilesM; ++j) {
                    signal(brsXMailbox + bM * j * tilesM, sP);
                }
            }

            #pragma unroll
            for (uint i = 0; i < bN; ++i) {
                accumulator(i) = __fdividef(__expf(accumulator(i) - mI), dI);
            }

            // Online softmax is complete
            // Eagerly write gate logits to global memory
            #pragma unroll
            for (unsigned int j = 0; j < bN; ++j) {
                gC(threadIdx.x, j) = accumulator(j);
            }

            // Begin loss computation and global token ordering construction
            constexpr auto wS = 32U; // warpSize
            ElementC cache[bN / wS]; // |cache| == 2
            using BlockReduce = cub::BlockReduce<ElementC, threads>;
            auto* __restrict__ cS = CAST_TO(typename BlockReduce::TempStorage, gateScratch);
            const auto sl = static_cast<float>(gArg.sl);
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
                    atomicAdd(gML + (threadIdx.x + i * wS), __fdividef(cache[i], sl));
                }
            }

            // Now do online top-k mask
            // Prep shared memory view tensors
            constexpr auto tkTiler = cute::Shape<cute::_2, cute::Int<bN>>{};
            // Ensures enough bytes per thread
            static_assert(SHARED_SIZE / threads >= cute::size(tkTiler));
            static_assert(cute::get<1>(sC.shape()) * (sizeof(typename decltype(sC)::value_type) / sizeof(uint8_t))
                >= cute::size(tkTiler));
            const auto tK = cute::make_tensor(cute::make_smem_ptr(CAST_TO(uint8_t, gateScratch)),
                cute::Layout<cute::Shape<cute::Int<bM>, cute::Int<cute::size(tkTiler)>>>{});
            // Get thread-owned slices: first for topK
            const auto topK = cute::local_tile(tK, tkTiler,
                cute::crd2idx(cute::make_coord(threadIdx.x, 0), tK.layout()));
            cutlass::AlignedArray<uint8_t, bN> rTopK{};
            __syncthreads();
            #pragma unroll
            for (uint i = 0; i < bN; ++i) {
                topK[i] = 0U;
            }

            auto sV = -cuda::std::numeric_limits<ElementC>::infinity();
            uint sIdx = 0U;
            auto mCw = ElementC(0);
            auto lSV = sV;
            auto lSIdx = sIdx;
            auto bool shouldSweep = true;

            #pragma unroll
            for (uint i = 0; i < k; ++i) {
                constexpr auto phases = 2U;
                const auto batonPrefix = phases * i / 2; // needed as we alternate between two buffers
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
                        if (accumulator(j) > lSV && !rTopK[j]) {
                            lSIdx = cute::get<0>(tileCoord) * bN + j;
                            lSV = accumulator(j);
                        }
                        // proposal
                        if (accumulator(j) > sV && !rTopK[j]) {
                            sIdx = cute::get<0>(tileCoord) * bN + j;
                            sV = accumulator(j);
                        }
                    }
                    shouldSweep = false;
                }
                if (cute::get<1>(tileCoord) > 0) {
                    awaitPayload(tkMailbox + flagPrefix, rTp, batonPrefix + 1);
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
                    const auto sP = RingTopKPayload{sV, sIdx, batonPrefix + 1};
                    signal(tkXMailbox + flagPrefix, sP);
                    // Now we await the results to return
                    awaitPayload(tkMailbox + flagPrefix, rTp, batonPrefix + 2);
                    sV = rTp.sV;
                    sIdx = rTp.sIdx;
                }
                else {
                    // Phase 0 ends with me, let's unblock everyone else in one go
                    const auto sP = RingTopKPayload{sV, sIdx, batonPrefix + 2};
                    auto* __restrict__ mailboxes = tkXMailbox;
                    for (uint j = 0; j < tilesM; ++j) {
                        mailboxes += phases * j * bM * tilesM;
                        signal(mailboxes + flagPrefix, sP);
                    }
                }

                if (sIdx / bN == cute::get<1>(tileCoord)) {
                    // Our proposal won in this round!
                    topK[sIdx % bN] = 1U;
                    // We need to sweep in the next round
                    shouldSweep = true;
                }
                mCw += sV;
            }

            // prefetch topK to registers, one last time :)
            #pragma unroll
            for(uint j = 0; j < bN; ++j) {
                rTopK[j] = topK[j];
            }

            // needed for reusing shared memory
            __syncthreads();
            using BlockScan = cub::BlockScan<uint8_t, threads>;
            static_assert(sizeof(typename BlockScan::TempStorage + sizeof(uint)) * bN <= SHARED_SIZE);
            auto* __restrict__ scanTempStorage = CAST_TO(typename BlockScan::TempStorage, gateScratch);
            auto* __restrict__ startIndices = CAST_TO(uint, scanTempStorage + bN);
            // Ensures we can safely use uint8_t without any concern for overflow
            static_assert(bM <= cuda::std::numeric_limits<uint8_t>::max());

            uint8_t cachedSelected = 0U;
            cutlass::AlignedArray<uint8_t, bN> myIndices{};
            // scan down the column
            #pragma unroll
            for (uint i = 0; i < bN; ++i) {
                uint8_t selected = 0U;
                BlockScan(scanTempStorage[i]).InclusiveSum(rTopK[i], myIndices[i], selected);
                cachedSelected = threadIdx.x == i ? selected : cachedSelected;
            }

            if (threadIdx.x < bN) {
                startIndices[threadIdx.x] = atomicAdd(eC + threadIdx.x, cachedSelected);
                atomicAdd(gMeC + threadIdx.x, __fdividef(static_cast<ElementC>(cachedSelected),
                    static_cast<ElementC>(sl)));
            }
            __syncthreads();
            #pragma unroll
            for (uint i = 0; i < bN; ++i) {
                if (rTopK[i]) {
                    tP[startIndices[i] + myIndices[i] - 1] =
                        TokenIdxTuple{bM * cute::get<0>(tileCoord) + threadIdx.x, mCw};
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
            class FrgTensorD,
            typename MatrixA,
            typename MatrixB,
            typename MatrixC,
            typename ElementC
        >
        __device__ __forceinline__
        void operator()(const MatrixA& activations,
            const MatrixB& weights, MatrixC& routing,
            FrgTensorD& accumulator,
            const unsigned int& tileIdx,
            GateArgs const& gArg,
            ElementC* __restrict__ const& gateScratch,
            const unsigned int& k) {
            static_assert(cuda::std::is_same_v<ElementC, float> && cuda::std::is_same_v<ElementC, mp_t>);
            typename BlockGEMM::CollectiveMainloop mainLoop{};
            cute::clear(accumulator);
            constexpr auto bM = cute::get<0>(typename BlockGEMM::BlockTiler{});
            constexpr auto bN = cute::get<1>(typename BlockGEMM::BlockTiler{});
            static_assert(k <= bN);
            static_assert(cute::size(accumulator) == bN);
            constexpr auto threads = BlockGEMM::GEMM::block_dim.x;

            // padded to fill bM
            const auto tilesM = cute::ceil_div(cute::get<0>(routing.shape()), bM);
            // padded to fill bN
            const auto tilesN = cute::ceil_div(cute::get<1>(routing.shape()), bN);

            const auto tileCoord = idx2crd(tileIdx, cute::Shape(tilesM, tilesN), cute::Stride(tilesN, 1));
            const auto ctaCoord = make_coord(cute::get<0>(tileCoord), cute::get<1>(tileCoord), cute::_);
            const auto gA = cute::local_tile(activations, typename BlockGEMM::BlockTiler{}, ctaCoord, cute::Step<cute::_1, cute::X,cute::_1>{});
            const auto gB = cute::local_tile(weights, typename BlockGEMM::BlockTiler{}, ctaCoord, cute::Step< cute::X,cute::_1,cute::_1>{});
            const auto gC = cute::local_tile(routing, typename BlockGEMM::BlockTiler{}, ctaCoord, cute::Step<cute::_1,cute::_1, cute::X>{});

            static_assert(cuda::std::numeric_limits<ElementC>::is_iec559, "IEEE 754 required");
            static_assert(cuda::std::numeric_limits<ElementC>::has_infinity);
            auto k_tile_iter = cute::make_coord_iterator(size<2>(gA));
            int k_tile_count = size<2>(gA);

            mainLoop(
                accumulator,
                gA,
                gB,
                accumulator,
                k_tile_iter, k_tile_count,
                cute::Underscore{},
                threadIdx.x,
                CAST_TO(char, gateScratch));
            __syncthreads();

            /// Epilogue
            static_assert(SHARED_SIZE % (threads * sizeof(ElementC) == 0));
            constexpr auto elems = SHARED_SIZE / (threads * sizeof(ElementC));
            static_assert(bN % elems == 0);
            constexpr auto trips = bN / elems;

            // Transposed layout in shared memory to minimize bank conflicts
            constexpr auto sCLay = cute::make_layout(cute::Shape<cute::Int<bM>, cute::Int<elems>>{});
            const auto sC = cute::make_tensor(cute::make_smem_ptr(gateScratch), sCLay);
            typename BlockGEMM::MMA tiledMMA{};
            const auto tCsC = tiledMMA.get_slice(threadIdx.x).partition_C(sC);
            // cache
            const auto nx = gArg.nx;
            auto* __restrict__ tP = gArg.tP();
            auto* __restrict__ eC = gArg.eC();
            auto* __restrict__ gMeC = gArg.gMeC();
            auto* __restrict__ gML = gArg.gML();

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
                // Necessary to ensure THREADSxElems half-tile is ready as values are scattered across threads
                __syncthreads();

                // Prefetch to registers
                #pragma unroll
                for (unsigned int j = 0; j < elems; ++j) {
                    accumulator(j + i * elems) = sC(threadIdx.x, j);
                }
            }

            // Handle padding before softmax
            #pragma unroll
            for (uint i = 0; i < bN; ++i) {
                accumulator(i) = i < nx ? accumulator(i) : -cuda::std::numeric_limits<ElementC>::infinity();
            }

            /// Reduce
            #pragma unroll
            for (uint i = 0; i < bN; ++i) {
                const auto pM = mI;
                mI = max(mI, accumulator(i));
                dI = fmaf(dI, __expf(pM - mI),__expf(accumulator(i) - mI));
            }

            #pragma unroll
            for (uint i = 0; i < bN; ++i) {
                accumulator(i) = __fdividef(__expf(accumulator(i) - mI), dI);
            }

            // Online softmax is complete
            // Eagerly write gate logits to global memory
            #pragma unroll
            for (unsigned int j = 0; j < bN; ++j) {
                gC(threadIdx.x, j) = accumulator(j);
            }

            // Begin loss computation and global token ordering construction
            constexpr auto wS = 32U; // warpSize
            ElementC cache[bN / wS];
            using BlockReduce = cub::BlockReduce<ElementC, threads>;
            auto* __restrict__ cS = CAST_TO(typename BlockReduce::TempStorage, gateScratch);
            const auto sl = static_cast<float>(gArg.sl);
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
                    atomicAdd(gML + (threadIdx.x + i * wS), __fdividef(cache[i], sl));
                }
            }

            // sum of the combine weights per token
            auto mCw = ElementC(0);
            // Now do online top-k mask
            // Prep shared memory view tensors
            constexpr auto tkTiler = cute::Shape<cute::_2, cute::Int<bN>>{};
            // Ensures enough bytes per thread
            static_assert(SHARED_SIZE / threads >= cute::size(tkTiler));
            static_assert(cute::get<1>(sC.shape()) * (sizeof(typename decltype(sC)::value_type) / sizeof(uint8_t))
                >= cute::size(tkTiler));
            const auto tK = cute::make_tensor(cute::make_smem_ptr(CAST_TO(uint8_t, gateScratch)),
                cute::Layout<cute::Shape<cute::Int<bM>, cute::Int<cute::size(tkTiler)>>>{});

            // Get thread-owned slices: first for topK
            const auto topK = cute::local_tile(tK, tkTiler,
                cute::crd2idx(cute::make_coord(threadIdx.x, 0), tK.layout()));
            cutlass::AlignedArray<uint8_t, bN> rTopK{};
            // Prior to reusing shared memory
            __syncthreads();
            #pragma unroll
            for (uint i = 0; i < bN; ++i) {
                topK[i] = 0U;
            }
            #pragma unroll
            for (uint i = 0; i < k; ++i) {
                auto sV = -cuda::std::numeric_limits<ElementC>::infinity();
                uint sIdx = 0U;
                #pragma unroll
                for(uint j = 0; j < bN; ++j) {
                    rTopK[j] = topK[j];
                }
                #pragma unroll
                for (uint j = 0; j < bN; ++j) {
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
            for(uint j = 0; j < bN; ++j) {
                rTopK[j] = topK[j];
            }
            // needed for reusing shared memory
            __syncthreads();
            using BlockScan = cub::BlockScan<uint8_t, threads>;
            static_assert(sizeof(typename BlockScan::TempStorage + sizeof(uint)) * bN <= SHARED_SIZE);
            auto* __restrict__ scanTempStorage = CAST_TO(typename BlockScan::TempStorage, gateScratch);
            auto* __restrict__ startIndices = CAST_TO(uint, scanTempStorage + bN);
            // Ensures we can safely use uint8_t without any concern for overflow
            static_assert(bM <= cuda::std::numeric_limits<uint8_t>::max());

            uint8_t cachedSelected = 0U;
            cutlass::AlignedArray<uint8_t, bN> myIndices{};
            // scan down the column
            #pragma unroll
            for (uint i = 0; i < bN; ++i) {
                uint8_t selected = 0U;
                BlockScan(scanTempStorage[i]).InclusiveSum(rTopK[i], myIndices[i], selected);
                cachedSelected = threadIdx.x == i ? selected : cachedSelected;
            }

            if (threadIdx.x < bN) {
                startIndices[threadIdx.x] = atomicAdd(eC + threadIdx.x, cachedSelected);
                atomicAdd(gMeC() + threadIdx.x, __fdividef(static_cast<ElementC>(cachedSelected),
                    static_cast<ElementC>(sl)));
            }
            __syncthreads();
            #pragma unroll
            for (uint i = 0; i < bN; ++i) {
                if (rTopK[i]) {
                    tP()[startIndices[i] + myIndices[i] - 1] =
                        TokenIdxTuple{bM * cute::get<0>(tileCoord) + threadIdx.x, mCw};
                }
            }
        }
    };

    template<
        unsigned int Arch,
        unsigned int blocks,
        GateReductionLevel g = GateReductionLevel::singleBlock,
        typename ElementC = float,
        typename MatrixA,
        typename MatrixB,
        typename MatrixC
    >
    __device__ __forceinline__
    void forward(const MatrixA& activations,
        const MatrixB& weights,
        MatrixC& routing,
        const unsigned int& k,
        ElementC* __restrict__ scratch){
        const auto gArg = GateArgs {
                bookkeeping.sl,
                bookkeeping.nx,
                bookkeeping.tP(),
                bookkeeping.eC(),
                bookkeeping.gMeC(),
                bookkeeping.gML(),
                bookkeeping.bRsP(),
                bookkeeping.rTp()
        };
        assert(__isShared(scratch));
        static_assert(cuda::std::is_same_v<mp_t, ElementC>);
        using ElementA = typename MatrixA::value_type;
        using ElementB = typename MatrixB::value_type;
        using Operation = BlockMM<Arch, ElementA, ElementB, ElementC>;
        using ctaTiler = typename Operation::BlockTiler; // (BLK_M, BLK_N, BLK_K)
        auto accumulator = cute::partition_fragment_C(typename Operation::MMA{}, typename Operation::TilerOut{});
        constexpr auto threads = Operation::GEMM::block_dim.x;
        constexpr auto bM = cute::get<0>(ctaTiler{});
        constexpr auto bN = cute::get<1>(ctaTiler{});
        constexpr auto elems = SHARED_SIZE / (threads * sizeof(ElementC));
        FusedGate<g, Operation> fusedGate{};

        const auto nTiles = cute::ceil_div(cute::get<0>(routing.shape()), bM) *
            cute::ceil_div(cute::get<1>(routing.shape()), bN);

        for (unsigned int i = blockIdx.x; i < nTiles; i += blocks) {
            fusedGate(activations, weights, routing, accumulator, i, gArg, nTiles, scratch, k);
        }

        __threadfence();
        __syncthreads();
        // Everyone syncs here prior to packet construction
        if (!threadIdx.x) {
            bookkeeping.deviceBlockade->arrive_and_wait();
        }
        __syncthreads();

        // Compute Gate loss
        auto* __restrict__ gBK = bookkeeping.gateBk();
        auto* __restrict__ gL = bookkeeping.gL();

        for (unsigned int i = threads * blockIdx.x + threadIdx.x; i < gArg.nx; i+= threads * blocks) {
            const auto me = gArg.gML[i];
            const auto ce = gArg.gMeC[i];
            atomicAdd(gL(),__fdividef(me * ce, static_cast<mp_t>(gArg.nx)));
        }

        // wipe flags clean for next iteration
        constexpr auto vF = sizeof(uint4) / sizeof(BookType);
        const auto fE = bookkeeping.brs / sizeof(BookType);
        const auto vBRs = fE / vF;
        for (unsigned int i = threads * blockIdx.x + threadIdx.x; i < vBRs; i+= threads * blocks) {
            gBK[i] = uint4{0U, 0U, 0U, 0U};
        }

        for (unsigned int i = threads * blockIdx.x + threadIdx.x + vBRs * vF; i < fE; i += threads * blocks) {
            CAST_TO(uint, gBK)[i] = 0U;
        }
    }
}
#endif //GATE_CUH
