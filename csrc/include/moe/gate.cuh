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
    /// Fused GEMM, softmax, topKMask, and loss, assuming blocks >= tiles.N and no bias.
    /// Supporting the latter is trivial; the former requires a completely new algorithm
    template<
        GateReductionLevel g = GateReductionLevel::multiBlock,
        typename BlockGEMM,
        unsigned int k
    >
    requires(k <= 16 && k > 0)
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
            Bookkeeping const& bk,
            ElementC* __restrict__ gateScratch, ElementC* __restrict__ lossScratch) {
            static_assert(cuda::std::is_same_v<ElementC, float> &&
                cuda::std::is_same_v<ElementC, maxPrecision> &&
                cuda::std::is_same_v<typename MatrixC::value_type, ElementC>);
            typename BlockGEMM::CollectiveMainloop mainLoop{};
            cute::clear(accumulator);
            constexpr auto bM = cute::get<0>(BlockGEMM::BlockTiler{});
            constexpr auto bN = cute::get<0>(BlockGEMM::BlockTiler{});
            constexpr auto threads = BlockGEMM::GEMM::block_dim.x;

            // padded to fill bM
            const auto tilesM = cute::ceil_div(cute::get<0>(routing.shape()), bM);
            // padded to fill bN
            const auto tilesN = cute::ceil_div(cute::get<1>(routing.shape()), bN);

            auto tileCoord = idx2crd(tileIdx, cute::Shape(tilesM, tilesN), cute::Stride(tilesN, 1));
            auto ctaCoord = make_coord(cute::get<0>(tileCoord), cute::get<1>(tileCoord), cute::_);
            auto gA = cute::local_tile(activations, typename BlockGEMM::BlockTiler{}, ctaCoord, cute::Step<cute::_1, cute::X,cute::_1>{});
            auto gB = cute::local_tile(weights, typename BlockGEMM::BlockTiler{}, ctaCoord, cute::Step< cute::X,cute::_1,cute::_1>{});
            auto gC = cute::local_tile(routing, typename BlockGEMM::BlockTiler{}, ctaCoord, cute::Step<cute::_1,cute::_1, cute::X>{});

            // Preset gC as a binary mask
            #pragma unroll
            for (unsigned int i = 0; i < bN; ++i) {
                gC(i + (threadIdx.x / bN) * bN, threadIdx.x % bN) = ElementC(1);
            }

            /// Pointers for flags needed in epilogue
            /// col-major indexing
            /// TODO use cute tensor
            const auto myTileOffset = bM * (cute::get<0>(tileCoord) + cute::get<1>(tileCoord) * tilesM) + threadIdx.x;
            const auto nextTileOffset = bM * (cute::get<0>(tileCoord) +
                (cute::get<1>(tileCoord) + 1 == tilesN ? 0 : cute::get<1>(tileCoord) + 1) * tilesM) + threadIdx.x;
            auto* tFlag = bk.bRSync() + myTileOffset;
            auto* nextFlag = bk.bRSync() + nextTileOffset;
            auto* values = bk.bRSoftM() + bM * cute::get<0>(tileCoord) + threadIdx.x;

            /// Below needed for assigning -infinity
            /// See https://stackoverflow.com/a/20016972
            static_assert(cuda::std::numeric_limits<ElementC>::is_iec559, "IEEE 754 required");
            static_assert(cuda::std::numeric_limits<ElementC>::has_infinity);
            if (!cute::get<0>(tileCoord)) {
                // We are on the extreme west and need to preset our flags and values
                *tFlag = 1U;
                *values.x = ElementC(0);
                *values.y = -cuda::std::numeric_limits<ElementC>::infinity();
            }

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
            cutlass::AlignedArray<ElementC, elems> rScratch{};
            static_assert(rScratch.size() >= 32);
            const auto residue = tilesN * bN - moeConfig.numExperts;
            static_assert(size(accumulator) % elems == 0);
            static_assert(elems % 32 == 0);
            constexpr auto trips = size(accumulator) / elems;

            // Transposed layout in shared memory to minimize bank conflicts
            constexpr auto sCLay = cute::make_layout(cute::Shape<cute::Int<bM>, cute::Int<elems>>{});
            auto sC = cute::make_tensor(cute::make_smem_ptr(gateScratch), sCLay);
            typename BlockGEMM::MMA tiledMMA{};
            auto tCsC = tiledMMA.get_slice(threadIdx.x).partition_C(sC);
            auto unsigned int padIterator = bN - residue;

            // Begin Block-Ring softmax
            ring::awaitTurn(tFlag, 1U);
            // Unpack message
            // using notation from https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf
            auto dI = *values.x;
            auto mI = *values.y;
            #pragma unroll
            for (unsigned int i = 0; i < trips; ++i) {
                #pragma unroll
                for (unsigned int j = 0; j < elems; ++j) {
                    tCsC(j) = accumulator(j + i * elems);
                }
                // Necessary to ensure THREADSx32 half-tile is ready as values are scattered across threads
                __syncthreads();

                // Prefetch to registers
                #pragma unroll
                for (unsigned int j = 0; j < elems; ++j) {
                    rScratch[j] = sC(threadIdx.x, j);
                }

                // Handle padding before softmax
                if (cute::get<1>(tileCoord) == tilesN - 1) {
                    #pragma unroll
                    for (; padIterator < (i + 1) * elems; ++padIterator) {
                        rScratch[padIterator] = -cuda::std::numeric_limits<ElementC>::infinity();
                    }
                }

                /// Reduce
                #pragma unroll
                for (unsigned int j = 0; j < rScratch.size(); ++j) {
                    const auto pM = mI;
                    mI = max(mI, rScratch[j]);
                    dI = fmaf(dI, __expf(pM - mI),__expf(rScratch[j] - mI));
                }
            }

            *values = float2{mI, dI};
            ring::signal(nextFlag);

            if (cute::get<1>(tileCoord) < tilesN - 1) {
                ring::awaitTurn(tFlag, 2U);
                dI = *values.x;
                mI = *values.y;
                // Eagerly notify neighbor
                *values = float2{mI, dI};
                ring::signal(nextFlag);
            }

            #pragma unroll
            for (unsigned int i = 0; i < trips; ++i) {
                #pragma unroll
                for (unsigned int j = 0; j < elems; ++j) {
                    tCsC(j) = accumulator(j + i * elems);
                }
                // Necessary to ensure THREADSx32 half-tile is ready as values are scattered across threads
                __syncthreads();

                // Prefetch to registers
                #pragma unroll
                for (unsigned int j = 0; j < elems; ++j) {
                    rScratch[j] = sC(threadIdx.x, j);
                }

                #pragma unroll
                for (unsigned int j = 0; j < elems; ++j) {
                    accumulator(j + i * elems) = __fdividef(__expf(rScratch[j] - mI), dI);
                }
            }

            // Write softmax results to global memory prior to
            cutlass::AlignedArray<cuda::std::pair<ElementC, unsigned int>, k> heap{};
            auto* extHeap = bk.bRsH() + k * (bM * cute::get<0>(tileCoord) + threadIdx.x);
            // Wait for heap to propagate
            if (cute::get<1>(tileCoord) > 0) {
                ring::awaitTurn(tFlag, 3U);
                // read from current values
                #pragma unroll
                for (unsigned int i = 0; i < k; ++i) {
                    heap[i] = extHeap[i];
                }
            }
            else {
                #pragma unroll
                for (unsigned int i = 0; i < k; ++i) {
                    heap[i] = cuda::std::pair<ElementC, unsigned int>{accumulator(i), i};
                }
            }

            // Now do "online" topKMask
            // Build binary min heap on register memory
            cuda::std::make_heap(heap.begin(), heap.end(), cuda::std::greater{});
            cuda::std::pop_heap(heap.begin(), heap.end(), cuda::std::greater{});
            // min element now at the end of the array
            #pragma unroll
            for (unsigned int i = k; i < size(accumulator); ++i) {
                if (accumulator(i) > heap.back().first) {
                    gC(threadIdx.x, heap.back().first) = ElementC(0);
                    // Insert new element
                    heap[k - 1] = cuda::std::pair<ElementC, unsigned int>{accumulator(i), i + bN * cute::get<1>(tileCoord)};
                    cuda::std::push_heap(rScratch.begin(), rScratch.end(), cuda::std::greater{});
                    cuda::std::pop_heap(rScratch.begin(), rScratch.end(), cuda::std::greater{});
                }
                else {
                    // applies mask
                    gC(threadIdx.x, i) = ElementC(0);
                }
            }
            // propagate heap to next block, do eagerly to unblock others
            #pragma unroll
            for (int i = 0; i < k; ++i) {
                extHeap[i] = heap[i];
            }
            ring::signal(nextFlag);

            // Meanwhile, reduce mE per https://arxiv.org/pdf/2006.16668
            // 1 Transpose accumulator values using shared memory
            #pragma unroll
            for (unsigned int i = 0; i < trips; ++i) {
                auto partialSum = ElementC(0);
                #pragma unroll
                for (unsigned int j = 0; j < elems; ++j) {
                    sC(threadIdx.x, j) = accumulator(j + i * elems);
                }
                // Necessary to ensure THREADSx32 half-tile is ready
                __syncthreads();

                // 32x1 sub-tile from 128x32 tile
                #pragma unroll
                for (unsigned int j = 0; j < elems; ++j) {
                    rScratch[j] = sC(j + (threadIdx.x / elems) * elems, threadIdx.x % elems);
                }

                /// Reduce
                #pragma unroll
                for (unsigned int j = 0; j < rScratch.size(); ++j) {
                    partialSum += rScratch[j];
                }
                atomicAdd_block(lossScratch + (i * elems + threadIdx.x % elems), partialSum);
                // Below is needed to avoid data races while loading the next batch
                __syncthreads();
            }

            // Now issue additions to global loss vector: mE
            if (threadIdx.x < bN) {
                // TODO optimize to use float2 for hopper
                atomicAdd(bk.gML() + (bN * cute::get<1>(tileCoord) + threadIdx.x),
                    __fdividef(lossScratch[threadIdx.x], static_cast<float>(moeConfig.seqLen)));
                lossScratch[threadIdx.x] = ElementC(0);
            }

            // await global completion of online mask
            ring::awaitTurn(tFlag, 4U);

            __syncthreads();
            //The binary mask is ready in global memory, so we need to swap those values with the logits instead.
            #pragma unroll
            for (unsigned int i = 0; i < trips; ++i) {
                auto partialSum = ElementC(0);
                #pragma unroll
                for (unsigned int j = 0; j < elems; ++j) {
                     rScratch[j] = gC(i * elems + (j + (threadIdx.x / bN) * bN), threadIdx.x % bN);
                }
                /// Reduce
                #pragma unroll
                for (unsigned int j = 0; j < rScratch.size(); ++j) {
                    partialSum += rScratch[j];
                }
                atomicAdd_block(lossScratch + (i * elems + threadIdx.x % elems), partialSum);
            }
            if (threadIdx.x < bN) {
                atomicAdd(bk.gMeC() + (bN * cute::get<1>(tileCoord) + threadIdx.x),
                    __fdividef(lossScratch[threadIdx.x], static_cast<float>(moeConfig.seqLen)));
                atomicAdd(bk.eC() + (bN * cute::get<1>(tileCoord) + threadIdx.x),
                    static_cast<unsigned int>(lossScratch[threadIdx.x]));
                lossScratch[threadIdx.x] = ElementC(0);
            }
            // Needed to avoid overwriting gC
            __syncthreads();
            // Copy logits from registers to gC
            #pragma unroll
            for (unsigned int i = 0; i < cute::size(accumulator); ++i) {
                gC(threadIdx.x, i) = accumulator(i);
            }

            // TODO token ids and combine weights
        }
    };

    // Special, nice case where N <= BLOCK_N
    template<
        typename BlockGEMM,
        unsigned int k
    >
    struct FusedGate<GateReductionLevel::singleBlock, BlockGEMM, k> {
        template<
            class FrgTensorD,
            typename MatrixA,
            typename MatrixB,
            typename MatrixC,
            typename ElementC,
            typename BlockScanTS
        >
        __device__ __forceinline__
        void operator()(const MatrixA& activations,
            const MatrixB& weights, MatrixC& routing,
            FrgTensorD& accumulator,
            const unsigned int& tileIdx,
            Bookkeeping const& bk,
            ElementC* __restrict__ const& gateScratch,
            ElementC* __restrict__ const& scratchpad) {
            static_assert(cuda::std::is_same_v<ElementC, float> && cuda::std::is_same_v<ElementC, maxPrecision>);
            typename BlockGEMM::CollectiveMainloop mainLoop{};
            cute::clear(accumulator);
            constexpr auto bM = cute::get<0>(BlockGEMM::BlockTiler{});
            constexpr auto bN = cute::get<1>(BlockGEMM::BlockTiler{});
            static_assert(k < bN);
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
                static_cast<char*>(static_cast<void*>(gateScratch)));
            __syncthreads();

            /// Epilogue
            static_assert(SHARED_SIZE % (threads * sizeof(ElementC) == 0));
            constexpr auto elems = SHARED_SIZE / (threads * sizeof(ElementC));
            cutlass::AlignedArray<ElementC, elems> rScratch{};
            static_assert(bN % elems == 0);
            const auto residue = tilesN * bN - bk.nx;
            constexpr auto trips = size(accumulator) / elems;

            // Transposed layout in shared memory to minimize bank conflicts
            constexpr auto sCLay = cute::make_layout(cute::Shape<cute::Int<bM>, cute::Int<elems>>{});
            const auto sC = cute::make_tensor(cute::make_smem_ptr(gateScratch), sCLay);
            typename BlockGEMM::MMA tiledMMA{};
            const auto tCsC = tiledMMA.get_slice(threadIdx.x).partition_C(sC);
            auto unsigned int padIterator = bN - residue;

            // Begin softmax
            // using notation from https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf
            auto dI = ElementC(0);
            auto mI = -cuda::std::numeric_limits<ElementC>::infinity();
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
                    rScratch[j + i * elems] = sC(threadIdx.x, j);
                }

                // Handle padding before softmax
                #pragma unroll
                for (; padIterator < (i + 1) * elems; ++padIterator) {
                    rScratch[padIterator] = -cuda::std::numeric_limits<ElementC>::infinity();
                }
            }

            /// Reduce
            #pragma unroll
            for (unsigned int j = 0; j < rScratch.size(); ++j) {
                const auto pM = mI;
                mI = max(mI, rScratch[j]);
                dI = fmaf(dI, __expf(pM - mI),__expf(rScratch[j] - mI));
            }

            #pragma unroll
            for (unsigned int j = 0; j < rScratch.size(); ++j) {
                accumulator(j) = __fdividef(__expf(rScratch[j] - mI), dI);
            }

            // Write gate logits to global memory
            #pragma unroll
            for (unsigned int j = 0; j < cute::size(accumulator); ++j) {
                gC(threadIdx.x, j) = accumulator(j);
            }

            // Reduce gate logits
            static_assert(bN % elems == 0);
            #pragma unroll
            for (unsigned int i = 0; i < trips; ++i) {
                #pragma unroll
                for (unsigned int j = 0; j < elems; ++j) {
                    sC(threadIdx.x, j) = accumulator(j + i * elems);
                }
                // Necessary to ensure THREADSx32 half-tile is ready
                __syncthreads();

                // 32x1 sub-tile from 128x32 tile
                #pragma unroll
                for (unsigned int j = 0; j < elems; ++j) {
                    rScratch[j] = sC(j + threadIdx.x / elems * elems, threadIdx.x % elems);
                }

                /// Reduce
                #pragma unroll
                for (unsigned int j = 1; j < rScratch.size(); ++j) {
                    rScratch[0] += rScratch[j];
                }
                atomicAdd_block(scratchpad + (i * elems + threadIdx.x % elems), rScratch[0]);
                // Below is needed to avoid data races while loading the next batch
                __syncthreads();
            }

            if (threadIdx.x < bN) {
                atomicAdd(bk.gML() + threadIdx.x,
                    __fdividef(scratchpad[threadIdx.x], static_cast<float>(bk.sl)));
                scratchpad[threadIdx.x] = ElementC(0);
            }
            __syncthreads();

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

            // Repurpose floating-point register scratchpad as temporary byte storage
            static_assert(bN * sizeof(uint8_t) <=
                decltype(rScratch)::kElements * sizeof(typename decltype(rScratch)::value_type));
            const auto rTopK = cute::make_tensor(CAST_TO(uint8_t, rScratch.data()),
                cute::Layout<cute::Shape<cute::_1, cute::Int<bN>>,
                            cute::Stride<cute::Int<bN>, cute::_1>>{});
            auto tIdx = 0U;
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
                    // prefetch from shared to registers;
                    // this repetitive copy is a precondition
                    // for the compiler to keep rTopK in registers
                    rTopK(j) = topK[j];
                }
                #pragma unroll
                for (uint j = 0; j < bN; ++j) {
                    if (accumulator(j) > sV && !topK[j]) {
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
            auto* scanTempStorage = CAST_TO(typename BlockScan::TempStorage, gateScratch);
            auto* startIndex = CAST_TO(uint, scanTempStorage + bN);

            #pragma unroll
            for (uint i = 0; i < bN; ++i) {
                // Update the global token ordering with the block collective
                uint8_t selected = 0U;
                uint8_t myIdx;
                BlockScan(scanTempStorage[i]).InclusiveSum(rTopK[i], myIdx, selected);
                if (!threadIdx.x) {
                    // get start index for global memory writes
                    startIndex[i] = atomicAdd(bk.eC() + i, selected);
                    atomicAdd(bk.gMeC() + i, __fdividef(selected, static_cast<ElementC>(bk.sl)));
                }
                __syncthreads();
                if (rTopK[i]) {
                    bk.tP()[startIndex[i] + myIdx - 1] = TokenIdxTuple{bM * cute::get<0>(tileCoord) + threadIdx.x, mCw};
                }
            }
        }
    };

    template<
        unsigned int Arch,
        unsigned int blocks,
        unsigned int k,
        GateReductionLevel g = GateReductionLevel::singleBlock,
        typename ElementC = float,
        typename MatrixA,
        typename MatrixB,
        typename MatrixC
    >
    __device__ __forceinline__
    void forward(Bookkeeping const& bk,
        const MatrixA& activations,
        const MatrixB& weights,
        MatrixC& routing,
        ElementC* __restrict__ scratch){
        assert(__isShared(scratch));
        static_assert(cuda::std::is_same_v<maxPrecision, ElementC>);
        using ElementA = typename MatrixA::value_type;
        using ElementB = typename MatrixB::value_type;
        using Operation = BlockMM<Arch, ElementA, ElementB, ElementC>;
        using ctaTiler = typename Operation::BlockTiler; // (BLK_M, BLK_N, BLK_K)
        auto accumulator = cute::partition_fragment_C(typename Operation::MMA{}, typename Operation::TilerOut{});
        constexpr auto threads = Operation::GEMM::block_dim.x;
        constexpr auto bM = cute::get<0>(ctaTiler{});
        constexpr auto bN = cute::get<1>(ctaTiler{});
        constexpr auto elems = SHARED_SIZE / (threads * sizeof(ElementC));
        __shared__ __align__(16) ElementC scratchpad[bN];
        #pragma unroll
        for (uint i = threadIdx.x; i < bN; i += threads) {
            scratchpad[i] = ElementC(0);
        }
        FusedGate<g, Operation, k> fusedGate{};

        const auto nTiles = cute::ceil_div(cute::get<0>(routing.shape()), bM) *
            cute::ceil_div(cute::get<1>(routing.shape()), bN);

        for (unsigned int i = blockIdx.x; i < nTiles; i += blocks) {
            fusedGate(activations, weights, routing, accumulator, i, bk, nTiles, scratch, scratchpad);
        }

        __threadfence();
        __syncthreads();
        // Everyone syncs here prior to packet construction
        if (!threadIdx.x) {
            bk.deviceBlockade->arrive_and_wait();
        }
        __syncthreads();

        // Compute Gate loss
        for (unsigned int i = threads * blockIdx.x + threadIdx.x; i < moeConfig.numExperts; i+= threads * blocks) {
            const auto me = bk.gML()[i];
            const auto ce = bk.gMeC()[i];
            atomicAdd(bk.gL(),__fdividef(me * ce, static_cast<maxPrecision>(moeConfig.numExperts)));
        }
    }
}
#endif //GATE_CUH
