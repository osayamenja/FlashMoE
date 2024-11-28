//
// Created by oja7 on 11/25/24.
//

#ifndef GATE_CUH
#define GATE_CUH

#include "../engine/processor/gemm.cuh"
#include "../definition/types.cuh"

namespace aristos::gate {
    enum class TripPredication {
        complete,
        partial
    };
    /// Fused GEMM, softmax and topKMask, assuming blocks >= tiles.N and no bias.
    /// Supporting the latter is trivial; the former requires a completely new algorithm
    template<
        typename BlockGEMM,
        class FrgTensorD,
        typename MatrixA,
        typename MatrixB,
        typename MatrixC,
        unsigned int k,
        unsigned int sharedSize,
        TripPredication tP = TripPredication::complete
    >
    requires(k <= 64 && k > 0 && (sharedSize == 16 * 1024 || sharedSize == 8 * 1024))
    __device__ __forceinline__
    void fGSTkM(const MatrixA& activations, const MatrixB& weights, MatrixC& routing,
        FrgTensorD accumulator, const unsigned int& tileIdx, typename BlockGEMM::MatrixCType* gateScratch) {
        using ElementC = toCT<typename BlockGEMM::c_value_type>;
        static_assert(cuda::std::is_same_v<ElementC, float>);
        typename BlockGEMM::CollectiveMainloop mainLoop{};
        cute::clear(accumulator);
        constexpr auto bM = cute::get<0>(BlockGEMM::BlockTiler{});
        constexpr auto bN = cute::get<0>(BlockGEMM::BlockTiler{});

        // M is padded, such that the below is correct
        const auto tilesM = cute::get<0>(routing.shape()) / bM;
        // We assert the below prior to this point
        const auto tilesN = cute::get<1>(routing.shape()) / bN;

        auto tileCoord = idx2crd(tileIdx, cute::Shape(tilesM, tilesN), cute::Stride(tilesN, 1));
        auto ctaCoord = make_coord(cute::get<0>(tileCoord), cute::get<1>(tileCoord), cute::_);
        auto gA = cute::local_tile(activations, typename BlockGEMM::BlockTiler{}, ctaCoord, cute::Step<cute::_1, cute::X,cute::_1>{});
        auto gB = cute::local_tile(weights, typename BlockGEMM::BlockTiler{}, ctaCoord, cute::Step< cute::X,cute::_1,cute::_1>{});
        // TODO Transpose
        auto gC = cute::local_tile(routing, typename BlockGEMM::BlockTiler{}, ctaCoord, cute::Step<cute::_1,cute::_1, cute::X>{});

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
        cutlass::AlignedArray<ElementC, cute::max(k, 32)> rScratch{};
        /// Below needed for assigning -infinity
        /// See https://stackoverflow.com/a/20016972
        static_assert(cuda::std::numeric_limits<ElementC>::is_iec559, "IEEE 754 required");
        static_assert(cuda::std::numeric_limits<ElementC>::has_infinity);
        static_assert(rScratch.size() >= 32 || size(accumulator) % rScratch.size() == 0);

        // Do Block-Ring softmax and write results to global memory
        /// Column-major indexing
        auto* tFlag = moeConfig.getBRSFlags() + (cute::get<0>(tileCoord) * bM +
            cute::get<1>(tileCoord) * bM * tilesM);
        auto* nextFlag = moeConfig.getBRSFlags() + (cute::get<0>(tileCoord) * bM +
            (cute::get<1>(tileCoord) + 1) % tilesN * bM * tilesM);
        auto* values = moeConfig.getBRSValues() + (cute::get<0>(tileCoord) * bM +
            cute::get<1>(tileCoord) * bM * tilesM);
        auto* nextValues = moeConfig.getBRSValues() + (cute::get<0>(tileCoord) * bM +
            (cute::get<1>(tileCoord) + 1) % tilesN * bM * tilesM);
        auto residue = tilesN * BLOCK_N - moeConfig.numExperts;

        static_assert(sharedSize % (ARISTOS_BLOCK_SIZE * sizeof(ElementC) == 0));
        constexpr auto elems = sharedSize / (ARISTOS_BLOCK_SIZE * sizeof(ElementC));
        static_assert(size(accumulator) % elems == 0);
        static_assert(elems % 32 == 0);
        constexpr auto trips = size(accumulator) / elems;
        constexpr auto predicatedTrips = cuda::std::conditional_t<tP == TripPredication::complete,
        cute::Int<trips>, cute::Int<trips / 2>>::value;

        // Transposed layout in shared memory to minimize bank conflicts
        constexpr auto sCLay = cute::make_layout(cute::Shape<cute::Int<elems>, cute::Int<ARISTOS_BLOCK_SIZE>>{},
            cute::LayoutRight{});
        auto sC = cute::make_tensor(cute::make_smem_ptr(gateScratch), sCLay);
        typename BlockGEMM::MMA tiledMMA{};
        auto tCsC = tiledMMA.get_slice(threadIdx.x).partition_C(sC);
        auto unsigned int padIterator = BLOCK_N - residue;
        while (atomicLoad(tFlag) != 1U){}
        // Unpack message
        // using notation from https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf
        auto dI = *values.x;
        auto mI = *values.y;
        #pragma unroll
        for (unsigned int i = 0; i < predicatedTrips; ++i) {
            #pragma unroll
            for (unsigned int j = 0; j < elems; ++j) {
                tCsC(j) = accumulator(j + i * elems);
            }
            // Necessary to ensure 128x32 half-tile is ready as values are scattered across threads
            __syncthreads();

            // Prefetch to registers
            #pragma unroll
            for (unsigned int j = 0; j < elems; ++j) {
                rScratch[j + i * elems] = tCsC(j, threadIdx.x);
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
                dI = __fmaf_ieee_rn(dI, __expf(__fsub_rn(pM, mI)),__expf(__fsub_rn(rScratch[j], mI)));
            }
        }

        *nextValues = float2{mI, dI};
        __threadfence();
        atomicAdd(nextFlag, 1U);

        if (cute::get<1>(tileCoord) < tilesN - 1) {
            while (atomicLoad(tFlag) != 2U){}
            dI = *values.x;
            mI = *values.y;
            // notify neighbor
            *nextValues = float2{mI, dI};
            __threadfence();
            atomicAdd(nextFlag, 1U);
        }

        #pragma unroll
        for (unsigned int i = 0; i < predicatedTrips; ++i) {
            #pragma unroll
            for (unsigned int j = 0; j < elems; ++j) {
                tCsC(j) = accumulator(j + i * elems);
            }
            // Necessary to ensure 128x32 half-tile is ready as values are scattered across threads
            __syncthreads();

            // Prefetch to registers
            #pragma unroll
            for (unsigned int j = 0; j < elems; ++j) {
                rScratch[j + i * elems] = tCsC(j, threadIdx.x);
            }

            #pragma unroll
            for (unsigned int j = 0; j < elems; ++j) {
                accumulator(j + i * elems) = __fdividef(__expf(__fsub_rn(rScratch[j], mI)), dI);
            }
        }
        // Now do topKMask

    }
    template<
        unsigned int Arch,
        unsigned int blocks,
        typename MatrixA,
        typename MatrixB,
        typename MatrixC,
        typename MatrixD,
        typename ElementC = float
    >
    __device__ __forceinline__
    void forward(const MatrixA& activations,
        const MatrixB& weights,
        MatrixC& routing,
        const MatrixD& bias,
        ElementC* scratch){
        assert(__isShared(scratch));
        using ElementA = typename MatrixA::value_type;
        using ElementB = typename MatrixB::value_type;
        using Operation = BlockMM<Arch, ElementA, ElementB, ElementC>;
        using ctaTiler = typename Operation::BlockTiler; // (BLK_M, BLK_N, BLK_K)
        auto accumulator = cute::partition_fragment_C(typename Operation::MMA{}, typename Operation::TilerOut{});

        const auto nTiles = cute::ceil_div(cute::get<0>(routing.shape()), cute::get<0>(ctaTiler{})) *
            cute::ceil_div(cute::get<1>(routing.shape()), cute::get<1>(ctaTiler{}));

        for (unsigned int i = blockIdx.x; i < nTiles; i += blocks) {
            fGSTkM<Operation>(activations, weights, routing, bias, accumulator, i, scratch);
        }
    }
}
#endif //GATE_CUH
