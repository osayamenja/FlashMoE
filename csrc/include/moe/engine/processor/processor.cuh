//
// Created by osayamen on 7/13/24.
//

#ifndef ARISTOS_COMPUTE_CUH
#define ARISTOS_COMPUTE_CUH

#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/collective/collective_mma.hpp>

#define CAST_TO(T, p) static_cast<T*>(static_cast<void*>(p))
#include "mmaConfig.cuh"
namespace aristos::processor{
    template <typename Element, typename ActivationFunction>
        requires(cuda::std::is_same_v<Element, cute::half_t> ||
        cuda::std::is_same_v<Element, cute::bfloat16_t> ||
        cuda::std::is_same_v<Element, cute::tfloat32_t> ||
        cuda::std::is_same_v<Element, float> ||
        cuda::std::is_same_v<Element, cute::float_e4m3_t> ||
        cuda::std::is_same_v<Element, cute::float_e5m2_t>)
    CUTE_DEVICE
    void fusedAddActivate(Element& accumulator, const Element& term, const ActivationFunction& op) {
        accumulator = op(accumulator + term);
    }

    // conversion operators are reinterpret casts, so technically should be free at runtime
    // Below is 2.5X faster
    template<>
    CUTE_DEVICE
    void fusedAddActivate(cute::half_t& accumulator, const cute::half_t& term,
        const cutlass::epilogue::thread::ReLU<cute::half_t>& op) {
        accumulator = __hfma_relu(__half(1.0f), accumulator.to_half(), term.to_half());
    }

    // Below is 2.5X faster
    template<>
    CUTE_DEVICE
    void fusedAddActivate(cute::bfloat16_t& accumulator, const cute::bfloat16_t& term,
        const cutlass::epilogue::thread::ReLU<cute::bfloat16_t>& op) {
        accumulator = cute::bfloat16_t(__hfma_relu(__nv_bfloat16(1.0f),
            accumulator.to_nv_bfloat16(), term.to_nv_bfloat16()));
    }

    // <=96 registers: <= 5 blocks
    template<class BlockMM, typename ActivationOp = cute::identity, unsigned int sharedSize,
    typename MatrixA, typename MatrixB, typename MatrixC, typename MatrixD,
    typename MatrixAx, typename MatrixBx, typename MatrixCx, typename MatrixDx>
    requires (cute::is_tensor_v<MatrixA>
        && cute::is_tensor_v<MatrixB>
        && cute::is_tensor_v<MatrixC>
        && cute::is_tensor_v<MatrixD>
        && cute::is_tensor_v<MatrixAx>
        && cute::is_tensor_v<MatrixBx>
        && cute::is_tensor_v<MatrixCx>
        && cute::is_tensor_v<MatrixDx>
        && cuda::std::is_same_v<typename MatrixC::value_type, typename MatrixAx::value_type>
        && cuda::std::is_same_v<typename MatrixA::value_type, typename MatrixAx::value_type>
        && cuda::std::is_same_v<typename MatrixB::value_type, typename MatrixBx::value_type>
        && cublasdx::is_complete_blas<BlockMM>::value
        && cublasdx::is_supported<BlockMM, cublasdx::sm_of<BlockMM>::value>::value
        && cublasdx::sm_of<BlockMM>::value >= MIN_ARCH)
    CUTE_DEVICE
    void expert(const MatrixA mA, const MatrixB mB, MatrixC mC, const MatrixD mD,
        const MatrixAx mAx, const MatrixBx mBx, MatrixCx mCx, const MatrixDx mDx) {
        static_assert(rank(mA) == 2 && rank(mB) == 2 && rank(mC) == 2 && rank(mD) == 2);
        static_assert(rank(mAx) == 2 && rank(mBx) == 2 && rank(mCx) == 2 && rank(mDx) == 2);
        using Parameters = CollectiveMMAConfig<BlockMM>;
        constexpr auto bM = cublasdx::size_of<BlockMM>::m;
        constexpr auto bN = cublasdx::size_of<BlockMM>::n;
        constexpr auto bK = cublasdx::size_of<BlockMM>::k;

        using blockTiler = cute::Shape<cute::Int<bM>, cute::Int<bN>, cute::Int<bK>>;

        using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
            cutlass::gemm::MainloopSm80CpAsyncUnpredicated<ARISTOS_PIPELINE_STAGES>,
            blockTiler,
            typename BlockMM::a_value_type,
            cute::Underscore,
            typename BlockMM::b_value_type,
            cute::Underscore,
            typename Parameters::mma_t,
            typename Parameters::gCopyA,
            typename Parameters::sLayA,
            typename Parameters::sCopyA,
            cute::identity,
            typename Parameters::gCopyB,
            typename Parameters::sLayB,
            typename Parameters::sCopyB,
            cute::identity
        >;

        typename Parameters::mma_t tiledMMA;
        using TilerOut = cute::Shape<cute::Int<bM>, cute::Int<bN>>;
        auto accum = cute::partition_fragment_C(tiledMMA, TilerOut{});
        cute::clear(accum);

        // Get the appropriate blocks for this thread block
        // use problem shape instead, p_MNK = (cute::ceil_div(M, bM), cute::ceil_div(N, bN), K)
        auto cta_coordX = cute::idx2crd(blockIdx.x, cute::Shape(cute::ceil_div(cute::get<0>(mC.shape()), bM),
        cute::ceil_div(cute::get<1>(mC.shape()), bN)));
        auto cta_coord = cute::make_coord(cute::get<0>(cta_coordX), cute::get<1>(cta_coordX), cute::_);
        auto gA = local_tile(mA, blockTiler{}, cta_coord, cute::Step<cute::_1, cute::X,cute::_1>{});  // (BLK_M,BLK_K,k)
        auto gB = local_tile(mB, blockTiler{}, cta_coord, cute::Step< cute::X,cute::_1,cute::_1>{});  // (BLK_N,BLK_K,k)
        auto gC = local_tile(mC, blockTiler{}, cta_coord, cute::Step<cute::_1,cute::_1, cute::X>{});  // (BLK_M,BLK_N)
        auto gD = local_tile(mD, blockTiler{}, cta_coord, cute::Step<cute::_1,cute::_1, cute::X>{});  // (BLK_M,BLK_N)

        auto k_tile_iter = cute::make_coord_iterator(size<2>(gA));
        int k_tile_count = size<2>(gA);

        extern __shared__ typename BlockMM::c_value_type scratch[];
        CollectiveMainloop collective_mma;
        collective_mma(
            accum,
            gA,
            gB,
            accum,
            k_tile_iter, k_tile_count,
            cute::Underscore{},
            threadIdx.x,
            CAST_TO(char, scratch));

        // Ensure shared memory is ready for reuse
        __syncthreads();

        // Epilogue
        auto tCgC = tiledMMA.get_slice(threadIdx.x).partition_C(gC);
        // Accounts for GEMMs that accumulate in types differing from input types,
        // given that the result moonlights as the input for the succeeding GEMM.
        auto gCStoreOp = cutlass::NumericConverter<typename decltype(tCgC)::value_type, typename decltype(accum)::value_type>{};

        auto tD = tiledMMA.get_slice(threadIdx.x).partition_C(gD);

        // Assume unary operator
        ActivationOp epilogueOp{};
        constexpr auto elemsBytes = (sharedSize / THREADS);
        constexpr auto trips = (size(accum) * sizeof(BlockMM::c_value_type)) / elemsBytes;
        constexpr auto elems = elemsBytes / sizeof(BlockMM::c_value_type);
        // Instead of shared memory, we could use 32 registers per trip, for the workspace instead.
        // We would be within the budget (32 + 64 <= 128) and, as a bonus, bypass the above barrier as well.
        // However, then we would be at the mercy of the compiler,
        // who may or may not reuse previous MMA register allocations (24 to be exact),
        // thus causing spills to local memory.

        CUTE_UNROLL
        for (int i = 0; i < trips; ++i) {
            // Prefetch from global to shared memory that will be reused per trip
            // Use addressing that minimizes bank conflicts in shared memory
            CUTE_UNROLL
            for (int j = 0; j < elems; ++j) {
                scratch[threadIdx.x + j * THREADS] = tD(j + i * elems);
            }
            // Fused Bias Add and Activation Function on register fragment
            // Also fuses copy to GMEM
            CUTE_UNROLL
            for (int j = 0; j < elems; ++j) {
                tCgC(j + i * elems) = gCStoreOp(fusedAddActivate(accum(j + i * elems),
                    scratch[threadIdx.x + j * THREADS], epilogueOp));
            }
        }

        __threadfence();
        // Signal that this tile is available and proceed to the next GEMM.
        __syncthreads();

        if (threadIdx.x == 0) {
            // Signal that this tile is available
            //atomicAdd(&syncP, 1);
            // Wait until all tiles are ready.
            //while (atomicCAS(&syncP, 0U, 0U) % kChunks != 0){}
        }
        __syncthreads();

        // Clear accumulator registers in preparation
        cute::clear(accum);

        gA = local_tile(mAx, blockTiler{}, cta_coord, cute::Step<cute::_1, cute::X,cute::_1>{});  // (BLK_M,BLK_K,k)
        gB = local_tile(mBx, blockTiler{}, cta_coord, cute::Step< cute::X,cute::_1,cute::_1>{});  // (BLK_N,BLK_K,k)
        auto gCx = local_tile(mCx, blockTiler{}, cta_coord, cute::Step<cute::_1,cute::_1, cute::X>{});  // (BLK_M,BLK_N)
        gD = local_tile(mDx, blockTiler{}, cta_coord, cute::Step<cute::_1,cute::_1, cute::X>{});  // (BLK_M,BLK_N)

        auto k_tile_iterX = cute::make_coord_iterator(size<2>(gA));
        k_tile_count = size<2>(gA);

        // Execute next GEMM now
        collective_mma(
            accum,
            gA,
            gB,
            accum,
            k_tile_iterX, k_tile_count,
            cute::Underscore{},
            threadIdx.x,
            CAST_TO(char, scratch));

        __syncthreads();

        // Epilogue
        tD = tiledMMA.get_slice(threadIdx.x).partition_C(gD);
        auto tCgCx = tiledMMA.get_slice(threadIdx.x).partition_C(gCx);

        CUTE_UNROLL
        for (int i = 0; i < trips; ++i) {
            // Prefetch
            CUTE_UNROLL
            for (int j = 0; j < elems; ++j) {
                scratch[threadIdx.x + j * THREADS] = tD(j + i * elems);
            }
            // Fused Bias Add on register fragment
            CUTE_UNROLL
            for (int j = 0; j < elems; ++j) {
                tCgCx(j + i * elems) = accum(j + i * elems) + scratch[threadIdx.x + j * THREADS];
            }
        }
        __threadfence();
        __syncthreads();
    }

    CUTE_DEVICE
    void start(){
        //TODO processor scheduling plane
    }
}
#endif //ARISTOS_COMPUTE_CUH
