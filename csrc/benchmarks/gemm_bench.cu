//
// Created by osayamen on 12/13/25.
//
#include <matx.h>
#include <cutlass/epilogue/thread/activation.h>

#include "../include/flashmoe/debug.cuh"
#include "../include/flashmoe/os/processor/gemm.cuh"
#include "../include/flashmoe/os/processor/mmaConfig.cuh"

#define RTOL 1e-3
#define ATOL 1e-4
template<typename BlockGEMM, int threads, typename MMA_C, typename Element, typename ElementC>
__device__ __forceinline__
void gemmMainloopV1(void* __restrict__ const& workspace,
    const Element* __restrict__ const& a,
    const Element* __restrict__ const& b,
    ElementC* __restrict__ const& c,
    const ElementC* __restrict__ const& bias,
    const int& M, const int& N, const int& K, const int& tileIdx) {
    constexpr auto bM = cute::get<0>(typename BlockGEMM::BlockTiler{});
    constexpr auto bN = cute::get<1>(typename BlockGEMM::BlockTiler{});
    constexpr auto bK = cute::get<2>(typename BlockGEMM::BlockTiler{});
    auto accumulator = cute::partition_fragment_C(typename BlockGEMM::MMA{}, typename BlockGEMM::TilerOut{});
    static_assert(cute::size(accumulator) == bN);
    // Instantiate mainloop
    typename BlockGEMM::CollectiveMainloop mainLoop{};
    cute::clear(accumulator);
    // Row-major
    const auto mA = cute::make_tensor(cute::make_gmem_ptr(a),
        make_layout(cute::make_shape(M, K), cute::LayoutRight{}));
    // Row-major, transposed
    const auto mB = make_tensor(cute::make_gmem_ptr(b),
        make_layout(cute::make_shape(N, K), cute::LayoutRight{}));
    // Row-major
    const auto mC = cute::make_tensor(cute::make_gmem_ptr(c),
            make_layout(cute::make_shape(M, N), cute::LayoutRight{}));
    const int tilesM = M / bM;
    const int tilesN = N / bN;
    const int tilesK = K / bK;
    const auto tileCoord = idx2crd(tileIdx, cute::Shape(tilesM, tilesN), cute::Stride(tilesN, 1));
    const auto ctaCoord = make_coord(cute::get<0>(tileCoord), cute::get<1>(tileCoord), cute::_);
    const auto gA = cute::local_tile(mA, typename BlockGEMM::BlockTiler{}, ctaCoord,
        cute::Step<cute::_1, cute::X,cute::_1>{});
    const auto gB = cute::local_tile(mB, typename BlockGEMM::BlockTiler{}, ctaCoord,
        cute::Step< cute::X,cute::_1,cute::_1>{});
    const auto gC = cute::local_tile(mC, typename BlockGEMM::BlockTiler{}, ctaCoord,
        cute::Step<cute::_1,cute::_1, cute::X>{});
    const auto k_tile_iter = cute::make_coord_iterator(tilesK);
    // prefetch bias
    constexpr auto trips = cute::ceil_div(bN, threads);
    ElementC biasCache[trips];
    const int biasOffset = tileIdx % tilesN;
    const auto* __restrict__ bP = bias + biasOffset;
    if constexpr (threads >= bN) {
        biasCache[0] = bP[threadIdx.x % bN];
    }
    else {
        // below is not strictly necessary, but it makes my life easier :)
        static_assert(bN % threads == 0);
        #pragma unroll
        for (int i = 0; i < trips; ++i) {
            biasCache[i] = bP[threadIdx.x + i * bN];
        }
    }
    mainLoop(
        accumulator,
        gA,
        gB,
        accumulator,
        k_tile_iter, tilesK,
        cute::Underscore{},
        threadIdx.x,
        CAST_TO(char, workspace));
    __syncthreads();

    typename BlockGEMM::MMA tiledMMA{};
    constexpr auto gCStoreOp = cutlass::NumericConverter<ElementC, MMA_C>{};
    // Assume elementwise operator
    typename BlockGEMM::FusedEpilogue epilogueOp{};
    constexpr auto elems = bN;
    constexpr auto sCLay = cute::make_layout(cute::Shape<cute::Int<bM>, cute::Int<elems>>{},
        cute::LayoutRight{});
    const auto sC = cute::make_tensor(cute::make_smem_ptr(CAST_TO(MMA_C, workspace)), sCLay);
    const auto rC = cute::make_tensor(cute::make_rmem_ptr(CAST_TO(ElementC, accumulator.data())),
        cute::Layout<cute::Shape<cute::_1, cute::Int<bN>>,
            cute::Stride<cute::Int<bN>, cute::_1>>{});
    const auto tCsC = tiledMMA.get_slice(threadIdx.x).partition_C(sC);
    #pragma unroll
    for (unsigned int j = 0; j < elems; ++j) {
        tCsC(j) = accumulator(j);
    }
    __syncthreads();
    const auto rIdx = threadIdx.x / elems * elems;
    const auto cIdx = threadIdx.x % elems;
    #pragma unroll
    for (unsigned int j = 0; j < elems; ++j) {
        accumulator(j) = sC(rIdx + j, cIdx);
    }
    // apply epilogue -> Activation(fragment_c + bias)
    #pragma unroll
    for (uint i = 0; i < trips; ++i) {
        #pragma unroll
        for (int j = 0; j < elems; ++j) {
            rC(j + i * elems) = gCStoreOp(epilogueOp(accumulator(j + i * elems), biasCache[i]));
        }
    }
    // Coalesced copy from registers to global memory
    #pragma unroll
    for (uint i = 0; i < trips; ++i) {
    #pragma unroll
        for (unsigned int j = 0; j < elems; ++j) {
            gC(rIdx + j, cIdx + i * elems) = rC(j + i * elems);
        }
    }
}

enum class GemmVersion {
    V1,
    V2
};

#define SC(T, v) static_cast<T>(v)

template<GemmVersion v, int bM, int bN, int bK, int pipeStages, int threads,
typename ActivationFunction, typename MMA_C,
typename Element, typename ElementC>
__global__ void gk(const Element* __restrict__ a, const Element* __restrict__ b,
    ElementC* __restrict__ c, const ElementC* __restrict__ bias,
    const __grid_constant__ int M, const __grid_constant__ int N, const int __grid_constant__ K) {
    static_assert(threads == bM);
    static_assert(bM % bN == 0);
    const int nTiles = (M / bM) * (N / bN);
    constexpr auto sharedSize = cute::max(bK * pipeStages * (bM + bN) * sizeof(ElementC),
        bM * bN * sizeof(MMA_C));
    __shared__ __align__(16) cuda::std::byte workspace[sharedSize];
        using V1G = flashmoe::BlockMM<ActivationFunction, Element, Element, MMA_C, bM, bN, bK, pipeStages, threads>;
    for (int tileIdx = SC(int, blockIdx.x); tileIdx < nTiles; tileIdx += SC(int, gridDim.x)) {
        if constexpr (v == GemmVersion::V1) {
            gemmMainloopV1<V1G, threads, MMA_C>(workspace, a, b, c, bias, M, N, K, tileIdx);
        }
        else {

        }
    }
}

template<typename Element>
using MXE = cuda::std::conditional_t<cuda::std::is_same_v<Element, cute::half_t>, matx::matxFp16,
        cuda::std::conditional_t<cuda::std::is_same_v<Element, cute::bfloat16_t>, matx::matxBf16, Element>>;

template<int runs, typename Activation, typename Element, typename ElementC>
__host__ __forceinline__
auto reference(void* const& a, void* const& b,
    void* const& bias, void* const& c_ref, void* const& c_ext,
    const int& M, const int& N, const int& K, matx::cudaExecutor& exec) {
    auto* mx_a = static_cast<Element*>(a);
    auto* mx_b = static_cast<Element*>(b);
    auto* mx_bias = static_cast<ElementC*>(bias);
    auto* mx_c_ref = static_cast<ElementC*>(c_ref);
    auto* mx_c_ext = static_cast<ElementC*>(c_ext);

    auto tA = matx::make_tensor<Element>(mx_a, {M, K});
    auto tB = matx::make_tensor<Element>(mx_b, {N, K});
    auto tC = matx::make_tensor<ElementC>(mx_c_ref, {M, N});
    auto tBias = matx::make_tensor<ElementC>(mx_bias, {N});
    // ReLU((a @ b) + bias)
    (tC = matx::apply(Activation{}, (matx::matmul(tA, tB.PermuteMatrix()) + tBias))).run(exec);
    auto result = matx::make_tensor<int>({});
    exec.sync();
    auto tCx = matx::make_tensor<ElementC>(mx_c_ext, {M, N});
    matx::allclose(result, tCx, tC, RTOL, ATOL, exec);
    exec.sync();
    // warmup
    for (int i = 0; i < 32; ++i) {
        (tC = matx::apply(Activation{}, (matx::matmul(tA, tB) + tBias))).run(exec);
    }
    exec.sync();
    exec.start_timer();
    for (int i = 0; i < runs; ++i) {
        (tC = matx::apply(Activation{}, (matx::matmul(tA, tB) + tBias))).run(exec);
    }
    exec.stop_timer();
    exec.sync();
    FLASHMOE_CHECK_CUDA(cudaPeekAtLastError());
    return std::make_tuple(result(), exec.get_time_ms() / static_cast<float>(runs));
}

template<int bM, int bN, int bK, int pipeStages, int threads, typename MMA_C, typename Element, typename ElementC>
__host__ __forceinline__
auto gk_test(Element* const& a, Element* const& b,
    ElementC* const& c, ElementC* const& c_ref, ElementC* const& bias,
    const int& M, const int& N, const int& K, matx::cudaExecutor& exec) {
    //using Act = cutlass::epilogue::thread::ReLU<ElementC>;
    using Act = cutlass::epilogue::thread::Identity<MMA_C>;
    using ActM = cutlass::epilogue::thread::Identity<MXE<ElementC>>;
    constexpr auto runs = 64;
    int bps = 0;
    auto kernel = gk<GemmVersion::V1, bM, bN, bK, pipeStages, threads, Act, MMA_C, Element, ElementC>;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kernel, threads, 0);
    const int blocks = min((M / bM) * (N / bN), bps * NUM_SMS);
    kernel<<<blocks, threads, 0, exec.getStream()>>>(a, b, c, bias, M, N, K);
    const auto [isCorrect, ref_time_ms] = reference<runs, ActM, MXE<Element>, MXE<ElementC>>(
        a, b, bias, c_ref, c, M, N, K, exec);
    // warmup
    for (int i = 0; i < 32; ++i) {
        kernel<<<blocks, threads, 0, exec.getStream()>>>(a, b, c, bias, M, N, K);
    }
    exec.sync();
    exec.start_timer();
    for (int i = 0; i < runs; ++i) {
        kernel<<<blocks, threads, 0, exec.getStream()>>>(a, b, c, bias, M, N, K);
    }
    exec.stop_timer();
    exec.sync();
    FLASHMOE_CHECK_CUDA(cudaPeekAtLastError());
    const auto k_time_ms = exec.get_time_ms() / static_cast<float>(runs);
    printf("%d, %d, %d, %s, %f, %f\n", M, N, K, isCorrect ? "Yes" : "No", k_time_ms, ref_time_ms);
}

template<typename T>
struct ConvFunctor {
    __device__ auto operator()(const float& x) const {
        return static_cast<T>(x);
    }
};
template<>
struct ConvFunctor<matx::matxFp16> {
    __device__ auto operator()(const float& x) const {
        return  __float2half(x);
    }
};
template<>
struct ConvFunctor<matx::matxBf16> {
    __device__ auto operator()(const float& x) const {
        return  __float2bfloat16(x);
    }
};

template<int bM, int bN, int bK, int pipeStages, int threads, typename MMA_C, typename Element, typename ElementC>
__host__ __forceinline__
void test_driver(const int& M, const int& N, const int& K, matx::cudaExecutor& exec) {
    Element* a = nullptr;
    Element* b = nullptr;
    ElementC* c = nullptr;
    ElementC* c_ref = nullptr;
    ElementC* bias = nullptr;
    auto stream = exec.getStream();
    cudaMallocAsync(&a, M * K * sizeof(Element), stream);
    cudaMallocAsync(&b, N * K * sizeof(Element), stream);
    cudaMallocAsync(&c, M * N * sizeof(ElementC), stream);
    cudaMallocAsync(&c_ref, M * N * sizeof(ElementC), stream);
    cudaMallocAsync(&bias, N * sizeof(ElementC), stream);

    using MX = MXE<Element>;
    using MXC = MXE<ElementC>;
    auto tA = matx::make_tensor<MX>(reinterpret_cast<MX*>(a), {M, K});
    (tA = matx::apply(ConvFunctor<MX>{}, matx::random<float>(tA.Shape(), matx::NORMAL))).run(exec);
    //(tA = matx::ones<ElementC>(tA.Shape())).run(exec);
    auto tB = matx::make_tensor<MX>(reinterpret_cast<MX*>(b), {N, K});
    (tB = matx::apply(ConvFunctor<MX>{}, matx::random<float>(tB.Shape(), matx::NORMAL))).run(exec);
    //(tB = matx::ones<ElementC>(tB.Shape())).run(exec);
    auto tBias = matx::make_tensor<MXC>(reinterpret_cast<MXC*>(bias), {N});
    (tBias = matx::ones<MXC>(tBias.Shape())).run(exec);
    //(tBias = matx::apply(ConvFunctor<MXC>{}, matx::random<float>(tBias.Shape(), matx::UNIFORM))).run(exec);
    gk_test<bM, bN, bK, pipeStages, threads, MMA_C, Element, ElementC>(a, b, c, c_ref, bias, M, N, K, exec);

    FLASHMOE_CHECK_CUDA(cudaPeekAtLastError());
    cudaFreeAsync(a, stream);
    cudaFreeAsync(b, stream);
    cudaFreeAsync(c, stream);
    cudaFreeAsync(c_ref, stream);
    cudaFreeAsync(bias, stream);
    cudaStreamSynchronize(stream);
}

__host__ __forceinline__
void test_parser(const int argc, char** argv) {
    int MNK = 128;
    using Element = float;
    using ElementC = float;
    using MMA_C = float;
    printf("M, N, K, Correct?, Kernel_Time(ms), Matx_Time(ms)\n");
    constexpr auto bK = 8 * (4 / sizeof(ElementC));
    if (argc > 1) {
        MNK = std::stoi(argv[1]);
    }
    cudaSetDevice(0);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    matx::cudaExecutor exec{stream, true};
    for (int i = MNK; i <= 4*1024; i *= 2) {
        test_driver<128, 64, bK, 2, 128, MMA_C, Element, ElementC>(i, i, i, exec);
    }
    cudaStreamDestroy(stream);
}

void work() {
    cudaSetDevice(0);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    matx::cudaExecutor exec{stream, true};
    constexpr auto M = 2;
    constexpr auto K = 2;
    constexpr auto N = 4;
    auto tA = matx::make_tensor<float>({M, K});
    tA(0,0) = 0.f;
    tA(0,1) = 1.f;
    tA(1,0) = 2.f;
    tA(1,1) = 3.f;
    print(tA);
    auto tB = matx::make_tensor<float>({N, K});
    tB(0,0) = 4.f;
    tB(0,1) = 5.f;
    tB(1,0) = 6.f;
    tB(1,1) = 7.f;
    tB(2,0) = 8.f;
    tB(2,1) = 9.f;
    tB(3,0) = 10.f;
    tB(3,1) = 11.f;
    (tB = matx::random<float>(tB.Shape(), matx::NORMAL)).run(exec);
    print(tB);
    auto tC = matx::make_tensor<float>({M, N});
    (tC = matx::matmul(tA, tB.PermuteMatrix())).run(exec);
    print(tC);
}
int main(int argc, char** argv) {
    test_parser(argc, argv);
}