//
// Created by osayamen on 12/22/25.
//

// place to experime
#include <random>
#include "common.cuh"

struct IndexSanitizer {
    const matx::index_t y;
    explicit IndexSanitizer(const matx::index_t& _y): y(_y){}
    __host__ __device__
    auto operator()(const matx::index_t& x) const {
        return x / y;
    }
};

struct F {
    float x;
    int y;
};
struct SplitFunctor {
    __host__ __device__
    auto operator()(const F& f) const {
        return f.y;
    }
};

int main() {
    volatile int* x = static_cast<int*>(std::malloc(sizeof(int)));
    cuda::atomic_ref<volatile int, cuda::thread_scope_system> a{*x};
    cuda::std::ignore = a.exchange(4, cuda::memory_order_acquire);
    using BLAS = decltype(
            cublasdx::Size<128, 128, 64>() +
            cublasdx::Precision<__half, __half, float>() +
            cublasdx::Type<cublasdx::type::real>() +
            cublasdx::Function<cublasdx::function::MM>() +
            cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major, cublasdx::row_major>() +
            cublasdx::Block() +
            cublasdx::MaxAlignment() +
            cublasdx::StaticBlockDim() +
            cublasdx::EnableInputStreaming() +
            cublasdx::SM<FLASHMOE_ARCH,
            FLASHMOE_ARCH >= 900 ? cublasdx::sm_modifier::arch_specific : cublasdx::sm_modifier::generic>());
    constexpr auto l = BLAS::get_layout_smem_c();
    cudaSetDevice(0);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    matx::cudaExecutor exec{stream};
    constexpr int S = 4; // 128
    constexpr int E = 2; // 8
    constexpr int H = 8; // 2048
    auto tA = matx::make_tensor<float>({S, H});
    tA.SetVals({{0, 1}, {2, 3}});
    auto tB = matx::make_tensor<float>({E, H});
    tB.SetVals({{1, 2}, {3, 4}});
    auto tC = matx::make_tensor<float>({S, E});
    (tC = matx::matmul(tA, tB.PermuteMatrix())).run(exec);
    print(tA); print(tB); print(tC);
}