//
// Created by osayamen on 12/22/25.
//

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
    cudaSetDevice(0);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    matx::cudaExecutor exec{stream};
    constexpr int S = 4;
    constexpr int E = 4;
    auto tA = matx::make_tensor<float>({S,E});
    constexpr int k = 2;
    std::random_device rnd;
    randUniform<FLASHMOE_ARCH>(tA.Data(), tA.TotalSize(), rnd(), -1.f, 1.f, stream);
    auto sIndices = matx::make_tensor<matx::index_t>(tA.Shape());
    auto topK_idx= sIndices.Slice<2>({0, 0}, {matx::matxEnd, k});
    auto ex_token_idx = matx::make_tensor<matx::index_t>({E, S});
    auto eCounts = matx::make_tensor<matx::index_t>({E});
    (sIndices = matx::argsort(tA, matx::SORT_DIR_DESC)).run(exec);
    // [S, E] -> [S, k]
    for (int i = 0; i < E; ++i) {
        auto eti_row = ex_token_idx.Slice<1>({i, 0}, {matx::matxDropDim, matx::matxEnd});
        auto eC = eCounts.Slice<0>({i}, {matx::matxDropDim});
        (matx::mtie(eti_row, eC) = matx::find_idx(topK_idx, matx::EQ<matx::index_t>{i})).run(exec);
        // converts indices from linearized [0, S*K) to [0, S)
        (eti_row = matx::apply(IndexSanitizer{k}, eti_row)).run(exec);
    }
    auto tf = matx::make_tensor<F>({2});
    tf(0) = F{0.9f, 0};
    tf(1) = F{-0.8f, 1};
    auto tf2 = matx::make_tensor<int>({2});
    (tf2 = matx::apply(SplitFunctor{}, tf)).run(exec);
    exec.sync();
    print(tA);print(topK_idx);print(ex_token_idx);print(eCounts);print(tf2);
    cudaStreamDestroy(stream);
}