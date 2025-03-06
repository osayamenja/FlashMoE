/******************************************************************************
 * Copyright (c) 2024, Osayamen Jonathan Aimuyo.
 ******************************************************************************/
#include <thrust/host_vector.h>

#include "include/bootstrap.cuh"
#include "include/moe/moe.cuh"
#include "correctness.cuh"
template<unsigned int M, unsigned int N = M>
requires(M > 0 && cutlass::is_pow2<M>::value && M == N)
__host__ __forceinline__
void makeIdentity(float* const& __restrict__ p, const bool print = false) {
    // we assume row major
    const auto t = cute::make_tensor(p,
        cute::Layout<cute::Shape<cute::Int<M>, cute::Int<N>>,
            cute::Stride<cute::Int<N>, cute::_1>>{});
    for (uint i = 0; i < M; ++i) {
        for (uint j = 0; j < N; ++j) {
            if (i == j) {
                t(i, j) = 1.0f;
            }
            else {
                t(i, j) = 0.0f;
            }
        }
    }

    if (print) {
        print_tensor(t);
    }
}

__host__ __forceinline__
void runOS() {
    // generate random input tile and eye weights
    constexpr auto S = aristos::ACC::S::value;
    constexpr auto H = aristos::ACC::H::value;
    constexpr auto E = aristos::ACC::E::value;
    constexpr auto P = aristos::ACC::P::value;
    constexpr auto dZ = S * H + P * H + cute::max(P, H) + E * H + S * E + S * H;
    void* p;
    CHECK_ERROR_EXIT(cudaMallocAsync(&p, dZ * sizeof(float), aristos::aristosStream));

    thrust::host_vector<float> activations(S * aristos::ACC::H::value);
    fill(activations.begin(), activations.end(), 1.0f);
    thrust::host_vector<float> expertWeights(2 * aristos::ACC::P::value * aristos::ACC::H::value);
    makeIdentity<aristos::ACC::P::value, aristos::ACC::H::value>(expertWeights.data());
    thrust::host_vector<float> bias(cute::max(aristos::ACC::P::value, aristos::ACC::H::value));
    fill(bias.begin(), bias.end(), 0.0f);
    thrust::host_vector<float> gateWeights(aristos::ACC::H::value * aristos::ACC::E::value);
    fill(gateWeights.begin(), gateWeights.end(), 1.0f);

    using clk = std::chrono::high_resolution_clock;
    std::chrono::duration<float> end {};
    const auto start = clk::now();
    aristos::initialize();
    printf("Number of local experts is %u\n", aristos::hostBookkeeping.nLx);
    end = clk::now() - start;
    printf("Initialize takes %fms\n", end.count() * 1000);
    aristos::finalize();
    CHECK_ERROR_EXIT(cudaFreeAsync(p, aristos::aristosStream));
}

int main() {
    runOS();
}
