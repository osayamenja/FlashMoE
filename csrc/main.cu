/******************************************************************************
 * Copyright (c) 2024, Osayamen Jonathan Aimuyo.
 ******************************************************************************/
#include <thrust/host_vector.h>
#include <torch/torch.h>

#include "include/bootstrap.cuh"
//#include "include/moe/moe.cuh"
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
    constexpr auto PX = aristos::ACC::PX::value;
    constexpr unsigned long aZ =  S * H;
    constexpr auto gwZ = aZ + PX * H;
    // scale this to number of experts
    constexpr auto bZ =  gwZ + P * H;
    constexpr auto b2Z =  bZ + P * H;
    constexpr auto dZ =  b2Z + cute::max(P, H);
    constexpr auto gZ = dZ + S * PX;
    constexpr auto cZ = gZ + S * H;
    cuda::std::byte* p;
    CHECK_ERROR_EXIT(cudaMallocAsync(&p, cZ * sizeof(float), aristos::aristosStream));
    auto* hP = std::calloc(dZ, sizeof(float));
    auto* fHp = static_cast<float*>(hP);
    using Element = aristos::ACC::Element;
    auto* __restrict__ eHp = static_cast<Element*>(hP);
    // Activations
    std::ranges::fill(fHp, fHp + aZ, 1.0f);
    // gate weights
    std::ranges::fill(fHp + aZ, fHp + aZ + E * H, 1.0f);
    // Expert weights
    // loop for number of experts
    makeIdentity<P, H>(fHp + gwZ);
    makeIdentity<P, H>(fHp + bZ);
    // bias
    std::ranges::fill(fHp + b2Z, fHp + dZ, 0.0f);
    /*const auto options = torch::TensorOptions().dtype(torch::kFloat32).
    layout(torch::kStrided).device(torch::kCPU);
    const auto a = torch::from_blob(fHp, {S, H}, options);
    const auto g = torch::from_blob(fHp + aZ, {H, E}, options);
    // torch reference of the gate function
    std::cout << softmax(matmul(a, g), 1) << std::endl;*/

    constexpr cutlass::NumericConverter<Element, float> conv{};
    for (uint i = 0; i < dZ; ++i) {
        eHp[i] = conv(fHp[i]);
    }
    CHECK_ERROR_EXIT(cudaMemcpyAsync(p, eHp, sizeof(Element) * dZ, cudaMemcpyHostToDevice,
        aristos::aristosStream));
    aristos::initialize();
    //aristos::moe::forwardHost(p, p + dZ * sizeof(Element));
    CHECK_ERROR_EXIT(cudaPeekAtLastError());
    CHECK_ERROR_EXIT(cudaMemcpyAsync(eHp, p + dZ * sizeof(Element), sizeof(Element) * (S * PX),
        cudaMemcpyDeviceToHost, aristos::aristosStream));
    cuda::std::array<aristos::TPS, aristos::ACC::S::value> b{};
    CHECK_ERROR_EXIT(cudaMemcpyAsync(b.data(), aristos::hostBookkeeping.tP(),
        sizeof(aristos::TPS) * b.size(), cudaMemcpyDeviceToHost,
        aristos::aristosStream));
    CHECK_ERROR_EXIT(cudaFreeAsync(p, aristos::aristosStream));
    aristos::finalize();
    /*const auto o = make_tensor(eHp,
        cute::Layout<cute::Shape<cute::Int<S>, cute::Int<PX>>, cute::Stride<cute::Int<PX>, cute::_1>>{});
    print_tensor(o);
    for (const auto &[idx, p] : b) {
        printf("{idx: %u, p: %f}\n", idx, p);
    }*/

    /*using clk = std::chrono::high_resolution_clock;
    std::chrono::duration<float> end {};
    const auto start = clk::now();
    aristos::initialize();
    printf("Number of local experts is %u\n", aristos::hostBookkeeping.nLx);
    end = clk::now() - start;
    printf("Initialize takes %fms\n", end.count() * 1000);
    aristos::finalize();*/
    std::free(hP);
}

int main() {
    runOS();
}
