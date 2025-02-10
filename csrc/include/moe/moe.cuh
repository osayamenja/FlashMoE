/******************************************************************************
 * Copyright (c) 2024, Osayamen Jonathan Aimuyo.
 ******************************************************************************/

#ifndef ARISTOS_MOE_CUH
#define ARISTOS_MOE_CUH

#include <torch/torch.h>

#include "gate.cuh"
#include "../os/os.cuh"
#include "../os/processor/processor.cuh"

namespace aristos::moe{
    template<
        unsigned int Arch,
        GateReductionLevel g = GateReductionLevel::singleBlock,
        DropTokens d = DropTokens::yes,
        CombineMode c = CombineMode::single,
        typename ActivationOp = cute::identity,
        typename ActivationOpX = cute::identity,
        typename ElementC = float,
        typename ElementA = cute::half_t,
        typename ElementB = ElementA,
        typename ElementD = ElementA,
        typename Activations,
        typename ExpertsUp,
        typename ExpertsDown,
        typename BiasUp,
        typename BiasDown,
        typename Gates,
        typename GateOut>
    requires(aristos::SupportedArch<Arch> && aristos::Matrix<Activations> &&
        aristos::Tensor<ExpertsUp> && aristos::Tensor<ExpertsDown> &&
        aristos::Tensor<BiasUp> && aristos::Tensor<BiasDown> &&
        aristos::Matrix<Gates> && aristos::Matrix<GateOut>)
    __global__ __maxnreg__(REGINALD) void forward(
        Activations const __grid_constant__ activations,
        ExpertsUp const __grid_constant__ expertsUp,
        ExpertsDown const __grid_constant__ expertsDown,
        BiasUp const __grid_constant__ biasUp,
        BiasDown const __grid_constant__ biasDown,
        Gates const __grid_constant__ gateWeights,
        GateOut const __grid_constant__ gateOutput) {
        constexpr auto blocks = Hardware<Arch>::blocks;
        constexpr auto processors = blocks - 1;
        __shared__ __align__(16) cuda::std::byte workspace[SHARED_SIZE];
        // wipe gTQHeads here and read the sequence bit, before the grid-wide barrier
        const auto gtQCl = bookkeeping.gtQCl;
        const auto sb = seqBit;
        auto* __restrict__ gtQHeads = bookkeeping.tQH();
        for (uint i = THREADS * blockIdx.x + threadIdx.x; i < gtQCl; i += blocks * THREADS) {
            gtQHeads[i] = 0U;
        }

        gate::forward<Arch, blocks, g, ElementC>(activations,
            gateWeights, gateOutput, CAST_TO(ElementC, workspace));
        if (blockIdx.x + 1 < blocks) {
            packet::encode<processors, d>(activations, gateOutput, workspace);
            processor::start<
                processors,
                Arch,
                c,
                ElementA,
                ElementB,
                ElementC,
                ElementD,
                ActivationOp,
                ActivationOpX>(CAST_TO(ElementD, workspace), sb);
        }
        else {
            os::start<processors, d>(workspace, activations,
                expertsUp, expertsDown, biasUp, biasDown, sb);
        }
    }

    template<
        unsigned int Arch
    >
    requires(aristos::SupportedArch<Arch>)
    __global__ __maxnreg__(REGINALD) void backward(){

    }

    template<typename Element>
    __host__ __forceinline__
    void dispatchKernel(const torch::Tensor& activations,
        const torch::Tensor& expertsUp, const torch::Tensor& expertsDown,
        const torch::Tensor& biasUp, const torch::Tensor& biasDown,
        const torch::Tensor& gateWeights, const torch::Tensor& gateOutput) {
        using ElementC = mp_t; // accumulate type
        const auto sl = hostBookkeeping.sl;
        const auto ed = hostBookkeeping.ed;
        const auto pd = hostBookkeeping.pd;
        const auto nLx = hostBookkeeping.nLx;
        const auto nx = hostBookkeeping.nx;
        const auto px = hostBookkeeping.px;
#if CHECK_TENSORS
        TORCH_CHECK(activations.scalar_type() == expertsUp.scalar_type());
        TORCH_CHECK(activations.scalar_type() == expertsDown.scalar_type());
        TORCH_CHECK(activations.scalar_type() == biasUp.scalar_type());
        TORCH_CHECK(activations.scalar_type() == biasDown.scalar_type());
        TORCH_CHECK(activations.scalar_type() == gateWeights.scalar_type());
        TORCH_CHECK(activations.scalar_type() == gateOutput.scalar_type());

        TORCH_CHECK(activations.is_contiguous())
        TORCH_CHECK(expertsUp.is_contiguous())
        TORCH_CHECK(expertsDown.is_contiguous())
        TORCH_CHECK(biasUp.is_contiguous())
        TORCH_CHECK(biasDown.is_contiguous())
        TORCH_CHECK(gateWeights.is_contiguous())
        TORCH_CHECK(gateOutput.is_contiguous())

        TORCH_CHECK(activations.is_cuda());
        TORCH_CHECK(expertsUp.is_cuda());
        TORCH_CHECK(expertsDown.is_cuda());
        TORCH_CHECK(biasUp.is_cuda());
        TORCH_CHECK(biasDown.is_cuda());
        TORCH_CHECK(gateWeights.is_cuda());
        TORCH_CHECK(gateOutput.is_cuda());

        TORCH_CHECK(activations.sizes()[0] == sl
            && activations.sizes()[1] == ed);
        TORCH_CHECK(gateWeights.size(0) == nx
            && gateWeights.size(1) == ed)
        TORCH_CHECK(gateOutput.size(0) == sl
            && gateWeights.size(1) == px)

        const auto eSzU = expertsUp.sizes();
        const auto eSzD = expertsDown.sizes();
        TORCH_CHECK(eSzU.size() == 3 && eSzD.size() == 3);
        TORCH_CHECK(eSzU[0] == nLx && eSzD[0] == nLx);
        // Weights are already transposed in memory
        TORCH_CHECK(eSzU[1] == pd && eSzD[1] == ed);
        TORCH_CHECK(eSzU[2] == ed && eSzD[3] == pd);
        const auto bSzU = biasUp.sizes();
        const auto bSzD = biasDown.sizes();
        TORCH_CHECK(bSzU.size() == 2 && bSzD.size() == 2);
        TORCH_CHECK(bSzU[0] == nLx && bSzD[0] == nLx);
        TORCH_CHECK(bSzU[1] == pd && bSzD[1] == ed);
#endif

        // Now construct cute tensors
        // Activations & Output
        const auto* __restrict__ aP = activations.const_data_ptr<Element>();
        const auto tA = cute::make_tensor(cute::make_gmem_ptr(aP),
            make_layout(cute::make_shape(sl, ed),
                cute::LayoutRight{}));

        // Expert weights
        const auto* __restrict__ ePu = expertsUp.const_data_ptr<Element>();
        const auto tXu = cute::make_tensor(cute::make_gmem_ptr(ePu),
            make_layout(make_shape(nLx, cute::make_shape(pd, ed)), cute::LayoutRight{}));
        const auto* __restrict__ ePd = expertsDown.const_data_ptr<Element>();
        const auto tXd = cute::make_tensor(cute::make_gmem_ptr(ePd),
            make_layout(make_shape(nLx, cute::make_shape(ed, pd)), cute::LayoutRight{}));

        // Bias
        const auto* __restrict__ bPu = biasUp.const_data_ptr<Element>();
        const auto tBu = cute::make_tensor(cute::make_gmem_ptr(bPu),
            make_layout(make_shape(nLx, cute::make_shape(1, pd)), cute::LayoutRight{}));
        const auto* __restrict__ bPd = biasDown.const_data_ptr<Element>();
        const auto tBd = cute::make_tensor(cute::make_gmem_ptr(bPd),
            make_layout(make_shape(nLx, cute::make_shape(1, ed)), cute::LayoutRight{}));
        // Gate
        const auto* __restrict__ gP = gateWeights.const_data_ptr<Element>();
        const auto tG = cute::make_tensor(cute::make_gmem_ptr(gP),
            make_layout(cute::make_shape(nx, ed), cute::LayoutRight{}));
        const auto* __restrict__ gOp = gateOutput.const_data_ptr<Element>();
        const auto tGo = cute::make_tensor(cute::make_gmem_ptr(gOp),
            make_layout(cute::make_shape(sl, px), cute::LayoutRight{}));
#if DECODE_MOE
        // Decode function id
        switch (hostBookkeeping.fId) {
            case 0: {
                constexpr auto g = GateReductionLevel::singleBlock;
                constexpr auto d = DropTokens::no;
                constexpr auto c = CombineMode::single;
                // Call forward pass
                moe::forward<700, g, d, c, ElementC, Element><<<Hardware<700>::blocks::value, THREADS>>>(
                    tA, tXu, tXd, tBu, tBd, tG, tGo);
            }
            break;
            case 1: {
                constexpr auto g = GateReductionLevel::singleBlock;
                constexpr auto d = DropTokens::no;
                constexpr auto c = CombineMode::multithreaded;
                // Call forward pass
                moe::forward<700, g, d, c, ElementC, Element><<<Hardware<700>::blocks::value, THREADS>>>(
                    tA, tXu, tXd, tBu, tBd, tG, tGo);
            }
            break;
            case 2: {
                constexpr auto g = GateReductionLevel::singleBlock;
                constexpr auto d = DropTokens::yes;
                constexpr auto c = CombineMode::single;
                // Call forward pass
                moe::forward<700, g, d, c, ElementC, Element><<<Hardware<700>::blocks::value, THREADS>>>(
                    tA, tXu, tXd, tBu, tBd, tG, tGo);
            }
            break;
            case 3: {
                constexpr auto g = GateReductionLevel::singleBlock;
                constexpr auto d = DropTokens::yes;
                constexpr auto c = CombineMode::multithreaded;
                // Call forward pass
                moe::forward<700, g, d, c, ElementC, Element><<<Hardware<700>::blocks::value, THREADS>>>(
                    tA, tXu, tXd, tBu, tBd, tG, tGo);
            }
            break;
            case 4: {
                constexpr auto g = GateReductionLevel::multiBlock;
                constexpr auto d = DropTokens::no;
                constexpr auto c = CombineMode::single;
                // Call forward pass
                moe::forward<700, g, d, c, ElementC, Element><<<Hardware<700>::blocks::value, THREADS>>>(
                    tA, tXu, tXd, tBu, tBd, tG, tGo);
            }
            break;
            case 5: {
                constexpr auto g = GateReductionLevel::multiBlock;
                constexpr auto d = DropTokens::no;
                constexpr auto c = CombineMode::multithreaded;
                // Call forward pass
                moe::forward<700, g, d, c, ElementC, Element><<<Hardware<700>::blocks::value, THREADS>>>(
                    tA, tXu, tXd, tBu, tBd, tG, tGo);
            }
            break;
            case 6: {
                constexpr auto g = GateReductionLevel::multiBlock;
                constexpr auto d = DropTokens::yes;
                constexpr auto c = CombineMode::single;
                // Call forward pass
                moe::forward<700, g, d, c, ElementC, Element><<<Hardware<700>::blocks::value, THREADS>>>(
                    tA, tXu, tXd, tBu, tBd, tG, tGo);
            }
            break;
            case 7: {
                constexpr auto g = GateReductionLevel::multiBlock;
                constexpr auto d = DropTokens::yes;
                constexpr auto c = CombineMode::multithreaded;
                // Call forward pass
                moe::forward<700, g, d, c, ElementC, Element><<<Hardware<700>::blocks::value, THREADS>>>(
                    tA, tXu, tXd, tBu, tBd, tG, tGo);
            }
            break;
            case 8: {
                constexpr auto g = GateReductionLevel::singleBlock;
                constexpr auto d = DropTokens::no;
                constexpr auto c = CombineMode::single;
                // Call forward pass
                moe::forward<800, g, d, c, ElementC, Element><<<Hardware<>::blocks::value, THREADS>>>(
                    tA, tXu, tXd, tBu, tBd, tG, tGo);
            }
            break;
            case 9: {
                constexpr auto g = GateReductionLevel::singleBlock;
                constexpr auto d = DropTokens::no;
                constexpr auto c = CombineMode::multithreaded;
                // Call forward pass
                moe::forward<800, g, d, c, ElementC, Element><<<Hardware<>::blocks::value, THREADS>>>(
                    tA, tXu, tXd, tBu, tBd, tG, tGo);
            }
            break;
            case 10: {
                constexpr auto g = GateReductionLevel::singleBlock;
                constexpr auto d = DropTokens::yes;
                constexpr auto c = CombineMode::single;
                // Call forward pass
                moe::forward<800, g, d, c, ElementC, Element><<<Hardware<>::blocks::value, THREADS>>>(
                    tA, tXu, tXd, tBu, tBd, tG, tGo);
            }
            break;
            case 11: {
                constexpr auto g = GateReductionLevel::singleBlock;
                constexpr auto d = DropTokens::yes;
                constexpr auto c = CombineMode::multithreaded;
                // Call forward pass
                moe::forward<800, g, d, c, ElementC, Element><<<Hardware<>::blocks::value, THREADS>>>(
                    tA, tXu, tXd, tBu, tBd, tG, tGo);
            }
            case 12: {
                constexpr auto g = GateReductionLevel::multiBlock;
                constexpr auto d = DropTokens::no;
                constexpr auto c = CombineMode::single;
                // Call forward pass
                moe::forward<800, g, d, c, ElementC, Element><<<Hardware<>::blocks::value, THREADS>>>(
                    tA, tXu, tXd, tBu, tBd, tG, tGo);
            }
            break;
            case 13: {
                constexpr auto g = GateReductionLevel::multiBlock;
                constexpr auto d = DropTokens::no;
                constexpr auto c = CombineMode::multithreaded;
                // Call forward pass
                moe::forward<800, g, d, c, ElementC, Element><<<Hardware<>::blocks::value, THREADS>>>(
                    tA, tXu, tXd, tBu, tBd, tG, tGo);
            }
            break;
            case 14: {
                constexpr auto g = GateReductionLevel::multiBlock;
                constexpr auto d = DropTokens::yes;
                constexpr auto c = CombineMode::single;
                // Call forward pass
                moe::forward<800, g, d, c, ElementC, Element><<<Hardware<>::blocks::value, THREADS>>>(
                    tA, tXu, tXd, tBu, tBd, tG, tGo);
            }
            break;
            case 15: {
                constexpr auto g = GateReductionLevel::multiBlock;
                constexpr auto d = DropTokens::yes;
                constexpr auto c = CombineMode::multithreaded;
                // Call forward pass
                moe::forward<800, g, d, c, ElementC, Element><<<Hardware<>::blocks::value, THREADS>>>(
                    tA, tXu, tXd, tBu, tBd, tG, tGo);
            }
            break;
            case 16: {
                constexpr auto g = GateReductionLevel::singleBlock;
                constexpr auto d = DropTokens::yes;
                constexpr auto c = CombineMode::single;
                // Call forward pass
                moe::forward<900, g, d, c, ElementC, Element><<<Hardware<900>::blocks::value, THREADS>>>(
                    tA, tXu, tXd, tBu, tBd, tG, tGo);
            }
            break;
            case 17: {
                constexpr auto g = GateReductionLevel::singleBlock;
                constexpr auto d = DropTokens::yes;
                constexpr auto c = CombineMode::multithreaded;
                // Call forward pass
                moe::forward<900, g, d, c, ElementC, Element><<<Hardware<900>::blocks::value, THREADS>>>(
                    tA, tXu, tXd, tBu, tBd, tG, tGo);
            }
            break;
            case 18: {
                constexpr auto g = GateReductionLevel::singleBlock;
                constexpr auto d = DropTokens::no;
                constexpr auto c = CombineMode::single;
                // Call forward pass
                moe::forward<900, g, d, c, ElementC, Element><<<Hardware<900>::blocks::value, THREADS>>>(
                    tA, tXu, tXd, tBu, tBd, tG, tGo);
            }
            break;
            case 19: {
                constexpr auto g = GateReductionLevel::singleBlock;
                constexpr auto d = DropTokens::no;
                constexpr auto c = CombineMode::multithreaded;
                moe::forward<900, g, d, c, ElementC, Element><<<Hardware<900>::blocks::value, THREADS>>>(
                    tA, tXu, tXd, tBu, tBd, tG, tGo);
            }
            break;
            case 20: {
                constexpr auto g = GateReductionLevel::multiBlock;
                constexpr auto d = DropTokens::yes;
                constexpr auto c = CombineMode::single;
                // Call forward pass
                moe::forward<900, g, d, c, ElementC, Element><<<Hardware<900>::blocks::value, THREADS>>>(
                    tA, tXu, tXd, tBu, tBd, tG, tGo);
            }
            break;
            case 21: {
                constexpr auto g = GateReductionLevel::multiBlock;
                constexpr auto d = DropTokens::yes;
                constexpr auto c = CombineMode::multithreaded;
                // Call forward pass
                moe::forward<900, g, d, c, ElementC, Element><<<Hardware<900>::blocks::value, THREADS>>>(
                    tA, tXu, tXd, tBu, tBd, tG, tGo);
            }
            break;
            case 22: {
                constexpr auto g = GateReductionLevel::multiBlock;
                constexpr auto d = DropTokens::no;
                constexpr auto c = CombineMode::single;
                // Call forward pass
                moe::forward<900, g, d, c, ElementC, Element><<<Hardware<900>::blocks::value, THREADS>>>(
                   tA, tXu, tXd, tBu, tBd, tG, tGo);
            }
            break;
            case 23: {
                constexpr auto g = GateReductionLevel::multiBlock;
                constexpr auto d = DropTokens::no;
                constexpr auto c = CombineMode::multithreaded;
                // Call forward pass
                moe::forward<900, g, d, c, ElementC, Element><<<Hardware<900>::blocks::value, THREADS>>>(
                    tA, tXu, tXd, tBu, tBd, tG, tGo);
            }
            break;
            default:
                reportError(false, "No such function exists!");
        }
#endif
    }

    __host__ __forceinline__
    void forwardHost(const torch::Tensor& activations,
        const torch::Tensor& expertsUp, const torch::Tensor& expertsDown,
        const torch::Tensor& biasUp, const torch::Tensor& biasDown,
        const torch::Tensor& gateWeights, const torch::Tensor& gateOutput){
        reportError(isInitialized, "Not initialized");
        switch (activations.scalar_type()) {
            case torch::kFloat: {
                if (at::globalContext().allowTF32CuBLAS() || at::globalContext().allowTF32CuDNN()) {
                    dispatchKernel<cute::tfloat32_t>(activations,expertsUp, expertsDown,
                        biasUp, biasDown, gateWeights, gateOutput);
                }
                else {
                    dispatchKernel<float>(activations, expertsUp, expertsDown,
                        biasUp, biasDown, gateWeights, gateOutput);
                }
            }
            break;
            case torch::kFloat16:
                dispatchKernel<cute::half_t>(activations, expertsUp, expertsDown,
                    biasUp, biasDown, gateWeights, gateOutput);
            break;
            case torch::kBFloat16:
                dispatchKernel<cute::bfloat16_t>(activations, expertsUp, expertsDown,
                    biasUp, biasDown, gateWeights, gateOutput);
            break;
            case torch::kFloat8_e4m3fn:
                dispatchKernel<cute::float_e4m3_t>(activations,expertsUp, expertsDown,
                    biasUp, biasDown, gateWeights, gateOutput);
            break;
            case torch::kFloat8_e5m2:
                dispatchKernel<cute::float_e5m2_t>(activations, expertsUp, expertsDown,
                    biasUp, biasDown, gateWeights, gateOutput);
            break;
            default:
                reportError(false, "Not supported!");
        }
    }

    template<typename Element>
    requires(aristos::TensorValueType<Element>)
    __host__ __forceinline__
    void backwardHost(){
        reportError(isInitialized, "Not initialized");
    }
}
#endif //ARISTOS_MOE_CUH
