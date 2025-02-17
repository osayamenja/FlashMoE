/******************************************************************************
 * Copyright (c) 2024, Osayamen Jonathan Aimuyo.
 ******************************************************************************/

#ifndef ARISTOS_MOE_CUH
#define ARISTOS_MOE_CUH

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
        typename Element
    >
    requires(aristos::TensorValueType<ElementC> &&
        aristos::TensorValueType<Element> &&
        cuda::std::is_invocable_r_v<ElementC, ActivationOp, ElementC> &&
        cuda::std::is_invocable_r_v<ElementC, ActivationOpX, ElementC>)
    __global__ __maxnreg__(REGINALD) void forward(
        const Element* __restrict__ iP, /* A, G, B, D*/ Element* __restrict__ oP /*G, O*/) {
        // Salami slice pointers
        const auto S = bookkeeping.sl;
        const auto P = bookkeeping.pd;
        const auto H = bookkeeping.ed;
        const auto E = bookkeeping.nx;
        const auto lE = bookkeeping.nLx;
        const auto* __restrict__ gP = iP + S * H;
        const auto* __restrict__ ePu = gP + H * E;
        const auto* __restrict__ ePd = ePu + lE * P * H;
        const auto* __restrict__ bU = ePd + lE * H * P;
        const auto* __restrict__ bd = bU + lE * P;
        auto* __restrict__ gOp = oP;
        auto* __restrict__ mOp = gOp + S * E;

        constexpr auto blocks = Hardware<Arch>::blocks::value;
        constexpr auto processors = blocks - 1;
        __shared__ __align__(16) cuda::std::byte workspace[SHARED_SIZE];
        // wipe buffers here and read the sequence bit, before the grid-wide barrier
        const auto gtQCl = bookkeeping.gtQCl;
        const auto sb = seqBit;
        auto* __restrict__ gtQHeads = bookkeeping.tQH();
        for (uint i = THREADS * blockIdx.x + threadIdx.x; i < gtQCl; i += blocks * THREADS) {
            gtQHeads[i] = 0U;
        }
        if constexpr (c == CombineMode::multithreaded) {
            // clear output buffer
            const auto sz = S * H;
            const auto vL = sz / sizeof(uint4);
            for (uint i = THREADS * blockIdx.x + threadIdx.x; i < vL; i += blocks * THREADS) {
                CAST_TO(uint4, mOp)[i] = uint4{0U, 0U, 0U, 0U};
            }
            // residue
            for (uint i = THREADS * blockIdx.x + threadIdx.x + vL * sizeof(uint); i < sz; i += blocks * THREADS) {
                mOp[i] = Element(0);
            }
        }

        const auto activations = cute::make_tensor(cute::make_gmem_ptr(iP),
            make_layout(cute::make_shape(S, H), cute::LayoutRight{}));
        const auto gateWeights = cute::make_tensor(cute::make_gmem_ptr(gP),
            make_layout(cute::make_shape(E, H), cute::LayoutRight{}));
        // Experts Weights
        const auto expertsUp = cute::make_tensor(cute::make_gmem_ptr(ePu),
            make_layout(make_shape(lE, cute::make_shape(P, H)), cute::LayoutRight{}));
        const auto expertsDown = cute::make_tensor(cute::make_gmem_ptr(ePd),
            make_layout(make_shape(lE, cute::make_shape(H, P)), cute::LayoutRight{}));
        // Bias
        // Broadcast from vector to matrix
        const auto biasUp = cute::make_tensor(cute::make_gmem_ptr(bU),
            make_layout(make_shape(lE, cute::make_shape(S, P)),
                make_stride(P, cute::Stride<cute::_0, cute::_1>{})));
        const auto biasDown = cute::make_tensor(cute::make_gmem_ptr(bd),
            make_layout(make_shape(lE, cute::make_shape(S, H)),
                make_stride(H, cute::Stride<cute::_0, cute::_1>{})));

        // Output
        const auto gateOutput = cute::make_tensor(cute::make_gmem_ptr(gOp),
            make_layout(cute::make_shape(S, E), cute::LayoutRight{}));
        const auto moeOutput = cute::make_tensor(cute::make_gmem_ptr(mOp),
            make_layout(make_layout(cute::make_shape(S, H), cute::LayoutRight{})));

        gate::forward<Arch, blocks, g, ElementC>(activations,
            gateWeights, gateOutput, bookkeeping.k, CAST_TO(ElementC, workspace));
        if (blockIdx.x + 1 < blocks) {
            constexpr auto cutoff = processors / ARISTOS_SUPER_BLOCK_SIZE * ARISTOS_SUPER_BLOCK_SIZE;
            if (blockIdx.x < cutoff) {
                packet::encode<cutoff, d, ARISTOS_SUPER_BLOCK_SIZE>(activations, CAST_TO(uint, workspace));
            }
            processor::start<
                processors,
                Arch,
                c,
                ActivationOp,
                ActivationOpX,
                ElementC,
                Element>(workspace, gateOutput, sb);
        }
        else {
            os::start<processors, d>(workspace, moeOutput, expertsUp, expertsDown, biasUp, biasDown, sb);
        }
    }

    template<
        unsigned int Arch
    >
    requires(aristos::SupportedArch<Arch>)
    __global__ __maxnreg__(REGINALD) void backward(){
    }

    template<
        typename T,
        typename ActivationOp = cutlass::epilogue::thread::ReLU<T>,
        typename ActivationOpX = cute::identity
    >
    __host__ __forceinline__
    void dispatchKernel(const void* __restrict__ iP, /* A, G, B, D*/ void* __restrict__ oP /*G, O*/) {
        using ElementC = GEA;
        // Decode function id
        switch (hostBookkeeping.fId) {
            case 0: {
                constexpr auto g = GateReductionLevel::singleBlock;
                constexpr auto d = DropTokens::no;
                constexpr auto c = CombineMode::single;
                using V = typename VCT<c, T>::Element;
                // Call forward pass
                moe::forward<ARISTOS_ARCH, g, d, c, ActivationOp, ActivationOpX, ElementC>
                 <<<Hardware<ARISTOS_ARCH>::blocks::value, THREADS>>>(
                     static_cast<const V*>(iP), static_cast<V*>(oP));
            }
            break;
            case 1: {
                constexpr auto g = GateReductionLevel::singleBlock;
                constexpr auto d = DropTokens::no;
                constexpr auto c = CombineMode::multithreaded;
                using V = typename VCT<c, T>::Element;
                // Call forward pass
                moe::forward<ARISTOS_ARCH, g, d, c, ActivationOp, ActivationOpX, ElementC>
                 <<<Hardware<ARISTOS_ARCH>::blocks::value, THREADS>>>(
                     static_cast<const V*>(iP), static_cast<V*>(oP));
            }
            break;
            case 2: {
                constexpr auto g = GateReductionLevel::singleBlock;
                constexpr auto d = DropTokens::yes;
                constexpr auto c = CombineMode::single;
                using V = typename VCT<c, T>::Element;
                // Call forward pass
                moe::forward<ARISTOS_ARCH, g, d, c, ActivationOp, ActivationOpX, ElementC>
                 <<<Hardware<ARISTOS_ARCH>::blocks::value, THREADS>>>(
                     static_cast<const V*>(iP), static_cast<V*>(oP));
            }
            break;
            case 3: {
                constexpr auto g = GateReductionLevel::singleBlock;
                constexpr auto d = DropTokens::yes;
                constexpr auto c = CombineMode::multithreaded;
                using V = typename VCT<c, T>::Element;
                // Call forward pass
                moe::forward<ARISTOS_ARCH, g, d, c, ActivationOp, ActivationOpX, ElementC>
                 <<<Hardware<ARISTOS_ARCH>::blocks::value, THREADS>>>(
                     static_cast<const V*>(iP), static_cast<V*>(oP));
            }
            case 4: {
                constexpr auto g = GateReductionLevel::multiBlock;
                constexpr auto d = DropTokens::no;
                constexpr auto c = CombineMode::single;
                using V = typename VCT<c, T>::Element;
                // Call forward pass
                moe::forward<ARISTOS_ARCH, g, d, c, ActivationOp, ActivationOpX, ElementC>
                 <<<Hardware<ARISTOS_ARCH>::blocks::value, THREADS>>>(
                     static_cast<const V*>(iP), static_cast<V*>(oP));
            }
            break;
            case 5: {
                constexpr auto g = GateReductionLevel::multiBlock;
                constexpr auto d = DropTokens::no;
                constexpr auto c = CombineMode::multithreaded;
                using V = typename VCT<c, T>::Element;
                // Call forward pass
                moe::forward<ARISTOS_ARCH, g, d, c, ActivationOp, ActivationOpX, ElementC>
                 <<<Hardware<ARISTOS_ARCH>::blocks::value, THREADS>>>(
                     static_cast<const V*>(iP), static_cast<V*>(oP));
            }
            break;
            case 6: {
                constexpr auto g = GateReductionLevel::multiBlock;
                constexpr auto d = DropTokens::yes;
                constexpr auto c = CombineMode::single;
                using V = typename VCT<c, T>::Element;
                // Call forward pass
                moe::forward<ARISTOS_ARCH, g, d, c, ActivationOp, ActivationOpX, ElementC>
                 <<<Hardware<ARISTOS_ARCH>::blocks::value, THREADS>>>(
                     static_cast<const V*>(iP), static_cast<V*>(oP));
            }
            break;
            case 7: {
                constexpr auto g = GateReductionLevel::multiBlock;
                constexpr auto d = DropTokens::yes;
                constexpr auto c = CombineMode::multithreaded;
                using V = typename VCT<c, T>::Element;
                // Call forward pass
                moe::forward<ARISTOS_ARCH, g, d, c, ActivationOp, ActivationOpX, ElementC>
                 <<<Hardware<ARISTOS_ARCH>::blocks::value, THREADS>>>(
                     static_cast<const V*>(iP), static_cast<V*>(oP));
            }
            break;
            default:
                reportError(false, "No such function exists!");
        }
    }

    __host__ __forceinline__
    void forwardHost(const void* __restrict__ iP, void* __restrict__ oP){
        reportError(isInitialized, "Not initialized");
        CHECK_ERROR_EXIT(cudaSetDevice(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE)));
        using relu = cutlass::epilogue::thread::ReLU<GEA>;
        using gelu = cutlass::epilogue::thread::GELU<GEA>;
        switch (hostBookkeeping.pfId) {
            case 0: {
                dispatchKernel<cute::tfloat32_t, relu>(iP, oP);
            }
            break;
            case 1: {
                dispatchKernel<cute::tfloat32_t, gelu>(iP, oP);
            }
            break;
            case 2:
                dispatchKernel<cute::half_t, relu>(iP, oP);
            break;
            case 3:
                dispatchKernel<cute::half_t, gelu>(iP, oP);
            break;
            case 4:
                dispatchKernel<cute::bfloat16_t, relu>(iP, oP);
            break;
            case 5:
                dispatchKernel<cute::bfloat16_t, gelu>(iP, oP);
            break;
            default:
                dispatchKernel<cute::half_t, relu>(iP, oP);
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
