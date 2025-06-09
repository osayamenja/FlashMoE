/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Kleos Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 */
//
// Created by oja7 on 11/5/24.
//

#ifndef MMACONFIG_CUH
#define MMACONFIG_CUH

#include <cute/arch/copy.hpp>
#include <cute/arch/copy_sm80.hpp>

namespace kleos {
    template<unsigned int Arch, typename TC, typename TA=TC, typename TB=TA>
    requires (Arch >= 700)
    struct MMAConfig {
        using mma = cute::TiledMMA<
                    cute::MMA_Atom<cute::UniversalFMA<TC, TA, TB>>,
                    cute::Layout<cute::Shape<cute::_16, cute::_8, cute::_1>>
        >;
    };

    template<>
    struct MMAConfig<700, cute::half_t> {
        using mma = cute::TiledMMA<
          cute::MMA_Atom<cute::SM70_8x8x4_F16F16F16F16_TN>,
          cute::Layout<cute::Shape<cute::_4, cute::_4, cute::_1>>,
        cute::Tile<cute::_32, cute::_32, cute::_8>
        >;
    };

    template<>
    struct MMAConfig<700, float, cute::half_t> {
        using mma = cute::TiledMMA<
          cute::MMA_Atom<cute::SM70_8x8x4_F32F16F16F32_TN>,
          cute::Layout<cute::Shape<cute::_4, cute::_4, cute::_1>>,
        cute::Tile<cute::_32, cute::_32, cute::_8>
        >;
    };

    template<>
    struct MMAConfig<800, cute::half_t> {
        using mma = cute::TiledMMA<
          cute::MMA_Atom<cute::SM80_16x8x8_F16F16F16F16_TN>,
          cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>,
        cute::Tile<cute::_32, cute::_32, cute::_16>
        >;
    };

    template<>
    struct MMAConfig<800, float, cute::half_t> {
        using mma = cute::TiledMMA<
          cute::MMA_Atom<cute::SM80_16x8x8_F32F16F16F32_TN>,
          cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>,
        cute::Tile<cute::_32, cute::_32, cute::_16>
        >;
    };

    template<>
    struct MMAConfig<800, float, cute::bfloat16_t> {
        using mma = cute::TiledMMA<
          cute::MMA_Atom<cute::SM80_16x8x8_F32BF16BF16F32_TN>,
          cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>,
        cute::Tile<cute::_32, cute::_32, cute::_16>
        >;
    };

    template<>
    struct MMAConfig<800, float, cute::tfloat32_t> {
        using mma = cute::TiledMMA<
          cute::MMA_Atom<cute::SM80_16x8x8_F32TF32TF32F32_TN>,
          cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>,
                        cute::Stride<cute::_2, cute::_1, cute::_1>>,
        cute::Tile<cute::_32, cute::_32, cute::_8>
        >;
    };

    template <unsigned int midSwizzle, unsigned int sizeK>
    requires((midSwizzle == 2 || midSwizzle == 3) && (sizeK >= 8 && sizeK <= 64))
    struct SwizzleAtom {
        using swizzleAtom =  decltype(
        cute::composition(cute::Swizzle<3, midSwizzle, 3>{},
                    cute::Layout<cute::Shape<cute::_8, cute::Int<sizeK>>,
                           cute::Stride<cute::Int<sizeK>, cute::_1>>{}));
    };

    template<typename Element, unsigned int Arch>
    using copyArch = cuda::std::conditional_t<sizeof(Element) >= 4 && Arch >= 800,
        cute::SM80_CP_ASYNC_CACHEALWAYS<Element>, cute::UniversalCopy<Element>>;

    template<typename Element>
    using sCopyLay = cuda::std::conditional_t<cuda::std::is_same_v<Element, float>,
    cute::UniversalCopy<float>, cuda::std::conditional_t<sizeof(Element) >= 4,
    cute::SM75_U32x4_LDSM_N, cute::SM75_U32x2_LDSM_N>>;

    template<typename Element> requires(sizeof(Element) >= 2)
    using VT = cuda::std::conditional_t<sizeof(Element) == 2, uint32_t, Element>;
    template<
        typename ElementA,
        typename ElementB,
        unsigned int Arch
    >
    struct CopyOp {
        using VTA = VT<ElementA>;
        using copyA = decltype(cute::make_tiled_copy(
            cute::Copy_Atom<copyArch<VTA, Arch>, ElementA>{},
            cute::Layout<cute::Shape<cute::_16, cute::_8>,
                cute::Stride<cute::_8, cute::_1>>{},
            cute::Layout<cute::Shape<cute::_1, cute::Int<sizeof(VTA) / sizeof(ElementA)>>>{}));

        using VTB = VT<ElementB>;
        using copyB = decltype(cute::make_tiled_copy(
            cute::Copy_Atom<copyArch<VTB, Arch>, ElementB>{},
            cute::Layout<cute::Shape<cute::_16, cute::_8>,
                cute::Stride<cute::_8, cute::_1>>{},
            cute::Layout<cute::Shape<cute::_1, cute::Int<sizeof(VTB) / sizeof(ElementB)>>>{}));
    };

    enum class LayoutOptimization {
        UseSwizzle,
        UseVanilla
    };

    template<typename T>
    requires (sizeof(T) == 2 || sizeof(T) == 4)
    using MiddleSwizzle = cute::Int<sizeof(T) == 2 ? 3 : 2>;

    template<
        unsigned int bM,
        unsigned int bN,
        unsigned int bK,
        unsigned int Arch,
        typename ElementA,
        typename ElementB,
        typename ElementC,
        LayoutOptimization lOpt = LayoutOptimization::UseVanilla
    >
    struct CollectiveMMAConfig{
        using copyAB = CopyOp<
            ElementA,
            ElementB,
            Arch
        >;

        using gCopyA = typename copyAB::copyA;
        using gCopyB = typename copyAB::copyB;

        using sCopyA = cute::Copy_Atom<cuda::std::conditional_t<Arch < 800,
        cute::AutoVectorizingCopyWithAssumedAlignment<8 * sizeof(ElementA)>,
        sCopyLay<VT<ElementA>>>, ElementA>;
        using sCopyB = cute::Copy_Atom<cuda::std::conditional_t<Arch < 800,
        cute::AutoVectorizingCopyWithAssumedAlignment<8 * sizeof(ElementB)>,
        sCopyLay<VT<ElementB>>>, ElementB>;

        using vSLayA = cute::Layout<cute::Shape<cute::Int<bM>, cute::Int<bK>>,
            cute::Stride<cute::Int<bK>, cute::_1>>;
        using sLayA = cuda::std::conditional_t<lOpt == LayoutOptimization::UseSwizzle,
        typename SwizzleAtom<MiddleSwizzle<ElementA>{}, bK>::swizzleAtom, vSLayA>;

        using vSLayB = cute::Layout<cute::Shape<cute::Int<bN>, cute::Int<bK>>,
            cute::Stride<cute::Int<bK>, cute::_1>>;
        using sLayB = cuda::std::conditional_t<lOpt == LayoutOptimization::UseSwizzle,
        typename SwizzleAtom<MiddleSwizzle<ElementB>{}, bK>::swizzleAtom, vSLayB>;

        using mma_t = typename MMAConfig<Arch, ElementC, ElementA, ElementB>::mma;
    };
}
#endif //MMACONFIG_CUH
