/******************************************************************************
 * Copyright (c) 2026, Osayamen Jonathan Aimuyo.
 ******************************************************************************/
//
// Created by Osayamen on 1/5/26.
//

#ifndef FLASHMOE_VT_CUH
#define FLASHMOE_VT_CUH

#include "constants.cuh"

namespace flashmoe {
  template<typename T>
  struct RawType {
    using type = T;
  };
  template<>
  struct RawType<__half> {
    using type = __half_raw;
  };
  template<>
  struct RawType<__nv_bfloat16> {
    using type = __nv_bfloat16_raw;
  };
  template<>
  struct RawType<__half2> {
    using type = __half2_raw;
  };
  template<>
  struct RawType<__nv_bfloat162> {
    using type = __nv_bfloat162_raw;
  };
  constexpr int MAX_ALIGNMENT = 16;

  template<typename T, int Alignment = MAX_ALIGNMENT>
  struct VectorTypeDescriptor {
    using VectorWidth = cute::Int<Alignment / sizeof(T)>;
    using VectorType = cutlass::AlignedArray<T, VectorWidth::value, Alignment>;
  };

  template<typename Element, int dim, int MAX_ALIGN = MAX_ALIGNMENT>
    requires(MAX_ALIGN <= MAX_ACCESS_ALIGNMENT && cutlass::is_pow2<MAX_ALIGN>::value && MAX_ALIGN >= 1)
  constexpr int ElementWidth = cute::min(dim, MAX_ALIGN / sizeof(Element));
  template<typename Element, int dim>
  constexpr uint32_t ElementAlignment = (cutlass::is_pow2<ElementWidth<Element, dim> >::value
                                           ? ElementWidth<Element, dim>
                                           : 1) * sizeof(Element);
  template<typename Element, int dim, int width>
  constexpr int ElementAlignmentForWidth = (cutlass::is_pow2<width>::value ? width : 1) * sizeof(Element);
}
#endif //FLASHMOE_VT_CUH
