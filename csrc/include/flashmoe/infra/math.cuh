//
// Created by osayamen on 1/18/26.
//

#ifndef FLASHMOE_MATH_CUH
#define FLASHMOE_MATH_CUH
#include <cutlass/fast_math.h>
namespace flashmoe
{
  // Also applies to shared memory banks
  template <typename Element>
  __device__ __forceinline__
  constexpr auto rTCL(uint const& len) {
    constexpr auto lineBytes = 128;
    return cutlass::round_up(len * sizeof(Element), lineBytes);
  }
}
#endif //FLASHMOE_MATH_CUH