//
// Created by osayamen on 1/18/26.
//

#ifndef FLASHMOE_MATH_CUH
#define FLASHMOE_MATH_CUH
#include <cuda/cmath>
namespace flashmoe
{
  // Also applies to shared memory banks
  template <typename Element>
  __device__ __forceinline__
  constexpr auto rTCL(uint const& len) {
    constexpr size_t lineBytes = 128;
    return cuda::round_up(len * sizeof(Element), lineBytes);
  }
}
#endif //FLASHMOE_MATH_CUH