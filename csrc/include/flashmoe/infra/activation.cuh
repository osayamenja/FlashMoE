//
// Created by osayamen on 1/15/26.
//

#ifndef FLASHMOE_ACTIVATION_CUH
#define FLASHMOE_ACTIVATION_CUH
#include <cublasdx.hpp>
#include <cutlass/epilogue/thread/activation.h>

namespace flashmoe {
  enum class Activation {
    identity,
    relu,
    gelu,
    silu // as many as cutlass supports
  };

  template<int a>
  consteval Activation defineAct() {
    static_assert(a >= 0 && a < 4);
    switch (a) {
      case 0: return Activation::identity;
      case 1: return Activation::silu;
      case 2: return Activation::gelu;
      case 3: return Activation::relu;
      default: return Activation::identity;
    }
  }

  template<typename Element, Activation a = Activation::identity>
  struct ActivationType {
    static_assert(a == Activation::identity);
    using AT = cublasdx::identity;
  };

  template<typename Element>
  struct ActivationType<Element, Activation::relu> {
    using AT = cutlass::epilogue::thread::ReLU<Element>;
  };

  template<typename Element>
  struct ActivationType<Element, Activation::gelu> {
    using AT = cutlass::epilogue::thread::GELU<Element>;
  };

  template<typename Element>
  struct ActivationType<Element, Activation::silu> {
    using AT = cutlass::epilogue::thread::SiLu<Element>;
  };
}
#endif //FLASHMOE_ACTIVATION_CUH
