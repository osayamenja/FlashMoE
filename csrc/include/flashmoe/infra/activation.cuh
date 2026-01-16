//
// Created by osayamen on 1/15/26.
//

#ifndef FLASHMOE_ACTIVATION_CUH
#define FLASHMOE_ACTIVATION_CUH
#include <cublasdx.hpp>
#include <cutlass/epilogue/thread/activation.h>
namespace flashmoe
{
    enum class Activation {
        identity,
        relu,
        gelu // as many as cutlass supports
    };

    template<Activation a = Activation::identity>
    struct ActivationType {
        static_assert(a == Activation::identity);
        using AT = cublasdx::identity;
    };

    template<>
    struct ActivationType<Activation::relu> {
        using AT = cublasdx::identity;

    };
}
#endif //FLASHMOE_ACTIVATION_CUH