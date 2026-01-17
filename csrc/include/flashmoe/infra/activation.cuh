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

    template<typename Element>
    struct ActivationType<Activation::relu> {
        using AT = cutlass::epilogue::thread::ReLU<Element>;
    };

    template<typename Element>
    struct ActivationType<Activation::gelu> {
        using AT = cutlass::epilogue::thread::GELU<Element>;
    };
}
#endif //FLASHMOE_ACTIVATION_CUH