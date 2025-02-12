//
// Created by oja7 on 12/22/24.
//

#ifndef ARCH_CUH
#define ARCH_CUH
#include <cute/numeric/integral_constant.hpp>

namespace aristos {
    template<unsigned int arch>
    concept SupportedArch = arch >= 700;

    __host__ __device__
    enum class Board {
        pcie,
        sxm,
    };
    // Data center GPUs only
    template<
        unsigned int Arch,
        unsigned int maxRegisters = 128,
        Board b = Board::pcie
    >
    requires (SupportedArch<Arch>)
    struct Hardware {
        static_assert(Arch == 800 && maxRegisters == 128 && b == Board::pcie,
            "Unregistered Arch");
        using blocks = cute::Int<4U * 108>;
    };

    template<>
    struct Hardware<800, 96> {
        using blocks = cute::Int<5U * 108>;
    };

    template<>
    struct Hardware<700> {
        using blocks = cute::Int<4U * 80>;
    };

    template<>
    struct Hardware<700, 96> {
        using blocks = cute::Int<5U * 80>;
    };

    // Hopper
    template<>
    struct Hardware<900, 128, Board::sxm> {
        using blocks = cute::Int<4U * 132>;
    };

    template<>
    struct Hardware<900, 128, Board::pcie> {
        using blocks = cute::Int<4U * 114>;
    };

    // Odd ones
    template<>
    struct Hardware<890> {
        using blocks = cute::Int<5U * 84>;
    };

    template<>
    struct Hardware<860> {
        using blocks = cute::Int<5U * 84>;
    };

    template<>
    struct Hardware<750> {
        // this may be less than the actual
        using blocks = cute::Int<3U * 40>;
    };
}
#endif //ARCH_CUH
