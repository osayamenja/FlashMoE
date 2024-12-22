//
// Created by oja7 on 12/22/24.
//

#ifndef ARCH_CUH
#define ARCH_CUH
#include <cute/numeric/integral_constant.hpp>

namespace aristos {
    template<unsigned int arch>
    concept SupportedArch = arch >= 700 && arch <= 900;

    enum class Board {
        pcie,
        sxm,
    };
    // Data center GPUs only
    template<unsigned int Arch = 800, unsigned int maxRegisters = 128, Board b = Board::pcie>
    struct Hardware {
        static_assert(Arch == 800 && maxRegisters == 128 && b == Board::pcie);
        using blocks = cute::Int<4 * 108>;
    };

    struct Hardware<800, 96> {
        using blocks = cute::Int<5 * 108>;
    };

    struct Hardware<700> {
        using blocks = cute::Int<4 * 80>;
    };

    struct Hardware<700, 96> {
        using blocks = cute::Int<5 * 80>;
    };

    // Hopper
    struct Hardware<900, 128, Board::sxm> {
        using blocks = cute::Int<4 * 132>;
    };

    struct Hardware<900, 128, Board::pcie> {
        using blocks = cute::Int<4 * 114>;
    };

    // Odd ones
    struct Hardware<890> {
        using blocks = cute::Int<5 * 84>;
    };

    struct Hardware<860> {
        using blocks = cute::Int<5 * 84>;
    };

    struct Hardware<750> {
        // this may be less than the actual
        using blocks = cute::Int<3 * 40>;
    };
}
#endif //ARCH_CUH
