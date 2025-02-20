//
// Created by oja7 on 12/22/24.
//

#ifndef ARCH_CUH
#define ARCH_CUH

#include <cute/numeric/integral_constant.hpp>
#define BASE_SHARED_SIZE 16 * 1024U
#define BASE_PIPE_STAGES 2U
namespace aristos {
    template<unsigned int arch>
    concept SupportedArch = arch >= 700;

    __host__ __device__
    enum class Board {
        pcie,
        sxm,
    };

    template<unsigned int blocks>
    requires(blocks > 0U)
    struct OSD {
        using osBlocks = cute::C<1U>;
        using processorBlocks = cute::C<blocks - osBlocks::value>;
        using threads = cute::C<128U>; // per block
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
        using arch = cute::C<800U>;
        using registers = cute::C<128U>;
        using blocks = cute::C<4U * 108>;
        using bKBase = cute::C<8U>;
        using rScratch = cute::C<32U>;
        using pipeStages = cute::C<BASE_PIPE_STAGES * 2>;
        using sharedMemory = cute::C<BASE_SHARED_SIZE * 2>;
        using OS = OSD<blocks::value>;
    };

    template<>
    struct Hardware<800, 96> {
        using arch = cute::C<800U>;
        using registers = cute::C<96U>;
        using blocks = cute::C<5U * 108>;
        using bKBase = cute::C<8U>;
        using rScratch = cute::C<32U>;
        using sharedMemory = cute::C<BASE_SHARED_SIZE>;
        using pipeStages = cute::C<BASE_PIPE_STAGES>;
        using OS = OSD<blocks::value>;
    };

    template<>
    struct Hardware<800, 255> {
        using arch = cute::C<800U>;
        using registers = cute::C<255U>;
        using blocks = cute::C<2U * 108>;
        using bKBase = cute::C<8U>;
        using rScratch = cute::C<64U>;
        using pipeStages = cute::C<BASE_PIPE_STAGES * 2>;
        using sharedMemory = cute::C<BASE_SHARED_SIZE * 2>;
        using OS = OSD<blocks::value>;
    };

    template<>
    struct Hardware<700> {
        using arch = cute::C<700U>;
        using registers = cute::C<128U>;
        using blocks = cute::C<4U * 80>;
        using bKBase = cute::C<16U>;
        using rScratch = cute::C<32U>;
        using pipeStages = cute::C<1U>;
        using sharedMemory = cute::C<BASE_SHARED_SIZE>;
        using OS = OSD<blocks::value>;
    };

    template<>
    struct Hardware<700, 96> {
        using arch = cute::C<700U>;
        using registers = cute::C<96U>;
        using blocks = cute::C<5U * 80>;
        using pipeStages = cute::C<1U>;
        using bKBase = cute::C<16U>;
        using rScratch = cute::C<32U>;
        using sharedMemory = cute::C<BASE_SHARED_SIZE>;
        using OS = OSD<blocks::value>;
    };

    template<>
    struct Hardware<700, 255> {
        // recommended
        using arch = cute::C<700U>;
        using registers = cute::C<255U>;
        using blocks = cute::C<2U * 80>;
        using pipeStages = cute::C<1U>;
        using bKBase = cute::C<32U>;
        using rScratch = cute::C<64U>;
        using sharedMemory = cute::C<BASE_SHARED_SIZE * 2>;
        using OS = OSD<blocks::value>;
    };

    // Hopper
    template<>
    struct Hardware<900, 128, Board::sxm> {
        using arch = cute::C<900U>;
        using registers = cute::C<128U>;
        using blocks = cute::C<4U * 132>;
        using bKBase = cute::C<8U>;
        using rScratch = cute::C<32U>;
        using pipeStages = cute::C<BASE_PIPE_STAGES * 2>;
        using sharedMemory = cute::C<BASE_SHARED_SIZE * 2>;
        using OS = OSD<blocks::value>;
    };

    template<>
    struct Hardware<900, 128, Board::pcie> {
        using arch = cute::C<900U>;
        using registers = cute::C<128U>;
        using blocks = cute::C<4U * 114>;
        using bKBase = cute::C<8U>;
        using rScratch = cute::C<32U>;
        using pipeStages = cute::C<BASE_PIPE_STAGES * 2>;
        using sharedMemory = cute::C<BASE_SHARED_SIZE * 2>;
        using OS = OSD<blocks::value>;
    };

    template<>
    struct Hardware<900, 255, Board::sxm> {
        using arch = cute::C<900U>;
        using registers = cute::C<255U>;
        using blocks = cute::C<2U * 132>;
        using bKBase = cute::C<8U>;
        using rScratch = cute::C<64U>;
        using pipeStages = cute::C<BASE_PIPE_STAGES * 2>;
        using sharedMemory = cute::C<BASE_SHARED_SIZE * 2>;
        using OS = OSD<blocks::value>;
    };

    template<>
    struct Hardware<900, 255, Board::pcie> {
        using arch = cute::C<900U>;
        using registers = cute::C<255U>;
        using blocks = cute::C<2U * 114>;
        using bKBase = cute::C<8U>;
        using rScratch = cute::C<64U>;
        using pipeStages = cute::C<BASE_PIPE_STAGES * 2>;
        using sharedMemory = cute::C<BASE_SHARED_SIZE * 2>;
        using OS = OSD<blocks::value>;
    };

    template<>
    struct Hardware<900, 96, Board::sxm> {
        using arch = cute::C<900U>;
        using registers = cute::C<96U>;
        using blocks = cute::C<5U * 132>;
        using bKBase = cute::C<8U>;
        using rScratch = cute::C<64U>;
        using pipeStages = cute::C<BASE_PIPE_STAGES * 2>;
        using sharedMemory = cute::C<BASE_SHARED_SIZE * 2>;
        using OS = OSD<blocks::value>;
    };

    template<>
    struct Hardware<900, 96, Board::pcie> {
        using arch = cute::C<900U>;
        using registers = cute::C<96U>;
        using blocks = cute::C<5U * 114>;
        using bKBase = cute::C<8U>;
        using rScratch = cute::C<64U>;
        using pipeStages = cute::C<BASE_PIPE_STAGES * 2>;
        using sharedMemory = cute::C<BASE_SHARED_SIZE * 2>;
        using OS = OSD<blocks::value>;
    };

}
#endif //ARCH_CUH
