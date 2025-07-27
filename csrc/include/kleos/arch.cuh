/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Kleos Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 */

//
// Created by oja7 on 12/22/24.
//

#ifndef ARCH_CUH
#define ARCH_CUH

#include <cute/numeric/integral_constant.hpp>
#define BASE_SHARED_SIZE 16 * 1024U
#define BASE_PIPE_STAGES 2U
#define MAX_THREADS_PER_SM 2048U
namespace kleos {
    template<unsigned int arch>
    concept SupportedArch = arch >= 700;

    __device__
    // Uses 48KB as upper bound rather than hardware maximum.
    // Avoids setting cudaFuncAttribute
    enum class UseSharedBound {
        yes,
        no
    };

    template<
        unsigned int Arch,
        unsigned int blocksPerSM,
        unsigned int sharedSlice,
        UseSharedBound u
    >
    requires(blocksPerSM <= MAX_THREADS_PER_SM / 128U && blocksPerSM > 0
        && sharedSlice == BASE_SHARED_SIZE || sharedSlice == BASE_SHARED_SIZE * 2)
    struct ArchShared {
        static_assert(Arch == 700 || Arch == 800 || Arch == 900, "Unregistered Config!");
    };

    template<
        unsigned int blocksPerSM,
        unsigned int sharedSlice,
        UseSharedBound u
    >
    struct ArchShared<700, blocksPerSM, sharedSlice, u> {
        using Max = cute::C<96 * 1024U>;
        using Spare = cute::C<u == UseSharedBound::yes ?
            cute::min(48 * 1024U - (sharedSlice + 1024U),
                (Max::value - blocksPerSM * (sharedSlice + 1024U)) / blocksPerSM)
                    :(Max::value - blocksPerSM * (sharedSlice + 1024U)) / blocksPerSM>;
    };
    template<
        unsigned int blocksPerSM,
        unsigned int sharedSlice,
        UseSharedBound u
    >
    struct ArchShared<800, blocksPerSM, sharedSlice, u> {
        using Max = cute::C<164 * 1024U>;
        using Spare = cute::C<u == UseSharedBound::yes ?
            cute::min(48 * 1024U - (sharedSlice + 1024U),
                (Max::value - blocksPerSM * (sharedSlice + 1024U)) / blocksPerSM)
                    :(Max::value - blocksPerSM * (sharedSlice + 1024U)) / blocksPerSM>;
    };

    template<
        unsigned int blocksPerSM,
        unsigned int sharedSlice,
        UseSharedBound u
    >
    struct ArchShared<900, blocksPerSM, sharedSlice, u> {
        using Max = cute::C<228 * 1024U>;
        using Spare = cute::C<u == UseSharedBound::yes ?
            cute::min(48 * 1024U - (sharedSlice + 1024U),
                (Max::value - blocksPerSM * (sharedSlice + 1024U)) / blocksPerSM)
                    :(Max::value - blocksPerSM * (sharedSlice + 1024U)) / blocksPerSM>;
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
        unsigned int maxRegisters = 128
    >
    requires (SupportedArch<Arch>)
    struct Hardware {
        static_assert(Arch == 800 && maxRegisters == 128,
            "Unregistered Arch");
        using blocksPerSM = cute::C<4U>;
        using arch = cute::C<800U>;
        using registers = cute::C<128U>;
        using blocks = cute::C<blocksPerSM::value * NUM_SMS>;
        using bKBase = cute::C<8U>;
        using rScratch = cute::C<32U>;
        using pipeStages = cute::C<BASE_PIPE_STAGES * 2>;
        using sharedMemory = cute::C<BASE_SHARED_SIZE * 2>;
        using spare = ArchShared<800U, blocksPerSM::value, sharedMemory::value, UseSharedBound::yes>::Spare;
        using OS = OSD<blocks::value>;
    };

    template<>
    struct Hardware<800, 96> {
        using blocksPerSM = cute::C<5U>;
        using arch = cute::C<800U>;
        using registers = cute::C<96U>;
        using blocks = cute::C<blocksPerSM::value * NUM_SMS>;
        using bKBase = cute::C<8U>;
        using rScratch = cute::C<32U>;
        using sharedMemory = cute::C<BASE_SHARED_SIZE>;
        using pipeStages = cute::C<BASE_PIPE_STAGES>;
        using spare = ArchShared<800U, blocksPerSM::value, sharedMemory::value, UseSharedBound::yes>::Spare;
        using OS = OSD<blocks::value>;
    };

    template<>
    struct Hardware<800, 255> {
        using blocksPerSM = cute::C<2U>;
        using arch = cute::C<800U>;
        using registers = cute::C<255U>;
        using blocks = cute::C<blocksPerSM::value * NUM_SMS>;
        using bKBase = cute::C<8U>;
        using rScratch = cute::C<64U>;
        using pipeStages = cute::C<BASE_PIPE_STAGES * 2>;
        using sharedMemory = cute::C<BASE_SHARED_SIZE * 2>;
        using spare = ArchShared<800U, blocksPerSM::value, sharedMemory::value, UseSharedBound::yes>::Spare;
        using OS = OSD<blocks::value>;
    };

    template<>
    struct Hardware<700> {
        using blocksPerSM = cute::C<4U>;
        using arch = cute::C<700U>;
        using registers = cute::C<128U>;
        using blocks = cute::C<blocksPerSM::value * NUM_SMS>;
        using bKBase = cute::C<16U>;
        using rScratch = cute::C<32U>;
        using pipeStages = cute::C<1U>;
        using sharedMemory = cute::C<BASE_SHARED_SIZE>;
        using spare = ArchShared<700U, blocksPerSM::value, sharedMemory::value, UseSharedBound::yes>::Spare;
        using OS = OSD<blocks::value>;
    };

    template<>
    struct Hardware<700, 96> {
        using blocksPerSM = cute::C<5U>;
        using arch = cute::C<700U>;
        using registers = cute::C<96U>;
        using blocks = cute::C<blocksPerSM::value * NUM_SMS>;
        using pipeStages = cute::C<1U>;
        using bKBase = cute::C<16U>;
        using rScratch = cute::C<32U>;
        using sharedMemory = cute::C<BASE_SHARED_SIZE>;
        using spare = ArchShared<700U, blocksPerSM::value, sharedMemory::value, UseSharedBound::yes>::Spare;
        using OS = OSD<blocks::value>;
    };

    template<>
    struct Hardware<700, 255> {
        // recommended
        using blocksPerSM = cute::C<2U>;
        using arch = cute::C<700U>;
        using registers = cute::C<255U>;
        using blocks = cute::C<blocksPerSM::value * NUM_SMS>;
        using pipeStages = cute::C<1U>;
        using bKBase = cute::C<32U>;
        using rScratch = cute::C<64U>;
        using sharedMemory = cute::C<BASE_SHARED_SIZE * 2>;
        using spare = ArchShared<700U, blocksPerSM::value, sharedMemory::value, UseSharedBound::yes>::Spare;
        using OS = OSD<blocks::value>;
    };

    // Hopper
    template<>
    struct Hardware<900, 128> {
        using blocksPerSM = cute::C<4U>;
        using arch = cute::C<900U>;
        using registers = cute::C<128U>;
        using blocks = cute::C<blocksPerSM::value * NUM_SMS>;
        using bKBase = cute::C<8U>;
        using rScratch = cute::C<32U>;
        using pipeStages = cute::C<BASE_PIPE_STAGES * 2>;
        using sharedMemory = cute::C<BASE_SHARED_SIZE * 2>;
        using spare = ArchShared<900U, blocksPerSM::value, sharedMemory::value, UseSharedBound::yes>::Spare;
        using OS = OSD<blocks::value>;
    };

    template<>
    struct Hardware<900, 255> {
        using blocksPerSM = cute::C<2U>;
        using arch = cute::C<900U>;
        using registers = cute::C<255U>;
        using blocks = cute::C<blocksPerSM::value * NUM_SMS>;
        using bKBase = cute::C<8U>;
        using rScratch = cute::C<64U>;
        using pipeStages = cute::C<BASE_PIPE_STAGES>;
        using sharedMemory = cute::C<BASE_SHARED_SIZE * 2>;
        using spare = ArchShared<900U, blocksPerSM::value, sharedMemory::value, UseSharedBound::yes>::Spare;
        using OS = OSD<blocks::value>;
    };

    template<>
    struct Hardware<900, 96> {
        using blocksPerSM = cute::C<5U>;
        using arch = cute::C<900U>;
        using registers = cute::C<96U>;
        using blocks = cute::C<blocksPerSM::value * NUM_SMS>;
        using bKBase = cute::C<8U>;
        using rScratch = cute::C<64U>;
        using pipeStages = cute::C<BASE_PIPE_STAGES * 2>;
        using sharedMemory = cute::C<BASE_SHARED_SIZE * 2>;
        using spare = ArchShared<900U, blocksPerSM::value, sharedMemory::value, UseSharedBound::yes>::Spare;
        using OS = OSD<blocks::value>;
    };
}
#endif //ARCH_CUH
