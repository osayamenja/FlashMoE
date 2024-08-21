//
// Created by Jonathan on 7/18/24.
//

#ifndef ARISTOS_PLATFORM_CUH
#define ARISTOS_PLATFORM_CUH

namespace aristos{
    template<uint capability>
    concept MinArch = requires{
        // Due to CUTLASS
        // https://github.com/NVIDIA/cutlass/blob/main/README.md
        capability >= 7;
    };
}
#endif //ARISTOS_PLATFORM_CUH
