//
// Created by Osayamen on 1/18/26.
//

#ifndef FLASHMOE_CONSTANTS_CUH
#define FLASHMOE_CONSTANTS_CUH
namespace flashmoe
{
#if (__CUDA_ARCH__ >= 1000) && (defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ >= 12) && (defined(__CUDACC_VER_MINOR__) && __CUDACC_VER_MINOR__ >= 9)
    constexpr int MAX_ACCESS_ALIGNMENT = 32;
#else
    constexpr int MAX_ACCESS_ALIGNMENT = 16;
#endif
    constexpr int WARP_SIZE = 32;
}

namespace flashmoe::scheduler
{
    constexpr int SCHEDULER_COUNT = WARP_SIZE; // warp_size
    // register state sizes
#if defined(FLASHMOE_ARCH)
    #if FLASHMOE_ARCH >= 1000
        constexpr int PROCESSOR_STATE_SIZE = 10;
    #elif FLASHMOE_ARCH >= 900
        constexpr int PROCESSOR_STATE_SIZE = 9;
    #else
        constexpr int PROCESSOR_STATE_SIZE = 8;
    #endif
#else
    constexpr int PROCESSOR_STATE_SIZE = 10; // 8 is recommended
#endif
    static_assert(PROCESSOR_STATE_SIZE <= 64);
    constexpr int MAX_PROCESSORS = WARP_SIZE * PROCESSOR_STATE_SIZE; // can be relaxed but with slower perf
    constexpr int WORK_SET_SIZE = 4;
    constexpr int QUEUE_STATE_SIZE = 2;
}
#endif //FLASHMOE_CONSTANTS_CUH