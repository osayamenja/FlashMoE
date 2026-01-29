//
// Created by Osayamen on 1/18/26.
//

#ifndef FLASHMOE_CONSTANTS_CUH
#define FLASHMOE_CONSTANTS_CUH
namespace flashmoe
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
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
    constexpr int PROCESSOR_STATE_SIZE = 8; // 8 is recommended
    static_assert(PROCESSOR_STATE_SIZE <= 64);
    constexpr int MAX_PROCESSORS = WARP_SIZE * PROCESSOR_STATE_SIZE; // can be relaxed but with slower perf
    constexpr int WORK_SET_SIZE = 4;
    constexpr int QUEUE_STATE_SIZE = 2;
}
#endif //FLASHMOE_CONSTANTS_CUH