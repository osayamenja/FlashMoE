/******************************************************************************
 * Copyright (c) 2024, Osayamen Jonathan Aimuyo.
 ******************************************************************************/

#include "include/bootstrap.cuh"
#include "include/types.cuh"
int main() {
    using Element = cute::half_t;
    // GPT3-MoE 1.3B
    const auto iC = aristos::InitialConfig{
        50 * 1024,
        24,
        512,
        4,
        2,
        2048,
        2048,
        2048 * 4,
        1,
        1,
        16,
        true,
        true,
    };
    aristos::initialize<Element>(iC);
    //aristos::moe::forwardHost();
    aristos::finalize();
    return 0;
}
