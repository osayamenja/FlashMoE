/******************************************************************************
 * Copyright (c) 2024, Osayamen Jonathan Aimuyo.
 ******************************************************************************/

#include "include/bootstrap.cuh"
#include "include/moe/moe.cuh"
#include "correctness.cuh"
//#include "eval.cuh"
int main() {
    /*aristos::testTopologyDiscovery();*/
    using clk = std::chrono::high_resolution_clock;
    std::chrono::duration<float> end {};
    const auto start = clk::now();
    aristos::initialize();
    end = clk::now() - start;
    printf("Initialize takes %fms\n", end.count() * 1000);
    aristos::finalize();
    aristos::evalExpert();
}
