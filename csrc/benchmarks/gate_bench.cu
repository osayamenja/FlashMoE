//
// Created by osayamen on 12/28/25.
//
// Benchmark and unit tests for the fused gate

#include <matx.h>
#include <cublasdx.hpp>

#include "common.cuh"
int main(int argc, char** argv) {
    // use MatX as the baseline to compute the Gate.
    // GEMM + Softmax -> verify routing output
    // argsort for topk selection of expert-tokens
    // -> verify expert counts and token indices
}