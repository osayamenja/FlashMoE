//
// Created by osayamen on 12/22/25.
//

// place to experiment
#include "debug.cuh"

int main() {
  CHECK_CUDA(cudaSetDevice(0));
  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0)); // Get properties for device i
  printf("Device name: %s\n", prop.name);
}