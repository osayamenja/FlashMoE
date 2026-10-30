// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda/std/bit>

#include "../include/flashmoe/gate.cuh"

constexpr int rows = 128;
constexpr int threads = 128;

void check(cudaError_t result) {
  if (result != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(result));
    std::exit(1);
  }
}

__host__ __device__
bool chosen(int token, int expert, int experts, int k, int pattern) {
  const int first = pattern == 0 ? 0 : token % experts;
  return (expert - first + experts) % experts < k;
}

__host__ __device__
float probability(int token, int expert) {
  if (token % 7 == 0) {
    return 0.f;
  }
  return cuda::std::bit_cast<float>(
    0x3e800000u | ((static_cast<unsigned>(token) & 65535u) << 4) | (expert & 15));
}

template<int columns>
__global__ void capacity(const flashmoe::gate::GateKernelArgs args,
                         const flashmoe::GateContext ctx, int pattern, bool reverse) {
  const int tilesM = (args.S + rows - 1) / rows;
  const int tiles = tilesM * (args.E / columns);
  for (int index = blockIdx.x; index < tiles; index += gridDim.x) {
    const int tile = reverse ? tiles - index - 1 : index;
    const int tileM = tile % tilesM;
    const int tileN = tile / tilesM;
    const int token = tileM * rows + threadIdx.x;
    int selected[columns];
    float probabilities[columns];
    for (int i = 0; i < columns; ++i) {
      const int expert = tileN * columns + i;
      selected[i] = token < args.S && chosen(token, expert, args.E, args.k, pattern);
      probabilities[i] = probability(token, expert);
    }
    flashmoe::gate::stageTile<rows, columns>(
      args, ctx, tileM, tileN, selected, probabilities, 1.0f);
  }
}

template<int columns>
void run(int tokens, int experts, int k, int limit, cudaStream_t stream, int sms) {
  const int stride = std::max(rows, (limit + rows - 1) / rows * rows);
  const size_t size = static_cast<size_t>(experts) * stride + 2;
  flashmoe::TPS *output, *staging;
  int *counts;
  check(cudaMallocAsync(&output, size * sizeof(*output), stream));
  check(cudaMallocAsync(&staging, sizeof(*staging) * (experts * tokens + 2), stream));
  check(cudaMemsetAsync(staging, 0xff, sizeof(*staging) * (experts * tokens + 2), stream));
  check(cudaMallocAsync(&counts, sizeof(*counts) * experts, stream));
  const flashmoe::GateContext ctx{
    .staging = staging + 1
  };
  const flashmoe::gate::GateKernelArgs args{
    .expertCounts = counts, .tokenIds = output + 1,
    .S = tokens, .E = experts, .k = k, .EC = limit, .roundEC = stride
  };

  std::vector<flashmoe::TPS> expected(size), actual(size);
  std::vector<int> demand(experts), actualDemand(experts);
  flashmoe::TPS stagingGuards[2];
  int launches = 0;
  // Reuse the same context across changing inputs, schedules and CUDA graph replays.
  for (int pattern : {0, 1, 0}) {
    std::memset(expected.data(), 0xff, size * sizeof(expected[0]));
    std::fill(demand.begin(), demand.end(), 0);
    for (int token = 0; token < tokens; ++token) {
      for (int expert = 0; expert < experts; ++expert) {
        if (chosen(token, expert, experts, k, pattern)) {
          const int slot = demand[expert]++;
          if (slot < limit) {
            expected[1 + static_cast<size_t>(expert) * stride + slot] =
              flashmoe::TPS{static_cast<unsigned>(token), probability(token, expert)};
          }
        }
      }
    }
    for (int blocks : {1, 3, sms * 2 + 1}) {
      for (bool reverse : {false, true}) {
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        check(cudaMemsetAsync(output, 0xff, size * sizeof(*output), stream));
        check(cudaMemsetAsync(counts, 0xa5, experts * sizeof(*counts), stream));
        capacity<columns><<<blocks, threads, 0, stream>>>(args, ctx, pattern, reverse);
        check(cudaGetLastError());
        flashmoe::gate::capacityKernel<<<experts, flashmoe::gate::capacityThreads, 0, stream>>>(
          args, ctx.staging);
        check(cudaGetLastError());
        check(cudaStreamEndCapture(stream, &graph));
        check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        for (int repeat = 0; repeat < 3; ++repeat) {
          check(cudaGraphLaunch(executable, stream));
          check(cudaMemcpyAsync(actual.data(), output, size * sizeof(*output),
                                cudaMemcpyDeviceToHost, stream));
          check(cudaMemcpyAsync(actualDemand.data(), counts, experts * sizeof(*counts),
                                cudaMemcpyDeviceToHost, stream));
          check(cudaMemcpyAsync(stagingGuards, staging, sizeof(*staging), cudaMemcpyDeviceToHost, stream));
          check(cudaMemcpyAsync(stagingGuards + 1, staging + experts * tokens + 1,
                                sizeof(*staging), cudaMemcpyDeviceToHost, stream));
          check(cudaStreamSynchronize(stream));
          if (actualDemand != demand ||
              std::memcmp(actual.data(), expected.data(), size * sizeof(*output)) != 0 ||
              std::memcmp(stagingGuards, expected.data(), sizeof(*staging)) != 0 ||
              std::memcmp(stagingGuards + 1, expected.data(), sizeof(*staging)) != 0) {
            std::fprintf(stderr, "Capacity mismatch S=%d E=%d k=%d EC=%d blocks=%d reverse=%d pattern=%d\n",
                         tokens, experts, k, limit, blocks, reverse, pattern);
            std::exit(1);
          }
          ++launches;
        }
        check(cudaGraphExecDestroy(executable));
        check(cudaGraphDestroy(graph));
      }
    }
  }
  std::printf("Passed S=%d E=%d k=%d EC=%d: %d launches\n", tokens, experts, k, limit, launches);
  check(cudaFreeAsync(output, stream));
  check(cudaFreeAsync(staging, stream));
  check(cudaFreeAsync(counts, stream));
  check(cudaStreamSynchronize(stream));
}

int main() {
  int device, sms;
  check(cudaGetDevice(&device));
  check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
  cudaStream_t stream;
  check(cudaStreamCreate(&stream));
  run<8>(64, 8, 1, 0, stream, sms);
  run<8>(256, 8, 1, 32, stream, sms);
  run<8>(384, 8, 3, 129, stream, sms);
  run<8>(384, 8, 8, 384, stream, sms);
  run<32>(512, 64, 4, 17, stream, sms);
  run<32>(rows * (threads + 5) + 17, 64, 2, 129, stream, sms);
  check(cudaStreamDestroy(stream));
}
