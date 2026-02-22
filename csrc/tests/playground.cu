//
// Created by osayamen on 12/22/25.
//

// place to experiment
#include <random>
#include "common.cuh"
#include "debug.cuh"

int main() {
  constexpr auto S = 256;
  constexpr auto E = 8;
  constexpr auto k = 1;
  constexpr auto EC = cute::ceil_div(S, E) * k;
  constexpr auto bM = 128;
  constexpr auto rEC = cute::ceil_div(EC, bM) * bM;
  const auto [counts, ids] = generate_token_ids_and_expert_counts(S, E, EC, rEC, k, 0.f, 0.001f);
  printMetadata(counts, ids, rEC, E);
}