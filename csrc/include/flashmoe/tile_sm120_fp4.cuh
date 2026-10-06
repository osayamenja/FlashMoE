/******************************************************************************
 * Copyright (c) 2026, Osayamen Jonathan Aimuyo.
 ******************************************************************************/
//
// Created by FlashMoE contributors on 10/5/26.
//
// Block-scaled FP4 tile GEMM for sm_120a (RTX 50 series, RTX PRO 6000 Blackwell, DGX Spark), for the two W4A4 recipes
//  - mxfp4: E2M1 elements + one UE8M0 scale per 32 K-elements, mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X
//  - nvfp4: E2M1 elements + one UE4M3 scale per 16 K-elements, mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X
//  both m16n8k64, see infra/fp4.cuh for the formats.
//  - FP4Mainloop: the structure of MXFP8Mainloop (tile_sm120_mxfp8.cuh): one CTA computes one bM x bN output tile
//    (gated: two bM x bN tiles, A is read once) with 256 threads = 8 warps (4 along M x 2 along N), cp.async multistage
//    pipeline, ldmatrix fragments from swizzled shared memory and scale factors (SF) straight out of the canonical
//    128x4 layout, see infra/mx.cuh. An m16n8k64 FP4 A fragment (32 E2M1 per thread) occupies the same bytes of the
//    same threads as an m16n8k32 FP8 fragment, so the shared memory tiles are byte for byte those of the MXFP8 mainloop
//    with a row of bK / 2 bytes.
//  - epilogues that work on the accumulator fragments: BF16 output, FP4 output (the A operand of the next GEMM) and
//    the fused gated variant. The global scales of nvfp4 enter through alpha (decode: gsA * gsB) and st (encode scale
//    of the output tensor, 1 / gs).
//
// The SF handling, the smem layouts and the tiled MMA follow cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp
// and builders/sm120_blockscaled_mma_builder.inl (the FP4, i.e., non mxf8f6f4, configuration), CUTLASS 4.4.1:
//
// Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
// following conditions are met: 1. Redistributions of source code must retain the above copyright notice, this list of
// conditions and the following disclaimer. 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation and/or other materials provided with the
// distribution. 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or
// promote products derived from this software without specific prior written permission.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
// INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef FLASHMOE_TILE_SM120_FP4_CUH
#define FLASHMOE_TILE_SM120_FP4_CUH

#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda/std/type_traits>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/epilogue/thread/activation.h>
#include <cute/tensor.hpp>
#include <cute/tensor_zip.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>

#include "infra/fp4.cuh"
#include "infra/mx.cuh"
#include "infra/structures.cuh"
#include "tile_sm120_mxfp8.cuh" // namespace detail: SF fragment partitioning and the epilogue helpers

namespace flashmoe::tile::sm120 {
  namespace detail {
    /// Transposes a 4 x 4 matrix of bytes that is distributed over the 4 lanes of a quad (lane % 4 = row), lane s
    /// holding its row as w = e[s][0] | e[s][1] << 8 | e[s][2] << 16 | e[s][3] << 24: afterwards lane q holds
    /// w = e[0][q] | e[1][q] << 8 | e[2][q] << 16 | e[3][q] << 24. Two exchange stages, 2 shuffles and 2 byte permutes.
    __device__ __forceinline__
    uint32_t quadTransposeBytes(const uint32_t& w) {
      const bool b0 = (threadIdx.x & 1U) != 0;
      const bool b1 = (threadIdx.x & 2U) != 0;
      // stage 1: swap the off-diagonal 2 x 2 blocks of bytes, lanes differ in bit 1
      const uint32_t r = __shfl_xor_sync(0xFFFFFFFFU, w, 2);
      const uint32_t n = __byte_perm(w, r, b1 ? 0x3276U : 0x5410U);
      // stage 2: transpose the 2 x 2 blocks, lanes differ in bit 0
      const uint32_t u = __shfl_xor_sync(0xFFFFFFFFU, n, 1);
      return __byte_perm(n, u, b0 ? 0x3715U : 0x6240U);
    }
  }

  /// Block-scaled FP4 tile GEMM, C[bM, bN] = A[bM, K] @ B[bN, K]^T with FP32 accumulation, where
  ///  - A is a row-major [M, K] E2M1 matrix (K-major, 2 elements per byte, the even k in the low nibble, K / 2 bytes per
  ///    row) and B a row-major [N, K] E2M1 matrix (K-major),
  ///  - sfa / sfb are their scale factors, one per 32 (mxfp4, UE8M0) or 16 (nvfp4, UE4M3) consecutive K elements of a
  ///    row, in the canonical 128x4 layout of mx::sfOffset() of the *whole matrix* (row 0 of the matrix is row 0 of the
  ///    SF buffer, k-block = k / sfVecSize),
  ///  - gated == true computes two accumulators from one A tile, acc = A @ B^T and acc2 = A @ B2^T, with B2 / sfb2 of
  ///    the same shape and layout as B / sfb.
  /// bM = 128 (one SF atom), threads = 256, bN is 64 or a multiple of 128 (bN of *each* B matrix when gated), bK is the
  /// number of K elements of a pipeline stage.
  /// Runtime requirements: M % bM == 0, N % bN == 0, K % bK == 0.
  ///
  /// The accumulator fragments follow a "Perm64" N layout of the tiled MMA: of every 64 columns of the tile, warp
  /// wn in {0, 1} owns the contiguous columns [32 wn, 32 wn + 32), i.e., exactly one mxfp4 / two nvfp4 scale blocks of
  /// the output, and the 4 lanes of a quad (same lane / 4) hold the 32 values of one row of those, which is what the
  /// quantizing epilogues exploit.
  ///
  /// The mainloop and the epilogues may be called repeatedly by the same CTA (persistent kernels) for different tiles
  /// and different matrices. All state is in the workspace (dynamic shared memory) and every cp.async group is
  /// drained before operator() returns. The mainloop begins with a block-wide barrier, so the workspace may be
  /// overwritten by anyone between calls; the epilogues do not touch the workspace. All threads of the CTA must call
  /// everything below, and the 256 participating threads are threadIdx.x in [0, 256).
  template <QuantRecipe r, int bM, int bN, int bK, int pipeStages, int threads, bool gated>
    requires(isFP4(r) && bM == 128 && threads == 256 && (bN == 64 || bN % 128 == 0) && (bK == 128 || bK == 256) &&
             pipeStages >= 2)
  struct FP4Mainloop {
    static constexpr QuantRecipe recipe = r;
    using ElementAB = cutlass::float_e2m1_t;
    using ElementSF = cuda::std::conditional_t<r == QuantRecipe::nvfp4, cutlass::float_ue4m3_t,
                                               cutlass::float_ue8m0_t>;
    using AccumType = float;
    static constexpr bool usesTma = false;
    static constexpr int VS = quantSFVecSize(r); // K elements per scale factor
    static constexpr int NSF = 64 / VS; // scale factors of one MMA (K = 64) along K
    using Threads = cute::Int<threads>;
    using TileShape = cute::Shape<cute::Int<bM>, cute::Int<bN>, cute::Int<bK>>;
    using PipeStages = cute::Int<pipeStages>;
    using MmaOp = cute::SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<ElementAB, ElementAB, float, ElementSF, VS>;
    // 4 x 2 warps. PermTileN = Perm64: warp wn owns 32 contiguous columns of every 64
    using PermN = cute::Layout<cute::Shape<cute::_8, cute::_2, cute::_4>, cute::Stride<cute::_1, cute::_32, cute::_8>>;
    using TiledMma = decltype(cute::make_tiled_mma(MmaOp{}, cute::Layout<cute::Shape<cute::_4, cute::_2, cute::_1>>{},
                                                   cute::Tile<cute::_128, PermN, cute::_64>{}));
    /// accumulator fragment of one bM x bN tile
    using Accumulator = decltype(cute::partition_fragment_C(TiledMma{}, cute::Shape<cute::Int<bM>, cute::Int<bN>>{}));

    // accumulator fragment: 4 * MMA_M * MMA_N values, element (v, i, j) at v + 4 * (i + MMA_M * j) with
    // v = 2 * rowHalf + c: row = 16 * (warp % 4) + lane / 4 + 8 * rowHalf + 64 * i,
    //                      col = 64 * (j / 4) + 32 * (warp / 4) + 8 * (j % 4) + 2 * (lane % 4) + c
    static constexpr int MMA_M = bM / 64;
    static constexpr int MMA_N = bN / 16;
    static constexpr int MMA_K = bK / 64;

    static constexpr int NB = gated ? 2 : 1; // number of B operands
    static constexpr int rowBytes = bK / 2; // bytes of a row of a k-tile
    static constexpr int kAtoms = bK / (mx::SF_K_TILE * VS); // SF atoms along K of a k-tile: 1, 2 or 4
    static constexpr int sfbAtoms = (bN + 127) / 128; // SF atoms (128 rows) of one B operand
    static constexpr int sfUnits = 1 + NB * sfbAtoms; // [A | B | B2], one unit = the kAtoms atoms of 128 rows
    static constexpr int sfChunksPerUnit = mx::SF_ATOM_BYTES * kAtoms / 16; // 16 B cp.async chunks
    static constexpr int sfChunks = sfChunksPerUnit * sfUnits;
    static constexpr int sfCopyIters = (sfChunks + threads - 1) / threads;
    static constexpr int bytesA = bM * rowBytes * pipeStages;
    static constexpr int bytesB = bN * rowBytes * pipeStages;
    static constexpr int sfBytesStage = mx::SF_ATOM_BYTES * kAtoms * sfUnits;
    static constexpr int offA = 0;
    static constexpr int offB = offA + bytesA;
    static constexpr int offB2 = offB + bytesB; // gated only
    static constexpr int offSF = offB + NB * bytesB;
    static constexpr int dataBytes = offSF + sfBytesStage * pipeStages;
    /// The workspace is aligned to 1024 B inside (128 B swizzle, TMA friendly), which costs up to 1023 B.
    static constexpr int Alignment = 1024;
    using SharedSize = cute::Int<dataBytes + Alignment>;
    static constexpr int sharedSize = SharedSize::value;

    // SMEM layouts: K-major, swizzled so that ldmatrix is conflict free. The MMA fragments are partitioned from
    // layouts over 4-bit elements, the cp.async copies work on the same bytes through layouts over bytes.
    using SmemAtom = cuda::std::conditional_t<bK == 256, cute::GMMA::Layout_K_SW128_Atom<cute::uint4_t>,
                                              cute::GMMA::Layout_K_SW64_Atom<cute::uint4_t>>;
    using SmemAtomBytes = cuda::std::conditional_t<bK == 256, cute::GMMA::Layout_K_SW128_Atom<uint8_t>,
                                                   cute::GMMA::Layout_K_SW64_Atom<uint8_t>>;
    using SmemLayoutA = decltype(cute::tile_to_shape(SmemAtom{},
      cute::make_shape(cute::Int<bM>{}, cute::Int<bK>{}, cute::Int<pipeStages>{}),
      cute::Step<cute::_1, cute::_2, cute::_3>{}));
    using SmemLayoutB = decltype(cute::tile_to_shape(SmemAtom{},
      cute::make_shape(cute::Int<bN>{}, cute::Int<bK>{}, cute::Int<pipeStages>{}),
      cute::Step<cute::_1, cute::_2, cute::_3>{}));
    using SmemLayoutABytes = decltype(cute::tile_to_shape(SmemAtomBytes{},
      cute::make_shape(cute::Int<bM>{}, cute::Int<rowBytes>{}, cute::Int<pipeStages>{}),
      cute::Step<cute::_1, cute::_2, cute::_3>{}));
    using SmemLayoutBBytes = decltype(cute::tile_to_shape(SmemAtomBytes{},
      cute::make_shape(cute::Int<bN>{}, cute::Int<rowBytes>{}, cute::Int<pipeStages>{}),
      cute::Step<cute::_1, cute::_2, cute::_3>{}));
    // SF: (row, k) -> byte inside the 512 B canonical atoms of a stage: (row % 32) * 16 + (row / 32 % 4) * 4 +
    // kBlock % 4 (+ 512 * (kBlock / 4)), 128 rows per atom, the VS K elements of a block share the byte (stride 0).
    // The K mode is ((VS, NSF), (4 / NSF, kAtoms)): one MMA, then the MMAs inside an atom, then the atoms.
    using SmemShapeSFK = cute::Shape<cute::Shape<cute::Int<VS>, cute::Int<NSF>>,
                                     cute::Shape<cute::Int<mx::SF_K_TILE / NSF>, cute::Int<kAtoms>>>;
    using SmemLayoutSFA = cute::Layout<
      cute::Shape<cute::Shape<cute::_32, cute::_4>, SmemShapeSFK>,
      cute::Stride<cute::Stride<cute::_16, cute::_4>,
                   cute::Stride<cute::Stride<cute::_0, cute::_1>, cute::Stride<cute::Int<NSF>,
                                                                                cute::Int<mx::SF_ATOM_BYTES>>>>>;
    using SmemLayoutSFB = cute::Layout<
      cute::Shape<cute::Shape<cute::_32, cute::Int<(bN < 128 ? bN : 128) / 32>, cute::Int<sfbAtoms>>, SmemShapeSFK>,
      cute::Stride<cute::Stride<cute::_16, cute::_4, cute::Int<mx::SF_ATOM_BYTES * kAtoms>>,
                   cute::Stride<cute::Stride<cute::_0, cute::_1>, cute::Stride<cute::Int<NSF>,
                                                                                cute::Int<mx::SF_ATOM_BYTES>>>>>;

    // gmem -> smem: 16 B cp.async per thread
    using GmemCopyAtom = cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, uint8_t>;
    static constexpr int chunksPerRow = rowBytes / 16;
    using GmemThr = cute::Layout<cute::Shape<cute::Int<threads / chunksPerRow>, cute::Int<chunksPerRow>>,
                                 cute::Stride<cute::Int<chunksPerRow>, cute::_1>>;
    using GmemVal = cute::Layout<cute::Shape<cute::_1, cute::_16>>;
    using GmemCopy = decltype(cute::make_tiled_copy(GmemCopyAtom{}, GmemThr{}, GmemVal{}));

    /// accumulator fragment, uninitialized (the mainloop clears it)
    __device__ __forceinline__
    static Accumulator makeAccumulator() {
      return cute::partition_fragment_C(TiledMma{}, cute::Shape<cute::Int<bM>, cute::Int<bN>>{});
    }

    /// Single B operand: acc = A[bM tile mTile] @ B[bN tile nTile]^T. tileCoord = (mTile, nTile, _), see
    /// tile::idx2Coord. a, b: E2M1 bytes, workspace: dynamic shared memory of at least sharedSize bytes.
    template <typename TileCoord>
      requires(!gated && cute::rank_v<TileCoord> == 3)
    __device__ __forceinline__
    void operator()(void* __restrict__ const& workspace,
                    const uint8_t* __restrict__ const& a, const uint8_t* __restrict__ const& sfa,
                    const uint8_t* __restrict__ const& b, const uint8_t* __restrict__ const& sfb,
                    Accumulator& acc, const int& M, const int& N, const int& K, const TileCoord& tileCoord) const {
      mainloop(workspace, a, sfa, b, sfb, nullptr, nullptr, acc, acc, M, N, K, tileCoord);
    }

    /// Gated: acc = A @ B^T, acc2 = A @ B2^T for the same (mTile, nTile) tile of both B matrices.
    template <typename TileCoord>
      requires(gated && cute::rank_v<TileCoord> == 3)
    __device__ __forceinline__
    void operator()(void* __restrict__ const& workspace,
                    const uint8_t* __restrict__ const& a, const uint8_t* __restrict__ const& sfa,
                    const uint8_t* __restrict__ const& b, const uint8_t* __restrict__ const& sfb,
                    const uint8_t* __restrict__ const& b2, const uint8_t* __restrict__ const& sfb2,
                    Accumulator& acc, Accumulator& acc2, const int& M, const int& N, const int& K,
                    const TileCoord& tileCoord) const {
      mainloop(workspace, a, sfa, b, sfb, b2, sfb2, acc, acc2, M, N, K, tileCoord);
    }

    // ------------------------------------------------------------------------------------------------------------
    // epilogues. alpha (and alphaG, alphaU): the factor of the accumulator that undoes the tensor scales, gsA * gsB for
    // nvfp4, 1 for mxfp4. st: the encode scale 1 / gs of the output tensor for nvfp4, unused for mxfp4.
    // ------------------------------------------------------------------------------------------------------------

    /// out[M, N] (row-major) tile = act(alpha * acc + bias). bias: [N] or nullptr. Vanilla GEMM1 epilogue.
    template <typename Act, typename ElementC, typename TileCoord>
      requires(!gated && detail::EpilogueElement<ElementC> && cute::rank_v<TileCoord> == 3)
    __device__ __forceinline__
    static void epilogueBF16(const Accumulator& acc, ElementC* __restrict__ const& out,
                             const ElementC* __restrict__ const& bias, const AccumType& alpha,
                             const int& M, const int& N, const TileCoord& tileCoord) {
      const auto origin = threadOrigin();
      const int m0 = cute::get<0>(tileCoord) * bM + origin.row;
      const int n0 = cute::get<1>(tileCoord) * bN;
      float2 bv[MMA_N];
      #pragma unroll
      for (int j = 0; j < MMA_N; ++j) {
        bv[j] = bias == nullptr ? make_float2(0.f, 0.f) :
          detail::loadPair(bias + n0 + origin.col + colOffset(j));
      }
      #pragma unroll
      for (int i = 0; i < MMA_M; ++i) {
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
          ElementC* const rowOut = out + static_cast<size_t>(m0 + 64 * i + 8 * h) * N + n0 + origin.vcol;
          #pragma unroll
          for (int jb = 0; jb < MMA_N / 4; ++jb) {
            // 4 words (column pairs) of this thread -> after the quad transpose, 16 contiguous bytes of the row
            uint32_t w[4];
            #pragma unroll
            for (int a = 0; a < 4; ++a) {
              w[a] = detail::packPair(out,
                detail::activate<Act>(fmaf(alpha, acc(accIdx(0, h, i, 4 * jb + a)), bv[4 * jb + a].x)),
                detail::activate<Act>(fmaf(alpha, acc(accIdx(1, h, i, 4 * jb + a)), bv[4 * jb + a].y)));
            }
            detail::quadTranspose(w);
            *reinterpret_cast<uint4*>(rowOut + 64 * jb) = make_uint4(w[0], w[1], w[2], w[3]);
          }
        }
      }
    }

    /// q[M, N / 2] (E2M1, row-major) and sfOut (canonical SF layout of an [M, N] matrix, N / VS k-blocks per row)
    /// = FP4(act(alpha * acc + bias)), i.e., the A operand of the next GEMM. bias: [N] or nullptr. acc is overwritten.
    template <typename Act, typename ElementC, typename TileCoord>
      requires(!gated && detail::EpilogueElement<ElementC> && cute::rank_v<TileCoord> == 3)
    __device__ __forceinline__
    static void epilogueFP4(Accumulator& acc, uint8_t* __restrict__ const& q, uint8_t* __restrict__ const& sfOut,
                            const ElementC* __restrict__ const& bias, const AccumType& alpha, const AccumType& st,
                            const int& M, const int& N, const TileCoord& tileCoord) {
      const auto origin = threadOrigin();
      const int n0 = cute::get<1>(tileCoord) * bN + origin.col;
      #pragma unroll
      for (int j = 0; j < MMA_N; ++j) {
        const float2 bv = bias == nullptr ? make_float2(0.f, 0.f) : detail::loadPair(bias + n0 + colOffset(j));
        #pragma unroll
        for (int i = 0; i < MMA_M; ++i) {
          #pragma unroll
          for (int h = 0; h < 2; ++h) {
            acc(accIdx(0, h, i, j)) = detail::activate<Act>(fmaf(alpha, acc(accIdx(0, h, i, j)), bv.x));
            acc(accIdx(1, h, i, j)) = detail::activate<Act>(fmaf(alpha, acc(accIdx(1, h, i, j)), bv.y));
          }
        }
      }
      quantizeStore(acc, q, sfOut, st, N, tileCoord);
    }

    /// Gated epilogue of GEMM0, the semantics of processor::fGET_gated, but in FP32 (no BF16 rounding of the
    /// activated gate in between):
    ///   y = swishAlpha * act(swishBeta * (alphaG * acc + bias)) * (alphaU * acc2 + biasV)
    /// followed by the FP4 quantization of y along N as in epilogueFP4. acc is overwritten.
    template <typename Act, typename ElementC, typename TileCoord>
      requires(gated && detail::EpilogueElement<ElementC> && cute::rank_v<TileCoord> == 3)
    __device__ __forceinline__
    static void epilogueGatedFP4(Accumulator& acc, const Accumulator& acc2,
                                 uint8_t* __restrict__ const& q, uint8_t* __restrict__ const& sfOut,
                                 const ElementC* __restrict__ const& bias, const ElementC* __restrict__ const& biasV,
                                 const AccumType& alphaG, const AccumType& alphaU,
                                 const AccumType& swishAlpha, const AccumType& swishBeta, const AccumType& st,
                                 const int& M, const int& N, const TileCoord& tileCoord) {
      const auto origin = threadOrigin();
      const int n0 = cute::get<1>(tileCoord) * bN + origin.col;
      #pragma unroll
      for (int j = 0; j < MMA_N; ++j) {
        const float2 bg = bias == nullptr ? make_float2(0.f, 0.f) : detail::loadPair(bias + n0 + colOffset(j));
        const float2 bu = biasV == nullptr ? make_float2(0.f, 0.f) : detail::loadPair(biasV + n0 + colOffset(j));
        #pragma unroll
        for (int i = 0; i < MMA_M; ++i) {
          #pragma unroll
          for (int h = 0; h < 2; ++h) {
            acc(accIdx(0, h, i, j)) = swishAlpha *
              detail::activate<Act>(fmaf(alphaG, acc(accIdx(0, h, i, j)), bg.x) * swishBeta) *
              fmaf(alphaU, acc2(accIdx(0, h, i, j)), bu.x);
            acc(accIdx(1, h, i, j)) = swishAlpha *
              detail::activate<Act>(fmaf(alphaG, acc(accIdx(1, h, i, j)), bg.y) * swishBeta) *
              fmaf(alphaU, acc2(accIdx(1, h, i, j)), bu.y);
          }
        }
      }
      quantizeStore(acc, q, sfOut, st, N, tileCoord);
    }

  private:
    struct Origin {
      int row; // row of the accumulator elements (v = 0, i = 0)
      int col; // column of the accumulator element (v = 0, j = 0)
      int vcol; // first column of the 8 consecutive columns the thread stores after the quad transpose
    };

    __device__ __forceinline__
    static Origin threadOrigin() {
      const int warp = static_cast<int>(threadIdx.x) / 32;
      const int lane = static_cast<int>(threadIdx.x) % 32;
      return {16 * (warp % 4) + lane / 4, 32 * (warp / 4) + 2 * (lane % 4), 32 * (warp / 4) + 8 * (lane % 4)};
    }

    // linear index of accumulator element (v, i, j), v = 2 * rowHalf + c
    __device__ __forceinline__
    static constexpr int accIdx(const int v, const int h, const int i, const int j) {
      return v + 2 * h + 4 * (i + MMA_M * j);
    }

    // column offset (w.r.t. the thread's origin column) of the MMA_N index j
    __device__ __forceinline__
    static constexpr int colOffset(const int j) {
      return 64 * (j / 4) + 8 * (j % 4);
    }

    /// FP4 quantization of the fragment y along N: for every row, every VS consecutive columns share a scale. The 32
    /// columns [64 jb + 32 wn, +32) of a row are the 8 values (4 atoms of 2) of the 4 lanes of a quad: mxfp4 has one
    /// scale for them, nvfp4 one for the atoms {0, 1} and one for {2, 3}. amax is thread-local and then reduced with 2
    /// xor shuffles. Each lane converts its 8 values to 4 bytes (atom a = columns 8 a + 2 q, 8 a + 2 q + 1 of a
    /// lane q) and a 4 x 4 byte transpose over the quad leaves lane q with the 4 bytes of the columns 8 q .. 8 q + 7.
    template <typename TileCoord>
    __device__ __forceinline__
    static void quantizeStore(const Accumulator& y, uint8_t* __restrict__ const& q, uint8_t* __restrict__ const& sfOut,
                              const AccumType& st, const int& N, const TileCoord& tileCoord) {
      const auto origin = threadOrigin();
      const int lane = static_cast<int>(threadIdx.x) % 32;
      const int wn = static_cast<int>(threadIdx.x) / 128;
      const int m0 = cute::get<0>(tileCoord) * bM + origin.row;
      const int nTile0 = cute::get<1>(tileCoord) * bN;
      const uint32_t numKBlocks = static_cast<uint32_t>(N / VS);
      #pragma unroll
      for (int i = 0; i < MMA_M; ++i) {
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int row = m0 + 64 * i + 8 * h;
          uint8_t* const rowQ = q + static_cast<size_t>(row) * (N / 2) + (nTile0 + origin.vcol) / 2;
          #pragma unroll
          for (int jb = 0; jb < MMA_N / 4; ++jb) {
            float x[8];
            #pragma unroll
            for (int a = 0; a < 4; ++a) {
              x[2 * a] = y(accIdx(0, h, i, 4 * jb + a));
              x[2 * a + 1] = y(accIdx(1, h, i, 4 * jb + a));
            }
            uint32_t w;
            uint8_t sf0 = 0, sf1 = 0;
            if constexpr (r == QuantRecipe::mxfp4) {
              w = fp4::quantize8<r>(x, st, sf0);
            }
            else {
              // |x| as an integer is monotonic and orders NaN > Inf > finite
              uint32_t amax0 = 0U, amax1 = 0U;
              #pragma unroll
              for (int t = 0; t < 4; ++t) {
                amax0 = max(amax0, __float_as_uint(x[t]) & 0x7FFFFFFFU);
                amax1 = max(amax1, __float_as_uint(x[4 + t]) & 0x7FFFFFFFU);
              }
              amax0 = max(amax0, __shfl_xor_sync(0xFFFFFFFFU, amax0, 1));
              amax1 = max(amax1, __shfl_xor_sync(0xFFFFFFFFU, amax1, 1));
              amax0 = max(amax0, __shfl_xor_sync(0xFFFFFFFFU, amax0, 2));
              amax1 = max(amax1, __shfl_xor_sync(0xFFFFFFFFU, amax1, 2));
              sf0 = fp4::nvfp4ScaleFromAmax(__uint_as_float(amax0), st);
              sf1 = fp4::nvfp4ScaleFromAmax(__uint_as_float(amax1), st);
              const float enc0 = fp4::blockEncodeScale<r>(sf0, st);
              const float enc1 = fp4::blockEncodeScale<r>(sf1, st);
              w = static_cast<uint32_t>(fp4::packE2M1x2(x[0] * enc0, x[1] * enc0)) |
                  (static_cast<uint32_t>(fp4::packE2M1x2(x[2] * enc0, x[3] * enc0)) << 8) |
                  (static_cast<uint32_t>(fp4::packE2M1x2(x[4] * enc1, x[5] * enc1)) << 16) |
                  (static_cast<uint32_t>(fp4::packE2M1x2(x[6] * enc1, x[7] * enc1)) << 24);
            }
            w = detail::quadTransposeBytes(w);
            *reinterpret_cast<uint32_t*>(rowQ + 32 * jb) = w;
            if (lane % 4 == 0) {
              const uint32_t kBlock = static_cast<uint32_t>((nTile0 + 64 * jb + 32 * wn) / VS);
              const size_t off = mx::sfOffset(static_cast<uint32_t>(row), kBlock, numKBlocks);
              if constexpr (r == QuantRecipe::mxfp4) {
                sfOut[off] = sf0;
              }
              else {
                // kBlock is even: the two scales are adjacent bytes of the same atom
                *reinterpret_cast<uint16_t*>(sfOut + off) = static_cast<uint16_t>(sf0 | (sf1 << 8));
              }
            }
          }
        }
      }
    }

    template <typename TileCoord>
    __device__ __forceinline__
    void mainloop(void* __restrict__ const& workspace,
                  const uint8_t* __restrict__ const& a, const uint8_t* __restrict__ const& sfa,
                  const uint8_t* __restrict__ const& b, const uint8_t* __restrict__ const& sfb,
                  const uint8_t* __restrict__ const& b2, const uint8_t* __restrict__ const& sfb2,
                  Accumulator& acc, Accumulator& acc2, const int& M, const int& N, const int& K,
                  const TileCoord& tileCoord) const {
#if !defined(FLASHMOE_MX_UE8M0_CVT)
      // The block-scaled mma (.kind::mxf4nvf4.block_scale) only exists in the architecture-specific instruction set
      // (sm_120a, ...). Other device targets, e.g., the PTX of a generic -arch=native pass, compile to a trap.
      __trap();
#else
      const int tid = static_cast<int>(threadIdx.x);
      const int mTile = cute::get<0>(tileCoord);
      const int nTile = cute::get<1>(tileCoord);
      const int kTiles = K / bK;
      const int sfAtomsK = K / (mx::SF_K_TILE * VS); // SF atoms along K

      // align the workspace
      uint8_t* smem = static_cast<uint8_t*>(workspace);
      smem += (Alignment - (static_cast<uint32_t>(__cvta_generic_to_shared(smem)) % Alignment)) % Alignment;
      // 4-bit element views (fragments) and byte views (cp.async) of the same tiles, (rows, bK, stages)
      const auto sA = cute::make_tensor(cute::make_smem_ptr<cute::uint4_t>(static_cast<void*>(smem + offA)),
                                        SmemLayoutA{});
      const auto sB = cute::make_tensor(cute::make_smem_ptr<cute::uint4_t>(static_cast<void*>(smem + offB)),
                                        SmemLayoutB{});
      const auto sB2 = cute::make_tensor(cute::make_smem_ptr<cute::uint4_t>(static_cast<void*>(smem + offB2)),
                                         SmemLayoutB{});
      const auto sAb = cute::make_tensor(cute::make_smem_ptr(smem + offA), SmemLayoutABytes{});
      const auto sBb = cute::make_tensor(cute::make_smem_ptr(smem + offB), SmemLayoutBBytes{});
      const auto sB2b = cute::make_tensor(cute::make_smem_ptr(smem + offB2), SmemLayoutBBytes{});
      ElementSF* const sfSmem = reinterpret_cast<ElementSF*>(smem + offSF);

      const auto mA = cute::make_tensor(cute::make_gmem_ptr(a),
        cute::make_layout(cute::make_shape(M, K / 2), cute::make_stride(K / 2, cute::_1{})));
      const auto mB = cute::make_tensor(cute::make_gmem_ptr(b),
        cute::make_layout(cute::make_shape(N, K / 2), cute::make_stride(K / 2, cute::_1{})));
      const auto mB2 = cute::make_tensor(cute::make_gmem_ptr(b2),
        cute::make_layout(cute::make_shape(N, K / 2), cute::make_stride(K / 2, cute::_1{})));
      const auto gA = cute::local_tile(mA, cute::make_shape(cute::Int<bM>{}, cute::Int<rowBytes>{}),
                                       cute::make_coord(mTile, cute::_)); // (bM, rowBytes, kTiles)
      const auto gB = cute::local_tile(mB, cute::make_shape(cute::Int<bN>{}, cute::Int<rowBytes>{}),
                                       cute::make_coord(nTile, cute::_)); // (bN, rowBytes, kTiles)
      const auto gB2 = cute::local_tile(mB2, cute::make_shape(cute::Int<bN>{}, cute::Int<rowBytes>{}),
                                        cute::make_coord(nTile, cute::_));

      // ---------------- producer ----------------
      GmemCopy gmemCopy;
      const auto thrCopy = gmemCopy.get_thread_slice(tid);
      const auto tAgA = thrCopy.partition_S(gA);
      const auto tAsA = thrCopy.partition_D(sAb);
      const auto tBgB = thrCopy.partition_S(gB);
      const auto tBsB = thrCopy.partition_D(sBb);
      const auto tB2gB2 = thrCopy.partition_S(gB2);
      const auto tB2sB2 = thrCopy.partition_D(sB2b);
      // SF: chunk c of the 16 B chunks of a stage = [A unit | B units | B2 units], a unit being the kAtoms atoms of
      // 128 rows of one row tile. The atoms of (row tile, k atom) are at ((rowTile * sfAtomsK) + kAtom) * 512, so a
      // unit of a k-tile is contiguous in gmem and in smem.
      const uint8_t* sfSrc[sfCopyIters];
      CUTE_UNROLL
      for (int it = 0; it < sfCopyIters; ++it) {
        const int c = tid + it * threads;
        sfSrc[it] = nullptr;
        if (c < sfChunks) {
          const int unit = c / sfChunksPerUnit;
          const size_t rowTile = unit == 0 ? static_cast<size_t>(mTile) :
            static_cast<size_t>(nTile) * bN / mx::SF_ROW_TILE + (unit - 1) % sfbAtoms;
          const uint8_t* const base = unit == 0 ? sfa : (unit <= sfbAtoms ? sfb : sfb2);
          sfSrc[it] = base + rowTile * sfAtomsK * mx::SF_ATOM_BYTES + (c % sfChunksPerUnit) * 16;
        }
      }
      auto loadTile = [&](const int& stage, const int& kt) {
        cute::copy(gmemCopy, tAgA(cute::_, cute::_, cute::_, kt), tAsA(cute::_, cute::_, cute::_, stage));
        cute::copy(gmemCopy, tBgB(cute::_, cute::_, cute::_, kt), tBsB(cute::_, cute::_, cute::_, stage));
        if constexpr (gated) {
          cute::copy(gmemCopy, tB2gB2(cute::_, cute::_, cute::_, kt), tB2sB2(cute::_, cute::_, cute::_, stage));
        }
        const size_t kOff = static_cast<size_t>(kt) * kAtoms * mx::SF_ATOM_BYTES;
        using Chunk = cute::uint128_t;
        CUTE_UNROLL
        for (int it = 0; it < sfCopyIters; ++it) {
          const int c = tid + it * threads;
          if (c < sfChunks) {
            cute::SM80_CP_ASYNC_CACHEGLOBAL<Chunk>::copy(
              *reinterpret_cast<const Chunk*>(sfSrc[it] + kOff),
              *reinterpret_cast<Chunk*>(sfSmem + stage * sfBytesStage + c * 16));
          }
        }
      };

      // ---------------- consumer ----------------
      TiledMma tiledMma;
      auto thrMma = tiledMma.get_thread_slice(tid);
      auto tCrA = thrMma.partition_fragment_A(sA(cute::_, cute::_, cute::Int<0>{})); // (MMA, MMA_M, MMA_K)
      auto tCrB = thrMma.partition_fragment_B(sB(cute::_, cute::_, cute::Int<0>{})); // (MMA, MMA_N, MMA_K)
      auto tCrB2 = thrMma.partition_fragment_B(sB2(cute::_, cute::_, cute::Int<0>{}));
      // SF smem views of one stage, pointer arithmetic only, to build the SF register fragments
      const auto sSFAv = cute::make_tensor(cute::make_smem_ptr(sfSmem), SmemLayoutSFA{});
      const auto sSFBv = cute::make_tensor(cute::make_smem_ptr(sfSmem), SmemLayoutSFB{});
      auto tCrSFA = detail::partitionFragmentSFA(sSFAv, thrMma); // (MMA, MMA_M, MMA_K)
      auto tCrSFB = detail::partitionFragmentSFB(sSFBv, thrMma); // (MMA, MMA_N, MMA_K)
      auto tCrSFB2 = detail::partitionFragmentSFB(sSFBv, thrMma);

      const auto smemTiledCopyA = cute::make_tiled_copy_A(
        cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, cute::uint4_t>{}, tiledMma);
      const auto smemThrCopyA = smemTiledCopyA.get_thread_slice(tid);
      const auto tCsA = smemThrCopyA.partition_S(cute::as_position_independent_swizzle_tensor(sA)); // (CPY, CPY_M, CPY_K, PIPE)
      auto tCrA_view = smemThrCopyA.retile_D(tCrA);
      const auto smemTiledCopyB = cute::make_tiled_copy_B(
        cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, cute::uint4_t>{}, tiledMma);
      const auto smemThrCopyB = smemTiledCopyB.get_thread_slice(tid);
      const auto tCsB = smemThrCopyB.partition_S(cute::as_position_independent_swizzle_tensor(sB));
      auto tCrB_view = smemThrCopyB.retile_D(tCrB);
      const auto tCsB2 = smemThrCopyB.partition_S(cute::as_position_independent_swizzle_tensor(sB2));
      auto tCrB2_view = smemThrCopyB.retile_D(tCrB2);
      const auto tileShapeMNK = cute::tile_shape(tiledMma);
      const auto smemTiledCopySFA = cute::make_tiled_copy_impl(
        cute::Copy_Atom<cute::UniversalCopy<ElementSF>, ElementSF>{}, detail::getLayoutSFATV(tiledMma),
        cute::make_shape(cute::size<0>(tileShapeMNK), cute::size<2>(tileShapeMNK)));
      const auto smemThrCopySFA = smemTiledCopySFA.get_thread_slice(tid);
      auto tCrSFA_view = smemThrCopySFA.retile_D(tCrSFA);
      const auto smemTiledCopySFB = cute::make_tiled_copy_impl(
        cute::Copy_Atom<cute::UniversalCopy<ElementSF>, ElementSF>{}, detail::getLayoutSFBTV(tiledMma),
        cute::make_shape(cute::size<1>(tileShapeMNK), cute::size<2>(tileShapeMNK)));
      const auto smemThrCopySFB = smemTiledCopySFB.get_thread_slice(tid);
      auto tCrSFB_view = smemThrCopySFB.retile_D(tCrSFB);
      auto tCrSFB2_view = smemThrCopySFB.retile_D(tCrSFB2);
      // bN == 64: a B tile uses half of an SF atom, rows [64 h, 64 h + 64) of the atom, h = nTile % 2
      const int sfbHalf = bN < 128 ? ((nTile * bN) % mx::SF_ROW_TILE / 32) * 4 : 0;

      cute::clear(acc);
      if constexpr (gated) {
        cute::clear(acc2);
      }
      // the workspace may have been used by anything since the previous call
      __syncthreads();
      // ---------------- prologue ----------------
      CUTE_UNROLL
      for (int s = 0; s < pipeStages - 1; ++s) {
        if (s < kTiles) {
          loadTile(s, s);
        }
        cute::cp_async_fence();
      }
      // ---------------- mainloop ----------------
      for (int kt = 0; kt < kTiles; ++kt) {
        cute::cp_async_wait<pipeStages - 2>();
        __syncthreads(); // tile kt is in smem and every warp is done with the stage that is refilled below
        {
          const int nk = kt + pipeStages - 1;
          if (nk < kTiles) {
            loadTile(nk % pipeStages, nk);
          }
          cute::cp_async_fence();
        }
        const int stage = kt % pipeStages;
        ElementSF* const sfStage = sfSmem + stage * sfBytesStage;
        const auto sSFA = cute::make_tensor(cute::make_smem_ptr(sfStage), SmemLayoutSFA{});
        const auto sSFB = cute::make_tensor(
          cute::make_smem_ptr(sfStage + mx::SF_ATOM_BYTES * kAtoms + sfbHalf), SmemLayoutSFB{});
        const auto sSFB2 = cute::make_tensor(
          cute::make_smem_ptr(sfStage + mx::SF_ATOM_BYTES * kAtoms * (1 + sfbAtoms) + sfbHalf), SmemLayoutSFB{});
        const auto tCsSFA = smemThrCopySFA.partition_S(sSFA); // (CPY, CPY_M, CPY_K)
        const auto tCsSFB = smemThrCopySFB.partition_S(sSFB);
        const auto tCsSFB2 = smemThrCopySFB.partition_S(sSFB2);
        const auto tCsA_s = tCsA(cute::_, cute::_, cute::_, stage);
        const auto tCsB_s = tCsB(cute::_, cute::_, cute::_, stage);
        const auto tCsB2_s = tCsB2(cute::_, cute::_, cute::_, stage);
        auto copyKBlock = [&](const auto& kb) {
          cute::copy(smemTiledCopyA, tCsA_s(cute::_, cute::_, kb), tCrA_view(cute::_, cute::_, kb));
          cute::copy(smemTiledCopyB, tCsB_s(cute::_, cute::_, kb), tCrB_view(cute::_, cute::_, kb));
          cute::copy(tCsSFA(cute::_, cute::_, kb), tCrSFA_view(cute::_, cute::_, kb));
          cute::copy(tCsSFB(cute::_, cute::_, kb), tCrSFB_view(cute::_, cute::_, kb));
          if constexpr (gated) {
            cute::copy(smemTiledCopyB, tCsB2_s(cute::_, cute::_, kb), tCrB2_view(cute::_, cute::_, kb));
            cute::copy(tCsSFB2(cute::_, cute::_, kb), tCrSFB2_view(cute::_, cute::_, kb));
          }
        };
        copyKBlock(cute::_0{});
        CUTE_UNROLL
        for (int kb = 0; kb < MMA_K; ++kb) {
          if (kb + 1 < MMA_K) {
            copyKBlock(kb + 1);
          }
          cute::gemm(tiledMma, cute::make_zip_tensor(tCrA(cute::_, cute::_, kb), tCrSFA(cute::_, cute::_, kb)),
                     cute::make_zip_tensor(tCrB(cute::_, cute::_, kb), tCrSFB(cute::_, cute::_, kb)), acc);
          if constexpr (gated) {
            cute::gemm(tiledMma, cute::make_zip_tensor(tCrA(cute::_, cute::_, kb), tCrSFA(cute::_, cute::_, kb)),
                       cute::make_zip_tensor(tCrB2(cute::_, cute::_, kb), tCrSFB2(cute::_, cute::_, kb)), acc2);
          }
        }
      }
      cute::cp_async_wait<0>();
#endif
    }
  };

  /// The FP4 expert GEMMs of FlashMoE (sm_12x: 99 KB of shared memory per block, 256 threads), picked with
  /// tests/gemmFP4.cu on an RTX 5090: bK = 128 and 3 stages everywhere,
  ///  - GEMM0, gated: 128 x (bN + bN) x 128 (A is read once for both B operands), 79 KB (mxfp4) / 84 KB (nvfp4)
  ///    for bN = 128; 128 x (64 + 64) was 11-15% slower;
  ///  - GEMM0 vanilla and GEMM1: 128 x bN x 128, 53 KB (mxfp4) / 56 KB (nvfp4) for bN = 128. bK = 256 with 2 stages
  ///    (K % 256 == 0 only) is within -2% .. +7% of it, i.e., not worth a runtime selection, and 4 stages is 2-5% slower.
  constexpr int FP4Threads = 256;

  template <QuantRecipe r, int bM, int bN, int threads, bool gated>
  using FP4GEMM0 = FP4Mainloop<r, bM, bN, 128, 3, threads, gated>;

  template <QuantRecipe r, int bM, int bN, int threads>
  using FP4GEMM1 = FP4Mainloop<r, bM, bN, 128, 3, threads, false>;
}
#endif //FLASHMOE_TILE_SM120_FP4_CUH
