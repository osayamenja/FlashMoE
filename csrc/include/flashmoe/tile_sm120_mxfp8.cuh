/******************************************************************************
 * Copyright (c) 2026, Osayamen Jonathan Aimuyo.
 ******************************************************************************/
//
// Created by FlashMoE contributors on 10/5/26.
//
// Block-scaled MXFP8 (E4M3 elements + one UE8M0 scale per 32 K-elements) tile GEMM for sm_120a (RTX 50 series,
// RTX PRO 6000 Blackwell, DGX Spark): mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.
//  - MXFP8Mainloop: one CTA computes one bM x bN output tile (gated: two bM x bN tiles, A is read once) with
//    256 threads = 8 warps (4 along M x 2 along N), cp.async multistage pipeline, ldmatrix fragments from swizzled
//    shared memory and scale factors (SF) straight out of the canonical 128x4 layout, see infra/mx.cuh.
//  - epilogues that work on the accumulator fragments: BF16 output, BF16 output fused with the combine (the tile rows
//    go straight to the output rows of their tokens), MXFP8 output (the A operand of the next GEMM) and the fused gated
//    variant.
//
// thrfrgSFA / thrfrgSFB / partitionFragmentSFA / partitionFragmentSFB / getLayoutSFATV / getLayoutSFBTV in
// namespace detail are adapted from cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp, CUTLASS 4.4.1:
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

#ifndef FLASHMOE_TILE_SM120_MXFP8_CUH
#define FLASHMOE_TILE_SM120_MXFP8_CUH

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

#include "infra/mx.cuh"
#include "infra/packed.cuh" // TPS
#include "infra/rvt.cuh" // RedAdd

namespace flashmoe::tile::sm120 {
  namespace detail {
    // ------------------------------------------------------------------------------------------------------------
    // Scale factor fragment partitioning, adapted from CUTLASS sm120_blockscaled_mma_tma.hpp (BSD-3-Clause)
    // ------------------------------------------------------------------------------------------------------------
    template <class SFATensor, class Atom, class TiledThr, class TiledPerm>
    CUTE_HOST_DEVICE constexpr
    auto thrfrgSFA(SFATensor&& sfatensor, cute::TiledMMA<Atom, TiledThr, TiledPerm>& mma) {
      using AtomShape_MNK = typename Atom::Shape_MNK;
      using AtomLayoutSFA_TV = typename Atom::Traits::SFALayout;
      auto permutation_mnk = TiledPerm{};
      auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
      auto t_tile = cute::make_tile(cute::get<0>(permutation_mnk), cute::get<2>(permutation_mnk));
      auto t_tensor = cute::logical_divide(sfatensor, t_tile);
      auto a_tile = cute::make_tile(cute::make_layout(cute::size<0>(AtomShape_MNK{})),
                                    cute::make_layout(cute::size<2>(AtomShape_MNK{})));
      auto a_tensor = cute::zipped_divide(t_tensor, a_tile);
      auto tv_tensor = a_tensor.compose(AtomLayoutSFA_TV{}, cute::_);
      auto thr_tile = cute::make_tile(cute::_, cute::make_tile(cute::make_layout(cute::size<1>(thr_layout_vmnk)),
                                                               cute::make_layout(cute::size<3>(thr_layout_vmnk))));
      return cute::zipped_divide(tv_tensor, thr_tile);
    }

    template <class SFBTensor, class Atom, class TiledThr, class TiledPerm>
    CUTE_HOST_DEVICE constexpr
    auto thrfrgSFB(SFBTensor&& sfbtensor, cute::TiledMMA<Atom, TiledThr, TiledPerm>& mma) {
      using AtomShape_MNK = typename Atom::Shape_MNK;
      using AtomLayoutSFB_TV = typename Atom::Traits::SFBLayout;
      auto permutation_mnk = TiledPerm{};
      auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
      auto t_tile = cute::make_tile(cute::get<1>(permutation_mnk), cute::get<2>(permutation_mnk));
      auto t_tensor = cute::logical_divide(sfbtensor, t_tile);
      auto a_tile = cute::make_tile(cute::make_layout(cute::size<1>(AtomShape_MNK{})),
                                    cute::make_layout(cute::size<2>(AtomShape_MNK{})));
      auto a_tensor = cute::zipped_divide(t_tensor, a_tile);
      auto tv_tensor = a_tensor.compose(AtomLayoutSFB_TV{}, cute::_);
      auto thr_tile = cute::make_tile(cute::_, cute::make_tile(cute::make_layout(cute::size<2>(thr_layout_vmnk)),
                                                               cute::make_layout(cute::size<3>(thr_layout_vmnk))));
      return cute::zipped_divide(tv_tensor, thr_tile);
    }

    template <class SFATensor, class ThrMma>
    CUTE_HOST_DEVICE constexpr
    auto partitionFragmentSFA(SFATensor&& sfatensor, ThrMma& thread_mma) {
      using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
      auto thr_tensor = cute::make_tensor(static_cast<SFATensor&&>(sfatensor).data(),
                                          thrfrgSFA(sfatensor.layout(), thread_mma));
      auto thr_vmnk = thread_mma.thr_vmnk_;
      auto thr_vmk = cute::make_coord(cute::get<0>(thr_vmnk),
                                      cute::make_coord(cute::get<1>(thr_vmnk), cute::get<3>(thr_vmnk)));
      auto partition_SFA = thr_tensor(thr_vmk, cute::make_coord(cute::_, cute::repeat<cute::rank<1, 1>(thr_tensor)>(cute::_)));
      return cute::make_fragment_like<ValTypeSF>(partition_SFA);
    }

    template <class SFBTensor, class ThrMma>
    CUTE_HOST_DEVICE constexpr
    auto partitionFragmentSFB(SFBTensor&& sfbtensor, ThrMma& thread_mma) {
      using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
      auto thr_tensor = cute::make_tensor(static_cast<SFBTensor&&>(sfbtensor).data(),
                                          thrfrgSFB(sfbtensor.layout(), thread_mma));
      auto thr_vmnk = thread_mma.thr_vmnk_;
      auto thr_vnk = cute::make_coord(cute::get<0>(thr_vmnk),
                                      cute::make_coord(cute::get<2>(thr_vmnk), cute::get<3>(thr_vmnk)));
      auto partition_SFB = thr_tensor(thr_vnk, cute::make_coord(cute::_, cute::repeat<cute::rank<1, 1>(thr_tensor)>(cute::_)));
      return cute::make_fragment_like<ValTypeSF>(partition_SFB);
    }

    template <class TiledMma>
    CUTE_HOST_DEVICE constexpr
    auto getLayoutSFATV(TiledMma& mma) {
      auto tile_shape_mnk = cute::tile_shape(mma);
      auto ref_A = cute::make_layout(cute::make_shape(cute::size<0>(tile_shape_mnk), cute::size<2>(tile_shape_mnk)));
      auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
      auto atile = cute::make_tile(cute::_,
        cute::make_tile(cute::make_layout(cute::make_shape(cute::size<1>(thr_layout_vmnk), cute::size<2>(thr_layout_vmnk)),
                                          cute::make_stride(cute::Int<1>{}, cute::Int<0>{})), cute::_));
      auto thridx_2_thrid = cute::right_inverse(thr_layout_vmnk);
      return thrfrgSFA(ref_A, mma).compose(atile, cute::_).compose(thridx_2_thrid, cute::_);
    }

    template <class TiledMma>
    CUTE_HOST_DEVICE constexpr
    auto getLayoutSFBTV(TiledMma& mma) {
      auto tile_shape_mnk = cute::tile_shape(mma);
      auto ref_B = cute::make_layout(cute::make_shape(cute::size<1>(tile_shape_mnk), cute::size<2>(tile_shape_mnk)));
      auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
      auto btile = cute::make_tile(cute::_,
        cute::make_tile(cute::make_layout(cute::make_shape(cute::size<1>(thr_layout_vmnk), cute::size<2>(thr_layout_vmnk)),
                                          cute::make_stride(cute::Int<0>{}, cute::Int<1>{})), cute::_));
      auto thridx_2_thrid = cute::right_inverse(thr_layout_vmnk);
      return thrfrgSFB(ref_B, mma).compose(btile, cute::_).compose(thridx_2_thrid, cute::_);
    }

    // ------------------------------------------------------------------------------------------------------------
    // element helpers of the epilogues
    // ------------------------------------------------------------------------------------------------------------
    __device__ __forceinline__
    float2 loadPair(const __nv_bfloat16* __restrict__ const& p) {
      return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(p));
    }

    __device__ __forceinline__
    float2 loadPair(const __half* __restrict__ const& p) {
      return __half22float2(*reinterpret_cast<const __half2*>(p));
    }

    // two values rounded to ElementC, as the bits of a 32-bit word (the first one in the low half)
    __device__ __forceinline__
    uint32_t packPair(const __nv_bfloat16*, const float& x, const float& y) {
      const __nv_bfloat162 v = __floats2bfloat162_rn(x, y);
      return *reinterpret_cast<const uint32_t*>(&v);
    }

    __device__ __forceinline__
    uint32_t packPair(const __half*, const float& x, const float& y) {
      const __half2 v = __floats2half2_rn(x, y);
      return *reinterpret_cast<const uint32_t*>(&v);
    }

    template <typename Element>
    concept EpilogueElement = cuda::std::is_same_v<Element, __nv_bfloat16> || cuda::std::is_same_v<Element, __half>;

    // ------------------------------------------------------------------------------------------------------------
    // element helpers of the combine epilogues (GEMM1 of a local task, see epilogueCombine)
    // ------------------------------------------------------------------------------------------------------------

    /// The 4-byte aligned pair of ElementC values at p as a 32-bit word (the first one in the low half)
    __device__ __forceinline__
    uint32_t loadWord(const __nv_bfloat16* __restrict__ const& p) {
      return *reinterpret_cast<const uint32_t*>(p);
    }

    __device__ __forceinline__
    uint32_t loadWord(const __half* __restrict__ const& p) {
      return *reinterpret_cast<const uint32_t*>(p);
    }

    __device__ __forceinline__
    float2 unpackPair(const __nv_bfloat16*, const uint32_t& w) {
      return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w));
    }

    __device__ __forceinline__
    float2 unpackPair(const __half*, const uint32_t& w) {
      return __half22float2(*reinterpret_cast<const __half2*>(&w));
    }

    /// The pair w scaled by the routing probability p and rounded to ElementC, the arithmetic of combine() in
    /// combine.cuh: pair -> float2, float2 multiply (__fmul2_rn from sm_100 on) with {p, p}, float2 -> pair, RN.
    template <typename ElementC>
    __device__ __forceinline__
    uint32_t scalePair(const ElementC* tag, const uint32_t& w, const float& p) {
      const float2 x = unpackPair(tag, w);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
      const float2 y = __fmul2_rn(x, make_float2(p, p));
#else
      const float2 y = make_float2(x.x * p, x.y * p);
#endif
      return packPair(tag, y.x, y.y);
    }

    /// red.global.add.noftz.v4.bf16x2 (f16x2) of 8 ElementC values, the 4 words w, to p (16 B aligned), the vector
    /// reduction of combine() (RedAdd<900, ElementC2, 4>)
    __device__ __forceinline__
    void redAdd8(__nv_bfloat16* __restrict__ const& p, const uint32_t (&w)[4]) {
      cutlass::AlignedArray<__nv_bfloat162, 4, 16> v;
      #pragma unroll
      for (int a = 0; a < 4; ++a) {
        v[a] = *reinterpret_cast<const __nv_bfloat162*>(&w[a]);
      }
      constexpr RedAdd<900, __nv_bfloat162, 4> op{};
      op(reinterpret_cast<__nv_bfloat162*>(p), v);
    }

    __device__ __forceinline__
    void redAdd8(__half* __restrict__ const& p, const uint32_t (&w)[4]) {
      cutlass::AlignedArray<__half2, 4, 16> v;
      #pragma unroll
      for (int a = 0; a < 4; ++a) {
        v[a] = *reinterpret_cast<const __half2*>(&w[a]);
      }
      constexpr RedAdd<900, __half2, 4> op{};
      op(reinterpret_cast<__half2*>(p), v);
    }

    /// What the combine epilogue needs from global memory, loaded into registers before the mainloop so that the loads
    /// are not on the critical path of the epilogue: the token index and routing probability of each of the rows of a
    /// thread, and the bias pairs of its columns (as 32-bit words, 0 is a pair of zeros in both formats)
    template <int rows, int cols>
    struct CombineRegs {
      TPS tps[rows];
      uint32_t bias[cols];
    };

    /// x * sigmoid(x) = x / (1 + 2^(-x log2(e))) with the hardware ex2 / rcp approximations (a few ulp), instead of
    /// cutlass::epilogue::thread::SiLu<float>, which calls the full precision exp and division.
    __device__ __forceinline__
    float siluFast(const float& x) {
      float e, r;
      asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e) : "f"(-x * 1.4426950408889634f));
      asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(1.0f + e));
      return x * r;
    }

    /// Activation of the epilogues: act(x), or the faster equivalent where there is one.
    template <typename Act>
    __device__ __forceinline__
    float activate(const float& x) {
      if constexpr (cuda::std::is_same_v<Act, cutlass::epilogue::thread::SiLu<float>>) {
        return siluFast(x);
      }
      else {
        constexpr Act act{};
        return act(x);
      }
    }

    /// Transposes a 4 x 4 matrix of 32-bit words that is distributed over the 4 lanes of a quad (lane % 4 = row):
    /// afterwards lane q holds w[s] = (the old w[q] of lane s). Two exchange stages, 4 shuffles.
    __device__ __forceinline__
    void quadTranspose(uint32_t (&w)[4]) {
      const bool b0 = (threadIdx.x & 1U) != 0;
      const bool b1 = (threadIdx.x & 2U) != 0;
      // stage 1: swap the off-diagonal 2 x 2 blocks, lanes differ in bit 1
      const uint32_t r0 = __shfl_xor_sync(0xFFFFFFFFU, b1 ? w[0] : w[2], 2);
      const uint32_t r1 = __shfl_xor_sync(0xFFFFFFFFU, b1 ? w[1] : w[3], 2);
      const uint32_t n0 = b1 ? r0 : w[0];
      const uint32_t n1 = b1 ? r1 : w[1];
      const uint32_t n2 = b1 ? w[2] : r0;
      const uint32_t n3 = b1 ? w[3] : r1;
      // stage 2: transpose the 2 x 2 blocks, lanes differ in bit 0
      const uint32_t u0 = __shfl_xor_sync(0xFFFFFFFFU, b0 ? n0 : n1, 1);
      const uint32_t u1 = __shfl_xor_sync(0xFFFFFFFFU, b0 ? n2 : n3, 1);
      w[0] = b0 ? u0 : n0;
      w[1] = b0 ? n1 : u0;
      w[2] = b0 ? u1 : n2;
      w[3] = b0 ? n3 : u1;
    }

    /// The same for a 4 x 4 matrix of 16-bit elements: lane q holds {lo = e0 | e1 << 16, hi = e2 | e3 << 16}.
    /// Afterwards lane q holds e[s] = (the old e[q] of lane s). Two shuffles.
    __device__ __forceinline__
    void quadTranspose(uint32_t& lo, uint32_t& hi) {
      const bool b0 = (threadIdx.x & 1U) != 0;
      const bool b1 = (threadIdx.x & 2U) != 0;
      // stage 1, on whole words
      const uint32_t r = __shfl_xor_sync(0xFFFFFFFFU, b1 ? lo : hi, 2);
      const uint32_t nLo = b1 ? r : lo; // (e0, e1)
      const uint32_t nHi = b1 ? hi : r; // (e2, e3)
      // stage 2: the element sent for the first / second block column is e0 or e1 / e2 or e3
      const uint32_t t = b0 ? ((nLo & 0xFFFFU) | (nHi << 16)) : ((nLo >> 16) | (nHi & 0xFFFF0000U));
      const uint32_t u = __shfl_xor_sync(0xFFFFFFFFU, t, 1);
      const uint32_t u0 = u & 0xFFFFU;
      const uint32_t u1 = u >> 16;
      lo = b0 ? (u0 | (nLo & 0xFFFF0000U)) : ((nLo & 0xFFFFU) | (u0 << 16));
      hi = b0 ? (u1 | (nHi & 0xFFFF0000U)) : ((nHi & 0xFFFFU) | (u1 << 16));
    }
  }

  /// Block-scaled MXFP8 tile GEMM, C[bM, bN] = A[bM, K] @ B[bN, K]^T with FP32 accumulation, where
  ///  - A is a row-major [M, K] E4M3 matrix (K-major) and B a row-major [N, K] E4M3 matrix (K-major),
  ///  - sfa / sfb are their UE8M0 scale factors, one per 32 consecutive K elements of a row, in the canonical 128x4
  ///    layout of mx::sfOffset() of the *whole matrix* (row 0 of the matrix is row 0 of the SF buffer),
  ///  - gated == true computes two accumulators from one A tile, acc = A @ B^T and acc2 = A @ B2^T, with B2 / sfb2 of
  ///    the same shape and layout as B / sfb.
  /// bM = 128 (one SF atom), threads = 256, bN is 64 or a multiple of 128 (bN of *each* B matrix when gated).
  /// Runtime requirements: M % bM == 0, N % bN == 0, K % 128 == 0 (a whole SF atom per 4 k-blocks).
  ///
  /// The accumulator fragments follow a "Perm64" N layout of the tiled MMA: of every 64 columns of the tile, warp
  /// wn in {0, 1} owns the contiguous columns [32 wn, 32 wn + 32), i.e., exactly one MXFP8 scale block of the output,
  /// and the 4 lanes of a quad (same lane / 4) hold the 32 values of one row of a scale block, which is what the
  /// quantizing epilogues exploit.
  ///
  /// The mainloop and the epilogues may be called repeatedly by the same CTA (persistent kernels) for different tiles
  /// and different matrices. All state is in the workspace (dynamic shared memory) and every cp.async group is
  /// drained before operator() returns. The mainloop begins with a block-wide barrier, so the workspace may be
  /// overwritten by anyone between calls; the epilogues do not touch the workspace. All threads of the CTA must call
  /// everything below, and the 256 participating threads are threadIdx.x in [0, 256).
  template <int bM, int bN, int bK, int pipeStages, int threads, bool gated>
    requires(bM == 128 && threads == 256 && (bN == 64 || bN % 128 == 0) && (bK == 64 || bK == 128) &&
             pipeStages >= 2)
  struct MXFP8Mainloop {
    using ElementAB = cutlass::float_e4m3_t;
    using ElementSF = cutlass::float_ue8m0_t;
    using AccumType = float;
    static constexpr bool usesTma = false;
    using Threads = cute::Int<threads>;
    using TileShape = cute::Shape<cute::Int<bM>, cute::Int<bN>, cute::Int<bK>>;
    using PipeStages = cute::Int<pipeStages>;
    using MmaOp = cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<ElementAB, ElementAB, float, ElementSF, 32>;
    // 4 x 2 warps. PermTileN = Perm64: warp wn owns 32 contiguous columns of every 64
    using PermN = cute::Layout<cute::Shape<cute::_8, cute::_2, cute::_4>, cute::Stride<cute::_1, cute::_32, cute::_8>>;
    using TiledMma = decltype(cute::make_tiled_mma(MmaOp{}, cute::Layout<cute::Shape<cute::_4, cute::_2, cute::_1>>{},
                                                   cute::Tile<cute::_128, PermN, cute::_32>{}));
    /// accumulator fragment of one bM x bN tile
    using Accumulator = decltype(cute::partition_fragment_C(TiledMma{}, cute::Shape<cute::Int<bM>, cute::Int<bN>>{}));

    // accumulator fragment: 4 * MMA_M * MMA_N values, element (v, i, j) at v + 4 * (i + MMA_M * j) with
    // v = 2 * rowHalf + c: row = 16 * (warp % 4) + lane / 4 + 8 * rowHalf + 64 * i,
    //                      col = 64 * (j / 4) + 32 * (warp / 4) + 8 * (j % 4) + 2 * (lane % 4) + c
    static constexpr int MMA_M = bM / 64;
    static constexpr int MMA_N = bN / 16;

    static constexpr int NB = gated ? 2 : 1; // number of B operands
    static constexpr int sfbAtoms = (bN + 127) / 128; // SF atoms (128 rows) of one B operand
    static constexpr int sfThreads = 32 * (1 + NB * sfbAtoms); // one thread copies 16 B of the SF atoms of a stage
    static_assert(sfThreads <= threads);
    static constexpr int bytesA = bM * bK * pipeStages;
    static constexpr int bytesB = bN * bK * pipeStages;
    static constexpr int sfBytesStage = mx::SF_ATOM_BYTES * (1 + NB * sfbAtoms);
    static constexpr int offA = 0;
    static constexpr int offB = offA + bytesA;
    static constexpr int offB2 = offB + bytesB; // gated only
    static constexpr int offSF = offB + NB * bytesB;
    static constexpr int dataBytes = offSF + sfBytesStage * pipeStages;
    /// The workspace is aligned to 1024 B inside (128 B swizzle, TMA friendly), which costs up to 1023 B.
    static constexpr int Alignment = 1024;
    using SharedSize = cute::Int<dataBytes + Alignment>;
    static constexpr int sharedSize = SharedSize::value;

    // SMEM layouts: K-major, swizzled so that ldmatrix is conflict free
    using SmemAtom = cuda::std::conditional_t<bK == 128, cute::GMMA::Layout_K_SW128_Atom<uint8_t>,
                                              cute::GMMA::Layout_K_SW64_Atom<uint8_t>>;
    using SmemLayoutA = decltype(cute::tile_to_shape(SmemAtom{},
      cute::make_shape(cute::Int<bM>{}, cute::Int<bK>{}, cute::Int<pipeStages>{}),
      cute::Step<cute::_1, cute::_2, cute::_3>{}));
    using SmemLayoutB = decltype(cute::tile_to_shape(SmemAtom{},
      cute::make_shape(cute::Int<bN>{}, cute::Int<bK>{}, cute::Int<pipeStages>{}),
      cute::Step<cute::_1, cute::_2, cute::_3>{}));
    // SF: (row, k) -> byte inside the 512 B canonical atoms: (row % 32) * 16 + (row / 32 % 4) * 4 + kBlock % 4,
    // 128 rows per atom, the 32 K elements of a block share the byte (stride 0)
    using SmemLayoutSFA = cute::Layout<
      cute::Shape<cute::Shape<cute::_32, cute::_4>, cute::Shape<cute::_32, cute::Int<bK / 32>>>,
      cute::Stride<cute::Stride<cute::_16, cute::_4>, cute::Stride<cute::_0, cute::_1>>>;
    using SmemLayoutSFB = cute::Layout<
      cute::Shape<cute::Shape<cute::_32, cute::Int<(bN < 128 ? bN : 128) / 32>, cute::Int<sfbAtoms>>,
                  cute::Shape<cute::_32, cute::Int<bK / 32>>>,
      cute::Stride<cute::Stride<cute::_16, cute::_4, cute::Int<mx::SF_ATOM_BYTES>>, cute::Stride<cute::_0, cute::_1>>>;

    // gmem -> smem: 16 B cp.async per thread
    using GmemCopyAtom = cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, uint8_t>;
    static constexpr int chunksPerRow = bK / 16;
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
    /// tile::idx2Coord. workspace: dynamic shared memory of at least sharedSize bytes.
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
    // epilogues
    // ------------------------------------------------------------------------------------------------------------

    /// out[M, N] (row-major) tile = act(acc + bias). bias: [N] or nullptr. Vanilla GEMM1 epilogue.
    template <typename Act, typename ElementC, typename TileCoord>
      requires(!gated && detail::EpilogueElement<ElementC> && cute::rank_v<TileCoord> == 3)
    __device__ __forceinline__
    static void epilogueBF16(const Accumulator& acc, ElementC* __restrict__ const& out,
                             const ElementC* __restrict__ const& bias,
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
                                      detail::activate<Act>(acc(accIdx(0, h, i, 4 * jb + a)) + bv[4 * jb + a].x),
                                      detail::activate<Act>(acc(accIdx(1, h, i, 4 * jb + a)) + bv[4 * jb + a].y));
            }
            detail::quadTranspose(w);
            *reinterpret_cast<uint4*>(rowOut + 64 * jb) = make_uint4(w[0], w[1], w[2], w[3]);
          }
        }
      }
    }

    /// Registers of epilogueCombine: for each of the 2 * MMA_M rows of the thread (row 16 * (warp % 4) + lane / 4 +
    /// 8 h + 64 i of the tile) the token index and routing probability, for each of its MMA_N column pairs the bias
    using CombineRegs = detail::CombineRegs<2 * MMA_M, MMA_N>;

    /// Loads the registers of epilogueCombine before the mainloop. tokenIds: the TPS of the bM rows of the tile, only
    /// the first tileSize are valid (the others are not read). bias: [N] or nullptr.
    template <typename ElementC, typename TileCoord>
      requires(!gated && detail::EpilogueElement<ElementC> && cute::rank_v<TileCoord> == 3)
    __device__ __forceinline__
    static CombineRegs prefetchCombine(const TPS* __restrict__ const& tokenIds,
                                       const ElementC* __restrict__ const& bias,
                                       const uint& tileSize, const TileCoord& tileCoord) {
      const auto origin = threadOrigin();
      const int n0 = cute::get<1>(tileCoord) * bN + origin.col;
      CombineRegs regs;
      #pragma unroll
      for (int i = 0; i < MMA_M; ++i) {
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int row = origin.row + 64 * i + 8 * h;
          regs.tps[2 * i + h] = row < static_cast<int>(tileSize) ? tokenIds[row] : TPS{0U, 0.f};
        }
      }
      #pragma unroll
      for (int j = 0; j < MMA_N; ++j) {
        regs.bias[j] = bias == nullptr ? 0U : detail::loadWord(bias + n0 + colOffset(j));
      }
      return regs;
    }

    /// The combine fused into the epilogue of GEMM1: the rows of the tile go to the output rows of their tokens,
    /// out[S, N] (row-major) row tokenIds[row].tokenIdx. With v = ElementC(act(acc + bias)) as in epilogueBF16:
    ///  - plural (top k > 1): out += ElementC(float(v) * probability), a vector reduction (red.global.add.v4.bf16x2),
    ///    the same rounding steps as GEMM1 into the heap followed by combine() (the order of the additions is not
    ///    deterministic);
    ///  - single (top k = 1): out = v, 16 B stores.
    /// Only the rows < tileSize are written. All threads execute the quad transposes, the stores are predicated.
    template <bool plural, typename Act, typename ElementC, typename TileCoord>
      requires(!gated && detail::EpilogueElement<ElementC> && cute::rank_v<TileCoord> == 3)
    __device__ __forceinline__
    static void epilogueCombine(const Accumulator& acc, ElementC* __restrict__ const& out, const CombineRegs& regs,
                                const uint& tileSize, const int& N, const TileCoord& tileCoord) {
      const auto origin = threadOrigin();
      const int n0 = cute::get<1>(tileCoord) * bN;
      float2 bv[MMA_N];
      #pragma unroll
      for (int j = 0; j < MMA_N; ++j) {
        bv[j] = detail::unpackPair(out, regs.bias[j]);
      }
      #pragma unroll
      for (int i = 0; i < MMA_M; ++i) {
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
          const bool live = origin.row + 64 * i + 8 * h < static_cast<int>(tileSize);
          const TPS t = regs.tps[2 * i + h];
          ElementC* const rowOut = out + static_cast<size_t>(t.tokenIdx) * N + n0 + origin.vcol;
          #pragma unroll
          for (int jb = 0; jb < MMA_N / 4; ++jb) {
            uint32_t w[4];
            #pragma unroll
            for (int a = 0; a < 4; ++a) {
              w[a] = detail::packPair(out,
                                      detail::activate<Act>(acc(accIdx(0, h, i, 4 * jb + a)) + bv[4 * jb + a].x),
                                      detail::activate<Act>(acc(accIdx(1, h, i, 4 * jb + a)) + bv[4 * jb + a].y));
            }
            detail::quadTranspose(w);
            if (live) {
              if constexpr (plural) {
                #pragma unroll
                for (int a = 0; a < 4; ++a) {
                  w[a] = detail::scalePair(out, w[a], t.probability);
                }
                detail::redAdd8(rowOut + 64 * jb, w);
              }
              else {
                *reinterpret_cast<uint4*>(rowOut + 64 * jb) = make_uint4(w[0], w[1], w[2], w[3]);
              }
            }
          }
        }
      }
    }

    /// q[M, N] (E4M3, row-major) and sfOut (canonical SF layout of an [M, N] matrix, N / 32 k-blocks per row)
    /// = MXFP8(act(acc + bias)), i.e., the A operand of the next GEMM. bias: [N] or nullptr. acc is overwritten.
    template <typename Act, typename ElementC, typename TileCoord>
      requires(!gated && detail::EpilogueElement<ElementC> && cute::rank_v<TileCoord> == 3)
    __device__ __forceinline__
    static void epilogueMXFP8(Accumulator& acc, uint8_t* __restrict__ const& q, uint8_t* __restrict__ const& sfOut,
                              const ElementC* __restrict__ const& bias,
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
            acc(accIdx(0, h, i, j)) = detail::activate<Act>(acc(accIdx(0, h, i, j)) + bv.x);
            acc(accIdx(1, h, i, j)) = detail::activate<Act>(acc(accIdx(1, h, i, j)) + bv.y);
          }
        }
      }
      quantizeStore(acc, q, sfOut, N, tileCoord);
    }

    /// Gated epilogue of GEMM0, the semantics of processor::fGET_gated, but in FP32 (no BF16 rounding of the
    /// activated gate in between):
    ///   y = alpha * act(beta * (acc + bias)) * (acc2 + biasV)
    /// followed by the MXFP8 quantization of y along N as in epilogueMXFP8. acc is overwritten.
    template <typename Act, typename ElementC, typename TileCoord>
      requires(gated && detail::EpilogueElement<ElementC> && cute::rank_v<TileCoord> == 3)
    __device__ __forceinline__
    static void epilogueGatedMXFP8(Accumulator& acc, const Accumulator& acc2,
                                   uint8_t* __restrict__ const& q, uint8_t* __restrict__ const& sfOut,
                                   const ElementC* __restrict__ const& bias, const ElementC* __restrict__ const& biasV,
                                   const AccumType& alpha, const AccumType& beta,
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
            acc(accIdx(0, h, i, j)) = alpha * detail::activate<Act>((acc(accIdx(0, h, i, j)) + bg.x) * beta) *
              (acc2(accIdx(0, h, i, j)) + bu.x);
            acc(accIdx(1, h, i, j)) = alpha * detail::activate<Act>((acc(accIdx(1, h, i, j)) + bg.y) * beta) *
              (acc2(accIdx(1, h, i, j)) + bu.y);
          }
        }
      }
      quantizeStore(acc, q, sfOut, N, tileCoord);
    }

  private:
    struct Origin {
      int row; // row of the accumulator elements (v = 0, i = 0)
      int col; // column of the accumulator element (v = 0, j = 0)
      int vcol; // first column of the 8 consecutive columns the thread stores after quadTranspose
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

    /// MXFP8 quantization of the fragment y along N: for every row, every 32 consecutive columns share a UE8M0 scale
    /// 2^ceil(log2(amax / 448)) (the recipe of mx::quantizeKernel). Those 32 columns are the 8 values of one row held
    /// by the 4 lanes of a quad: amax is thread-local over 8 values and then reduced with 2 xor shuffles.
    __device__ __forceinline__
    static void quantizeStore(const Accumulator& y, uint8_t* __restrict__ const& q, uint8_t* __restrict__ const& sfOut,
                              const int& N, const auto& tileCoord) {
      const auto origin = threadOrigin();
      const int lane = static_cast<int>(threadIdx.x) % 32;
      const int wn = static_cast<int>(threadIdx.x) / 128;
      const int m0 = cute::get<0>(tileCoord) * bM + origin.row;
      const int nTile0 = cute::get<1>(tileCoord) * bN;
      const uint32_t numKBlocks = static_cast<uint32_t>(N / mx::BLOCK_SIZE);
      #pragma unroll
      for (int i = 0; i < MMA_M; ++i) {
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int r = m0 + 64 * i + 8 * h;
          uint8_t* const rowQ = q + static_cast<size_t>(r) * N + nTile0 + origin.vcol;
          #pragma unroll
          for (int jb = 0; jb < MMA_N / 4; ++jb) {
            float x[8];
            uint32_t amaxBits = 0U; // |x| as an integer is monotonic and orders NaN > Inf > finite
            #pragma unroll
            for (int a = 0; a < 4; ++a) {
              x[2 * a] = y(accIdx(0, h, i, 4 * jb + a));
              x[2 * a + 1] = y(accIdx(1, h, i, 4 * jb + a));
              amaxBits = max(amaxBits, max(__float_as_uint(x[2 * a]) & 0x7FFFFFFFU,
                                           __float_as_uint(x[2 * a + 1]) & 0x7FFFFFFFU));
            }
            amaxBits = max(amaxBits, __shfl_xor_sync(0xFFFFFFFFU, amaxBits, 1));
            amaxBits = max(amaxBits, __shfl_xor_sync(0xFFFFFFFFU, amaxBits, 2));
            const uint8_t sf = mx::ue8m0FromAmax(__uint_as_float(amaxBits));
            // NaN / Inf block: the scale is NaN and the payload irrelevant
            const float inv = sf == mx::UE8M0_NAN ? 1.0f : mx::ue8m0ToRcpFloat(sf);
            // 4 pairs of E4M3 (atoms 0..3) -> after the quad transpose, 8 contiguous bytes of the row
            uint32_t lo = mx::packE4M3x2(x[0] * inv, x[1] * inv) | (mx::packE4M3x2(x[2] * inv, x[3] * inv) << 16);
            uint32_t hi = mx::packE4M3x2(x[4] * inv, x[5] * inv) | (mx::packE4M3x2(x[6] * inv, x[7] * inv) << 16);
            detail::quadTranspose(lo, hi);
            *reinterpret_cast<uint2*>(rowQ + 64 * jb) = make_uint2(lo, hi);
            if (lane % 4 == 0) {
              const uint32_t kBlock = static_cast<uint32_t>((nTile0 + 64 * jb + 32 * wn) / mx::BLOCK_SIZE);
              sfOut[mx::sfOffset(static_cast<uint32_t>(r), kBlock, numKBlocks)] = sf;
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
      // The block-scaled mma (.kind::mxf8f6f4.block_scale) only exists in the architecture-specific instruction set
      // (sm_120a, ...). Other device targets, e.g., the PTX of a generic -arch=native pass, compile to a trap.
      __trap();
#else
      const int tid = static_cast<int>(threadIdx.x);
      const int mTile = cute::get<0>(tileCoord);
      const int nTile = cute::get<1>(tileCoord);
      const int kTiles = K / bK;
      const int sfAtomsK = K / (mx::SF_K_TILE * mx::BLOCK_SIZE); // SF atoms along K

      // align the workspace
      uint8_t* smem = static_cast<uint8_t*>(workspace);
      smem += (Alignment - (static_cast<uint32_t>(__cvta_generic_to_shared(smem)) % Alignment)) % Alignment;
      const auto sA = cute::make_tensor(cute::make_smem_ptr(smem + offA), SmemLayoutA{}); // (bM, bK, stages)
      const auto sB = cute::make_tensor(cute::make_smem_ptr(smem + offB), SmemLayoutB{}); // (bN, bK, stages)
      const auto sB2 = cute::make_tensor(cute::make_smem_ptr(smem + offB2), SmemLayoutB{});
      ElementSF* const sfSmem = reinterpret_cast<ElementSF*>(smem + offSF);

      const auto mA = cute::make_tensor(cute::make_gmem_ptr(a),
        cute::make_layout(cute::make_shape(M, K), cute::make_stride(K, cute::_1{})));
      const auto mB = cute::make_tensor(cute::make_gmem_ptr(b),
        cute::make_layout(cute::make_shape(N, K), cute::make_stride(K, cute::_1{})));
      const auto mB2 = cute::make_tensor(cute::make_gmem_ptr(b2),
        cute::make_layout(cute::make_shape(N, K), cute::make_stride(K, cute::_1{})));
      const auto gA = cute::local_tile(mA, cute::make_shape(cute::Int<bM>{}, cute::Int<bK>{}),
                                       cute::make_coord(mTile, cute::_)); // (bM, bK, kTiles)
      const auto gB = cute::local_tile(mB, cute::make_shape(cute::Int<bN>{}, cute::Int<bK>{}),
                                       cute::make_coord(nTile, cute::_)); // (bN, bK, kTiles)
      const auto gB2 = cute::local_tile(mB2, cute::make_shape(cute::Int<bN>{}, cute::Int<bK>{}),
                                        cute::make_coord(nTile, cute::_));

      // ---------------- producer ----------------
      GmemCopy gmemCopy;
      const auto thrCopy = gmemCopy.get_thread_slice(tid);
      const auto tAgA = thrCopy.partition_S(gA);
      const auto tAsA = thrCopy.partition_D(sA);
      const auto tBgB = thrCopy.partition_S(gB);
      const auto tBsB = thrCopy.partition_D(sB);
      const auto tB2gB2 = thrCopy.partition_S(gB2);
      const auto tB2sB2 = thrCopy.partition_D(sB2);
      // SF: thread t < sfThreads copies 16 B t of the SF atoms of a stage, [A atom | B atoms | B2 atoms], each
      // 512 B = 32 threads. The atom of (row tile, k atom) is at ((rowTile * sfAtomsK) + kAtom) * 512
      const uint8_t* sfSrc = nullptr;
      if (tid < sfThreads) {
        const int which = tid / 32;
        const size_t rowTile = which == 0 ? static_cast<size_t>(mTile) :
          static_cast<size_t>(nTile) * bN / mx::SF_ROW_TILE + (which - 1) % sfbAtoms;
        const uint8_t* const base = which == 0 ? sfa : (which <= sfbAtoms ? sfb : sfb2);
        sfSrc = base + rowTile * sfAtomsK * mx::SF_ATOM_BYTES + (tid % 32) * 16;
      }
      auto loadTile = [&](const int& stage, const int& kt) {
        cute::copy(gmemCopy, tAgA(cute::_, cute::_, cute::_, kt), tAsA(cute::_, cute::_, cute::_, stage));
        cute::copy(gmemCopy, tBgB(cute::_, cute::_, cute::_, kt), tBsB(cute::_, cute::_, cute::_, stage));
        if constexpr (gated) {
          cute::copy(gmemCopy, tB2gB2(cute::_, cute::_, cute::_, kt), tB2sB2(cute::_, cute::_, cute::_, stage));
        }
        if (tid < sfThreads) {
          const size_t kAtom = static_cast<size_t>(kt) * bK / (mx::SF_K_TILE * mx::BLOCK_SIZE);
          using Chunk = cute::uint128_t;
          cute::SM80_CP_ASYNC_CACHEGLOBAL<Chunk>::copy(
            *reinterpret_cast<const Chunk*>(sfSrc + kAtom * mx::SF_ATOM_BYTES),
            *reinterpret_cast<Chunk*>(sfSmem + stage * sfBytesStage + tid * 16));
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

      const auto smemTiledCopyA = cute::make_tiled_copy_A(cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, uint8_t>{}, tiledMma);
      const auto smemThrCopyA = smemTiledCopyA.get_thread_slice(tid);
      const auto tCsA = smemThrCopyA.partition_S(cute::as_position_independent_swizzle_tensor(sA)); // (CPY, CPY_M, CPY_K, PIPE)
      auto tCrA_view = smemThrCopyA.retile_D(tCrA);
      const auto smemTiledCopyB = cute::make_tiled_copy_B(cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, uint8_t>{}, tiledMma);
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
      constexpr int kBlocksPerTile = bK / mx::BLOCK_SIZE;
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
        // bK == 64: one SF atom (4 k-blocks) serves two k-tiles, this one uses bytes [2 (kt % 2), 2 (kt % 2) + 2)
        const int kbOff = bK == 64 ? 2 * (kt % 2) : 0;
        ElementSF* const sfStage = sfSmem + stage * sfBytesStage;
        const auto sSFA = cute::make_tensor(cute::make_smem_ptr(sfStage + kbOff), SmemLayoutSFA{});
        const auto sSFB = cute::make_tensor(cute::make_smem_ptr(sfStage + mx::SF_ATOM_BYTES + sfbHalf + kbOff),
                                            SmemLayoutSFB{});
        const auto sSFB2 = cute::make_tensor(
          cute::make_smem_ptr(sfStage + mx::SF_ATOM_BYTES * (1 + sfbAtoms) + sfbHalf + kbOff), SmemLayoutSFB{});
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
        for (int kb = 0; kb < kBlocksPerTile; ++kb) {
          if (kb + 1 < kBlocksPerTile) {
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

  /// The MXFP8 expert GEMMs of FlashMoE (sm_12x: 99 KB of shared memory per block, 256 threads), picked with
  /// tests/gemmMXFP8.cu on an RTX 5090:
  ///  - GEMM0, gated: 128 x (bN + bN) x 64, 3 stages (A is read once for both B operands); vanilla: bK 128, 2 stages
  ///  - GEMM1: 128 x bN x 128, 2 stages
  constexpr int MXFP8Threads = 256;

  template <int bM, int bN, int threads, bool gated>
  using MXFP8GEMM0 = MXFP8Mainloop<bM, bN, gated ? 64 : 128, gated ? 3 : 2, threads, gated>;

  template <int bM, int bN, int threads>
  using MXFP8GEMM1 = MXFP8Mainloop<bM, bN, 128, 2, threads, false>;
}
#endif //FLASHMOE_TILE_SM120_MXFP8_CUH
