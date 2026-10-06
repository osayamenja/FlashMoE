/******************************************************************************
 * Copyright (c) 2026, Osayamen Jonathan Aimuyo.
 ******************************************************************************/
//
// Created by osayamen on 1/4/26.
//

#ifndef FLASHMOE_DISPATCH_CUH
#define FLASHMOE_DISPATCH_CUH
#include <cuda/std/bit>
#include <cuda/ptx>
#include <nvshmem.h>

#include "infra/fp4.cuh"
#include "infra/heap.cuh"
#include "infra/math.cuh"
#include "infra/packed.cuh"
#include "infra/signal.cuh"
#include "infra/structures.cuh"
#include "infra/trace.cuh"
#include "infra/vt.cuh"

namespace flashmoe
{
  /// Quantizing dispatch, see quantizeTokens: the number of 16 B chunks that every lane of a warp loads back to back
  /// and quantizes together.
  constexpr int QUANT_CHUNK_GROUP = 4;

  /// Offset of the scale of the k-block kBlock of a row from that of the k-block 0 in the canonical layout of
  /// mx::sfOffset: sfOffset(row, kBlock, n) == sfOffset(row, 0, n) + sfBlockOffset(kBlock)
  __device__ __forceinline__
  constexpr size_t sfBlockOffset(const int& kBlock) {
    return static_cast<size_t>(kBlock / mx::SF_K_TILE) * mx::SF_ATOM_BYTES + kBlock % mx::SF_K_TILE;
  }

  /// Global stores that are executed only if pred holds. They are predicated instructions and not branches, so that
  /// the quantization of several chunks stays one straight line of code.
  __device__ __forceinline__
  void storeByteIf(const bool& pred, uint8_t* const& ptr, const uint32_t& v) {
    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %0, 0;\n\t@p st.global.u8 [%1], %2;\n}"
      :: "r"(static_cast<uint32_t>(pred)), "l"(ptr), "r"(v) : "memory");
  }

  __device__ __forceinline__
  void storeWordIf(const bool& pred, uint8_t* const& ptr, const uint32_t& v) {
    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %0, 0;\n\t@p st.global.u32 [%1], %2;\n}"
      :: "r"(static_cast<uint32_t>(pred)), "l"(ptr), "r"(v) : "memory");
  }

  __device__ __forceinline__
  void storeWord2If(const bool& pred, uint8_t* const& ptr, const uint32_t& lo, const uint32_t& hi) {
    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %0, 0;\n\t@p st.global.v2.u32 [%1], {%2, %3};\n}"
      :: "r"(static_cast<uint32_t>(pred)), "l"(ptr), "r"(lo), "r"(hi) : "memory");
  }

  /// fp4::ue8m0FromAmaxFP4 with selects instead of branches: the branches end the straight line of code that
  /// quantizeChunks wants. Same results, see the bit-identity test.
  __device__ __forceinline__
  uint8_t ue8m0FromAmaxFP4Select(const float& amax) {
    const uint32_t bits = __float_as_uint(amax) & 0x7FFFFFFFU;
    const uint32_t biased = bits >> 23;
    const int e = static_cast<int>(biased) - 2 + ((bits & 0x7FFFFFU) > 0x400000U ? 1 : 0);
    uint32_t sfByte = static_cast<uint32_t>(max(e, static_cast<int>(mx::UE8M0_MIN)));
    sfByte = biased == 0xFFU ? static_cast<uint32_t>(mx::UE8M0_NAN) : sfByte;
    return static_cast<uint8_t>(sfByte);
  }

  /// The factor the elements of a block with the UE8M0 scale e are multiplied by before the conversion, 2^(127 - e)
  /// (mx::ue8m0ToRcpFloat), and 1 for a NaN / Inf block (scale mx::UE8M0_NAN), with selects instead of branches.
  __device__ __forceinline__
  float ue8m0EncodeScaleSelect(const uint8_t& e) {
    uint32_t bits = (254U - e) << 23;
    bits = e == 254 ? 0x00400000U : bits;
    bits = e == mx::UE8M0_NAN ? 0x3F800000U : bits;
    return __uint_as_float(bits);
  }

  /// Two bf16 in the low (first) and the high (second) half of a 32 bit word
  __device__ __forceinline__
  __nv_bfloat162 asBF162(const uint32_t& bits) {
    return *reinterpret_cast<const __nv_bfloat162*>(&bits);
  }

  __device__ __forceinline__
  uint32_t asBits(const __nv_bfloat162& v) {
    return *reinterpret_cast<const uint32_t*>(&v);
  }

  /// The maxima of |x| over the even and the odd elements of the 8 bf16 of a 16 B chunk, in the low and the high half of
  /// the result. NaN propagates: the result is NaN if an element is NaN.
  __device__ __forceinline__
  __nv_bfloat162 chunkAbsMax(const uint4& raw) {
    const auto a0 = __habs2(asBF162(raw.x));
    const auto a1 = __habs2(asBF162(raw.y));
    const auto a2 = __habs2(asBF162(raw.z));
    const auto a3 = __habs2(asBF162(raw.w));
    return __hmax2_nan(__hmax2_nan(a0, a1), __hmax2_nan(a2, a3));
  }

  /// Block-scaled recipe applied while a token is copied to its packet. Every lane converts `groups` 16 B chunks of a
  /// row (raw), the chunks k0, k0 + 32, ..., i.e., 8 consecutive bf16 [8 k, 8 k + 8) each:
  /// recipe mxfp8: 8 E4M3 bytes at dst + 8 k and the UE8M0 scale of the 32 elements to sf, in the canonical layout
  /// (mx::sfOffset) of a [rows, H] matrix, where sfRow = mx::sfOffset(row, 0, H / 32) is the offset of the row.
  /// Bit-identical to mx::quantizeMXFP8.
  /// recipe mxfp4 / nvfp4: 8 E2M1 (two elements per byte, the even element in the low nibble) = 4 B at dst + 4 k and
  /// the scale of the 32 (UE8M0) or 16 (UE4M3) elements to sf. st is the encode scale 1 / gs of the nvfp4 token tensor
  /// and unused for mxfp4. Bit-identical to fp4::quantizeFP4 with the same st.
  /// The lanes of a scale block (4 for mxfp8 and mxfp4, 2 for nvfp4) exchange their amax with shuffles: all 32 lanes of
  /// the warp must call this function together, with the same arguments except for raw and k0. The chunks past the
  /// end of the row (k >= chunks) may hold any bits: they only share scale blocks with each other, as H is a multiple
  /// of 32, and they are not stored.
  /// The groups are independent and there is no branch in here, so that they are interleaved by the compiler: the
  /// amax of all of them, then the shuffles of all of them (two chunks per shuffle, packed as bf16x2), then the
  /// conversions and the stores. Only the NaN-ness and the magnitude of the amax matter to the scales, so that the
  /// amax is the NaN propagating maximum of the bf16, which is the same as the maximum of the integer |bits|
  /// (monotonic in |x|, NaN > Inf > finite) that the quantizer kernels use.
  template <QuantRecipe r, typename Element, int groups>
    requires(isBlockScaled(r) && cuda::std::is_same_v<Element, __nv_bfloat16>)
  __device__ __forceinline__
  void quantizeChunks(const uint4 (&raw)[groups], const int& k0, const int& chunks,
                      uint8_t* __restrict__ const& dst, uint8_t* __restrict__ const& sf, const size_t& sfRow,
                      const float& st) {
    constexpr int vec = 16 / sizeof(Element); // elements per lane
    constexpr int lanesPerBlock = quantSFVecSize(r) / vec;
    constexpr int pairs = (groups + 1) / 2;
    static_assert(vec == 8 && WARP_SIZE % lanesPerBlock == 0);
    __nv_bfloat162 amax[pairs]; // the amax of the chunks 2 i (low half) and 2 i + 1 (high half)
    #pragma unroll
    for (int i = 0; i < pairs; ++i) {
      // (even, odd) maxima of the two chunks -> (chunk 2 i, chunk 2 i + 1) with the maxima of the even and the odd
      // elements merged
      const uint32_t a = asBits(chunkAbsMax(raw[2 * i]));
      const uint32_t b = asBits(chunkAbsMax(raw[cute::min(2 * i + 1, groups - 1)]));
      amax[i] = __hmax2_nan(asBF162(__byte_perm(a, b, 0x5410U)), asBF162(__byte_perm(a, b, 0x7632U)));
    }
    #pragma unroll
    for (int o = lanesPerBlock / 2; o > 0; o >>= 1) {
      #pragma unroll
      for (int i = 0; i < pairs; ++i) {
        amax[i] = __hmax2_nan(amax[i], asBF162(__shfl_xor_sync(0xFFFFFFFFU, asBits(amax[i]), o)));
      }
    }
    // 32 chunks are 32 / lanesPerBlock scale blocks, a multiple of the 4 k-blocks of an atom of the layout of the
    // scales: the scale of the group g is that of the group 0 plus g atom rows
    static_assert((WARP_SIZE / lanesPerBlock) % mx::SF_K_TILE == 0);
    constexpr size_t sfGroupBytes = static_cast<size_t>(WARP_SIZE / lanesPerBlock / mx::SF_K_TILE) * mx::SF_ATOM_BYTES;
    uint8_t* __restrict__ const sfLane = sf + sfRow + sfBlockOffset(k0 / lanesPerBlock);
    const bool leader = k0 % lanesPerBlock == 0;
    #pragma unroll
    for (int g = 0; g < groups; ++g) {
      const int k = k0 + g * WARP_SIZE;
      const bool valid = k < chunks;
      // a bf16 is the upper half of the float with the same value
      const uint32_t amaxPair = asBits(amax[g / 2]);
      const float blockAmax = __uint_as_float(g % 2 == 0 ? amaxPair << 16 : amaxPair & 0xFFFF0000U);
      const uint32_t words[4] = {raw[g].x, raw[g].y, raw[g].z, raw[g].w};
      float x[vec];
      #pragma unroll
      for (int i = 0; i < vec / 2; ++i) {
        x[2 * i] = __uint_as_float(words[i] << 16);
        x[2 * i + 1] = __uint_as_float(words[i] & 0xFFFF0000U);
      }
      uint8_t sfByte;
      if constexpr (r == QuantRecipe::mxfp8) {
        sfByte = mx::ue8m0FromAmax(blockAmax);
        // NaN / Inf block: the scale is NaN, the payload is irrelevant but kept finite-saturated
        const float inv = ue8m0EncodeScaleSelect(sfByte);
        uint32_t w[vec / 4];
        #pragma unroll
        for (int i = 0; i < vec / 4; ++i) {
          const uint32_t lo = mx::packE4M3x2(x[4 * i] * inv, x[4 * i + 1] * inv);
          const uint32_t hi = mx::packE4M3x2(x[4 * i + 2] * inv, x[4 * i + 3] * inv);
          w[i] = lo | (hi << 16);
        }
        storeWord2If(valid, dst + static_cast<size_t>(k) * vec, w[0], w[1]);
      }
      else {
        float enc;
        if constexpr (r == QuantRecipe::mxfp4) {
          sfByte = ue8m0FromAmaxFP4Select(blockAmax);
          enc = ue8m0EncodeScaleSelect(sfByte);
        }
        else {
          sfByte = fp4::nvfp4ScaleFromAmax(blockAmax, st);
          enc = fp4::blockEncodeScale<r>(sfByte, st);
        }
        uint32_t w = 0U;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
          w |= static_cast<uint32_t>(fp4::packE2M1x2(x[2 * i] * enc, x[2 * i + 1] * enc)) << (8 * i);
        }
        // 8 E2M1 = 4 B
        storeWordIf(valid, dst + static_cast<size_t>(k) * (vec / 2), w);
      }
      storeByteIf(valid && leader, sfLane + g * sfGroupBytes, static_cast<uint32_t>(sfByte));
    }
  }

  /// Quantizing copy of the `count` tokens that a CTA dispatches to the packet of one expert (recipe mxfp8, mxfp4,
  /// nvfp4): the row p in [0, count) of the CTA is the row first + p * stride of the packet and holds the token
  /// tokenIds[first + p * stride].tokenIdx. dst is the E4M3 (H bytes per row) or E2M1 (H / 2 bytes per row) packet
  /// and sf its scale factor region, see quantizeChunks.
  /// Warp per token: the warp w of the CTA takes the rows w, w + warps, ... (warps = threads / 32). The 16 B chunks of
  /// a token are split into units of chunkGroup chunks per lane (a lane takes the chunks lane, lane + 32, ...). The
  /// loads of a unit are issued back to back, one unit before it is quantized: a lane always has chunkGroup loads in
  /// flight and chunkGroup independent quantizations to interleave (the previous version dispatched the tokens one
  /// after the other, each with a dependent load -> amax shuffle -> quantize -> store chain per lane; deeper
  /// prefetching did not help). The ids of the tokens are loaded two tokens ahead of the loads. The last unit of a row,
  /// when H / 8 is not a multiple of chunkGroup * 32, is quantized as if it were full: the chunks past the end of
  /// the row are loaded from the last chunk, they only share scale blocks with each other (H is a multiple of 32)
  /// and are not stored.
  /// Nothing here synchronizes the CTA, the caller does that before the packet is announced; the control flow is
  /// uniform across the warp, as quantizeChunks is warp-collective.
  template <int threads, QuantRecipe r, int chunkGroup, typename Element>
    requires(isBlockScaled(r) && cuda::std::is_same_v<Element, __nv_bfloat16>)
  __device__ __forceinline__
  void quantizeTokens(const Element* __restrict__ const& tokens, const TPS* __restrict__ const& tokenIds,
                      uint8_t* __restrict__ const& dst, uint8_t* __restrict__ const& sf, const int& first,
                      const int& stride, const int& count, const int& H, const float& st) {
    constexpr int vec = 16 / sizeof(Element); // elements per lane
    constexpr int warps = threads / WARP_SIZE;
    constexpr int span = WARP_SIZE * chunkGroup; // chunks of a unit
    static_assert(threads % WARP_SIZE == 0 && chunkGroup >= 1);
    // the shuffle tells the compiler that the warp index, and so the control flow below, is uniform across the warp
    const int warp = __shfl_sync(0xFFFFFFFFU, static_cast<int>(threadIdx.x) / WARP_SIZE, 0);
    const int lane = static_cast<int>(threadIdx.x) % WARP_SIZE;
    const int chunks = H / vec; // 16 B chunks of a row
    const int unitsPerRow = cute::ceil_div(chunks, span);
    const size_t rowBytes = static_cast<size_t>(H) * quantElementBits(r) / 8;
    const int sfBlocks = H / quantSFVecSize(r);
    // this warp quantizes the rows (slots) warp + s * warps, s < slots
    const int slots = count > warp ? (count - warp + warps - 1) / warps : 0;
    const auto tokenOf = [&](const int& s) {
      return s < slots ? tokenIds[first + (warp + s * warps) * stride].tokenIdx : 0U;
    };
    uint4 bufA[chunkGroup];
    uint4 bufB[chunkGroup];
    // loads: the unit u of the slot s with the chunks [u * span, (u + 1) * span) of the token idCur,
    // the tokens of the slots s + 1 and s + 2 are idNext and idNext2
    int fs = 0;
    int fu = 0;
    uint32_t idCur = tokenOf(0);
    uint32_t idNext = tokenOf(1);
    uint32_t idNext2 = tokenOf(2);
    const auto fetch = [&](uint4 (&b)[chunkGroup]) {
      if (fs < slots) {
        const auto* __restrict__ src = reinterpret_cast<const uint4*>(tokens + static_cast<size_t>(idCur) * H);
        #pragma unroll
        for (int g = 0; g < chunkGroup; ++g) {
          b[g] = __ldg(src + cute::min(fu * span + g * WARP_SIZE + lane, chunks - 1));
        }
      }
      if (fu + 1 == unitsPerRow) {
        fu = 0;
        ++fs;
        idCur = idNext;
        idNext = idNext2;
        idNext2 = tokenOf(fs + 2);
      }
      else {
        ++fu;
      }
    };
    // quantization: the unit u of the slot s
    int s = 0;
    int u = 0;
    int intraIdx = first + warp * stride; // row of the packet of the slot
    uint8_t* __restrict__ row = dst + static_cast<size_t>(intraIdx) * rowBytes;
    size_t sfRow = mx::sfOffset(intraIdx, 0, sfBlocks);
    // quantizes the unit that is in cur and fetches the unit that follows it into next
    const auto step = [&](uint4 (&cur)[chunkGroup], uint4 (&next)[chunkGroup]) {
      fetch(next);
      quantizeChunks<r, Element>(cur, u * span + lane, chunks, row, sf, sfRow, st);
      if (u + 1 == unitsPerRow) {
        u = 0;
        ++s;
        intraIdx = first + (warp + s * warps) * stride;
        row = dst + static_cast<size_t>(intraIdx) * rowBytes;
        sfRow = mx::sfOffset(intraIdx, 0, sfBlocks);
      }
      else {
        ++u;
      }
    };
    fetch(bufA);
    // two units per trip, so that the buffers swap their roles without copies
    while (s < slots) {
      step(bufA, bufB);
      if (s >= slots) {
        break;
      }
      step(bufB, bufA);
    }
  }

  // MoE tokens dispatch
  // recipe == mxfp8: tokens are quantized to E4M3 while they are copied, see quantizeTokens.
  // The packet holds the E4M3 rows with an H-byte pitch and its scale factors are written to the scale factor region
  // of the symmetric heap, see Heap::advanceSF.
  // recipe == mxfp4 / nvfp4: the same with E2M1 rows, i.e., an H / 2-byte pitch, see quantizeTokens. st is the
  // nvfp4 encode scale of the tokens, 1 / inputScale.
  template <
    Topology topo,
    int threads, // not necessarily required, but may help perf a little?
    int bM, // for GEMM0
    int bN, // for GEMM0
    QuantRecipe recipe = QuantRecipe::none,
    int batch = cute::min(bM, 8), // number of tokens dispatched in one go by a CTA
    typename Element
  >
  __forceinline__ __device__
  void dispatch(const int& H, const int& E,
                const Heap& symHeap,
                const int& EC, const int& roundEC, // EC rounded up to bM
                const int& epRank, const int& world,
                const int& superBlockSize, const int& blocks,
                const Element* __restrict__ const& tokens, // [S, H]
                uint64_t* __restrict__ signals, // [world, num_local_experts]
                const int* __restrict__ const& expertCounts, // [E]
                const TPS* __restrict__ const& _tokenIds, // [E, roundEC] -> generated by ::gate::forward
                uint* __restrict__ const& dispatchSync, // [E]
                const PEL* __restrict__ const& expertLookupInfo, // [E]
                cuda::std::byte* __restrict__ const& workspace, // shared memory
                const uint16_t& stateNumber,
                const float& st = 1.f) { // nvfp4: encode scale of the tokens, 1 / inputScale
    // Assumptions are below:
    // workspace is in shared memory;
    // ceil(EC / bM) * num_local_experts <= UINT16_MAX
    // tokens and tokensId are at least 32-byte aligned.
    // we require superblocks to be equally sized
    // Block-scaled recipes: bM == 128 and H % 128 == 0, see initialize()
    constexpr bool quantize = isBlockScaled(recipe);
    static_assert(!quantize || bM == mx::SF_ROW_TILE);
    // quantized packets: bytes of a row, and K elements per scale factor
    [[maybe_unused]] constexpr int sfVec = quantSFVecSize(recipe);
    [[maybe_unused]] const size_t rowBytes = static_cast<size_t>(H) * quantElementBits(recipe) / 8;
    const auto numSuperBlocks = blocks / superBlockSize;
    const int superBlockIdx = static_cast<int>(blockIdx.x) / superBlockSize;
    if (superBlockIdx >= numSuperBlocks) {
      return;
    }
    const int lBid = static_cast<int>(blockIdx.x) % superBlockSize;
    const bool isLeader = !lBid && !threadIdx.x;
    /// Populate Data Structures
    size_t offset = 0;
    auto* __restrict__ enL = reinterpret_cast<PEL*>(workspace);
    for (int i = threadIdx.x; i < E; i += threads) {
      enL[i] = expertLookupInfo[i];
    }
    offset += rTCL<PEL>(E);
    auto* __restrict__ seC = reinterpret_cast<int*>(workspace + offset);
    for (int i = threadIdx.x; i < E; i += threads) {
      seC[i] = expertCounts[i];
    }
    // peer total tokens
    auto* __restrict__ sPTT = seC + E;
    for (int i = threadIdx.x; i < world; i += threads) {
      sPTT[i] = 0;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < E; i += threads) {
      const auto peer = enL[i].peer;
      const auto nTokens = cute::min(seC[i], EC);
      const auto tilesM = cute::ceil_div(nTokens, bM);
      atomicAdd_block(sPTT + peer, tilesM);
    }
    __syncthreads();
    // Update lookup table
    for (int i = threadIdx.x; i < E; i += threads) {
      auto lInfo = enL[i];
      lInfo.eC = seC[i];
      // assert(sPTT[lInfo.peer] <= UINT16_MAX)
      lInfo.pTTt = static_cast<uint16_t>(sPTT[lInfo.peer]);
      enL[i] = lInfo;
    }
    __syncthreads();
    const auto* __restrict__ expertLookup = enL;
    uint32_t rTID [batch];
    constexpr auto eWidth = ElementWidth<Element, bN, MAX_ACCESS_ALIGNMENT>; // 256B in Blackwell, 128B others
    using VTD = VectorTypeDescriptor<typename RawType<Element>::type, ElementAlignmentForWidth<Element, bN, eWidth>>;
    using VectorElement = VTD::VectorType;
    static_assert(cuda::std::is_trivially_copyable_v<VectorElement>);
    // below is fine because we enforce that H % bN == 0 and bN % vectorWidth == 0
    const int vH = H / VTD::VectorWidth::value;
    const auto* __restrict__ vTokens = reinterpret_cast<const VectorElement*>(tokens);

    const auto tokenIds = make_tensor(cute::make_gmem_ptr(_tokenIds),
      cute::make_layout(cute::make_shape(E, roundEC), cute::LayoutRight{}));
    static_assert(batch >= 1 && bM % batch == 0);
    for (int expertIdx = superBlockIdx; expertIdx < E; expertIdx += numSuperBlocks) {
      const auto lI = expertLookup[expertIdx];
      const auto flagOffset = epRank * lI.nLocalExperts + lI.expertLocalIdx;
      // note that for DropTokens:no, we set EC to be the entire sequence length (S) to accommodate the worst-case
      const auto routedTokens = cute::min(lI.eC, EC);
      FLASHMOE_TRACE_EVENT_T0(dispatchBegin, expertIdx, routedTokens);
      auto* __restrict__ peerHeap = reinterpret_cast<VectorElement*>(topo == Topology::MIXED && lI.isRemote ?
        symHeap.advance<0, 0>(lI.peer, lI.expertLocalIdx): lI.remoteSHeap +
        symHeap.advanceOffset<0, 1>(epRank, lI.expertLocalIdx));
      // scale factors of the packet: either in the symmetric heap of the peer, or in our send buffer
      // from where they are transferred via RDMA
      [[maybe_unused]] uint8_t* __restrict__ peerSF = nullptr;
      if constexpr (quantize) {
        peerSF = reinterpret_cast<uint8_t*>(topo == Topology::MIXED && lI.isRemote ?
          symHeap.advanceSF<0, 0>(lI.peer, lI.expertLocalIdx) :
          lI.remoteSHeap + symHeap.advanceSFOffset<0, 1>(epRank, lI.expertLocalIdx));
      }
      if (routedTokens) {
        // tokens of this CTA
        const auto cTokens = routedTokens / superBlockSize + (lBid < routedTokens % superBlockSize);
        if constexpr (quantize) {
          // warp per token quantize + copy: the packet is E4M3 (H-byte row pitch) or E2M1 (H / 2 bytes)
          quantizeTokens<threads, recipe, QUANT_CHUNK_GROUP>(tokens,
            _tokenIds + static_cast<size_t>(expertIdx) * roundEC, reinterpret_cast<uint8_t*>(peerHeap), peerSF, lBid,
            superBlockSize, static_cast<int>(cTokens), H, st);
        }
        // coalesced vectorized copy of the tokens as they are, all of them are done above for the quantized recipes
        const auto partition = quantize ? 0 : cTokens;
        const auto trips = partition / batch;
        for (int i = 0; i < trips; ++i) {
          #pragma unroll
          for (int j = 0; j < batch; ++j) {
            // intentionally redundant copies here to avoid any memory completion points (CTA barriers) within this loop
            const auto v = tokenIds(expertIdx, lBid + (j + i * batch) * superBlockSize);
            // unpack
            rTID[j] = v.tokenIdx;
          }
          // Communicate these tokens
          // TODO For SM 90 and above, optimize the below using _TMA + warp splitting_
          // Assume a CTA of 4 warps needs to copy 8 tokens
          // Currently, _all warps_ in the CTA cooperate to copy each token, thus taking #tokens = 8 iterations
          // With warp splitting, each warp uses TMA to copy a single token (GMEM->SMEM->GMEM)
          // so that, for this example, it would take # tokens / # warps == 2 iterations.
          // Technically, a single thread could issue the TMA call but a warp is preferred to sidestep divergence
          #pragma unroll
          for (int j = 0; j < batch; ++j) {
            const auto tokenIdx = rTID[j];
            const auto intraIdx = lBid + (j + i * batch) * superBlockSize;
            auto* __restrict__ localPH = peerHeap + intraIdx * vH;
            const auto* __restrict__ aP = vTokens + tokenIdx * vH;
            // coalesced vectorized copy
            for (int k = static_cast<int>(threadIdx.x); k < vH; k += threads) {
              const auto v = cuda::ptx::ld_nc_L1_no_allocate_L2_256B(cuda::ptx::space_global, aP + k);
              cuda::ptx::st(cuda::ptx::space_global, localPH + k, v);
            }
          }
        }
        // residue
        if (const auto residue = partition - trips * batch; residue) {
          #pragma unroll
          for (int j = 0; j < batch; ++j) {
            if (j < residue) {
              const auto v = tokenIds(expertIdx, lBid + (j + trips * batch) * superBlockSize);
              rTID[j] = v.tokenIdx;
            }
          }
          #pragma unroll
          for (int j = 0; j < batch; ++j) {
            if (j < residue) {
              const auto tokenIdx = rTID[j];
              const auto intraIdx = lBid + (j + trips * batch) * superBlockSize;
              auto* __restrict__ localPH = peerHeap + intraIdx * vH;
              const auto* __restrict__ aP = vTokens + tokenIdx * vH;
              for (int k = static_cast<int>(threadIdx.x); k < vH; k += threads) {
                const auto v = cuda::ptx::ld_nc_L1_no_allocate_L2_256B(cuda::ptx::space_global, aP + k);
                cuda::ptx::st(cuda::ptx::space_global, localPH + k, v);
              }
            }
          }
        }
        __syncthreads();
        if (!threadIdx.x) {
          bool shouldNotify = false;
          if (superBlockSize > 1) {
            // last block notifies
            cuda::atomic_ref<uint, cuda::thread_scope_device> dS{*(dispatchSync + expertIdx)};
            shouldNotify = dS.fetch_add(1, cuda::memory_order_acq_rel) + 1 == superBlockSize;
            if (shouldNotify) {
              // acq_rel above allows us to safely do the below.
              // We assume this counter is used _once_ (this function)
              // per kernel because we defer grid-wide synchronization to kernel teardown.
              dS.store(0, cuda::memory_order_relaxed);
            }
          }
          else {
            shouldNotify = true;
          }
          if (shouldNotify) {
            // let's finalize this transfer.
            const auto sigPayload = SignalPayload<PacketStage::initial>{
              routedTokens,
              lI.pTTt,
              stateNumber
            };
            if (topo == Topology::MIXED && lI.isRemote) {
              if constexpr (quantize) {
                // The scale factors of the rows [0, ceil(routedTokens / 128) * 128) are a prefix of the
                // canonical layout. Send them ahead of the packet: the fence orders them before the packet and
                // its signal, which is what the receiver waits on.
                const auto sfRows = cute::ceil_div(routedTokens, static_cast<uint>(mx::SF_ROW_TILE)) *
                                    mx::SF_ROW_TILE;
                nvshmem_putmem_nbi(symHeap.advanceSF<0, 1>(epRank, lI.expertLocalIdx), peerSF,
                                   mx::sfBytes(sfRows, H / sfVec), lI.pe);
                nvshmem_fence();
              }
              // do RDMA transfer + signal
              nvshmem_putmem_signal_nbi(
                symHeap.advance<0, 1>(epRank, lI.expertLocalIdx),
                peerHeap,
                quantize ? static_cast<size_t>(routedTokens) * rowBytes : sizeof(VectorElement) * routedTokens * vH,
                signals + flagOffset,
                cuda::std::bit_cast<uint64_t>(sigPayload),
                NVSHMEM_SIGNAL_SET,
                lI.pe);
            }
            else {
              cuda::atomic_ref<uint64_t, cuda::thread_scope_system> sf{*(lI.remoteSFlags + flagOffset)};
              // we've done the DMA transfer already, so we set the signal only
              sf.store(cuda::std::bit_cast<uint64_t>(sigPayload), cuda::memory_order_release);
            }
          }
        }
      }
      else if (isLeader) {
        // single thread sends a noop control message to unblock the remote peer
        // Pack payload into a single signal word
        const auto sigPayload = SignalPayload<PacketStage::initial>{
          0U, // noop as there are no tokens to be processed
          lI.pTTt,
          stateNumber
        };
        if (topo == Topology::MIXED && lI.isRemote) {
          // transmit signal via RDMA
          nvshmemx_signal_op(signals + flagOffset,
          cuda::std::bit_cast<uint64_t>(sigPayload), NVSHMEM_SIGNAL_SET, lI.pe);
        }
        else {
          cuda::atomic_ref<uint64_t, cuda::thread_scope_system> sf{*(lI.remoteSFlags + flagOffset)};
          sf.store(cuda::std::bit_cast<uint64_t>(sigPayload), cuda::memory_order_release);
        }
      }
      FLASHMOE_TRACE_EVENT_T0(dispatchEnd, expertIdx, 0);
    }
  }
}
#endif //FLASHMOE_DISPATCH_CUH
