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
  /// MXFP8 recipe applied while a token is copied to its packet: quantizes the row src[0, H) to E4M3 bytes dst[0, H)
  /// and writes the UE8M0 scale of every 32 elements to sf, the canonical layout (mx::sfOffset) of a [rows, H] matrix,
  /// at row sfRow. Bit-identical to mx::quantizeMXFP8.
  /// Every thread converts 8 consecutive elements (16 B of input); the 4 adjacent threads that share a block of
  /// 32 elements exchange their amax with shuffles, so all `threads` threads of the CTA must call this function
  /// together, with the same arguments except for threadIdx. The loop bound is uniform across the CTA,
  /// the lanes past the end of the row participate in the shuffles with amax = 0.
  template <int threads, typename Element>
    requires(sizeof(Element) == 2 && mx::MXInput<Element>)
  __device__ __forceinline__
  void quantizeToken(const Element* __restrict__ const& src, uint8_t* __restrict__ const& dst,
                     uint8_t* __restrict__ const& sf, const int& sfRow, const int& H) {
    constexpr int vec = 16 / sizeof(Element); // elements per thread
    constexpr int lanesPerBlock = mx::BLOCK_SIZE / vec;
    static_assert(vec == 8 && threads % lanesPerBlock == 0);
    const int chunks = H / vec;
    const int numKBlocks = H / mx::BLOCK_SIZE;
    const int trips = cute::ceil_div(chunks, threads);
    for (int t = 0; t < trips; ++t) {
      const int k = static_cast<int>(threadIdx.x) + t * threads;
      const bool valid = k < chunks;
      float x[vec];
      uint32_t amaxBits = 0U; // |x| as an integer is monotonic in |x| and orders NaN > Inf > finite
      if (valid) {
        alignas(16) Element e[vec];
        *reinterpret_cast<uint4*>(e) = __ldg(reinterpret_cast<const uint4*>(src) + k);
        #pragma unroll
        for (int i = 0; i < vec; ++i) {
          x[i] = mx::detail::toFloat(e[i]);
          amaxBits = max(amaxBits, __float_as_uint(x[i]) & 0x7FFFFFFFU);
        }
      }
      #pragma unroll
      for (int o = lanesPerBlock / 2; o > 0; o >>= 1) {
        amaxBits = max(amaxBits, __shfl_xor_sync(0xFFFFFFFFU, amaxBits, o));
      }
      if (valid) {
        const uint8_t sfByte = mx::ue8m0FromAmax(__uint_as_float(amaxBits));
        // NaN / Inf block: the scale is NaN, the payload is irrelevant but kept finite-saturated
        const float inv = sfByte == mx::UE8M0_NAN ? 1.0f : mx::ue8m0ToRcpFloat(sfByte);
        uint32_t w[vec / 4];
        #pragma unroll
        for (int i = 0; i < vec / 4; ++i) {
          const uint32_t lo = mx::packE4M3x2(x[4 * i] * inv, x[4 * i + 1] * inv);
          const uint32_t hi = mx::packE4M3x2(x[4 * i + 2] * inv, x[4 * i + 3] * inv);
          w[i] = lo | (hi << 16);
        }
        *reinterpret_cast<uint2*>(dst + static_cast<size_t>(k) * vec) = make_uint2(w[0], w[1]);
        if (k % lanesPerBlock == 0) {
          sf[mx::sfOffset(sfRow, k / lanesPerBlock, numKBlocks)] = sfByte;
        }
      }
    }
  }

  /// MXFP4 / NVFP4 recipe applied while a token is copied to its packet: quantizes the row src[0, H) to E2M1 (two
  /// elements per byte, the even element in the low nibble) dst[0, H / 2) and writes the scale of every VS elements,
  /// UE8M0 per 32 (mxfp4) or UE4M3 per 16 (nvfp4), to sf, the canonical layout (mx::sfOffset) of a [rows, H] matrix, at
  /// row sfRow. st is the encode scale 1 / gs of the nvfp4 token tensor and unused for mxfp4.
  /// Bit-identical to fp4::quantizeFP4 with the same st.
  /// Every thread converts 8 consecutive elements (16 B of input) into 4 B of output with fp4::quantize8, which is
  /// warp-collective: the lanes of a scale block (4 for mxfp4, 2 for nvfp4) exchange their amax with shuffles, so all
  /// `threads` threads of the CTA must call this function together, with the same arguments except for threadIdx.
  /// The loop bound is uniform across the CTA, the lanes past the end of the row participate with zeros.
  template <int threads, QuantRecipe r, typename Element>
    requires(isFP4(r) && sizeof(Element) == 2 && mx::MXInput<Element>)
  __device__ __forceinline__
  void quantizeTokenFP4(const Element* __restrict__ const& src, uint8_t* __restrict__ const& dst,
                        uint8_t* __restrict__ const& sf, const int& sfRow, const int& H, const float& st) {
    constexpr int vec = 16 / sizeof(Element); // elements per thread
    constexpr int vs = quantSFVecSize(r);
    constexpr int lanesPerBlock = vs / vec;
    static_assert(vec == 8 && threads % WARP_SIZE == 0 && WARP_SIZE % lanesPerBlock == 0);
    const int chunks = H / vec;
    const int numKBlocks = H / vs;
    const int trips = cute::ceil_div(chunks, threads);
    for (int t = 0; t < trips; ++t) {
      const int k = static_cast<int>(threadIdx.x) + t * threads;
      const bool valid = k < chunks;
      float x[vec] = {};
      if (valid) {
        alignas(16) Element e[vec];
        *reinterpret_cast<uint4*>(e) = __ldg(reinterpret_cast<const uint4*>(src) + k);
        #pragma unroll
        for (int i = 0; i < vec; ++i) {
          x[i] = mx::detail::toFloat(e[i]);
        }
      }
      uint8_t sfByte = 0;
      const uint32_t w = fp4::quantize8<r>(x, st, sfByte);
      if (valid) {
        // 8 E2M1 = 4 B
        *reinterpret_cast<uint32_t*>(dst + static_cast<size_t>(k) * (vec / 2)) = w;
        if (k % lanesPerBlock == 0) {
          sf[mx::sfOffset(sfRow, k / lanesPerBlock, numKBlocks)] = sfByte;
        }
      }
    }
  }

  /// quantizeToken / quantizeTokenFP4 by recipe; st is only used by nvfp4
  template <int threads, QuantRecipe r, typename Element>
    requires(isBlockScaled(r) && sizeof(Element) == 2 && mx::MXInput<Element>)
  __device__ __forceinline__
  void quantizeTokenRow(const Element* __restrict__ const& src, uint8_t* __restrict__ const& dst,
                        uint8_t* __restrict__ const& sf, const int& sfRow, const int& H, const float& st) {
    if constexpr (r == QuantRecipe::mxfp8) {
      quantizeToken<threads>(src, dst, sf, sfRow, H);
    }
    else {
      quantizeTokenFP4<threads, r>(src, dst, sf, sfRow, H, st);
    }
  }

  // MoE tokens dispatch
  // recipe == mxfp8: tokens are quantized to E4M3 while they are copied, see quantizeToken.
  // The packet holds the E4M3 rows with an H-byte pitch and its scale factors are written to the scale factor region
  // of the symmetric heap, see Heap::advanceSF.
  // recipe == mxfp4 / nvfp4: the same with E2M1 rows, i.e., an H / 2-byte pitch, see quantizeTokenFP4. st is the
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
        const auto partition = routedTokens / superBlockSize + (lBid < routedTokens % superBlockSize);
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
            if constexpr (quantize) {
              // coalesced quantize + copy: the packet is E4M3 (H-byte row pitch) or E2M1 (H / 2 bytes)
              quantizeTokenRow<threads, recipe>(tokens + static_cast<size_t>(tokenIdx) * H,
                reinterpret_cast<uint8_t*>(peerHeap) + static_cast<size_t>(intraIdx) * rowBytes, peerSF, intraIdx, H, st);
            }
            else {
              auto* __restrict__ localPH = peerHeap + intraIdx * vH;
              const auto* __restrict__ aP = vTokens + tokenIdx * vH;
              // coalesced vectorized copy
              for (int k = static_cast<int>(threadIdx.x); k < vH; k += threads) {
                const auto v = cuda::ptx::ld_nc_L1_no_allocate_L2_256B(cuda::ptx::space_global, aP + k);
                cuda::ptx::st(cuda::ptx::space_global, localPH + k, v);
              }
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
              if constexpr (quantize) {
                quantizeTokenRow<threads, recipe>(tokens + static_cast<size_t>(tokenIdx) * H,
                  reinterpret_cast<uint8_t*>(peerHeap) + static_cast<size_t>(intraIdx) * rowBytes, peerSF, intraIdx, H,
                  st);
              }
              else {
                auto* __restrict__ localPH = peerHeap + intraIdx * vH;
                const auto* __restrict__ aP = vTokens + tokenIdx * vH;
                for (int k = static_cast<int>(threadIdx.x); k < vH; k += threads) {
                  const auto v = cuda::ptx::ld_nc_L1_no_allocate_L2_256B(cuda::ptx::space_global, aP + k);
                  cuda::ptx::st(cuda::ptx::space_global, localPH + k, v);
                }
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
