/******************************************************************************
 * Copyright (c) 2026, Osayamen Jonathan Aimuyo.
 ******************************************************************************/

//
// In-kernel task tracer for moe::forward, enabled with -DFLASHMOE_TRACE=1 (CMake: -DFLASHMOE_TRACE=ON).
//
// With FLASHMOE_TRACE=0 (the default) every FLASHMOE_TRACE_* hook expands to nothing, so Context, KernelArgs and the
// kernels are unchanged.
//
// With FLASHMOE_TRACE=1, Context carries a trace::Buffers, and every CTA appends 16-byte records (trace::Record) to its
// own segment of one global buffer:
//   * timestamps come from %globaltimer, which is aligned across SMs; its resolution on sm_120 is 32 ns;
//   * the segment of block b starts at records + b * capacity; the OS block (the last one) has osCapacity slots;
//   * a block-local atomic counter in shared memory hands out slots, the final count is published to counters[b]
//     when the block exits. Records beyond a segment's capacity are dropped (the counter still counts them);
//   * a null Buffers::records turns the hooks into a branch, so a traced build can be timed without recording;
//   * FLASHMOE_TRACE_MASK selects the event groups (bit 0 block, 1 processor, 2 scheduler, 3 subscriber, 4 dispatch).
// The kernel must call FLASHMOE_TRACE_BEGIN (moe::forward does), the hooks read the segment from shared memory.
// A hook can change how the compiler unrolls the code around it: compare the time of a traced build that records
// nothing (null buffers) with the untraced build.
// The hooks sit at stable points of the dispatch / OS (scheduler, subscriber) / processor code, see Event for the
// payload of each event. tests/trace_summary.py turns a trace file into the timeline statistics.
//

#ifndef FLASHMOE_TRACE_CUH
#define FLASHMOE_TRACE_CUH

#ifndef FLASHMOE_TRACE
#define FLASHMOE_TRACE 0
#endif

#ifndef FLASHMOE_TRACE_MASK
#define FLASHMOE_TRACE_MASK 31
#endif

#if FLASHMOE_TRACE
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

namespace flashmoe::trace {
  // Each record is {%globaltimer ns, a, b}. The upper 8 bits of a hold the Event, the lower 24 bits the first payload.
  enum class Event : uint8_t {
    none = 0,
    blockBegin = 1, // [every block, at kernel entry] a: 1 if OS block, 0 otherwise
    blockEnd = 2, // [every block, at kernel exit] a: 1 if OS block, 0 otherwise
    // processor
    procWaitBegin = 3, // [thread 0] starts polling the doorbell
    procDoorbell = 4, // [thread 0] doorbell observed. a: interrupt, b: task queue slot
    // [thread 0] task is in shared memory and the CTA synchronized. a: taskType << 16 | localExpert, b: tileIdx
    procTaskLoaded = 5,
    procMainDone = 6, // [thread 0] the GEMM main loop returned (GEMM tasks only)
    procTaskDone = 7, // [thread 0] task finished. a: taskType << 16 | tileSize, b: tileIdx
    // scheduler
    // [assigning lane] one schedule() call: count doorbells rung back to back for the tasks in the task queue slots
    // qIdx, qIdx + stride, ... (stride = number of subscriber threads, if a bit 23 is set) or qIdx, qIdx + 1, ...
    // a: strided << 23 | qIdx, b: count << 24 | duration in units of 32 ns. The record's time is the end of the call.
    // The doorbells are not recorded one by one: a hook next to every release store of the unrolled doorbell loops changed
    // the generated scheduler code (half as many fence sites in the SASS) and made the kernel 60% slower, measured with
    // the buffers set to null.
    schedBatch = 8,
    schedSweep = 9, // [lane 0] one pass over all task queues. a: tasks scheduled in the pass, b: tasks scheduled so far
    schedScan = 10, // [lane 0] one sweep of the processor readiness queue. a: ready processors found, b: tasks waiting
    // subscriber
    // [lane 0 of the warp] a first-stage packet is decoded into GEMM0 tasks. a: peer << 12 | localExpert, b: routedTokens
    subPacket = 11,
    subCombine = 12, // [lane] a combine task is generated. a: tileIdx << 12 | localExpert, b: tokenBatchStart
    // dispatch
    dispatchBegin = 13, // [thread 0] a block starts dispatching an expert. a: expert, b: routedTokens
    dispatchEnd = 14, // [thread 0] a block finished dispatching an expert and signaled it. a: expert
  };

  struct __align__(16) Record {
    uint64_t t;
    uint32_t a;
    uint32_t b;
  };
  static_assert(sizeof(Record) == 16);

  struct Buffers {
    Record* records = nullptr; // [blocks - 1, capacity] followed by [osCapacity]
    uint32_t* counters = nullptr; // [blocks]
    uint32_t capacity = 0; // records per processor / dispatch block
    uint32_t osCapacity = 0; // records of the OS block
  };

  // File format, little endian: FileHeader, then for each block {uint32 count, uint32 dropped, count x Record}.
  constexpr char FILE_MAGIC[8] = {'F', 'M', 'T', 'R', '0', '0', '0', '1'};
  struct FileHeader {
    char magic[8];
    uint32_t blocks;
    uint32_t recordBytes;
    uint32_t capacity;
    uint32_t osCapacity;
    uint32_t reserved[2];
  };

  struct BlockState {
    Record* base;
    uint32_t cap;
    uint32_t n;
  };

  __device__ __forceinline__
  BlockState& blockState() {
    __shared__ BlockState s;
    return s;
  }

  __device__ __forceinline__
  uint64_t globalTimer() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
  }

  // Event groups, selectable with FLASHMOE_TRACE_MASK (bit i enables group i, default: all) to reduce the perturbation
  constexpr int GROUP_BLOCK = 0, GROUP_PROCESSOR = 1, GROUP_SCHEDULER = 2, GROUP_SUBSCRIBER = 3, GROUP_DISPATCH = 4;
  consteval int groupOf(const Event ev) {
    return ev <= Event::blockEnd ? GROUP_BLOCK : ev <= Event::procTaskDone ? GROUP_PROCESSOR :
      ev <= Event::schedScan ? GROUP_SCHEDULER : ev <= Event::subCombine ? GROUP_SUBSCRIBER : GROUP_DISPATCH;
  }

  template <Event ev>
  __device__ __forceinline__
  void record(const uint32_t a, const uint32_t b) {
    if constexpr (((FLASHMOE_TRACE_MASK >> groupOf(ev)) & 1) == 0) {
      return;
    }
    auto& s = blockState();
    if (s.base == nullptr) {
      return;
    }
    const auto t = globalTimer();
    const auto slot = atomicAdd_block(&s.n, 1U);
    if (slot < s.cap) {
      reinterpret_cast<uint4*>(s.base)[slot] = make_uint4(static_cast<uint32_t>(t), static_cast<uint32_t>(t >> 32),
        (static_cast<uint32_t>(ev) << 24) | (a & 0xFFFFFFU), b);
    }
  }

  /// Start time of an interval that batch() closes; 0 if this block does not record
  __device__ __forceinline__
  uint64_t stamp() {
    if constexpr (((FLASHMOE_TRACE_MASK >> GROUP_SCHEDULER) & 1) == 0) {
      return 0UL;
    }
    return blockState().base == nullptr ? 0UL : globalTimer();
  }

  /// schedBatch, see Event
  __device__ __forceinline__
  void batch(const uint64_t& t0, const bool& strided, const uint32_t& qIdx, const uint32_t& count) {
    if constexpr (((FLASHMOE_TRACE_MASK >> GROUP_SCHEDULER) & 1) == 0) {
      return;
    }
    auto& s = blockState();
    if (s.base == nullptr) {
      return;
    }
    const auto t = globalTimer();
    const uint64_t ticks = (t - t0) >> 5;
    const uint32_t dur = ticks > 0xFFFFFFUL ? 0xFFFFFFU : static_cast<uint32_t>(ticks);
    record<Event::schedBatch>((static_cast<uint32_t>(strided) << 23) | (qIdx & 0x7FFFFFU),
      ((count > 255U ? 255U : count) << 24) | dur);
  }

  /// All threads of the kernel call this first, it contains a block-wide barrier
  __device__ __forceinline__
  void begin(const Buffers& buffers) {
    auto& s = blockState();
    if (!threadIdx.x) {
      s.n = 0U;
      const bool isOS = blockIdx.x == gridDim.x - 1;
      s.cap = isOS ? buffers.osCapacity : buffers.capacity;
      s.base = buffers.records == nullptr ? nullptr : buffers.records + static_cast<size_t>(blockIdx.x) * buffers.capacity;
    }
    __syncthreads();
    if (!threadIdx.x) {
      record<Event::blockBegin>(blockIdx.x == gridDim.x - 1, 0U);
    }
  }

  /// Thread 0 calls this after the last barrier of the block
  __device__ __forceinline__
  void end(const Buffers& buffers) {
    if (!threadIdx.x) {
      record<Event::blockEnd>(blockIdx.x == gridDim.x - 1, 0U);
      if (buffers.counters != nullptr) {
        buffers.counters[blockIdx.x] = blockState().n;
      }
    }
  }

  __host__ __forceinline__
  Buffers allocate(const uint32_t& blocks, const uint32_t& capacity, const uint32_t& osCapacity) {
    Buffers b{};
    b.capacity = capacity;
    b.osCapacity = osCapacity;
    const auto records = static_cast<size_t>(blocks - 1) * capacity + osCapacity;
    if (cudaMalloc(&b.records, records * sizeof(Record)) != cudaSuccess ||
      cudaMalloc(&b.counters, sizeof(uint32_t) * blocks) != cudaSuccess) {
      printf("trace: cannot allocate %zu MiB\n", records * sizeof(Record) >> 20);
      abort();
    }
    return b;
  }

  __host__ __forceinline__
  void reset(const Buffers& b, const uint32_t& blocks, cudaStream_t stream) {
    cudaMemsetAsync(b.counters, 0, sizeof(uint32_t) * blocks, stream);
  }

  __host__ __forceinline__
  void release(const Buffers& b) {
    cudaFree(b.records);
    cudaFree(b.counters);
  }

  /// Copies the buffers to the host and writes them in the file format above. Returns false on an I/O or CUDA error.
  __host__ __forceinline__
  bool writeFile(const Buffers& b, const uint32_t& blocks, const char* const& path, cudaStream_t stream) {
    cudaStreamSynchronize(stream);
    std::vector<uint32_t> counters(blocks);
    if (cudaMemcpy(counters.data(), b.counters, sizeof(uint32_t) * blocks, cudaMemcpyDeviceToHost) != cudaSuccess) {
      return false;
    }
    FILE* f = fopen(path, "wb");
    if (f == nullptr) {
      return false;
    }
    FileHeader h{};
    for (int i = 0; i < 8; ++i) {
      h.magic[i] = FILE_MAGIC[i];
    }
    h.blocks = blocks;
    h.recordBytes = sizeof(Record);
    h.capacity = b.capacity;
    h.osCapacity = b.osCapacity;
    fwrite(&h, sizeof(h), 1, f);
    std::vector<Record> segment;
    for (uint32_t i = 0; i < blocks; ++i) {
      const auto cap = i == blocks - 1 ? b.osCapacity : b.capacity;
      const uint32_t count = counters[i] < cap ? counters[i] : cap;
      const uint32_t header[2] = {count, counters[i] - count};
      fwrite(header, sizeof(header), 1, f);
      segment.resize(count);
      if (count > 0 && cudaMemcpy(segment.data(), b.records + static_cast<size_t>(i) * b.capacity,
        sizeof(Record) * count, cudaMemcpyDeviceToHost) != cudaSuccess) {
        fclose(f);
        return false;
      }
      fwrite(segment.data(), sizeof(Record), count, f);
    }
    return fclose(f) == 0;
  }
}

// Hooks. ev is an enumerator of trace::Event.
// The _T0 variant records on thread 0 of the block only, _LANE0 on lane 0 of each warp.
#define FLASHMOE_TRACE_BEGIN(ctx) ::flashmoe::trace::begin((ctx).traceBuffers)
#define FLASHMOE_TRACE_END(ctx) ::flashmoe::trace::end((ctx).traceBuffers)
#define FLASHMOE_TRACE_EVENT(ev, a, b) \
  ::flashmoe::trace::record<::flashmoe::trace::Event::ev>(static_cast<uint32_t>(a), static_cast<uint32_t>(b))
#define FLASHMOE_TRACE_EVENT_T0(ev, a, b) do { if (!threadIdx.x) { FLASHMOE_TRACE_EVENT(ev, a, b); } } while (0)
#define FLASHMOE_TRACE_EVENT_LANE0(ev, a, b) \
  do { if (!(threadIdx.x % 32)) { FLASHMOE_TRACE_EVENT(ev, a, b); } } while (0)
// FLASHMOE_TRACE_STAMP(t) declares t, FLASHMOE_TRACE_BATCH(t, ...) records the schedBatch that ends now
#define FLASHMOE_TRACE_STAMP(t) const auto t = ::flashmoe::trace::stamp()
#define FLASHMOE_TRACE_BATCH(t, strided, qIdx, count) ::flashmoe::trace::batch(t, strided, qIdx, count)
#else
#define FLASHMOE_TRACE_BEGIN(ctx) ((void)0)
#define FLASHMOE_TRACE_END(ctx) ((void)0)
#define FLASHMOE_TRACE_EVENT(ev, a, b) ((void)0)
#define FLASHMOE_TRACE_EVENT_T0(ev, a, b) ((void)0)
#define FLASHMOE_TRACE_EVENT_LANE0(ev, a, b) ((void)0)
#define FLASHMOE_TRACE_STAMP(t) ((void)0)
#define FLASHMOE_TRACE_BATCH(t, strided, qIdx, count) ((void)0)
#endif
#endif //FLASHMOE_TRACE_CUH
