//
// Created by osayamen on 1/10/26.
//
// unit tests for the scheduler

#include <cuda/atomic>
#include <cub/cub.cuh>
#include "debug.cuh"
#include "../include/flashmoe/scheduler.cuh"

__device__ __forceinline__ uint32_t lane_id() { return threadIdx.x & 31u; }
__device__ __forceinline__ uint32_t warp_id() { return threadIdx.x >> 5; }

__device__ __forceinline__ void jitter_spin(uint32_t seed) {
    // Deterministic “random” delay to create adversarial interleaving.
    // Uses clock64 busy-wait; keep it small-ish.
    seed = seed * 1664525u + 1013904223u;
    const uint32_t spins = (seed >> 24) & 0x3Fu; // 0..63
    const unsigned long long start = clock64();
    while ((clock64() - start) < static_cast<unsigned long long>(spins)) {
        /* spin */
    }
}
struct __align__(8) MockTask {
    uint32_t v;
    uint16_t parentIdx;
    uint8_t reproduce = 0;
    uint8_t consumed;
};
// --- roles ---
template<int producers>
__device__ void run_producer(
    MockTask* __restrict__ const& taskQueue,
    uint32_t* __restrict__ const& tQHeads, // shared: length Q (doorbells)
    uint32_t* __restrict__ const& interrupts, // shared: length Q
    const uint32_t& q) {
    uint32_t cursor = 0;
    uint32_t interrupt = 0;
    cuda::atomic_ref<uint32_t, cuda::thread_scope_block> inr{*(interrupts + q)};
    cuda::atomic_ref<uint32_t, cuda::thread_scope_block> tqh{*(tQHeads + q)};
    // producers will only produce one task but will wait until interrupt before terminating.
    // this confirms that the scheduler did interrupt them.
    MockTask task{};
    task.consumed = 0;
    task.reproduce = 1;
    task.v = 1;
    task.parentIdx = q; // to ensure unique slot for reproduced tasks
    // Delay to simulate realistic execution as the producer waits for remote tokens before
    // generating task descriptors
    jitter_spin((q + 1u) * 0x9E3779B9u + cursor + q);
    // produce one task
    // qIdx = q + cursor * producers
    const auto qIdx = flashmoe::DQ::next<flashmoe::DQType::stride, producers>(q, cursor);
    taskQueue[qIdx] = task;
    cuda::std::ignore = tqh.fetch_add(1, cuda::memory_order_release);
    cursor++;
    interrupt = inr.load(cuda::memory_order_acquire);
    while (!interrupt) {
        interrupt = inr.load(cuda::memory_order_acquire);
    }
}

// Consumer CTA.
// polls its signal slot until:
//   - interrupt==1  OR
//   - task_queue_index != previous.task_queue_index
// Then it immediately sets status[pid]=Ready (release), and if not interrupt, processes the task
struct __align__(8) ConsumerState {
    MockTask task;
    uint16_t interrupt;
};
template<int threads>
__device__ void run_consumer(MockTask* __restrict__ const& taskQueue,
    const uint32_t& pid,
    uint32_t* __restrict__ const& statusQ, // init {observed}
    flashmoe::TQSignal* __restrict__ const& tqSignalQ, // init {0,0}
    uint32_t* __restrict__ gTQHeads, // init{0}
    const uint32_t N,
    const uint32_t tN){

    // Track previous task_queue_index to detect changes.
    uint32_t prevSig = 0U;
    cuda::atomic_ref<uint64_t, cuda::thread_scope_device> doorbell{*reinterpret_cast<uint64_t*>(tqSignalQ + pid)};
    using SQT = cuda::std::underlying_type_t<flashmoe::ReadySignal>;
    cuda::atomic_ref<SQT, cuda::thread_scope_device> sqd{*(statusQ + pid)};
    bool keepWorking = true;
    __shared__ ConsumerState state;
    while (keepWorking) {
        MockTask task{};
        if (!threadIdx.x) {
            auto sig = cuda::std::bit_cast<flashmoe::TQSignal>(doorbell.load(cuda::memory_order_acquire));
            while (sig.signal == prevSig && sig.interrupt == 0) {
                sig = cuda::std::bit_cast<flashmoe::TQSignal>(doorbell.load(cuda::memory_order_acquire));
            }
            ConsumerState c{};
            keepWorking = sig.interrupt == 0;
            c.interrupt = sig.interrupt;
            if (keepWorking) {
                // notify scheduler
                sqd.store(flashmoe::ready, cuda::memory_order_relaxed);
                prevSig = sig.signal;
                const auto qIdx = sig.decodeSig();
                // simulated work
                task = taskQueue[qIdx];
                task.v += 1;
                task.consumed = 1;
                taskQueue[qIdx] = task; // for verification
                c.task = task;
            }
            else {
                // we need to clear out our mailbox to ensure it's ready for s subsequent iteration
                constexpr auto TQSZero = cuda::std::bit_cast<uint64_t>(flashmoe::TQSignal{0, 0});
                doorbell.store(TQSZero, cuda::memory_order_relaxed);
            }
            state = c;
        }
        __syncthreads();
        if (threadIdx.x > 0) {
            keepWorking = state.interrupt == 0;
            task = state.task;
        }
        if (keepWorking && task.reproduce) {
            auto nextTask = MockTask{};
            nextTask.reproduce = 0;
            nextTask.consumed = 0;
            nextTask.v = task.v;
            // produce tN tasks afterward
            const uint32_t base = N + task.parentIdx * tN;
            auto* __restrict__ tQNext = taskQueue + base;
            for (uint32_t i = threadIdx.x; i < tN; i += threads) {
                tQNext[i] = nextTask;
            }
            __syncthreads();
            if (!threadIdx.x) {
                cuda::atomic_ref<uint32_t, cuda::thread_scope_device> gth{*(gTQHeads + task.parentIdx)};
                cuda::std::ignore = gth.fetch_add(tN, cuda::memory_order_release);
            }
        }
    }
}

// -------------------- The kernel --------------------
template<int threads = 128>
__global__ void testSchedulerE2E(
    MockTask* __restrict__ taskQueue, // length = N + M*tN   (or whatever your scheduler uses)
    uint32_t* __restrict__ statusQueue, // length = P, init Observed(0)
    flashmoe::TQSignal* __restrict__ tqSignalQ, // length = P, init {0,0}
    uint32_t* __restrict__ gTQHeads, // length = M, init 0
    const uint32_t tN) {
    constexpr int producers = threads - flashmoe::scheduler::SCHEDULER_COUNT;
    const uint32_t num_primary_tasks = producers;
    const uint32_t num_spawned_tasks = producers * tN;
    // each producer produces one which gets spawned as tN additional tasks.
    const auto taskBoundV = num_primary_tasks + num_spawned_tasks;
    constexpr uint32_t L_bitset = flashmoe::nSI<flashmoe::scheduler::SCHEDULER_COUNT>(producers);
    static_assert(threads > flashmoe::scheduler::SCHEDULER_COUNT);
    using namespace flashmoe;

    const uint32_t b = blockIdx.x;
    const auto num_consumers = gridDim.x - 1;
    // CTA0 = scheduler + primary producers (multiple warps).
    if (b == 0) {
        // Dynamic shared layout for scheduler inputs.
        extern __shared__ uint32_t smem[];
        size_t off = 0;

        auto* readyQ = smem + off;
        off += num_consumers;
        auto* scratch = smem + off;
        off += num_consumers;
        auto* tQHeads_sh = smem + off;
        off += producers;
        auto* interrupts = smem + off;
        off += producers;
        auto* bitset = reinterpret_cast<BitSet*>(smem + off);
        static_assert(cuda::std::is_same_v<decltype(bitset->storage), uint32_t>);
        off += L_bitset;
        auto* taskBound = smem + off;
        off += 1;

        // Initialize shared inputs once per launch (warp0 is enough).
        // readyQ[i] = i, scratch[i]=1
        for (uint32_t i = threadIdx.x; i < num_consumers; i += threads) {
            readyQ[i] = i;
            scratch[i] = 1u;
        }
        // tQHeads[q]=0, intr[q]=0
        for (uint32_t i = threadIdx.x; i < producers; i += threads) {
            tQHeads_sh[i] = 0u;
            interrupts[i] = 0u;
        }
        // bitset init to 0
        for (uint32_t i = threadIdx.x; i < L_bitset; i += threads) {
            bitset[i] = BitSet{};
        }
        if (threadIdx.x == 0) {
            *taskBound = taskBoundV;
        }
        __syncthreads();

        if (warp_id() == 0) {
            // call scheduler
            scheduler::start<producers>(scratch, bitset, num_consumers, tN,
                num_primary_tasks, producers, interrupts, tQHeads_sh, gTQHeads, taskBound,
                readyQ, statusQueue, tqSignalQ);
        }
        else {
            run_producer<producers>(taskQueue, tQHeads_sh, interrupts, threadIdx.x - scheduler::SCHEDULER_COUNT);
        }
        return;
    }

    // Consumer CTAs everyone else
    const uint32_t pid = b - 1;
    run_consumer<threads>(taskQueue, pid, statusQueue, tqSignalQ, gTQHeads, num_primary_tasks, tN);
}

template<int threads>
__global__ void verify(const MockTask* __restrict__ tQ, const uint n, uint* __restrict__ failures) {
    const uint tid = blockIdx.x * threads + threadIdx.x;
    uint tally = 0;
    for (uint i = tid; i < n; i += threads * gridDim.x) {
        const auto task = tQ[i];
        tally += (task.consumed == 0) ? 1U : 0U;
    }
    __syncthreads();
    using BlockReduce = cub::BlockReduce<uint, threads>;
    __shared__ typename BlockReduce::TempStorage ts;
    const uint total = BlockReduce(ts).Sum(tally);
    if (threadIdx.x == 0) {
        atomicAdd(failures, total);
    }
}
// ./testScheduler <consumers> <n_spawned>
int main(const int argc, char** argv) {
    uint32_t blocks = 64;
    uint32_t n_spawned = 64;
    uint32_t post_runs = 0;
    if (argc > 1) {
        blocks = std::stoi(argv[1]);
    }
    if (argc > 2) {
        n_spawned = std::stoi(argv[2]);
    }
    if (argc > 3) {
        post_runs = std::stoi(argv[3]);
    }
    constexpr uint threads = 128;
    if (blocks < 2) {
        throw std::invalid_argument("blocks must be >= 2");
    }
    MockTask* taskQ = nullptr;
    flashmoe::TQSignal* tqs = nullptr;
    uint32_t* statusQueue = nullptr;
    uint32_t* tQHeads = nullptr;
    uint* failures = nullptr;

    constexpr auto producers = threads - flashmoe::scheduler::SCHEDULER_COUNT;
    constexpr auto num_primary_tasks = producers;
    const auto num_spawned_tasks = producers * n_spawned;
    const auto totalTasks = num_primary_tasks + num_spawned_tasks;
    const auto consumers = blocks - 1;
    if (consumers > flashmoe::scheduler::SCHEDULER_COUNT * flashmoe::scheduler::MAX_PROCESSORS) {
        throw std::invalid_argument(std::string("blocks - 1 must be <= ")
            .append(std::to_string(flashmoe::scheduler::MAX_PROCESSORS)));
    }

    cudaSetDevice(0);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMallocAsync(&taskQ, sizeof(MockTask) * totalTasks, stream);
    cudaMallocAsync(&statusQueue, sizeof(uint32_t) * consumers, stream);
    static_assert(flashmoe::ReadySignal::observed == 0);
    cudaMemsetAsync(statusQueue, flashmoe::ReadySignal::observed,
        sizeof(uint32_t) * consumers, stream);
    cudaMallocAsync(&tQHeads, sizeof(uint32_t) * producers, stream);
    cudaMemsetAsync(tQHeads, 0, sizeof(uint32_t) * producers, stream);
    cudaMallocAsync(&tqs, sizeof(flashmoe::TQSignal) * consumers, stream);
    cudaMemsetAsync(tqs, 0, sizeof(flashmoe::TQSignal) * consumers, stream);

    constexpr auto bitSetSize = flashmoe::nSI<flashmoe::scheduler::SCHEDULER_COUNT>(producers);
    const size_t sharedSize = sizeof(uint32_t) * (2 * (producers + consumers) + bitSetSize + 1);
    testSchedulerE2E<threads><<<blocks, threads, sharedSize, stream>>>(taskQ, statusQueue, tqs, tQHeads, n_spawned);
    CHECK_CUDA(cudaPeekAtLastError());
    cudaMallocAsync(&failures, sizeof(uint), stream);
    cudaMemsetAsync(failures, 0, sizeof(uint), stream);
    const uint vBlocks = cuda::ceil_div(totalTasks, threads);
    verify<threads><<<vBlocks, threads, 0, stream>>>(taskQ, totalTasks, failures);
    uint h_failures;
    cudaMemcpyAsync(&h_failures, failures, sizeof(uint), cudaMemcpyDeviceToHost, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    for (int i = 0; i < post_runs; ++i) {
        testSchedulerE2E<threads><<<blocks, threads, sharedSize, stream>>>(taskQ, statusQueue, tqs, tQHeads, n_spawned);
    }
    cudaFreeAsync(taskQ, stream);
    cudaFreeAsync(tqs, stream);
    cudaFreeAsync(tQHeads, stream);
    cudaFreeAsync(statusQueue, stream);
    cudaFreeAsync(failures, stream);
    const float failure_percent = (static_cast<float>(h_failures) / static_cast<float>(totalTasks)) * 100.f;
    printf("blocks,threads,producers,consumers,totalTasks,error(%%)\n");
    printf("%d,%d,%d,%d,%d,%f\n", blocks, threads, producers, consumers, totalTasks,failure_percent);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    cudaStreamDestroy(stream);
}
