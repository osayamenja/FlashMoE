//
// Created by oja7 on 12/20/24.
//

#ifndef BENCH_CUH
#define BENCH_CUH

#define THREADS_PER_WARP 32
#define THREADS_PER_BLOCK 256
#define NANO_TO_MICRO (cuda::std::nano::den / cuda::std::micro::den)
#define SEQ 23U

#include <cuda/atomic>
#include <cuda/annotated_ptr>
#include <cub/cub.cuh>
#include <cute/tensor.hpp>

#include "include/util/indexing.cuh"
#include "include/util/atomics.cuh"
#include "include/definition/types.cuh"

namespace aristos {
    __global__ void benchAtomics(__grid_constant__ const int iter, unsigned int* flag, bool skip = false, bool shouldPersist = false){
        // initialization
        using Nano = cuda::std::chrono::duration<double, cuda::std::nano>;
        unsigned int* pUnderTest = flag;
        __shared__ unsigned int sharedFlag;
        if(shouldPersist){
            associate_access_property(flag, cuda::access_property::persisting{});
            //pUnderTest = &sharedFlag;
        }
        __syncthreads();
        atomicExch(&sharedFlag, 0);
        Nano a_flag = Nano::zero(), a_cas = Nano::zero(), a_or = Nano::zero(), a_and = Nano::zero();
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device> aFlag (*pUnderTest);
        #pragma unroll
        for(int i = 0; i < iter; ++i){
            uint64_t start, end;
            asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
            aFlag.load();
            asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
            a_flag += static_cast<cuda::std::chrono::duration<double, cuda::std::nano>>(end - start);

            asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
            atomicCAS(pUnderTest, 0, 0);
            asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
            a_cas += static_cast<cuda::std::chrono::duration<double, cuda::std::nano>>(end - start);

            asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
            atomicOr(pUnderTest, 0U);
            asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
            a_or += static_cast<cuda::std::chrono::duration<double, cuda::std::nano>>(end - start);

            asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
            atomicAnd(pUnderTest, 1U);
            asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
            a_and += static_cast<cuda::std::chrono::duration<double, cuda::std::nano>>(end - start);

        }
        using BlockReduce = cub::BlockReduce<Nano, THREADS_PER_BLOCK>;
        __shared__ BlockReduce::TempStorage temp_storage;

        a_flag = BlockReduce(temp_storage).Reduce(a_flag, cub::Max());
        a_cas = BlockReduce(temp_storage).Reduce(a_cas, cub::Max());
        a_or = BlockReduce(temp_storage).Reduce(a_or, cub::Max());
        a_and = BlockReduce(temp_storage).Reduce(a_and, cub::Max());
        cuda::std::chrono::duration_cast<cuda::std::chrono::microseconds, double>(a_flag);

        if(block::threadID() == 0 && !skip){
            printf("Block Id is %u, a_flag: {T: %f, V: %d}, a_cas: {T: %f, V:%u}, a_or: {T: %f, V:%u}, a_and: {T: %f, V: %u},"
                   "isShared: %s\n",
                   aristos::grid::blockID(),
                   cuda::std::chrono::duration_cast<cuda::std::chrono::duration<double, cuda::std::micro>>(a_flag / (iter*1.0)).count(),
                   aFlag.load(),
                   cuda::std::chrono::duration_cast<cuda::std::chrono::duration<double, cuda::std::micro>>(a_cas/(iter*1.0)).count(),
                   atomicCAS(pUnderTest, 0, 0),
                   cuda::std::chrono::duration_cast<cuda::std::chrono::duration<double, cuda::std::micro>>(a_or/(iter*1.0)).count(),
                   atomicOr(pUnderTest, 0U),
                   cuda::std::chrono::duration_cast<cuda::std::chrono::duration<double, cuda::std::micro>>(a_and/(iter*1.0)).count(),
                   atomicAnd(pUnderTest, 1U),
                   (shouldPersist)? "Yes" : "No");
        }
    }

    extern constexpr int peers = 1;
    extern constexpr int stages = 2;
    extern constexpr int cells = 2;

    extern constexpr int capacity = 1;
    extern constexpr int k = 0;
    extern constexpr int embedDim = 0;
    extern constexpr int tokens = capacity * (embedDim + k + 2);

    extern constexpr int peerStride = stages * cells * tokens;
    extern constexpr int stageStride = cells * tokens;
    extern constexpr int cellStride = tokens;
    extern constexpr int tokenStride = (embedDim + k + 2);

    template<typename T>
    CUTE_DEVICE
    T* getTokenPointer(T* const& __restrict__ addr, unsigned int const& peer, unsigned int const& stage, unsigned int const& cell, unsigned int const& token){
        return addr + ((peer * peerStride) + (stage * stageStride) + (cell * cellStride) + (token * tokenStride));
    }

    __global__ void benchTen(unsigned int* foo, bool skip = false){
        auto t = make_tensor(cute::make_gmem_ptr(foo), make_shape(peers, make_shape(cute::make_shape(stages, cells), tokens)), cute::LayoutRight{});
        double cuteDuration = 0.0, rawDuration = 0.0;
        CUTE_UNROLL
        for(unsigned int i = 0; i < 1024; ++i){
            uint64_t start = 0, end = 0;
            asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
            &t(0, make_coord(cute::make_coord(0,1),0));
            asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
            cuteDuration = static_cast<double>(end - start);

            asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
            getTokenPointer(foo, 0, 0, 1, 0);
            asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
            rawDuration = static_cast<double>(end - start);
        }
        if(block::threadID() == 0 && !skip){
            printf("Block Id is %u, cute_t: {T: %f, V: %d}, raw_t: {T: %f, V:%d}\n",
                   block::threadID(),
                   cuteDuration,
                   t(0, make_coord(cute::make_coord(0,1),0)),
                   rawDuration,
                   *getTokenPointer(foo, 0, 0, 1, 0));
        }
    }

    __host__ __forceinline__
    void testBenchTen() {
        unsigned int* p;
        constexpr std::array<unsigned int, 4> arr {{1234, 56, 78, 0}};
        CUTE_CHECK_ERROR(cudaMalloc(&p, sizeof(unsigned int) * arr.size()));
        CUTE_CHECK_ERROR(cudaMemcpy(p, arr.data(), sizeof(unsigned int) * arr.size(), cudaMemcpyHostToDevice));
        benchTen<<<1,1>>>(p);
        CUTE_CHECK_LAST();
    }

    __device__ __inline__ unsigned int phases = 0U;
    __device__ __inline__ unsigned int makeshiftBarrier = 0U;
    __global__ void benchBarrier(cuda::barrier<cuda::thread_scope_device>* __restrict__ bar,
        CUTE_GRID_CONSTANT const unsigned int n, CUTE_GRID_CONSTANT const bool skip = false){
        double makeshiftDuration = 0.0, quoDuration = 0.0;
        constexpr auto iter = 1024;
        CUTE_UNROLL
        for(unsigned int i = 0; i < iter; ++i){
            uint64_t start = 0, end = 0;
            asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
            if(!block::threadID()){
                /// Arrive
                if(auto nextPhase = aristos::atomicLoad(&phases) + 1;
                    (atomicAdd(&makeshiftBarrier, 1U) + 1) == n * nextPhase){
                    atomicAdd(&phases, 1U);
                    /// Could execute completion function here
                }
                else{
                    /// You could do some other task prior to waiting
                    /// Wait
                    while(aristos::atomicLoad(&phases) != nextPhase){
                    }
                    /// Could execute completion function here
                }
            }
            __syncthreads();
            asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
            makeshiftDuration += static_cast<double>(end - start) / static_cast<double>(iter*NANO_TO_MICRO);
        }

        CUTE_UNROLL
        for(unsigned int i = 0; i < iter; ++i){
            uint64_t start = 0, end = 0;
            asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
            if(!block::threadID()){
                bar->arrive_and_wait();
            }
            __syncthreads();
            asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
            quoDuration += static_cast<double>(end - start) / static_cast<double>(iter*NANO_TO_MICRO);
        }

        using BlockReduce = cub::BlockReduce<double, ARISTOS_BLOCK_SIZE>;
        __shared__ BlockReduce::TempStorage temp_storage;

        makeshiftDuration = BlockReduce(temp_storage).Reduce(makeshiftDuration, cub::Max());
        quoDuration = BlockReduce(temp_storage).Reduce(quoDuration, cub::Max());
        if(aristos::block::threadID() == 0 && !skip){
            printf("Block Id is %u, makeshiftTime: {T: %f}, quoTime: {T: %f}\n", aristos::grid::blockID(),
                makeshiftDuration, quoDuration);
        }
    }

    __host__ __forceinline__
    void testBenchBarrier() {
        constexpr unsigned int n = 32;
        cuda::barrier<cuda::thread_scope_device>* deviceBarrier;
        const auto hostBarrier = new cuda::barrier<cuda::thread_scope_device>{n};
        CUTE_CHECK_ERROR(cudaMalloc(&deviceBarrier, sizeof(cuda::barrier<cuda::thread_scope_device>)));
        CUTE_CHECK_ERROR(cudaMemcpy(deviceBarrier, hostBarrier, sizeof(cuda::barrier<cuda::thread_scope_device>), cudaMemcpyHostToDevice));
        benchBarrier<<<n, ARISTOS_BLOCK_SIZE>>>(deviceBarrier, n);
        CUTE_CHECK_LAST();
        delete hostBarrier;
        cudaFree(deviceBarrier);
    }
}
#endif //BENCH_CUH
