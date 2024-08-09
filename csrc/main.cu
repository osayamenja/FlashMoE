#include <iostream>
#include <array>
#include <atomic>

#include <cuda_runtime.h>
#include <cuda/std/array>
#include <cuda/std/chrono>
#include <cuda/atomic>
#include <cuda/barrier>

// Heap things
#include <cuda/std/__algorithm/make_heap.h>
#include <cuda/std/__algorithm/pop_heap.h>

#include <nvshmemx.h>
#include <nvshmem.h>
#include <host/nvshmemx_api.h>

#include "include/aristos.cuh"
#include <cooperative_groups.h>
#include <functional>

#include <cub/cub.cuh>
#include <cute/tensor.hpp>
#include <cute/config.hpp>

#include <torch/torch.h>

#define THREADS_PER_WARP 32
#define THREADS_PER_BLOCK 256

__global__ void aristos_sandbox(int *destination, const int my_pe, const int n_pes, const bool skip = true) {
    cuda::std::array<int, 4>activations{{my_pe + 1, 2,my_pe + 3, 4}};
    int n_elems = 2;
    int k = 0;
    const auto start{cuda::std::chrono::high_resolution_clock::now()};
    for(int i = 0; i < n_pes; ++i){
        for(uint j = 0; j < n_elems; ++j){
            destination[j] = activations[k++];
        }
        nvshmemx_int_put_block((destination + ((my_pe+1)*n_elems)),
                        destination,
                        n_elems,
                        (i % n_pes));
    }
    const auto end{cuda::std::chrono::high_resolution_clock::now()};
    const cuda::std::chrono::duration<double> elapsed_seconds{end - start};

    if (!skip && cute::thread(0)){
        printf("PE %d.t0 Elapsed time is %lld\n", my_pe,
               cuda::std::chrono::duration_cast<cuda::std::chrono::microseconds>(elapsed_seconds).count());
    }
    __syncthreads();
}
void nvshmem_test(){
    std::cout << "Hello, World!" << std::endl;
    int my_pe_node;
    std::array<int, 6> msg{};
    cudaStream_t stream;

    nvshmem_init();
    my_pe_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    int my_pe = nvshmem_my_pe();
    int n_pe = nvshmem_n_pes();
    int dev_count;
    CUTE_CHECK_ERROR(cudaGetDeviceCount(&dev_count));
    cudaDeviceProp prop{};
    CUTE_CHECK_ERROR(cudaGetDeviceProperties(&prop, my_pe_node % dev_count));
    int clockrate;
    CUTE_CHECK_ERROR(cudaDeviceGetAttribute(&clockrate, cudaDevAttrClockRate, my_pe_node));

    fprintf(stderr, "mype: %d mype_node: %d device name: %s bus id: %d n_pes: %d\n", my_pe, my_pe_node,
            prop.name, prop.pciBusID, n_pe);
    cudaSetDevice(my_pe_node);
    cudaStreamCreate(&stream);

    auto destination_void = nvshmem_calloc(msg.size(), sizeof(int));
    auto destination = static_cast<int*>(destination_void);

    for (int i = 0; i < 5; ++i) {
        aristos_sandbox<<<1, 1>>>(destination, my_pe, n_pe);
        CUTE_CHECK_ERROR(cudaPeekAtLastError());
    }
    aristos_sandbox<<<1, 1>>>(destination, my_pe, n_pe, false);
    CUTE_CHECK_ERROR(cudaPeekAtLastError());
    CUTE_CHECK_ERROR(cudaMemcpyAsync(msg.data(), destination, msg.size() * sizeof(int), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    cudaDeviceSynchronize();
    std::stringstream result_stream;
    result_stream << "PE " << nvshmem_my_pe() << " buffer: { ";
    for(auto i: msg){
        result_stream << i << ", ";
    }
    result_stream << "}\n";
    std::cout << std::string(result_stream.str());
    nvshmem_free(destination);
    nvshmem_finalize();
}

__global__ void sandbox(){
    cuda::std::array<int, 100> task_counts = {};
    cuda::std::fill_n(task_counts.begin(), 90, 2);
}

void sandbox_runner(){
    sandbox<<<1,1>>>();
    CUTE_CHECK_ERROR(cudaDeviceSynchronize());
}

template<class TV>
__global__ void nvsh_test_kernel(TV* dest){
    dest[7] = TV(4.0);
    cute::print("Here is destination[7]: %f", (__half2float(dest[7])));
}

__global__ void heap_things(){
    cuda::std::array<int, 6> v = {{3, 2, 4, 1, 5, 9}};
    printf("initially, v is :{%d, %d, %d, %d, %d, %d}\n", v[0], v[1], v[2], v[3], v[4], v[5]);
    cuda::std::make_heap(v.begin(), v.end(), cuda::std::greater<>{});
    printf("after make_heap, v is :{%d, %d, %d, %d, %d, %d}\n", v[0], v[1], v[2], v[3], v[4], v[5]);
    cuda::std::pop_heap(v.begin(), v.end(), cuda::std::greater<>{});
    printf("after pop_heap, v is :{%d, %d, %d, %d, %d, %d}\n", v[0], v[1], v[2], v[3], v[4], v[5]);
}

__global__ void heap_things_pair(){
    cuda::std::array<cuda::std::pair<int, int>, 5> a {{{3, 0}, {2, 1}, {4, 2}, {1, 3}, {5, 4}}};
    printf("initially, a is :{%d, %d, %d, %d, %d}\n", a[0].first, a[1].first, a[2].first, a[3].first, a[4].first);
    cuda::std::make_heap(a.begin(), a.end(), cuda::std::greater<>{});
    printf("after make_heap, a is :{%d, %d, %d, %d, %d}\n", a[0].first, a[1].first, a[2].first, a[3].first, a[4].first);
    cuda::std::pop_heap(a.begin(), a.end(), cuda::std::greater<>{});
    printf("after pop_heap, a is :{%d, %d, %d, %d, %d}\n",  a[0].first, a[1].first, a[2].first, a[3].first, a[4].first);
}

void nvsh_test_ptr(){
    nvshmem_init();
    int my_pe_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    CUTE_CHECK_ERROR(cudaSetDevice(my_pe_node));
    using TensorType = __half;
    auto* destination = static_cast<TensorType*>(nvshmem_calloc(4, sizeof(float)));
    CUTE_CHECK_ERROR(cudaPeekAtLastError());
    nvsh_test_kernel<<<1,1>>>(destination);
    CUTE_CHECK_ERROR(cudaPeekAtLastError());
    CUTE_CHECK_ERROR(cudaDeviceSynchronize());
    nvshmem_free(destination);
    nvshmem_finalize();
}

template<aristos::Matrix T, unsigned int k=2>
void fused_top_idx(T input){
    using MatrixType = typename decltype(input)::value_type;
    cute::array<cuda::std::pair<MatrixType, int>, k> window{};
    auto my_slice = input(0, cute::_);
    CUTE_UNROLL
    for(int i = 0; i < k; ++i){
        window[i] = cuda::std::pair<MatrixType, int>{my_slice(i), i};
    }
    cuda::std::make_heap(window.begin(), window.end(), cuda::std::greater<>{});
    cuda::std::pop_heap(window.begin(), window.end(), cuda::std::greater<>{});
    // min element now at the end of the array
    for(int i = k; i < cute::size(my_slice); ++i){
        auto min_elem = window.back();
        if(cuda::std::pair<MatrixType, int>{my_slice(i), i} > min_elem){
            my_slice(min_elem.second) = MatrixType(0);
            window[k - 1] = cuda::std::pair<MatrixType, int>{my_slice(i), i};
            cuda::std::push_heap(window.begin(), window.end(), cuda::std::greater<>{});
            cuda::std::pop_heap(window.begin(), window.end(), cuda::std::greater<>{});
        }
        else{
            my_slice(i) = 0;
        }
    }
}
template<int Bits, bool Signed>
struct cute::is_integral<cutlass::integer_subbyte<Bits, Signed>> : cute::true_type {};

__global__ void memory_heterogeneity(void* symmetric,  uint64_t* flags, int my_pe){
    // Arrange
    const int peer = !my_pe;
    constexpr int k = 2;
    constexpr int n_tok = 2;
    constexpr int total_mem = n_tok + k;
    auto* scratchpad = static_cast<cute::half_t*>(symmetric);
    auto* scratchpad_begin = scratchpad + (my_pe*total_mem);
    scratchpad_begin[0] = cute::half_t(0.67);
    scratchpad_begin[1] = cute::half_t(0.02);
    auto* trailer = static_cast<uint_fast16_t*>(static_cast<void*>((scratchpad_begin + n_tok)));
    trailer[0] = 2;
    trailer[1] = 5;
    constexpr uint64_t set_flag = 1;

    // Send
    nvshmemx_putmem_signal_nbi_block(static_cast<void*>(scratchpad_begin),
                              static_cast<void*>(scratchpad_begin),
                              (total_mem * 2),
                              (flags + my_pe),
                              set_flag,
                              NVSHMEM_SIGNAL_SET,
                              peer);
    nvshmem_signal_wait_until((flags + my_pe), NVSHMEM_CMP_EQ, set_flag);

    // Reconstitute
    auto* floats = scratchpad + (peer*total_mem);
    cute::print("Received floats: %f, %f", floats[0].operator float(), floats[1].operator float());
    auto* r_trailers = static_cast<uint_fast16_t*>(static_cast<void*>(floats + n_tok));
    cute::print("Received trailers: %d, %d", r_trailers[0], r_trailers[1]);
}

__device__ __constant__ cuda::atomic<unsigned int, cuda::thread_scope_device> last{1};
__device__ __constant__ cuda::atomic<unsigned int, cuda::thread_scope_device> db{0};

__global__ void play_kernel(int n){
    printf("I am Thread %llu in Block %llu, x=%u, y=%u, z=%u, threadIdx.x=%u\n",
           cg::grid_group::thread_rank(),
           cg::grid_group::block_rank(),
           blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x);
    if(last.fetch_add(1) == n){
        db++;
        printf("Thread %d in Block %d is last: %d\n", aristos::grid_tid(), aristos::bid(), last.load());
    }
    if(cooperative_groups::grid_group::thread_rank() == 15){
        printf("Block_dim.x %d, Block_dim.y %d, Block_dim.z %d\n", cg::grid_group::dim_blocks().x,
               cg::grid_group::dim_blocks().y, cg::grid_group::dim_blocks().z);
        printf("Threads in a block is: %d\n", cg::thread_block::num_threads());
    }

}

__global__ void play2(int* sync_grid, cuda::barrier<cuda::thread_scope_device>* b){
    auto t = cute::make_tensor(cute::make_gmem_ptr(sync_grid),
                               cute::make_shape(1, cg::grid_group::num_threads()));
    cuda::atomic_ref<int, cuda::thread_scope_device> a(t(cg::thread_block::thread_rank()));
    a.fetch_add(1);
    printf("I am Thread %llu -> %u in Block %llu -> %u, where my index is %u -> %u\n",
           cg::grid_group::thread_rank(),
           aristos::grid_tid(),
           cg::grid_group::block_rank(),
           aristos::bid(),
           cg::thread_block::thread_rank(),
           aristos::block_tid());
    b->arrive_and_wait();
    if(cg::thread_block::thread_rank() == 0){
        printf("AS EXPECTED, atomic result is %d and t[0] is %d\n", a.load(), t(0));
    }
}

__global__ void bench_cg(CUTE_GRID_CONSTANT const int iter, unsigned int* sync_p, unsigned int* h_sync_p){
    // initialization
    cuda::associate_access_property(&last, cuda::access_property::persisting{});
    cuda::associate_access_property(sync_p, cuda::access_property::persisting{});
    cuda::associate_access_property(h_sync_p, cuda::access_property::persisting{});
    auto start = cuda::std::chrono::high_resolution_clock::now();
    auto end = cuda::std::chrono::high_resolution_clock::now();
    auto elapsed_seconds{end - start};
    size_t dev_atomic = 0, a_add = 0, a_cas = 0, freq_at = 0, a_inc = 0;
    size_t x;
    /*auto t = aristos::grid_tid();
    auto tt = t;*/
    CUTE_UNROLL
    for(int i = 0; i < iter; ++i){
        start = cuda::std::chrono::high_resolution_clock::now();
        last++;
        elapsed_seconds = cuda::std::chrono::high_resolution_clock::now() - start;
        dev_atomic += cuda::std::chrono::duration_cast<cuda::std::chrono::microseconds>(elapsed_seconds).count();

        start = cuda::std::chrono::high_resolution_clock::now();
        x = atomicAdd(sync_p, 0U); // equivalent to a load
        elapsed_seconds = cuda::std::chrono::high_resolution_clock::now() - start;
        a_add += cuda::std::chrono::duration_cast<cuda::std::chrono::microseconds>(elapsed_seconds).count();

        start = cuda::std::chrono::high_resolution_clock::now();
        atomicInc(h_sync_p, 0U); // equivalent to a load
        elapsed_seconds = cuda::std::chrono::high_resolution_clock::now() - start;
        a_inc += cuda::std::chrono::duration_cast<cuda::std::chrono::microseconds>(elapsed_seconds).count();

        start = cuda::std::chrono::high_resolution_clock::now();
        atomicCAS(sync_p, 0U, 0U); // equivalent to a load
        elapsed_seconds = cuda::std::chrono::high_resolution_clock::now() - start;
        a_cas += cuda::std::chrono::duration_cast<cuda::std::chrono::microseconds>(elapsed_seconds).count();

        start = cuda::std::chrono::high_resolution_clock::now();
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device>(*sync_p).load();
        elapsed_seconds = cuda::std::chrono::high_resolution_clock::now() - start;
        freq_at += cuda::std::chrono::duration_cast<cuda::std::chrono::microseconds>(elapsed_seconds).count();
    }
    // warp all-reduce
//    dev_atomic = __reduce_max_sync(0xffffffff, dev_atomic);
//    a_add = __reduce_max_sync(0xffffffff, a_add);
//    a_cas = __reduce_max_sync(0xffffffff, a_cas);
//    freq_at = __reduce_max_sync(0xffffffff, freq_at);
    // Specialize BlockReduce for a 1D block of 128 threads of type int
    using BlockReduce = cub::BlockReduce<size_t, THREADS_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    dev_atomic = BlockReduce(temp_storage).Reduce(dev_atomic, cub::Max());
    a_add = BlockReduce(temp_storage).Reduce(a_add, cub::Max());
    a_inc = BlockReduce(temp_storage).Reduce(a_inc, cub::Max());
    a_cas = BlockReduce(temp_storage).Reduce(a_cas, cub::Max());
    freq_at = BlockReduce(temp_storage).Reduce(freq_at, cub::Max());

    if(aristos::grid_tid() == 0){
        printf("dev_atomic: {T: %f, V: %d}, a_add: {T: %f, V:%u}, a_inc: {T: %f, V:%u}, a_cas: {T: %f, V: %u}, freq_at: {T: %f, V: %u},"
               " x is %llu\n",
               dev_atomic / (iter*1.0),
               last.load(),
               a_add/(iter*1.0),
               atomicAdd(sync_p, 0),
               a_inc / (iter * 1.0),
               atomicExch(sync_p, 0),
               a_cas/(iter*1.0),
               atomicCAS(sync_p, 0, 0),
               freq_at/(iter*1.0),
               cuda::atomic_ref<unsigned int, cuda::thread_scope_device>(*sync_p).load(),
               x);
    }

}

__global__ void testArgs(){
    __shared__ aristos::SenderConfig d_c;
    if(aristos::block_tid() == 0){
        d_c = aristos::SenderConfig(moeConfig);
    }
    __threadfence_block();
    __syncthreads();
    if(aristos::block_tid() == 1){
        d_c.dump();
        printf("Before, flags[0] is %lu; ",d_c.flags[0]);
        d_c.flags[0] = 56;
        printf("Now, flags[0] is %lu\n",d_c.flags[0]);
        printf("is Heap in Global Memory? %s\n", __isGlobal(d_c.sHeap) ? "Yes" : "No");
        printf("is d_c in Shared Memory? %s\n", __isShared(&d_c) ? "Yes" : "No");
    }
}
// local memory for days
__global__ void testTen(std::byte* p, size_t* q, int* r){
    __shared__ aristos::SenderConfig s;
}

template<unsigned int bM=128, unsigned int bN=128, unsigned int bK=4, unsigned int bP=3>
__global__ void occupancyTestKernel(){
    __shared__ float sharedA[cute::cosize_v<decltype(cute::make_layout(cute::make_shape(cute::Int<bM>{}, cute::Int<bK>{}, cute::Int<bP>{})))>];
    __shared__ float sharedB[cute::cosize_v<decltype(cute::make_layout(cute::make_shape(cute::Int<bN>{}, cute::Int<bK>{}, cute::Int<bP>{})))>];
}

void aristosInit(unsigned int seqLen, unsigned int embedDim, unsigned int hiddenProjDim,
                 unsigned int k) {
    // initialize nvshmem
    int my_pe = 0;

    // initialize moeConfig
    auto c = aristos::Config();
    int numSMs = 0;
    CUTE_CHECK_ERROR(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, my_pe));

    unsigned int bM = 128;
    unsigned int bN = 128;
    unsigned int bK = 4;
    unsigned int bP = 3;
    int numBlocks = 0;
    int blockSize = 128; // 256 is too high, since SM can only hold <= 2048 threads
    int minCommunicatorBlocks = 2; // We can be flexible here
    int minBlocks = (cute::ceil_div(seqLen, bM) * cute::ceil_div(hiddenProjDim, bN)) + 2;
    CUTE_CHECK_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &numBlocks,
            occupancyTestKernel<128, 128, 4, 3>,
            blockSize,
            0));
    assert(minBlocks <= (numBlocks * numSMs));

    // Good to go!
    // Let's do some initialization
}

void forwardHost(){
}

void backwardHost(){
}

int main() {
    /*cute::array<int, 4> u = {1,2,3,4};
    cute::array<int, 4> v = {4,5,6,7};
    cute::tuple<cute::array<int , 4>, cute::array<int, 4>> t = {u, v};
    std::cout << cute::get<1>(t)[0] << std::endl;
    std::cout << cute::tuple_size<decltype(t)>::value << std::endl;

    cute::array<int*, 2> u_ptr = {};
    cute::array<int*, 2> v_ptr = {};
    std::cout << cute::half_t(0.8).operator float() << std::endl;
    auto ff = cute::float_e4m3_t(4.25f);
    std::cout << ff.operator float() << std::endl;
    heap_things_pair<<<1,1>>>();
    CUTE_CHECK_ERROR(cudaPeekAtLastError());
    CUTE_CHECK_ERROR(cudaDeviceSynchronize());

    cute::complex<float> e_i = cute::complex(89.0);
    auto a_temp = std::array<int, 4>{78, 89, 91};
    auto a_t = cute::make_tensor(a_temp.data(), cute::make_layout(cute::make_shape(2, 2)));
    std::cout << (cute::is_complex<typename cute::iterator_traits<decltype(a_t.data())>::value_type>::value) << std::endl;
    auto a_t0 = a_t(0, cute::_);
    std::cout << a_t0(0) << std::endl;
    std::cout << cute::size(a_t) << std::endl;

    cuda::std::pair<int, int> p = {8, 9};
    cuda::std::pair<int, int> q = {8, 3};

    std::cout << (p < q) << std::endl;
    cuda::std::array<cuda::std::pair<int, int>, 4> a {};
    cuda::std::array<int, 4> a = {{1,67,3,4}};
    std::cout << *(a.begin() + 1) << std::endl;
    std::cout << *a.begin() << std::endl;
    cuda::std::array<cuda::std::pair<int, int>, 5> a {{{3, 0}, {2, 1}, {4, 2}, {1, 3}, {5, 4}}};
    auto a_t = cute::make_tensor(a.data(), cute::make_layout(cute::make_shape(1, 5)));
    auto my_slice = a_t(0, cute::_);
    int p = my_slice(0).first;
    auto pp = cuda::std::pair<int, int>{0, 0};

    cuda::std::array<int, 5> a_test {{3, 2, 4, 1, 5}};
    auto b_t = cute::make_tensor(a_test.data(), cute::make_layout(cute::make_shape(1, 5)));
    printf("initially, a_test is :{%d, %d, %d, %d, %d}\n", a_test[0], a_test[1], a_test[2], a_test[3], a_test[4]);
    fused_top_idx(b_t);
    printf("after fused_top_idx, a_test is :{%d, %d, %d, %d, %d}\n", a_test[0], a_test[1], a_test[2], a_test[3], a_test[4]);

    auto hh = cute::half_t(0);
    using hh_t = decltype(hh);
    auto u8 = cute::uint1_t(0).storage
    static_assert(cuda::std::is_integral<decltype(cute::uint1_t(0).storage)>()); // 2 and 4 int and float no bueno, neither does nvidia types. stdints

    static_assert(cuda::std::same_as<decltype(hh_t(0)), cute::half_t>);
    cute::array<int, 2> aa{};
    std::cout << aa[0]++ << std::endl;
    std::cout << aa[0] << std::endl;
    void* parent = calloc(4, 4);
    const int k = 2;
    const int n_tok = 2;
    auto* scratchpad = static_cast<cute::half_t*>(parent);
    scratchpad[0] = cute::half_t(0.67);
    scratchpad[1] = cute::half_t(0.02);
    auto* trailer = static_cast<uint_fast16_t*>(static_cast<void*>((scratchpad + n_tok)));
    trailer[0] = 2;
    trailer[1] = 5;

    void* recipient = calloc(4, 4);
    std::memcpy(recipient, parent, 4*4);

    auto* float_payload = static_cast<cute::half_t*>(recipient);
    cute::print("Payload is %f, %f\n", float_payload[0].operator float(), float_payload[1].operator float());
    const auto* metadata = static_cast<uint_fast16_t*>(static_cast<void*>((float_payload + n_tok)));
    cute::print("Trailer is %d, %d", metadata[0], metadata[1]);
    free(recipient);
    free(parent);*/
    /*int* p;
    dim3 dimGrid(2,2, 2);
    dim3 dimBlock(2,2);
    auto n_threads = dimGrid.x * dimGrid.y * dimGrid.z * dimBlock.x * dimBlock.y * dimBlock.z;
    std::cout << "n_threads is " << n_threads <<  std::endl;
    cuda::barrier<cuda::thread_scope_device>* b;
    // OR (recommended actually) you can do the below instead of MallocManaged()
    // auto host_b = new cuda::barrier<cuda::thread_scope_device>{n_threads};
    // CUTE_CHECK_ERROR(cudaMalloc(&b, sizeof(cuda::barrier<cuda::thread_scope_device>)));
    // CUTE_CHECK_LAST();
    // CUTE_CHECK_ERROR(cudaMemcpy(b, host_b, sizeof(cuda::barrier<cuda::thread_scope_device>), cudaMemcpyHostToDevice));
    // CUTE_CHECK_LAST();
    CUTE_CHECK_ERROR(cudaMallocManaged(&p, sizeof(int)*n_threads));
    CUTE_CHECK_ERROR(cudaMallocManaged(&b, sizeof(cuda::barrier<cuda::thread_scope_device>)));
    new (b) cuda::barrier<cuda::thread_scope_device>{n_threads};
    CUTE_CHECK_LAST();
    CUTE_CHECK_ERROR(cudaMemset(p, 0, (sizeof(int)*n_threads)));
    CUTE_CHECK_LAST();
    play2<<<dimGrid,dimBlock>>>(p, b);
    CUTE_CHECK_LAST();*/
    /*unsigned int* p;
    unsigned int* h_p;
    CUTE_CHECK_ERROR(cudaMalloc(&p, sizeof(unsigned int)*2));
    CUTE_CHECK_ERROR(cudaMemset(p, 0, sizeof(unsigned int)*2));
    CUTE_CHECK_ERROR(cudaMalloc(&h_p, sizeof(unsigned int)));
    CUTE_CHECK_ERROR(cudaMemset(h_p, 0, sizeof(unsigned int)));
    CUTE_CHECK_LAST();
    bench_cg<<<1,THREADS_PER_BLOCK>>>(1024, p, h_p);
    CUTE_CHECK_LAST();*/

//    int p_bytes = 4;
//    int k = 4;
//    auto t = cute::make_counting_tensor(cute::make_layout(cute::make_shape(2, 10), cute::LayoutRight{}));
//
//    auto payloads = cute::composition(t, cute::make_shape(cute::_, p_bytes));
//    auto numRouting = t(cute::_, (p_bytes));
//    auto tokenIds = t(cute::_, cute::size<1>(t) - 1);
//    auto experts = cute::make_tensor(t.data() + p_bytes + 1, cute::make_shape(2, k), cute::make_stride(1 + p_bytes + 5, 1));
//    cute::print_tensor(t);
//    cute::print_tensor(payloads);
//    cute::print_tensor(numRouting);
//    cute::print_tensor(experts);
//    cute::print_tensor(tokenIds);
    /*void* p;
    uint64_t* f;
    unsigned int* s;
    CUTE_CHECK_ERROR(cudaMalloc(&p, sizeof(unsigned int)));
    CUTE_CHECK_ERROR(cudaMemset(p, 0, sizeof(unsigned int)));
    CUTE_CHECK_ERROR(cudaMalloc(&f, sizeof(unsigned int)));
    CUTE_CHECK_ERROR(cudaMemset(f, 0, sizeof(unsigned int)));
    CUTE_CHECK_ERROR(cudaMalloc(&s, sizeof(unsigned int)));
    CUTE_CHECK_ERROR(cudaMemset(s, 0, sizeof(unsigned int)));

    using json = nlohmann::json;
    auto c = aristos::Config(f,
                             p,
                             s,
                             s,
                             1,
                             0,
                             3,
                             2,
                             1,
                             1024,
                             2,
                             2,
                             2048);
    CUTE_CHECK_ERROR(cudaMemcpyToSymbol(moeConfig, &c, sizeof(aristos::Config)));
    testArgs<<<2, 32>>>();
    CUTE_CHECK_LAST();
    CUTE_CHECK_ERROR(cudaFree(p));
    CUTE_CHECK_ERROR(cudaFree(f));
    CUTE_CHECK_ERROR(cudaFree(s));*/
//    torch::Tensor tensor = torch::rand({2, 3});
//    std::cout << tensor << std::endl;
    play_kernel<<<dim3{4,4}, dim3{2,2}>>>(32);
    CUTE_CHECK_LAST();
    play_kernel<<<8, dim3{2,2}>>>(32);
    CUTE_CHECK_LAST();
    return 0;
}