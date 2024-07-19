#include <iostream>
#include <array>

#include <cuda_runtime.h>
#include <cublasdx.hpp>
#include <cuda/std/array>
#include <cuda/std/chrono>

// Heap things
#include <cuda/std/__algorithm/make_heap.h>
#include <cuda/std/__algorithm/pop_heap.h>

#include <nvshmemx.h>
#include <host/nvshmemx_api.h>

#include "include/aristos.cuh"

#define THREADS_PER_WARP 32

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

template<class GEMM, class ValueType = typename GEMM::value_type, unsigned int epsilon = 2>
__global__ void aristos_kernel(const ValueType alpha,
                               const ValueType* routing_weights,
                               const ValueType* shard_spec,
                               ValueType* c,
                               int my_pe){
    // GEMM to map tokens to peers
    extern __shared__ ValueType smem[];
    ValueType* sa = smem;
    ValueType* sb = smem + GEMM::a_size;
    ValueType* sc = smem + GEMM::a_size + GEMM::b_size;

    auto dM = cublasdx::size_of<GEMM>::m;
    const auto dN = cublasdx::size_of<GEMM>::n;
    auto dK = cublasdx::size_of<GEMM>::k;

    auto sa_tensor = cute::make_tensor(sa, cute::make_layout(cute::make_shape(dM, dK)));
    cute::Tensor sb_tensor = cute::make_tensor(sb, cute::make_layout(cute::make_shape(dK, dN)));
    cute::Tensor sc_tensor = cute::make_tensor(sc, cute::make_layout(cute::make_shape(dM, dN)));
    cute::Tensor act_tensor = cute::make_tensor(cute::make_gmem_ptr(routing_weights), cute::make_shape(dM, dK));
    cute::Tensor spec_tensor = cute::make_tensor(cute::make_gmem_ptr(shard_spec), cute::make_shape(dK, dN));

    cute::copy(act_tensor, sa_tensor);
    cute::copy(spec_tensor, sb_tensor);

    __syncthreads();
    GEMM().execute(alpha, sa, sb, 0.0, sc);
    __syncthreads();

    if(cute::thread0()){
        cute::print_tensor(sc_tensor);
    }

    cute::array_aligned<ValueType, cublasdx::size_of<GEMM>::m> tokens = {};
    // Iterate through each token slice and send to corresponding GPU.
    CUTE_UNROLL
    for(unsigned int n = 0; n < cute::size<1>(sc_tensor); ++n){
        if (cute::thread0()){
            cute::print("Sending { ");
        }
        for(unsigned int m = 0; m < cute::size<0>(sc_tensor); ++m){
            if(sc_tensor(m, n) > 0){
                tokens[m] = sc_tensor(m, n);
                if(cute::thread0()){
                    cute::print("%d, ", m);
                }
            }
        }
        if(cute::thread0()){
            cute::print("} to Peer %d...\n", n);
        }
    }

    // Initialize algo inputs
    cuda::std::array<int, dN> task_counts = {};
    cuda::std::fill_n(task_counts.begin(), dN, 2);
    int mu = 2 * dN;
    bool should_timeout = false;
    auto snapshot_now = cuda::std::chrono::high_resolution_clock::now();
    auto timeout_duration = cuda::std::chrono::duration<float, cuda::std::milli>
            (cuda::std::chrono::microseconds(1));

    int var_epsilon = epsilon;
    while(dN > 0 && !should_timeout){
        if(var_epsilon == 0){
            snapshot_now = cuda::std::chrono::high_resolution_clock::now();
            --var_epsilon; // Negates the if-condition
        }
        should_timeout = (var_epsilon < 0 &&
                          (cuda::std::chrono::high_resolution_clock::now() - snapshot_now) >= timeout_duration);

    }
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
    nvshmem_putmem_signal_nbi(static_cast<void*>(scratchpad_begin),
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

int main() {
//    cute::array<int, 4> u = {1,2,3,4};
//    cute::array<int, 4> v = {4,5,6,7};
//    cute::tuple<cute::array<int , 4>, cute::array<int, 4>> t = {u, v};
//    std::cout << cute::get<1>(t)[0] << std::endl;
//    std::cout << cute::tuple_size<decltype(t)>::value << std::endl;
//
//    cute::array<int*, 2> u_ptr = {};
//    cute::array<int*, 2> v_ptr = {};
//    std::cout << cute::half_t(0.8).operator float() << std::endl;
//    auto ff = cute::float_e4m3_t(4.25f);
//    std::cout << ff.operator float() << std::endl;
//    heap_things_pair<<<1,1>>>();
//    CUTE_CHECK_ERROR(cudaPeekAtLastError());
//    CUTE_CHECK_ERROR(cudaDeviceSynchronize());

//    cute::complex<float> e_i = cute::complex(89.0);
//    auto a_temp = std::array<int, 4>{78, 89, 91};
//    auto a_t = cute::make_tensor(a_temp.data(), cute::make_layout(cute::make_shape(2, 2)));
//    std::cout << (cute::is_complex<typename cute::iterator_traits<decltype(a_t.data())>::value_type>::value) << std::endl;
//    auto a_t0 = a_t(0, cute::_);
//    std::cout << a_t0(0) << std::endl;
//    std::cout << cute::size(a_t) << std::endl;
//
//    cuda::std::pair<int, int> p = {8, 9};
//    cuda::std::pair<int, int> q = {8, 3};
//
//    std::cout << (p < q) << std::endl;
//    cuda::std::array<cuda::std::pair<int, int>, 4> a {};
//    cuda::std::array<int, 4> a = {{1,67,3,4}};
//    std::cout << *(a.begin() + 1) << std::endl;
//    std::cout << *a.begin() << std::endl;
//    cuda::std::array<cuda::std::pair<int, int>, 5> a {{{3, 0}, {2, 1}, {4, 2}, {1, 3}, {5, 4}}};
//    auto a_t = cute::make_tensor(a.data(), cute::make_layout(cute::make_shape(1, 5)));
//    auto my_slice = a_t(0, cute::_);
//    int p = my_slice(0).first;
//    auto pp = cuda::std::pair<int, int>{0, 0};
//
//    cuda::std::array<int, 5> a_test {{3, 2, 4, 1, 5}};
//    auto b_t = cute::make_tensor(a_test.data(), cute::make_layout(cute::make_shape(1, 5)));
//    printf("initially, a_test is :{%d, %d, %d, %d, %d}\n", a_test[0], a_test[1], a_test[2], a_test[3], a_test[4]);
//    fused_top_idx(b_t);
//    printf("after fused_top_idx, a_test is :{%d, %d, %d, %d, %d}\n", a_test[0], a_test[1], a_test[2], a_test[3], a_test[4]);

//    auto hh = cute::half_t(0);
//    using hh_t = decltype(hh);
//    auto u8 = cute::uint1_t(0).storage
//    static_assert(cuda::std::is_integral<decltype(cute::uint1_t(0).storage)>()); // 2 and 4 int and float no bueno, neither does nvidia types. stdints
//
//    static_assert(cuda::std::same_as<decltype(hh_t(0)), cute::half_t>);
//    cute::array<int, 2> aa{};
//    std::cout << aa[0]++ << std::endl;
//    std::cout << aa[0] << std::endl;
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
    free(parent);
    return 0;
}