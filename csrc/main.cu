#include <iostream>
#include <array>

#include <cuda_runtime.h>
#include <cublasdx.hpp>
#include <cuda/std/array>
#include <cuda/std/chrono>
// Heap things
#include <cuda/std/__algorithm/make_heap.h>
#include <cuda/std/__algorithm/pop_heap.h>

#include <nvshmem.h>
#include <nvshmemx.h>
#include <nvtx3/nvtx3.hpp>

#include "aristos.cuh"

#define HANDLE_CUDA_ERROR(x)                                      \
{ const auto err = x;                                             \
  if( err != cudaSuccess )                                        \
  { printf("Error: %s\n", cudaGetErrorString(err)); exit(-1); } \
};

#ifndef CUDA_CHECK_AND_EXIT
#    define CUDA_CHECK_AND_EXIT(error)                                                                      \
        {                                                                                                   \
            auto status = static_cast<cudaError_t>(error);                                                  \
            if (status != cudaSuccess) {                                                                    \
                std::cout << cudaGetErrorString(status) << " " << __FILE__ << ":" << __LINE__ << std::endl; \
                std::exit(status);                                                                          \
            }                                                                                               \
        }
#endif

#define THREADS_PER_WARP 32
__global__ void aristos_sandbox(int *destination, int my_pe, int n_pes, bool skip = true) {
    cuda::std::array<int, 4>activations{{my_pe + 1, 2,my_pe + 3, 4}};
    int n_elems = 2;
    int k = 0;
    const auto start{cuda::std::chrono::high_resolution_clock::now()};
    for(uint i = 0; i < n_pes; ++i){
        for(uint j = 0; j < n_elems; ++j){
            destination[j] = activations[k++];
        }
        nvshmem_int_put((destination + ((my_pe+1)*n_elems)),
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
    CUDA_CHECK_AND_EXIT(cudaGetDeviceCount(&dev_count))
    cudaDeviceProp prop{};
    CUDA_CHECK_AND_EXIT(cudaGetDeviceProperties(&prop, my_pe_node % dev_count))
    int clockrate;
    CUDA_CHECK_AND_EXIT(cudaDeviceGetAttribute(&clockrate, cudaDevAttrClockRate, my_pe_node))

    fprintf(stderr, "mype: %d mype_node: %d device name: %s bus id: %d n_pes: %d\n", my_pe, my_pe_node,
            prop.name, prop.pciBusID, n_pe);
    cudaSetDevice(my_pe_node);
    cudaStreamCreate(&stream);

    int *destination = (int*) nvshmem_calloc(msg.size(), sizeof(int));
    for (int i = 0; i < 5; ++i) {
        aristos_sandbox<<<1, 1>>>(destination, my_pe, n_pe);
        CUDA_CHECK_AND_EXIT(cudaPeekAtLastError())
    }
    aristos_sandbox<<<1, 1>>>(destination, my_pe, n_pe, false);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError())
    HANDLE_CUDA_ERROR(cudaMemcpyAsync(msg.data(), destination, msg.size() * sizeof(int), cudaMemcpyDeviceToHost, stream))
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

    cute::Tensor sa_tensor = cute::make_tensor(sa, cute::make_layout(cute::make_shape(dM, dK)));
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
            var_epsilon--; // Negates the if-condition
        }
        should_timeout = (var_epsilon < 0 &&
                          (cuda::std::chrono::high_resolution_clock::now() - snapshot_now) >= timeout_duration);

    }
}

template<unsigned int Arch = 700>
void aristos_kernel_run(){
    int my_pe_node;
    nvshmem_init();
    my_pe_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    int my_pe = nvshmem_my_pe();
    int n_pes = nvshmem_n_pes();
    int dev_count;
    CUDA_CHECK_AND_EXIT(cudaGetDeviceCount(&dev_count))
    cudaDeviceProp prop{};
    CUDA_CHECK_AND_EXIT(cudaGetDeviceProperties(&prop, my_pe_node % dev_count))
    int clockrate;
    CUDA_CHECK_AND_EXIT(cudaDeviceGetAttribute(&clockrate, cudaDevAttrClockRate, my_pe_node))

    fprintf(stderr, "mype: %d mype_node: %d device name: %s bus id: %d n_pes: %d\n", my_pe, my_pe_node,
            prop.name, prop.pciBusID, n_pes);
    cudaSetDevice(my_pe_node);

    constexpr unsigned int M = 4;
    constexpr unsigned int N = 2;
    constexpr unsigned int K = 4;

    // Choose precision (__half, float, double) and type (real or complex).
    using precision = float;
    constexpr auto type = cublasdx::type::real;

    // Choose transpose mode for A and B: non_transposed, transposed, or conj_transposed.
    constexpr auto a_transpose_mode = cublasdx::transpose_mode::non_transposed;
    constexpr auto b_transpose_mode = cublasdx::transpose_mode::non_transposed;

    // Define the local matrix multiplication operation.
    using BlockMM  = decltype(cublasdx::Size<M, N, K>() +
                              cublasdx::Precision<precision>() +
                              cublasdx::Type<type>() +
                              cublasdx::Function<cublasdx::function::MM>() +
                              cublasdx::TransposeMode<a_transpose_mode, b_transpose_mode>() +
                              cublasdx::Block() +
                              cublasdx::SM<Arch>());

    using value_type = typename BlockMM::value_type;


    // Allocate device memory
    value_type* abc;
    auto size = BlockMM::a_size + BlockMM::b_size + BlockMM::c_size;
    auto size_bytes = size * sizeof(value_type);
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&abc, size_bytes));

    value_type* a = abc;
    value_type* b = a + BlockMM::a_size;
    value_type* c = b + BlockMM::b_size;

    // Init host data
    const std::array<float, M*K> a_host = {{0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1 , 0, 0}};
    const std::array<float, K*N> b_host = {{0, 1, 1, 1, 1, 0, 0, 0}};

    // Copy host data to device
    CUDA_CHECK_AND_EXIT(cudaMemcpy(a, a_host.data(), BlockMM::a_size * sizeof (value_type), cudaMemcpyHostToDevice))
    CUDA_CHECK_AND_EXIT(cudaMemcpy(b, b_host.data(), BlockMM::b_size * sizeof (value_type), cudaMemcpyHostToDevice))

    aristos_kernel<BlockMM><<<1, BlockMM::block_dim, BlockMM::shared_memory_size>>>
    (1.0f, a, b, c, my_pe);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError())
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize())
    CUDA_CHECK_AND_EXIT(cudaFree(abc))

    // Device vector is in global memory https://stackoverflow.com/a/71449425
}

__global__ void sandbox(){
    cuda::std::array<int, 100> task_counts = {};
    cuda::std::fill_n(task_counts.begin(), 90, 2);
}

void sandbox_runner(){
    sandbox<<<1,1>>>();
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize())
}

template<class TV>
__global__ void nvsh_test_kernel(TV* dest){
    dest[7] = TV(4.0);
    cute::print("Here is destination[7]: %f", (__half2float(dest[7])));

}

__global__ void heap_things(){
    cuda::std::array<int, 6> v = {3, 2, 4, 1, 5, 9};
    printf("initially, v is :{%d, %d, %d, %d, %d, %d}\n", v[0], v[1], v[2], v[3], v[4], v[5]);
    cuda::std::make_heap(v.begin(), v.end(), cuda::std::greater<>{});
    printf("after make_heap, v is :{%d, %d, %d, %d, %d, %d}\n", v[0], v[1], v[2], v[3], v[4], v[5]);
    cuda::std::pop_heap(v.begin(), v.end(), cuda::std::greater<>{});
    printf("after pop_heap, v is :{%d, %d, %d, %d, %d, %d}\n", v[0], v[1], v[2], v[3], v[4], v[5]);
}

void nvsh_test_ptr(){
    nvshmem_init();
    int my_pe_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    CUDA_CHECK_AND_EXIT(cudaSetDevice(my_pe_node))
    using TensorType = __half;
//    auto* destination = static_cast<TensorType*>(nvshmem_calloc(4, sizeof(float)));
    auto* destination = static_cast<TensorType*>(nvshmem_calloc(4, sizeof(float)));
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError())
    nvsh_test_kernel<<<1,1>>>(destination);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError())
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize())
    nvshmem_free(destination);
    nvshmem_finalize();
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
    heap_things<<<1,1>>>();
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError())
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize())
    return 0;
}
