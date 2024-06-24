#include <array>
#include <iostream>

#include <cuda/std/array>
#include <cuda/std/chrono>
#include <cute/util/debug.hpp>
#include <nvshmemx.h>
#include <nvtx3/nvtx3.hpp>

#define HANDLE_CUDA_ERROR(x)                                      \
{ const auto err = x;                                             \
  if( err != cudaSuccess )                                        \
  { printf("Error: %s\n", cudaGetErrorString(err)); exit(-1); } \
};
#define THREADS_PER_WARP 32

// Ring shuffle
__global__ void aristos(int *destination, int my_pe, int n_pes, bool skip = true) {
    cuda::std::array<cuda::std::array<int, 4>, 4>activations{{{1, 2,3, 4}}};
    const auto start{cuda::std::chrono::high_resolution_clock::now()};
    nvshmemx_int_put_nbi_warp(destination,
                          reinterpret_cast<const int *>(activations.data()), 4, ((my_pe + 1) % n_pes));
    const auto end{cuda::std::chrono::high_resolution_clock::now()};
    const cuda::std::chrono::duration<double> elapsed_seconds{end - start};

    if (!skip && cute::thread(0)){
        printf("Elapsed time is %lld\n",
               cuda::std::chrono::duration_cast<cuda::std::chrono::microseconds>(elapsed_seconds).count());
        printf("Item[0][0] is %d\n", activations[0][0]);
    }
}
int main() {
    {
        NVTX3_FUNC_RANGE()
        std::cout << "Hello, World!" << std::endl;
        int my_pe_node;
        std::array<int, 4> msg{};
        cudaStream_t stream;

        nvshmem_init();
        my_pe_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
        cudaSetDevice(my_pe_node);
        cudaStreamCreate(&stream);

        int *destination = (int *) nvshmem_calloc(4, sizeof(int));
        for (int i = 0; i < 20; ++i) {
            nvtx3::scoped_range test_run("Aristos Test Run");
            aristos<<<1, 1, 0, stream>>>(destination, nvshmem_my_pe(), nvshmem_n_pes());
        }
        {
            nvtx3::scoped_range range("Aristos");
            aristos<<<1, THREADS_PER_WARP, 0, stream>>>(destination, nvshmem_my_pe(), nvshmem_n_pes(), false);
        }
        nvshmemx_barrier_all_on_stream(stream);
        HANDLE_CUDA_ERROR(cudaMemcpyAsync(msg.data(), destination, 4 * sizeof(int), cudaMemcpyDeviceToHost, stream))
        cudaStreamSynchronize(stream);
        std::cout << "PE " << nvshmem_my_pe()
                  << " received message " << msg[2] << std::endl;

        nvshmem_free(destination);
        nvshmem_finalize();
    }
    return 0;
}
