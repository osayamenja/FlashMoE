cmake_minimum_required(VERSION 3.27)
project(csrc CUDA)

set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CUDA_STANDARD 17)

include(CheckCompilerFlag)
check_compiler_flag(CUDA -t4 NVCC_THREADS)

find_package(CUDAToolkit REQUIRED)

add_executable(csrc main.cu)

# add dependencies
include(cmake/CPM.cmake)
# This will automatically clone CCCL from GitHub and make the exported cmake targets available
CPMAddPackage(
        NAME CCCL
        GITHUB_REPOSITORY nvidia/cccl
        GIT_TAG main # Fetches the latest commit on the main branch
)

target_link_libraries(csrc PRIVATE CCCL::CCCL)

set_target_properties(csrc PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

## NVSHMEM Business
set(NVSHMEM_HOME "$ENV{NVSHMEM_HOME}")
find_package(NVSHMEM REQUIRED HINTS ${NVSHMEM_HOME}/lib/cmake/nvshmem)
add_library(nvshmem_host ALIAS nvshmem::nvshmem_host)
add_library(nvshmem_device ALIAS nvshmem::nvshmem_device)
target_link_libraries(csrc PUBLIC nvshmem_host nvshmem_device)
target_link_libraries(csrc PUBLIC CUDA::cudart CUDA::cuda_driver)

target_compile_options(csrc
        PRIVATE $<$<CONFIG:Debug>:-O0;-g;>
        $<$<AND:$<BOOL:${NVSHMEM_VERBOSE}>,$<COMPILE_LANGUAGE:CUDA>>:-Xptxas -v>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-O0;-g;-G>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${NVCC_THREADS}>>:-t4>
)