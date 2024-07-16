cmake_minimum_required(VERSION 3.27)

# Set c++ and c compilers
execute_process(COMMAND "which" "g++" OUTPUT_VARIABLE CPP_COMP_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND "which" "gcc" OUTPUT_VARIABLE C_COMP_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
set(CMAKE_CXX_COMPILER "${CPP_COMP_PATH}")
set(CMAKE_C_COMPILER "${C_COMP_PATH}")

project(csrc CUDA CXX)

# flags
set(ARISTOS_CUDA_CXX_FLAGS "${ARISTOS_CUDA_CXX_FLAGS} -Wall -Wextra")
set(ARISTOS_CUDA_CXX_FLAGS "${ARISTOS_CUDA_CXX_FLAGS} -fno-strict-aliasing")
if(NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "^aarch64")
    set(ARISTOS_CUDA_CXX_FLAGS "${ARISTOS_CUDA_CXX_FLAGS} -m64")
endif ()
set(ARISTOS_CUDA_CXX_FLAGS "${ARISTOS_CUDA_CXX_FLAGS} -Wno-unknown-pragmas")
set(ARISTOS_CUDA_CXX_FLAGS "${ARISTOS_CUDA_CXX_FLAGS} -Wno-switch")
set(ARISTOS_CUDA_CXX_FLAGS "${ARISTOS_CUDA_CXX_FLAGS} -Wno-unused-but-set-parameter")
set(ARISTOS_CUDA_CXX_FLAGS "${ARISTOS_CUDA_CXX_FLAGS} -Wno-sign-compare")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ARISTOS_CUDA_CXX_FLAGS}")

set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xfatbin -compress-all") # Compress all fatbins
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --display_error_number") # Show error/warning numbers
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler \"${ARISTOS_CUDA_CXX_FLAGS}\"")

include(CheckCompilerFlag)
check_compiler_flag(CUDA -t4 NVCC_THREADS)

find_package(CUDAToolkit REQUIRED)

#MathDx business
set(NvidiaCutlass_ROOT $ENV{CUTLASS_PATH})
set(cublasdx_CUTLASS_ROOT $ENV{CUTLASS_PATH})
find_package(mathdx REQUIRED COMPONENTS cublasdx CONFIG)

add_executable(csrc main.cu
        include/communication/queue.cuh
        include/communication/packet.cuh
        include/moe/expert.cuh
        include/aristos.cuh
        include/moe/gate.cuh
        include/compute/compute.cuh
        include/algorithm/algorithm.cuh
        include/util/indexing.cuh
        include/util/tensor.cuh)

# add dependencies
include(cmake/CPM.cmake)
## This will automatically clone CCCL from GitHub and make the exported cmake targets available
CPMAddPackage(
        NAME CCCL
        GITHUB_REPOSITORY nvidia/cccl
        GIT_TAG main # Fetches the latest commit on the main branch
)

CPMAddPackage(
        NAME NVTX
        GITHUB_REPOSITORY NVIDIA/NVTX
        GIT_TAG v3.1.0-c-cpp
        GIT_SHALLOW TRUE)

target_link_libraries(csrc PRIVATE CCCL::CCCL nvtx3-cpp)
#target_link_libraries(csrc PRIVATE nvtx3-cpp)

set_target_properties(csrc PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

## NVSHMEM Business
set(NVSHMEM_HOME "$ENV{NVSHMEM_HOME}")
find_package(NVSHMEM REQUIRED HINTS ${NVSHMEM_HOME}/lib/cmake/nvshmem)
add_library(nvshmem_host ALIAS nvshmem::nvshmem_host)
add_library(nvshmem_device ALIAS nvshmem::nvshmem_device)
target_link_libraries(csrc PUBLIC nvshmem_host nvshmem_device)
target_link_libraries(csrc PUBLIC CUDA::cudart CUDA::cuda_driver CUDA::cublas mathdx::cublasdx)

target_compile_options(csrc
        PRIVATE
        $<$<CONFIG:Debug>:-O0;-g;>
        $<$<AND:$<BOOL:${NVSHMEM_VERBOSE}>,$<COMPILE_LANGUAGE:CUDA>>:-Xptxas -v>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-O0;-g;-G>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${NVCC_THREADS}>>:-t4>
        "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>"
        # Required to support std::tuple in device code
        "$<$<BOOL:True>:$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--expt-relaxed-constexpr>>"
)
