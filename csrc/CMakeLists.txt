cmake_minimum_required(VERSION 3.27)

# Set compilers and obtain compute capability, assuming the running process sees homogenous GPUs
# Add COMMAND_ECHO STDOUT to see the commands in standard out
execute_process(COMMAND "which" "g++"
        COMMAND_ERROR_IS_FATAL ANY
        OUTPUT_VARIABLE CPP_COMP_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND "which" "gcc"
        COMMAND_ERROR_IS_FATAL ANY
        OUTPUT_VARIABLE C_COMP_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND "nvidia-smi" "--id=0" "--query-gpu=compute_cap" "--format=csv,noheader"
        COMMAND_ERROR_IS_FATAL ANY
        OUTPUT_VARIABLE COMPUTE_CAPABILITY
        OUTPUT_STRIP_TRAILING_WHITESPACE)

string(REGEX REPLACE "(\.[0-9])" "0" COMPUTE_CAPABILITY ${COMPUTE_CAPABILITY})
message(STATUS "GPU 0 Compute Capability: ${COMPUTE_CAPABILITY}")

#set(ENV{CPM_SOURCE_CACHE} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/deps")
set(ENV{CUTLASS_NVCC_ARCHS} "${COMPUTE_CAPABILITY}")
set(CMAKE_CXX_COMPILER "${CPP_COMP_PATH}" CACHE PATH "g++ Compiler Path")
set(CMAKE_C_COMPILER "${C_COMP_PATH}" CACHE PATH "gcc Compiler Path")

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
# Silence nvtx deprecation warning from legacy version used in libtorch

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
        include/moe/engine/subscriber.cuh
        include/moe/moe.cuh
        include/aristos.cuh
        include/moe/engine/processor.cuh
        include/moe/algorithm/algorithm.cuh
        include/moe/util/indexing.cuh
        include/moe/definition/tensor.cuh
        include/moe/engine/publisher.cuh
        include/moe/definition/types.cuh
        include/moe/definition/memory_layout.cuh
        include/moe/util/atomics.cuh
        include/moe/definition/values.cuh
        include/moe/engine/decider/decider.cuh
        include/moe/engine/decider/comps/edge.cuh
        include/moe/engine/decider/comps/expert.cuh
        include/moe/engine/decider/comps/group.cuh
        include/moe/engine/decider/comps/worker.cuh
        include/moe/engine/decider/comps/args.cuh
        include/moe/engine/decider/comps/functions.cuh
        include/moe/util/debug.cuh
        include/moe/engine/decider/comps/niche.cuh
)

# add dependencies
#TODO Pin versions for release build and use main for debug
include(cmake/CPM.cmake)
## This will automatically clone CCCL from GitHub and make the exported cmake targets available
CPMAddPackage(
        NAME CCCL
        GITHUB_REPOSITORY nvidia/cccl
        GIT_TAG main # Fetches the latest commit on the main branch
)

#Libtorch business
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

target_link_libraries(csrc PRIVATE CCCL::CCCL)

set_target_properties(csrc PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS   ON
)

## NVSHMEM Business
set(NVSHMEM_HOME "$ENV{NVSHMEM_HOME}")
find_package(NVSHMEM REQUIRED HINTS ${NVSHMEM_HOME}/lib/cmake/nvshmem)
add_library(nvshmem_host ALIAS nvshmem::nvshmem_host)
add_library(nvshmem_device ALIAS nvshmem::nvshmem_device)
target_link_libraries(csrc PUBLIC nvshmem_host nvshmem_device)
target_link_libraries(csrc PUBLIC CUDA::cudart CUDA::cuda_driver CUDA::cublas CUDA::nvtx3 mathdx::cublasdx
        "${TORCH_LIBRARIES}")
target_link_libraries(csrc PRIVATE atomic)

target_compile_options(csrc
        PRIVATE
        $<$<CONFIG:Debug>:-O0;-g;>
        $<$<AND:$<BOOL:${NVSHMEM_VERBOSE}>,$<COMPILE_LANGUAGE:CUDA>>:-Xptxas -v>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-O0;-g;-G>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${NVCC_THREADS}>>:-t4>
        $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas -v>
        #[[$<$<COMPILE_LANGUAGE:CUDA>:-t0>
        # Release config
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas -O3>]]
        # Required to support std::tuple in device code
        $<$<BOOL:True>:$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--expt-relaxed-constexpr>>
)
