cmake_minimum_required(VERSION 3.27)

# Add COMMAND_ECHO STDOUT to see the commands in standard out
# WAR to use updated gcc on NERSC machines
if(DEFINED ENV{NERSC_HOST})
    execute_process(COMMAND "which" "g++"
            COMMAND_ERROR_IS_FATAL ANY
            OUTPUT_VARIABLE CPP_COMP_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND "which" "gcc"
            COMMAND_ERROR_IS_FATAL ANY
            OUTPUT_VARIABLE C_COMP_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(CMAKE_CXX_COMPILER ${CPP_COMP_PATH})
    set(CMAKE_C_COMPILER ${C_COMP_PATH})
endif ()

project(csrc CUDA CXX)

# flags
set(ARISTOS_CUDA_CXX_FLAGS "${ARISTOS_CUDA_CXX_FLAGS} -Wall -Wextra -Wsuggest-attribute=const")
set(ARISTOS_CUDA_CXX_FLAGS "${ARISTOS_CUDA_CXX_FLAGS} -fno-strict-aliasing")
if(NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "^aarch64")
    set(ARISTOS_CUDA_CXX_FLAGS "${ARISTOS_CUDA_CXX_FLAGS} -m64")
endif ()
set(ARISTOS_CUDA_CXX_FLAGS "${ARISTOS_CUDA_CXX_FLAGS} -Wno-unknown-pragmas -Wnull-dereference -Wnarrowing")
set(ARISTOS_CUDA_CXX_FLAGS "${ARISTOS_CUDA_CXX_FLAGS} -Wno-switch -Wfloat-equal -Wduplicated-branches -Wformat=2")
set(ARISTOS_CUDA_CXX_FLAGS "${ARISTOS_CUDA_CXX_FLAGS} -Wno-unused-but-set-parameter")
set(ARISTOS_CUDA_CXX_FLAGS "${ARISTOS_CUDA_CXX_FLAGS} -Wno-sign-compare -v")
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

add_executable(csrc main.cu
        include/engine/subscriber.cuh
        include/moe/moe.cuh
        include/aristos.cuh
        include/engine/processor/processor.cuh
        include/util/indexing.cuh
        include/definition/tensor.cuh
        include/definition/types.cuh
        include/definition/memory_layout.cuh
        include/util/atomics.cuh
        include/engine/decider/decider.cuh
        include/engine/decider/comps/edge.cuh
        include/engine/decider/comps/expert.cuh
        include/engine/decider/comps/group.cuh
        include/engine/decider/comps/worker.cuh
        include/engine/decider/comps/args.cuh
        include/engine/decider/comps/functions.cuh
        include/util/debug.cuh
        include/engine/decider/comps/niche.cuh
        include/topo/topo.cuh
        include/engine/processor/mmaConfig.cuh
        include/prep.cuh
        include/engine/scheduler.cuh
        include/moe/gate.cuh
        include/engine/processor/gemm.cuh
        include/moe/packet.cuh
        bench.cuh
        eval.cuh
)
set_target_properties(csrc PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS   ON
)

string(SUBSTRING "${CMAKE_CUDA_ARCHITECTURES_NATIVE}" 0 2 COMPUTE_CAPABILITY) # xx-real -> xx
message(STATUS "GPU 0 Compute Capability: ${COMPUTE_CAPABILITY}")
set(ENV{CUTLASS_NVCC_ARCHS} "${COMPUTE_CAPABILITY}")

set(ENV{CPM_SOURCE_CACHE} "./cmake/cache")
set(CCCL_ENABLE_UNSTABLE ON)
include(cmake/CPM.cmake)

# add dependencies
## This will automatically clone CCCL from GitHub and make the exported cmake targets available
CPMAddPackage(
        NAME CCCL
        GITHUB_REPOSITORY nvidia/cccl
        GIT_TAG main # Fetches the latest commit on the main branch
)
if(CCCL_ADDED)
    target_link_libraries(csrc PRIVATE CCCL::CCCL)
endif()

#CUTLASS business
CPMAddPackage(
        NAME CUTLASS
        GITHUB_REPOSITORY nvidia/cutlass
        GIT_TAG main
        DOWNLOAD_ONLY TRUE
        OPTIONS
        "CUTLASS_NVCC_ARCHS=${COMPUTE_CAPABILITY}"
)
if(CUTLASS_ADDED)
    # header-only
    target_include_directories(csrc SYSTEM PRIVATE "${CUTLASS_SOURCE_DIR}/include")
endif ()

CPMAddPackage(
        NAME FMT
        GITHUB_REPOSITORY fmtlib/fmt
        GIT_TAG 11.0.2
        DOWNLOAD_ONLY
)
if(FMT_ADDED)
    target_link_libraries(csrc PRIVATE fmt::fmt)
endif ()

set(NvidiaCutlass_ROOT "${CUTLASS_SOURCE_DIR}")
set(mathdx_CUTLASS_ROOT "${CUTLASS_SOURCE_DIR}")

set(MATHDX_VER 24.08)
set(MATHDX_URL_PREFIX "https://developer.download.nvidia.com/compute/cublasdx/redist/cublasdx")
set(MATHDX_URL "${MATHDX_URL_PREFIX}/nvidia-mathdx-${MATHDX_VER}.0.tar.gz")
CPMFindPackage(
        NAME mathdx
        VERSION "${MATHDX_VER}"
        URL "${MATHDX_URL}"
        FIND_PACKAGE_ARGUMENTS "REQUIRED COMPONENTS cublasdx CONFIG"
)
target_link_libraries(csrc PRIVATE mathdx::cublasdx)

target_link_libraries(csrc PRIVATE CUDA::cudart CUDA::cuda_driver CUDA::nvml CUDA::cublas CUDA::nvtx3)
target_link_libraries(csrc PRIVATE atomic)

## NVSHMEM Business
find_package(NVSHMEM REQUIRED HINTS "$ENV{NVSHMEM_HOME}/lib/cmake/nvshmem")
target_link_libraries(csrc PRIVATE nvshmem::nvshmem)

# Link Cray's GPU-accelerated MPICH
if(DEFINED ENV{LINK_GTL} AND DEFINED ENV{NERSC_HOST} AND "$ENV{NERSC_HOST}" STREQUAL "perlmutter")
    find_library(GTL
            NAMES libmpi_gtl_cuda.so.0
            HINTS /opt/cray/pe/lib64/
            REQUIRED)
    target_link_libraries(csrc PRIVATE "${GTL}")
endif ()

target_compile_options(csrc PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas -v;--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-t0; --generate-line-info>
        $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode=arch=compute_${COMPUTE_CAPABILITY},code=sm_${COMPUTE_CAPABILITY}>
)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(csrc PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Og;-g;>
            $<$<COMPILE_LANGUAGE:CUDA>:-O0; -g; -G>
    )
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(csrc PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-O3>
            $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode=arch=compute_${COMPUTE_CAPABILITY},code=lto_${COMPUTE_CAPABILITY}>
    )
endif ()
