# Copyright (c) 2025, Osayamen Jonathan Aimuyo
# All rights reserved.
#
# This file is part of the FlashMoE Project and is licensed under the BSD 3-Clause License.
# See the LICENSE file in the root directory for full terms.

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

project(flashmoe CUDA CXX)

# flags
set(FLASHMOE_CUDA_CXX_FLAGS "${FLASHMOE_CUDA_CXX_FLAGS} -Wall -Wextra")
set(FLASHMOE_CUDA_CXX_FLAGS "${FLASHMOE_CUDA_CXX_FLAGS} -fno-strict-aliasing")
if(NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "^aarch64")
    set(FLASHMOE_CUDA_CXX_FLAGS "${FLASHMOE_CUDA_CXX_FLAGS} -m64")
endif ()
set(FLASHMOE_CUDA_CXX_FLAGS "${FLASHMOE_CUDA_CXX_FLAGS} -Wno-unknown-pragmas -Wnull-dereference -Wnarrowing")
set(FLASHMOE_CUDA_CXX_FLAGS "${FLASHMOE_CUDA_CXX_FLAGS} -Wno-switch -Wduplicated-branches -Wformat=2")
set(FLASHMOE_CUDA_CXX_FLAGS "${FLASHMOE_CUDA_CXX_FLAGS} -Wno-unused-but-set-parameter")
set(FLASHMOE_CUDA_CXX_FLAGS "${FLASHMOE_CUDA_CXX_FLAGS} -Wno-sign-compare -v")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLASHMOE_CUDA_CXX_FLAGS}")

set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xfatbin -compress-all") # Compress all fatbins
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --display_error_number") # Show error/warning numbers
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler \"${FLASHMOE_CUDA_CXX_FLAGS}\"")

# Query number of SMs
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(SM)

include(CheckCompilerFlag)
check_compiler_flag(CUDA -t4 NVCC_THREADS)

find_package(CUDAToolkit REQUIRED)
string(SUBSTRING "${CMAKE_CUDA_ARCHITECTURES_NATIVE}" 0 2 COMPUTE_CAPABILITY) # xx-real -> xx
math(EXPR GPU_ARCH "${COMPUTE_CAPABILITY} * 10" OUTPUT_FORMAT DECIMAL)
message(STATUS "GPU 0 Compute Capability: ${COMPUTE_CAPABILITY}")
set(ENV{CUTLASS_NVCC_ARCHS} "${COMPUTE_CAPABILITY}")

set(ENV{CPM_SOURCE_CACHE} "./cmake/cache")
set(CCCL_ENABLE_UNSTABLE ON)
include(cmake/CPM.cmake)
# add dependencies
CPMAddPackage(
        NAME CCCL
        GITHUB_REPOSITORY nvidia/cccl
        VERSION 3.1.3
)
#CUTLASS
CPMAddPackage(
        NAME CUTLASS
        GITHUB_REPOSITORY nvidia/cutlass
        GIT_TAG main
        DOWNLOAD_ONLY TRUE
        OPTIONS
        "CUTLASS_NVCC_ARCHS=${COMPUTE_CAPABILITY}"
)

set(FMT_SYSTEM_HEADERS ON)
CPMAddPackage(
        NAME FMT
        GITHUB_REPOSITORY fmtlib/fmt
        GIT_TAG 11.1.4
        OPTIONS
        "FMT_SYSTEM_HEADERS=ON"
)

CPMAddPackage(
        NAME NVTX3
        GITHUB_REPOSITORY NVIDIA/NVTX
        GIT_TAG v3.1.1-c-cpp
        GIT_SHALLOW TRUE
)

# MatX
set(MATX_NVTX_FLAGS ON)
set(MATX_ENABLE_FILEIO ON)
CPMAddPackage(
        NAME MATX
        GITHUB_REPOSITORY NVIDIA/MatX
        GIT_TAG main
        DOWNLOAD_ONLY TRUE
        OPTIONS
)
## NVSHMEM
find_package(NVSHMEM REQUIRED HINTS "$ENV{NVSHMEM_HOME}/lib/cmake/nvshmem")
# cuBLASDx
find_package(mathdx REQUIRED COMPONENTS cublasdx CONFIG)


# Pre-compile constants
# Read JSON file content
file(READ "flashmoe_config.json" FLASHMOE_CONFIG)

string(JSON CAP_FACTOR_V GET ${FLASHMOE_CONFIG} "capacity_factor")
message(STATUS "capacity_factor: ${CAP_FACTOR_V}")

string(JSON DROP_TOKENS_V GET ${FLASHMOE_CONFIG} "drop_tokens")
message(STATUS "drop_tokens: ${DROP_TOKENS_V}")

string(JSON E_TOP_K_V GET ${FLASHMOE_CONFIG} "expert_top_k")
message(STATUS "expert_top_k: ${E_TOP_K_V}")

string(JSON GLOBAL_BATCH_V GET ${FLASHMOE_CONFIG} "global_batch")
message(STATUS "global_batch: ${GLOBAL_BATCH_V}")

string(JSON IS_TRAINING_V GET ${FLASHMOE_CONFIG} "is_training")
message(STATUS "is_training: ${IS_TRAINING_V}")

string(JSON HIDDEN_ACT_V GET ${FLASHMOE_CONFIG} "hidden_act")
message(STATUS "hidden_act: ${HIDDEN_ACT_V}")

string(JSON HIDDEN_SIZE_V GET ${FLASHMOE_CONFIG} "hidden_size")
message(STATUS "hidden_size: ${HIDDEN_SIZE_V}")

string(JSON I_SIZE_V GET ${FLASHMOE_CONFIG} "intermediate_size")
message(STATUS "intermediate_size: ${I_SIZE_V}")

string(JSON MINI_BATCH_V GET ${FLASHMOE_CONFIG} "mini_batch")
message(STATUS "mini_batch: ${MINI_BATCH_V}")

string(JSON MOE_FREQ_V GET ${FLASHMOE_CONFIG} "moe_frequency")
message(STATUS "moe_frequency: ${MOE_FREQ_V}")

string(JSON NUM_EXPERTS_V GET ${FLASHMOE_CONFIG} "num_experts")
message(STATUS "num_experts: ${NUM_EXPERTS_V}")

string(JSON NUM_LAYERS_V GET ${FLASHMOE_CONFIG} "num_layers")
message(STATUS "num_layers: ${NUM_LAYERS_V}")

string(JSON SEQ_LEN_V GET ${FLASHMOE_CONFIG} "sequence_len")
message(STATUS "sequence_len: ${SEQ_LEN_V}")

string(JSON DTYPE_V GET ${FLASHMOE_CONFIG} "torch_dtype")
message(STATUS "torch_dtype: ${DTYPE_V}")

string(JSON VOCAB_SIZE_V GET ${FLASHMOE_CONFIG} "vocab_size")
message(STATUS "vocab_size: ${VOCAB_SIZE_V}")

function(flash_target_common tgt)
    target_link_libraries(${tgt} PRIVATE CUDA::cudart CUDA::cuda_driver CUDA::nvtx3)
    target_link_libraries(${tgt} PRIVATE atomic)
    if(CCCL_ADDED)
        target_link_libraries(${tgt} PRIVATE CCCL::CCCL)
    endif()
    if(CUTLASS_ADDED)
        target_include_directories(${tgt} SYSTEM PRIVATE "${CUTLASS_SOURCE_DIR}/include")
    endif ()
    if(FMT_ADDED)
        target_link_libraries(${tgt} PRIVATE fmt::fmt)
    endif ()
    if(NVTX3_ADDED)
        target_link_libraries(${tgt} PRIVATE nvtx3-cpp)
        string(FIND "$ENV{CMAKE_PREFIX_PATH}" "nvtx3" INDEX)
        if(INDEX EQUAL -1)
            # append nvtx3 to prefix path to remove libtorch error
            set(ENV{CMAKE_PREFIX_PATH} "$ENV{CMAKE_PREFIX_PATH}:${NVTX3_SOURCE_DIR}")
        endif()
    endif ()
    if(MATX_ADDED)
        target_include_directories(${tgt} SYSTEM PRIVATE "${MATX_SOURCE_DIR}/include")
    endif ()
    if(DEFINED ENV{NERSC_HOST} AND "$ENV{NERSC_HOST}" STREQUAL "perlmutter")
        target_link_libraries(${tgt} PRIVATE nvshmem::nvshmem_host nvshmem::nvshmem_device)
    else ()
        target_link_libraries(${tgt} PRIVATE nvshmem::nvshmem)
    endif ()
    target_link_libraries(${tgt} PRIVATE mathdx::cublasdx)
    target_compile_options(${tgt} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
            $<$<COMPILE_LANGUAGE:CUDA>:-t0; --generate-line-info>
            $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode=arch=compute_${COMPUTE_CAPABILITY},code=sm_${COMPUTE_CAPABILITY}>
    )
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(${tgt} PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-O0;-g;>
                $<$<COMPILE_LANGUAGE:CUDA>:-O0; -g; -G>
        )
    elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(${tgt} PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-O3>
                $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode=arch=compute_${COMPUTE_CAPABILITY},code=lto_${COMPUTE_CAPABILITY}>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas -v;--expt-relaxed-constexpr; -diag-suppress 186>
        )
    endif ()
    if (${tgt} STREQUAL "fb")
        target_compile_definitions(${tgt}
                PRIVATE
                FLASHMOE_ARCH=${GPU_ARCH}
                CAP_FACTOR=${CAP_FACTOR_V}
                DROP_TOKENS=${DROP_TOKENS_V}
                E_TOP_K=${E_TOP_K_V}
                GLOBAL_BATCH=${GLOBAL_BATCH_V}
                IS_TRAINING=${IS_TRAINING_V}
                HIDDEN_ACT=${HIDDEN_ACT_V}
                HIDDEN_SIZE=${HIDDEN_SIZE_V}
                I_SIZE=${I_SIZE_V}
                MOE_FREQ=${MOE_FREQ_V}
                MINI_BATCH=${MINI_BATCH_V}
                NUM_EXPERTS=${NUM_EXPERTS_V}
                NUM_LAYERS=${NUM_LAYERS_V}
                SEQ_LEN=${SEQ_LEN_V}
                DTYPE=${DTYPE_V}
                VOCAB_SIZE=${VOCAB_SIZE_V}
                NUM_SMS=${CUDA_NUM_SMS}
                FLASHMOE_NVTX=1
        )
    endif ()
endfunction()

add_executable(fb benchmarks/flash_bench.cu
        include/flashmoe/flashmoe.cuh
)
set_target_properties(fb PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
flash_target_common(fb)

add_executable(gb
        benchmarks/gemm_bench.cu
        include/flashmoe/os/processor/gemm.cuh
        include/flashmoe/os/processor/mmaConfig.cuh
)
flash_target_common(gb)
