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
add_library(flashmoe_warnings INTERFACE)

target_compile_options(flashmoe_warnings INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:
        -Wall -Wextra
        -fno-strict-aliasing
        -Wno-unknown-pragmas
        -Wnull-dereference
        -Wnarrowing
        -Wno-switch
        -Wduplicated-branches
        -Wformat=2
        -Wno-unused-but-set-parameter
        -Wno-sign-compare
        >

        $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcompiler=-Wall,-Wextra
        -Xcompiler=-fno-strict-aliasing
        -Xcompiler=-Wno-unknown-pragmas,-Wnull-dereference,-Wnarrowing
        -Xcompiler=-Wno-switch,-Wduplicated-branches,-Wformat=2
        -Xcompiler=-Wno-unused-but-set-parameter
        -Xcudafe --display_error_number
        -Xcompiler=-Wno-sign-compare
        >
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# Query number of SMs
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(SM)

include(CheckCompilerFlag)
check_compiler_flag(CUDA -t4 NVCC_THREADS)

find_package(CUDAToolkit REQUIRED)
string(REGEX MATCH "^[0-9]+" ARCH "${CMAKE_CUDA_ARCHITECTURES_NATIVE}") # xx-real -> xx
math(EXPR GPU_ARCH "${ARCH} * 10" OUTPUT_FORMAT DECIMAL)
message(STATUS "GPU 0 Compute Capability: ${ARCH}")

set(ARCH_TAG "${ARCH}")
if(ARCH GREATER_EQUAL 90)
    string(APPEND ARCH_TAG "a") # accelerated arch-specific instruction set
endif()
set(ENV{CUTLASS_NVCC_ARCHS} "${ARCH_TAG}")

set(ENV{CPM_SOURCE_CACHE} "./cmake/cache")
set(CCCL_ENABLE_UNSTABLE ON)
include(cmake/CPM.cmake)
# CCCL
CPMAddPackage(
        NAME CCCL
        GITHUB_REPOSITORY nvidia/cccl
        VERSION 3.1.3
)
#CUTLASS
CPMAddPackage(
        NAME CUTLASS
        GITHUB_REPOSITORY nvidia/cutlass
        GIT_TAG v4.4.1
        DOWNLOAD_ONLY TRUE
        OPTIONS
        "CUTLASS_NVCC_ARCHS=${ARCH_TAG}"
)

CPMAddPackage(
        NAME NVTX3
        GITHUB_REPOSITORY NVIDIA/NVTX
        GIT_TAG v3.1.1-c-cpp
        GIT_SHALLOW TRUE
)

# MatX
set(MATX_NVTX_FLAGS ON)
CPMAddPackage(
        NAME MATX
        GITHUB_REPOSITORY NVIDIA/MatX
        GIT_TAG v0.9.4
)
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    message(FATAL_ERROR "Set -DCMAKE_BUILD_TYPE=Release (or Debug).")
endif()

## NVSHMEM
find_package(NVSHMEM REQUIRED)
# cuBLASDx
find_package(mathdx REQUIRED COMPONENTS cublasdx CONFIG)
find_package(mathdx REQUIRED COMPONENTS curanddx CONFIG)
include(CheckCompilerFlag)
function(flash_target_test tgt)
    target_link_libraries(${tgt} PRIVATE CUDA::cudart CUDA::cuda_driver CUDA::nvtx3)
    target_link_libraries(${tgt} PRIVATE atomic)
    if(CCCL_ADDED)
        target_link_libraries(${tgt} PRIVATE CCCL::CCCL)
    endif()
    if(CUTLASS_ADDED)
        target_include_directories(${tgt} SYSTEM PRIVATE "${CUTLASS_SOURCE_DIR}/include")
    endif ()
    target_compile_options(${tgt} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
            $<$<COMPILE_LANGUAGE:CUDA>:-t0> #use sm_xxa for xx >= 90
            $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode=arch=compute_${ARCH_TAG},code=sm_${ARCH_TAG}>
    )
    target_compile_options(${tgt} PRIVATE
            $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-O0;-g>
            $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-O0;-g;-G;--expt-relaxed-constexpr;-Xptxas=-v>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-Xptxas -v>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:--expt-relaxed-constexpr;-Xptxas=-v>
    )
endfunction()
function(flash_target_common tgt)
    target_link_libraries(${tgt} PRIVATE CUDA::cudart CUDA::cuda_driver atomic)
    if(NVTX3_ADDED)
        target_link_libraries(${tgt} PRIVATE nvtx3-cpp)
    endif ()
    if(CCCL_ADDED)
        target_link_libraries(${tgt} PRIVATE CCCL::CCCL)
    endif()
    if(CUTLASS_ADDED)
        target_include_directories(${tgt} SYSTEM PRIVATE "${CUTLASS_SOURCE_DIR}/include")
    endif()
    if(MATX_ADDED)
        target_link_libraries(${tgt} PRIVATE matx::matx)
    endif()
    target_link_libraries(${tgt} PRIVATE mathdx::cublasdx mathdx::curanddx)
    target_compile_options(${tgt} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
            $<$<COMPILE_LANGUAGE:CUDA>:-t0>
    )
    # Debug/Release opts
    target_compile_options(${tgt} PRIVATE
            $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-O0;-g>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-Xptxas -v>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:SHELL:-gencode=arch=compute_${ARCH_TAG},code=sm_${ARCH_TAG}>
    )
    target_compile_definitions(${tgt} PRIVATE
            FLASHMOE_ARCH=${GPU_ARCH}
            NUM_SMS=${CUDA_NUM_SMS}
            FLASHMOE_NVTX=1
    )
endfunction()
function(flash_target_lto tgt)
    target_link_libraries(${tgt} PRIVATE CUDA::cudart CUDA::cuda_driver CUDA::nvml atomic)
    target_link_libraries(${tgt} PRIVATE flashmoe_warnings)
    if(NVTX3_ADDED)
        target_link_libraries(${tgt} PRIVATE nvtx3-cpp)
    endif ()
    if(CCCL_ADDED)
        target_link_libraries(${tgt} PRIVATE CCCL::CCCL)
    endif()
    if(CUTLASS_ADDED)
        target_include_directories(${tgt} SYSTEM PRIVATE "${CUTLASS_SOURCE_DIR}/include")
    endif()
    if(MATX_ADDED)
        target_link_libraries(${tgt} PRIVATE matx::matx)
    endif()
    target_link_libraries(${tgt} PRIVATE mathdx::cublasdx mathdx::curanddx)
    target_compile_options(${tgt} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr> #--Werror=all-warnings
            $<$<COMPILE_LANGUAGE:CUDA>:-t0; -lineinfo>
            $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode=arch=compute_${ARCH_TAG},code=sm_${ARCH_TAG}>
    )
    # Debug/Release opts
    target_compile_options(${tgt} PRIVATE
            $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-O0;-g>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-Xptxas -v; -Xcudafe --display_error_number>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:SHELL:-gencode=arch=compute_${ARCH_TAG},code=lto_${ARCH_TAG}>
    )
    target_link_options(${tgt} PRIVATE
            $<$<CONFIG:Release>:$<DEVICE_LINK:-Xptxas -v>>
    )
    target_compile_definitions(${tgt} PRIVATE
            FLASHMOE_ARCH=${GPU_ARCH}
            NUM_SMS=${CUDA_NUM_SMS}
            FLASHMOE_NVTX=1
    )
endfunction()
add_executable(testFlashMoE
        tests/flashmoe.cu
)
set_target_properties(testFlashMoE PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
flash_target_lto(testFlashMoE)
# WAR for build failure
set_target_properties(nvshmem::nvshmem_host PROPERTIES
        IMPORTED_LOCATION_RELEASE "$ENV{NVSHMEM_LIB_HOME}/libnvshmem_host.so.3"
)
target_link_libraries(testFlashMoE PRIVATE nvshmem::nvshmem_host nvshmem::nvshmem_device)
find_package(MPI REQUIRED COMPONENTS CXX)
target_link_libraries(testFlashMoE PRIVATE MPI::MPI_CXX)

add_executable(testScheduler
        tests/scheduler.cu
)
flash_target_test(testScheduler)

add_executable(testGEMM
        tests/gemm.cu
)
flash_target_common(testGEMM)

set(PYBIND11_FINDPYTHON ON)
CPMAddPackage(
        NAME pybind11
        GITHUB_REPOSITORY pybind/pybind11
        VERSION 3.0.2
        DOWNLOAD_ONLY TRUE
)
add_executable(playground
        tests/playground.cu
)
flash_target_common(playground)
find_package(Python3 COMPONENTS Development)
target_include_directories(playground PRIVATE "${pybind11_SOURCE_DIR}/include" ${Python3_INCLUDE_DIRS})
target_link_libraries(playground PRIVATE nvshmem::nvshmem_host nvshmem::nvshmem_device)

add_executable(testGate
        tests/gate.cu
)
flash_target_common(testGate)

add_executable(testCombine
        tests/combine.cu
)
flash_target_common(testCombine)

add_executable(gemmMNK
        tests/gemmMNK.cu
)
flash_target_common(gemmMNK)

add_executable(testGatedGEMM
        tests/gatedGEMM.cu
)
flash_target_common(testGatedGEMM)
