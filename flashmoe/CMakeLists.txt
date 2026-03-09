cmake_minimum_required(VERSION 3.27)
project(flashmoe_v010 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

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

find_package(CUDAToolkit REQUIRED)

set(FLASHMOE_KERNELS_SOURCE "" CACHE PATH "Path to flashmoe kernels source tree (optional)")

if (FLASHMOE_KERNELS_SOURCE)
  set(FLASHMOE_SOURCE_DIR ${FLASHMOE_KERNELS_SOURCE})
endif()

if (NOT DEFINED GENERATED_SRC)
  message(FATAL_ERROR "Need -DGENERATED_SRC")
endif()

if (NOT DEFINED TARGET_MODULE_NAME)
  message(FATAL_ERROR "Need -DTARGET_MODULE_NAME")
endif()

if (NOT DEFINED ARCH)
  message(FATAL_ERROR "Need -DARCH")
endif()

if(${ARCH} LESS 700)
  message(FATAL_ERROR "Unsupported ARCH")
endif()

set(ARCH_TAG "${ARCH}")
if(ARCH GREATER_EQUAL 90)
    string(APPEND ARCH_TAG "a") # accelerated arch-specific instruction set
endif()

# ---- CPM ----
include(${FLASHMOE_SOURCE_DIR}/cmake/CPM.cmake)

#pybind11
set(PYBIND11_FINDPYTHON ON)
CPMAddPackage(
  NAME pybind11
  GITHUB_REPOSITORY pybind/pybind11
  VERSION 3.0.1
)

#CCCL
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

#NVSHMEM
find_package(NVSHMEM REQUIRED)

# cuBLASDx
find_package(mathdx REQUIRED COMPONENTS cublasdx CONFIG)

if(NOT pybind11_ADDED)
    message(FATAL_ERROR "pybind11 not found")
endif()
pybind11_add_module(${TARGET_MODULE_NAME} MODULE ${GENERATED_SRC})

target_include_directories(${TARGET_MODULE_NAME} PRIVATE
  ${FLASHMOE_KERNELS_SOURCE}/include
)

set_target_properties(${TARGET_MODULE_NAME} PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_options(${TARGET_MODULE_NAME} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-03>
            $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr; -Xptxas -v; -t0>
            $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode=arch=compute_${ARCH_TAG},code=sm_${ARCH_TAG}>
            $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode=arch=compute_${ARCH_TAG},code=lto_${ARCH_TAG}>
)

target_link_libraries(${TARGET_MODULE_NAME} PRIVATE CUDA::cudart flashmoe_warnings)
target_compile_definitions(${TARGET_MODULE_NAME} PRIVATE
            FLASHMOE_ARCH=${ARCH}
)
set_target_properties(nvshmem::nvshmem_host PROPERTIES
        IMPORTED_LOCATION_RELEASE "$ENV{NVSHMEM_LIB_HOME}/libnvshmem_host.so.3"
)
target_link_libraries(${TARGET_MODULE_NAME} PRIVATE nvshmem::nvshmem_host nvshmem::nvshmem_device)
target_link_libraries(${TARGET_MODULE_NAME} PRIVATE mathdx::cublasdx)
if(CCCL_ADDED)
        target_link_libraries(${TARGET_MODULE_NAME} PRIVATE CCCL::CCCL)
endif()
if(CUTLASS_ADDED)
    target_include_directories(${TARGET_MODULE_NAME} SYSTEM PRIVATE "${CUTLASS_SOURCE_DIR}/include")
endif()
