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
        include/os/subscriber.cuh
        include/moe/moe.cuh
        include/aristos.cuh
        include/os/processor/processor.cuh
        include/indexing.cuh
        include/types.cuh
        include/atomics.cuh
        include/os/decider/decider.cuh
        include/os/decider/comps/edge.cuh
        include/os/decider/comps/expert.cuh
        include/os/decider/comps/group.cuh
        include/os/decider/comps/worker.cuh
        include/os/decider/comps/args.cuh
        include/os/decider/comps/functions.cuh
        include/debug.cuh
        include/os/decider/comps/niche.cuh
        include/topo.cuh
        include/os/processor/mmaConfig.cuh
        include/bootstrap.cuh
        include/os/scheduler.cuh
        include/moe/gate.cuh
        include/os/processor/gemm.cuh
        include/os/packet.cuh
        eval.cuh
        include/arch.cuh
        include/os/os.cuh
        include/moe/expert.cuh
        include/throughput.cuh
        correctness.cuh
        include/os/sync.cuh
        include/moe/fffn.cuh
)
set_target_properties(csrc PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS   ON
)

string(SUBSTRING "${CMAKE_CUDA_ARCHITECTURES_NATIVE}" 0 2 COMPUTE_CAPABILITY) # xx-real -> xx
math(EXPR GPU_ARCH "${COMPUTE_CAPABILITY} * 10U" OUTPUT_FORMAT DECIMAL)
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
    set(cublasdx_CUTLASS_ROOT "${CUTLASS_SOURCE_DIR}")
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

set(MATHDX_VER 25.01)
CPMFindPackage(
        NAME mathdx
        VERSION "${MATHDX_VER}"
        FIND_PACKAGE_ARGUMENTS "REQUIRED COMPONENTS cublasdx CONFIG"
)
target_link_libraries(csrc PRIVATE mathdx::cublasdx)

CPMAddPackage(
        NAME NVTX3
        GITHUB_REPOSITORY NVIDIA/NVTX
        GIT_TAG v3.1.0-c-cpp
        GIT_SHALLOW TRUE
)

if(NVTX3_ADDED)
    target_link_libraries(csrc PRIVATE nvtx3-cpp)
    string(FIND "$ENV{CMAKE_PREFIX_PATH}" "nvtx3" INDEX)
    if(INDEX EQUAL -1)
        # append nvtx3 to prefix path to remove libtorch error
        set(ENV{CMAKE_PREFIX_PATH} "$ENV{CMAKE_PREFIX_PATH}:${NVTX3_SOURCE_DIR}")
    endif()
endif ()

# libTorch
set(USE_SYSTEM_NVTX ON)
set(CAFFE2_USE_CUDNN ON)
find_package(Torch REQUIRED HINTS "$ENV{TORCH_ROOT}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_link_libraries(csrc PRIVATE "${TORCH_LIBRARIES}")

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
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-t0; --generate-line-info>
        $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode=arch=compute_${COMPUTE_CAPABILITY},code=sm_${COMPUTE_CAPABILITY}>
)

# Pre-compile constants
# Read JSON file content
file(READ "aristos_config.json" ARISTOS_CONFIG)

string(JSON CAP_FACTOR_V GET ${ARISTOS_CONFIG} "capacity_factor")
message(STATUS "capacity_factor: ${CAP_FACTOR_V}")

string(JSON DROP_TOKENS_V GET ${ARISTOS_CONFIG} "drop_tokens")
message(STATUS "drop_tokens: ${DROP_TOKENS_V}")

string(JSON E_TOP_K_V GET ${ARISTOS_CONFIG} "expert_top_k")
message(STATUS "expert_top_k: ${E_TOP_K_V}")

string(JSON GLOBAL_BATCH_V GET ${ARISTOS_CONFIG} "global_batch")
message(STATUS "global_batch: ${GLOBAL_BATCH_V}")

string(JSON IS_TRAINING_V GET ${ARISTOS_CONFIG} "is_training")
message(STATUS "is_training: ${IS_TRAINING_V}")

string(JSON HIDDEN_ACT_V GET ${ARISTOS_CONFIG} "hidden_act")
message(STATUS "hidden_act: ${HIDDEN_ACT_V}")

string(JSON HIDDEN_SIZE_V GET ${ARISTOS_CONFIG} "hidden_size")
message(STATUS "hidden_size: ${HIDDEN_SIZE_V}")

string(JSON I_SIZE_V GET ${ARISTOS_CONFIG} "intermediate_size")
message(STATUS "intermediate_size: ${I_SIZE_V}")

string(JSON MINI_BATCH_V GET ${ARISTOS_CONFIG} "mini_batch")
message(STATUS "mini_batch: ${MU_BATCH_V}")

string(JSON MOE_FREQ_V GET ${ARISTOS_CONFIG} "moe_frequency")
message(STATUS "moe_frequency: ${MOE_FREQ_V}")

string(JSON NUM_EXPERTS_V GET ${ARISTOS_CONFIG} "num_experts")
message(STATUS "num_experts: ${NUM_EXPERTS_V}")

string(JSON NUM_LAYERS_V GET ${ARISTOS_CONFIG} "num_layers")
message(STATUS "num_layers: ${NUM_LAYERS_V}")

string(JSON SEQ_LEN_V GET ${ARISTOS_CONFIG} "sequence_len")
message(STATUS "sequence_len: ${SEQ_LEN_V}")

string(JSON DTYPE_V GET ${ARISTOS_CONFIG} "torch_dtype")
message(STATUS "torch_dtype: ${DTYPE_V}")

string(JSON VOCAB_SIZE_V GET ${ARISTOS_CONFIG} "vocab_size")
message(STATUS "vocab_size: ${VOCAB_SIZE_V}")

target_compile_definitions(csrc
        PRIVATE
        ARISTOS_ARCH=${GPU_ARCH}
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
)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(csrc PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-O0;-g;>
            $<$<COMPILE_LANGUAGE:CUDA>:-O0; -g; -G>
    )
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(csrc PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-O3>
            $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode=arch=compute_${COMPUTE_CAPABILITY},code=lto_${COMPUTE_CAPABILITY}>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas -v;--expt-relaxed-constexpr>
    )
endif ()
