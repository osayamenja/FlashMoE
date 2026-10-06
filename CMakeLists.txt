cmake_minimum_required(VERSION 3.28)

project(flashmoe LANGUAGES CXX CUDA)

# -------------------------------------------------------
# Make CPM available
# -------------------------------------------------------
if(NOT COMMAND CPMAddPackage)
  include(${CMAKE_CURRENT_LIST_DIR}/csrc/cmake/CPM.cmake)
endif()

# -------------------------------------------------------
# Header-only FlashMoE library
# -------------------------------------------------------
add_library(flashmoe INTERFACE)
add_library(flashmoe::flashmoe ALIAS flashmoe)

target_include_directories(flashmoe INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/csrc/include>
  $<INSTALL_INTERFACE:include>
)

target_compile_features(flashmoe INTERFACE cxx_std_20)

# -------------------------------------------------------
# CCCL
#
# Must come before find_package(NVSHMEM): NVSHMEMConfig.cmake calls find_dependency(CCCL) on the toolkit's copy,
# which creates the CCCL::CCCL target first and shadows CCCL 3.1.3, so <cuda/memory> etc. are not found
# with CUDA toolkits that bundle an older CCCL (e.g. 13.0).
# -------------------------------------------------------
CPMAddPackage(
  NAME CCCL
  GITHUB_REPOSITORY nvidia/cccl
  VERSION 3.1.3
)

# -------------------------------------------------------
# Required dependencies discoverable via find_package()
# -------------------------------------------------------

find_package(CUDAToolkit REQUIRED)
find_package(NVSHMEM REQUIRED)
find_package(mathdx REQUIRED COMPONENTS cublasdx CONFIG)

target_link_libraries(flashmoe INTERFACE
  CUDA::cudart
  CUDA::cuda_driver
  mathdx::cublasdx
)
target_link_libraries(flashmoe INTERFACE atomic)

# NVSHMEM imported target names differ across installs
if(TARGET nvshmem::nvshmem_host AND TARGET nvshmem::nvshmem_device)
    # WAR for build failure
  if (NOT DEFINED ENV{NVSHMEM_LIB_HOME})
    message(FATAL_ERROR "NVSHMEM_LIB_HOME not defined")
  endif ()
  set_target_properties(nvshmem::nvshmem_host PROPERTIES
    IMPORTED_LOCATION_RELEASE "$ENV{NVSHMEM_LIB_HOME}/libnvshmem_host.so"
  )
  target_link_libraries(flashmoe INTERFACE
    nvshmem::nvshmem_device
    nvshmem::nvshmem_host
  )
else()
  message(FATAL_ERROR "NVSHMEM found but no supported CMake targets detected")
endif()

# -------------------------------------------------------
# Architecture helpers
#
# Rules:
#   - minimum supported arch is 70
#   - any arch >= 90 must use the 'a' suffix
#   - any arch < 90 must not use the 'a' suffix
# -------------------------------------------------------
function(flashmoe_parse_arch arch source out_normalized)
  # Strip optional suffixes like -real / -virtual
  string(REGEX REPLACE "-(real|virtual)$" "" _arch "${arch}")

  string(REGEX MATCH "^([0-9]+)(a?)$" _matched "${_arch}")
  if(NOT _matched)
    message(FATAL_ERROR
      "flashmoe found unsupported CUDA architecture entry '${arch}' from ${source}. "
      "Expected values like 70, 80, 89, 90a, 100a."
    )
  endif()

  set(_num "${CMAKE_MATCH_1}")
  set(_suffix "${CMAKE_MATCH_2}")

  if(_num LESS 70)
    if(source MATCHES "NATIVE")
      message(FATAL_ERROR
        "flashmoe detected native CUDA architecture '${arch}', which is below the "
        "minimum supported architecture 70 (Volta). Please set "
        "CMAKE_CUDA_ARCHITECTURES explicitly to a supported value."
      )
    else()
      message(FATAL_ERROR
        "flashmoe requires CUDA architectures to be at least 70 (Volta), "
        "but found '${arch}' from ${source}."
      )
    endif()
  endif()

  if(_num GREATER_EQUAL 90)
    if(NOT _suffix STREQUAL "a")
      message(FATAL_ERROR
        "flashmoe requires the 'a' suffix for any architecture >= 90, "
        "but found '${arch}' from ${source}. Use '${_num}a' instead."
      )
    endif()
    set(${out_normalized} "${_num}a" PARENT_SCOPE)
  else()
    if(_suffix STREQUAL "a")
      message(FATAL_ERROR
        "flashmoe found architecture '${arch}', but the 'a' suffix is only valid "
        "for architectures >= 90."
      )
    endif()
    set(${out_normalized} "${_num}" PARENT_SCOPE)
  endif()
endfunction()

function(flashmoe_normalize_arch_list raw_arches source out_arches)
  set(_normalized_arches "")

  foreach(_arch IN LISTS raw_arches)
    flashmoe_parse_arch("${_arch}" "${source}" _norm)
    list(APPEND _normalized_arches "${_norm}")
  endforeach()

  list(REMOVE_DUPLICATES _normalized_arches)
  set(${out_arches} "${_normalized_arches}" PARENT_SCOPE)
endfunction()

function(flashmoe_resolve_arches out_arches out_cutlass_arches)
  set(_raw_arches "")
  set(_source "")

  if(DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    if(CMAKE_CUDA_ARCHITECTURES STREQUAL "native")
      if(DEFINED CMAKE_CUDA_ARCHITECTURES_NATIVE AND NOT CMAKE_CUDA_ARCHITECTURES_NATIVE STREQUAL "")
        set(_raw_arches "${CMAKE_CUDA_ARCHITECTURES_NATIVE}")
        set(_source "CMAKE_CUDA_ARCHITECTURES_NATIVE (via CMAKE_CUDA_ARCHITECTURES=native)")
      else()
        message(FATAL_ERROR
          "flashmoe: CMAKE_CUDA_ARCHITECTURES is set to 'native', but "
          "CMAKE_CUDA_ARCHITECTURES_NATIVE is empty or unavailable. "
          "Please set CMAKE_CUDA_ARCHITECTURES explicitly, for example 80 or 90a."
        )
      endif()
    else()
      set(_raw_arches "${CMAKE_CUDA_ARCHITECTURES}")
      set(_source "CMAKE_CUDA_ARCHITECTURES")
    endif()
  elseif(DEFINED CMAKE_CUDA_ARCHITECTURES_NATIVE AND NOT CMAKE_CUDA_ARCHITECTURES_NATIVE STREQUAL "")
    set(_raw_arches "${CMAKE_CUDA_ARCHITECTURES_NATIVE}")
    set(_source "CMAKE_CUDA_ARCHITECTURES_NATIVE")
  else()
    message(FATAL_ERROR
      "flashmoe could not determine CUDA architectures. "
      "Set CMAKE_CUDA_ARCHITECTURES explicitly (for example 80, 89, 90a, or 80;90a)."
    )
  endif()

  flashmoe_normalize_arch_list("${_raw_arches}" "${_source}" _normalized_arches)

  set(${out_arches} "${_normalized_arches}" PARENT_SCOPE)
  set(${out_cutlass_arches} "${_normalized_arches}" PARENT_SCOPE)
endfunction()

flashmoe_resolve_arches(FLASHMOE_EFFECTIVE_CUDA_ARCHS FLASHMOE_CUTLASS_ARCHS)
message(STATUS "FlashMoE effective CUDA architectures: ${FLASHMOE_EFFECTIVE_CUDA_ARCHS}")

# -------------------------------------------------------
# CPM dependencies
# -------------------------------------------------------

# CCCL (fetched at the top of this file, before find_package(NVSHMEM/mathdx))
if(TARGET CCCL::CCCL)
  target_link_libraries(flashmoe INTERFACE CCCL::CCCL)
else()
  message(FATAL_ERROR "Unable to link CCCL")
endif()

# CUTLASS (header-only)
CPMAddPackage(
  NAME CUTLASS
  GITHUB_REPOSITORY nvidia/cutlass
  GIT_TAG v4.4.1
  DOWNLOAD_ONLY TRUE
)
if(DEFINED CUTLASS_SOURCE_DIR AND EXISTS "${CUTLASS_SOURCE_DIR}/include")
  target_include_directories(flashmoe INTERFACE
    ${CUTLASS_SOURCE_DIR}/include
  )
else()
  message(FATAL_ERROR "Unable to include CUTLASS")
endif()

CPMAddPackage(
        NAME NVTX3
        GITHUB_REPOSITORY NVIDIA/NVTX
        GIT_TAG v3.1.1-c-cpp
        GIT_SHALLOW TRUE
)
if(TARGET nvtx3-cpp)
     target_link_libraries(flashmoe INTERFACE nvtx3-cpp)
endif ()
# -------------------------------------------------------
# NVSHMEM requires relocatable device code
# -------------------------------------------------------
target_compile_options(flashmoe INTERFACE
  $<$<COMPILE_LANGUAGE:CUDA>:--relocatable-device-code=true>
)

# Helper for consumers
function(FlashMoESetRDC tgt)
  if(NOT TARGET ${tgt})
    message(FATAL_ERROR "target '${tgt}' does not exist")
  endif()
  set_target_properties(${tgt} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
endfunction()
function(FlashMoEAddOptions tgt)
  if(NOT TARGET ${tgt})
    message(FATAL_ERROR "flashmoe_add_options: target '${tgt}' does not exist")
  endif()

  get_target_property(_tgt_archs ${tgt} CUDA_ARCHITECTURES)
  if(NOT _tgt_archs OR _tgt_archs STREQUAL "NOTFOUND")
    set(_tgt_archs "${FLASHMOE_EFFECTIVE_CUDA_ARCHS}")
    set(_arch_source "target '${tgt}' inherited FlashMoE resolved architectures")
  elseif(_tgt_archs STREQUAL "native")
    if(DEFINED CMAKE_CUDA_ARCHITECTURES_NATIVE AND NOT CMAKE_CUDA_ARCHITECTURES_NATIVE STREQUAL "")
      set(_tgt_archs "${CMAKE_CUDA_ARCHITECTURES_NATIVE}")
      set(_arch_source "CMAKE_CUDA_ARCHITECTURES_NATIVE (via target '${tgt}' CUDA_ARCHITECTURES=native)")
    else()
      message(FATAL_ERROR
        "flashmoe_add_options: target '${tgt}' uses CUDA_ARCHITECTURES=native, "
        "but CMAKE_CUDA_ARCHITECTURES_NATIVE is empty or unavailable."
      )
    endif()
  else()
    set(_arch_source "target '${tgt}' CUDA_ARCHITECTURES")
  endif()

  flashmoe_normalize_arch_list("${_tgt_archs}" "${_arch_source}" _normalized_tgt_archs)

  target_compile_options(${tgt} INTERFACE
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

    # CUDA options
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>

    # Debug
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-O0;-g>
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-O0;-g;-G>

    # Release
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3>
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3>
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-Xptxas=-v>
  )
endfunction()