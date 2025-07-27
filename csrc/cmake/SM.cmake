# Copyright (c) 2025, Osayamen Jonathan Aimuyo
# SM.cmake - Detects the number of Streaming Multiprocessors on GPU 0

if(NOT DEFINED CUDA_NUM_SMS)
    try_run(SM_RUN_RESULT SM_COMPILE_RESULT
            "${CMAKE_BINARY_DIR}/sm_detect_run"
            "${CMAKE_SOURCE_DIR}/cmake/sm.cu"
            CMAKE_FLAGS -DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
            RUN_OUTPUT_VARIABLE NUM_SMS
            COMPILE_OUTPUT_VARIABLE COMPILE_LOG
    )

    if(NOT SM_COMPILE_RESULT)
        message(FATAL_ERROR "Failed to compile sm.cu:\n${COMPILE_LOG}")
    endif()

    if(NOT SM_RUN_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to run SM detector executable.")
    endif()

    set(CUDA_NUM_SMS "${NUM_SMS}" CACHE INTERNAL "Detected SM count")
    message(STATUS "GPU 0 Number of SMs: ${CUDA_NUM_SMS}")
endif()
