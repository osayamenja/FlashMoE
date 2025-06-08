/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Kleos Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 */

//
// Created by osayamen on 9/10/24.
//

#ifndef CSRC_DEBUG_CUH
#define CSRC_DEBUG_CUH

#include <iostream>
#include <typeinfo>
#include <cxxabi.h>
#if !defined(KLEOS_CHECK_CUDA)
#  define KLEOS_CHECK_CUDA(e)                                \
do {                                                         \
    cudaError_t code = (e);                                  \
    if (code != cudaSuccess) {                               \
        fprintf(stderr, "<%s:%d> %s:\n    %s: %s\n",         \
            __FILE__, __LINE__, #e,                          \
            cudaGetErrorName(code),                          \
            cudaGetErrorString(code));                       \
        fflush(stderr);                                      \
        exit(1);                                             \
    }                                                        \
} while (0)
#endif

#define KLEOS_ASSERT(predicate, errmsg)                    \
do {                                                         \
    if (!(predicate)) {                                      \
        std::cerr   << "Error: " errmsg                      \
                    << " [" << __PRETTY_FUNCTION__           \
                    << __FILE__ << ":" << __LINE__ << "]"    \
                    << std::endl;                            \
        std::quick_exit(EXIT_FAILURE);                       \
    }                                                        \
} while (0)


#if !defined(CHECK_LAST)
# define CHECK_LAST() KLEOS_CHECK_CUDA(cudaPeekAtLastError()); KLEOS_CHECK_CUDA(cudaDeviceSynchronize())
#endif

namespace kleos{
    template<typename T>
    __host__ __forceinline__
    void printType() {
        // Get the mangled name
        const char* mangledName = typeid(T).name();

        // Demangle the name
        int status;
        char* demangledName = abi::__cxa_demangle(mangledName, nullptr, nullptr, &status);

        // Print the demangled name
        if (status == 0) {
            std::cout << "Demangled name: " << demangledName << std::endl;
        } else {
            std::cerr << "Demangling failed!" << std::endl;
        }
        // Free the memory allocated by abi::__cxa_demangle
        free(demangledName);
    }

    __device__ __forceinline__
    bool isRegisterMemory(const void* p) {
        return !(__isShared(p) &&
            __isLocal(p) &&
            __isConstant(p) &&
            __isGlobal(p) &&
            __isGridConstant(p));
    }
}
#endif //CSRC_DEBUG_CUH
