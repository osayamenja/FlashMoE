/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Flashmoe Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 */

//
// Created by osayamen on 9/10/24.
//

#ifndef CSRC_DEBUG_CUH
#define CSRC_DEBUG_CUH

#include <cstdio>
#if !defined(CHECK_CUDA)
#  define CHECK_CUDA(e)                                      \
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

#define FLASHMOE_ASSERT(predicate, errmsg)                    \
do {                                                         \
    if (!(predicate)) {                                      \
        std::cerr   << "Error: " errmsg                      \
                    << " [" << __PRETTY_FUNCTION__           \
                    << __FILE__ << ":" << __LINE__ << "]"    \
                    << std::endl;                            \
        std::quick_exit(EXIT_FAILURE);                       \
    }                                                        \
} while (0)
#endif //CSRC_DEBUG_CUH
