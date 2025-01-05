//
// Created by osayamen on 9/10/24.
//

#ifndef CSRC_DEBUG_CUH
#define CSRC_DEBUG_CUH

#include <iostream>
#include <typeinfo>
#include <cxxabi.h>
namespace aristos{
    template<typename T>
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
    bool isRegister(const void* p) {
        return !(__isShared(p) &&
            __isLocal(p) &&
            __isConstant(p) &&
            __isGlobal(p) &&
            __isGridConstant(p));
    }
}
#endif //CSRC_DEBUG_CUH
