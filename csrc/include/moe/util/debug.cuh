//
// Created by osayamen on 9/10/24.
//

#ifndef CSRC_DEBUG_CUH
#define CSRC_DEBUG_CUH

#include <iostream>
#include <typeinfo>
#include <cxxabi.h>
namespace aristos{
    template<typename S>
    concept Streamable = requires (S s){
        {std::cout << s} -> std::same_as<std::ostream&>;
    };

    template<typename C, unsigned int k=0>
    concept StreamableContainer = requires(C c){
        requires Streamable<typename decltype(c)::value_type>
                && (std::same_as<C, std::vector<typename decltype(c)::value_type>>
                || ( k > 0 && std::same_as<C, std::array<typename decltype(c)::value_type, k>>));
    };

    /// For some container c,
    /// Use like printContainer<c.size()>(c) for array; otherwise, printContainer(c)
    template<unsigned int k=0, typename C> requires StreamableContainer<C, k>
    void printContainer(const C& c) {
        std::stringstream outStream;
        std::cout << '[';
        for (auto const &i: c)
            std::cout << ' ' << i << ",";
        std::cout << ']' << std::endl;
    }

    template<typename K, typename V, unsigned int k=0>
    requires Streamable<K> && StreamableContainer<V,k>
    void printMapCV(const std::unordered_map<K, V>& m) {
        std::stringstream outStream;
        std::cout << '[';
        for (const auto & i: m){
            std::cout << i.first << ": ";
            printContainer(i.second);
            std::cout << ", ";
        }
        std::cout << ']' << std::endl;
    }

    template<typename K, typename V>
    requires Streamable<K> && Streamable<V>
    void printMap(const std::unordered_map<K, V>& m) {
        std::stringstream outStream;
        std::cout << '[';
        for (const auto & i: m){
            std::cout << "{" << i.first << ": " << i.second << "}, ";
        }
        std::cout << ']' << std::endl;
    }

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
}
#endif //CSRC_DEBUG_CUH
