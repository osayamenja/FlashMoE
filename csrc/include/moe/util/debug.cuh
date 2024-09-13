//
// Created by osayamen on 9/10/24.
//

#ifndef CSRC_DEBUG_CUH
#define CSRC_DEBUG_CUH

#include <iostream>
namespace aristos{
    template<typename S>
    concept Streamable = requires (S s){
        {std::cout << s} -> std::same_as<std::ostream&>;
    };

    template<typename C, unsigned int k=0>
    concept StreamableContainer = requires(C c){
        Streamable<typename decltype(c)::value_type>
                && (std::same_as<C, std::vector<typename decltype(c)::value_type>>
                || ( k > 0 && std::same_as<C, std::array<typename decltype(c)::value_type, k>>));
    };

    /// For some container c,
    /// Use like printContainer<c.size()>(c) for non-array; otherwise, printContainer(c)
    template<unsigned int k=0, typename C> requires StreamableContainer<C, k>
    void printContainer(const C& c) {
        std::stringstream outStream;
        std::cout << '[';
        for (auto const &i: c)
            std::cout << ' ' << i << ",";
        std::cout << ']';
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
}
#endif //CSRC_DEBUG_CUH
