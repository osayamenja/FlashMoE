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

    template<typename C, typename S, unsigned int k=0>
    concept StreamableContainer = Streamable<S>
                                  && (std::same_as<C, std::conditional_t<k == 0, std::vector<S>, std::array<S, k>>>);

    /// For some container c,
    /// Use like printContainer<decltype(c),decltype(c)::value_type>(c) for non-array
    /// otherwise, printContainer<decltype(c),decltype(c)::value_type, c.size()>(c)
    template<typename C, typename V, unsigned int k=0> requires StreamableContainer<C, V, k>
    void printContainer(const C& c){
        //TODO use printf
        std::stringstream outStream;
        std::cout << '[';
        for (auto& i : c)
            std::cout << ' ' << i << ",";
        std::cout << ']' << std::endl;
    }
}
#endif //CSRC_DEBUG_CUH
