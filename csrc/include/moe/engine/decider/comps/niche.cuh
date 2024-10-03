//
// Created by osayamen on 9/15/24.
//

#ifndef CSRC_NICHE_CUH
#define CSRC_NICHE_CUH

namespace aristos{
    __forceinline__
    std::vector<unsigned int> subsets(const std::vector<size_t>& parents, const unsigned int& myRank){
        std::vector<unsigned int> platoon{};
        for(unsigned int i = 0; i < parents.size(); ++i){
            if (parents[i] == parents[myRank]) {
                platoon.emplace_back(i);
            }
        }
        return platoon;
    }

    template<typename T> requires std::equality_comparable<T>
    __forceinline__ bool dualSetCompare(const T& v00,
                                        const T& v01,
                                        const T& v10,
                                        const T& v11){
        return ((v00 == v10 && v01 == v11) || (v00 == v11 && v01 == v10));
    }

    enum Stability : unsigned short{
            STABLE = 0,
            EXPERIMENTAL = 1
        };
}
#endif //CSRC_NICHE_CUH
