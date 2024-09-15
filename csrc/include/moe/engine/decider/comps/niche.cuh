//
// Created by osayamen on 9/15/24.
//

#ifndef CSRC_NICHE_CUH
#define CSRC_NICHE_CUH

namespace aristos{
    template<typename P, typename K = P::value_type>
    __forceinline__
    std::unordered_map<K, std::vector<int>> subsets(P& parents, int n){
        std::unordered_map<K, std::vector<int>> sets{};
        for(int i = 0; i < n; ++i){
            sets[parents[i]].push_back(i);
        }
        return sets;
    }

    template<typename T> requires std::equality_comparable<T>
    __forceinline__ bool dualSetCompare(const T& v00,
                                        const T& v01,
                                        const T& v10,
                                        const T& v11){
        return ((v00 == v10 && v01 == v11) || (v00 == v11 && v01 == v10));
    }
}
#endif //CSRC_NICHE_CUH
