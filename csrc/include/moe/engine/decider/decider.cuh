/******************************************************************************
 * Copyright (c) 2024, Osayamen Jonathan Aimuyo.
 ******************************************************************************/

#ifndef CSRC_DECIDER_CUH
#define CSRC_DECIDER_CUH

#include <vector>
#include <algorithm>
#include <boost/pending/disjoint_sets.hpp>
#include <cute/config.hpp>

#include "comps/edge.cuh"
#include "comps/expert.cuh"
#include "comps/group.cuh"
#include "comps/worker.cuh"

namespace aristos::decider{
    /// Generates DP-EP groups [D,G] -> Devices to Groups
    using djs = boost::disjoint_sets_with_storage<boost::identity_property_map,
            boost::identity_property_map, boost::find_with_path_halving>;
    __forceinline__
    std::vector<int> group(){
    }

    /// Generates EP spec [E,D] -> Experts to Devices
    __forceinline__
    std::vector<int> assign(){

    }

    template<typename P, typename K = P::value_type>
    std::unordered_map<K, std::vector<int>> subsets(P& parents, int n){
        std::unordered_map<K, std::vector<int>> sets{};
        for(int i = 0; i < n; ++i){
            sets[parents[i]].push_back(i);
        }
        return sets;
    }
}
#endif //CSRC_DECIDER_CUH
