/******************************************************************************
 * Copyright (c) 2024, Osayamen Jonathan Aimuyo.
 ******************************************************************************/

#ifndef CSRC_DECIDER_CUH
#define CSRC_DECIDER_CUH

#include <vector>
#include <algorithm>
#include <numeric>
#include <queue>
#include <boost/pending/disjoint_sets.hpp>
#include <cute/config.hpp>

#include "comps/edge.cuh"
#include "comps/expert.cuh"
#include "comps/group.cuh"
#include "comps/worker.cuh"

namespace aristos::decider{
    /// Generates DP-EP groups [D,G] -> Devices to Groups
    using DisjointSet = boost::disjoint_sets_with_storage<boost::identity_property_map,
            boost::identity_property_map, boost::find_with_path_halving>;
    using AdjMatrix = std::vector<std::vector<std::pair<double, double>>>;
    constexpr unsigned int p2pFreq = 4;
    __forceinline__
    std::vector<int> group(const AdjMatrix& adjMatrix,
                           const std::vector<Worker>& workers,
                           const std::vector<Expert>& experts,
                           bool doParetoSweep = false){
        auto totalCost = std::accumulate(experts.begin(), experts.end(), 0U,
                                         [](unsigned int accum, const Expert& next){
            return accum + next.cost;
        });
        auto infeasibleGroups = std::unordered_set<unsigned int>{};
        for(const auto& w: workers){
            if(w.memoryCapacity < experts.size())
                infeasibleGroups.insert(w.id);
        }
        DisjointSet groups(workers.size());
        std::priority_queue<Edge, std::vector<Edge>, std::greater<>> candidateEdges;
        std::priority_queue<Edge> externalEdges;
        auto groupInfo = std::unordered_map<int, Group>{};
        auto effectiveWorld = workers.size() - infeasibleGroups.size();

    }

    /// Generates EP spec [E,D] -> Experts to Devices
    __forceinline__
    std::vector<int> assign(){

    }

    template<typename P, typename K = P::value_type>
    __forceinline__
    std::unordered_map<K, std::vector<int>> subsets(P& parents, int n){
        std::unordered_map<K, std::vector<int>> sets{};
        for(int i = 0; i < n; ++i){
            sets[parents[i]].push_back(i);
        }
        return sets;
    }
}
#endif //CSRC_DECIDER_CUH
