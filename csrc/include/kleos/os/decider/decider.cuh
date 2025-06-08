/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Kleos Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 */

#ifndef CSRC_DECIDER_CUH
#define CSRC_DECIDER_CUH

#include <vector>
#include <algorithm>
#include <queue>
#include <ranges>
#include <set>
#include <boost/pending/disjoint_sets.hpp>
#include <cute/tensor.hpp>

#include "../../types.cuh"
#include "comps/edge.cuh"
#include "comps/niche.cuh"
#include "comps/expert.cuh"
#include "comps/group.cuh"
#include "comps/worker.cuh"

namespace kleos{
    /// Necessary to use path halving to ensure amortized "practical constant" time
    using DisjointSet = boost::disjoint_sets_with_storage<boost::identity_property_map,
            boost::identity_property_map, boost::find_with_path_halving>;
    /// Generates DP-EP groups [D, G] -> Devices to Groups
    /// Complexity 𝓞(|E|*log(|E|) + |V|(|V|-1)) => 𝓞(|V|^2*log(|V|^2)),
    /// where |V| = |workers| and |E| = number of edges = |V|*(|V| - 1)
    template <
        JobType jT = JobType::training
    >
    struct Decider {
        static_assert(jT == JobType::training);
        template<
            typename AdjMatrix
        >
        requires(cute::is_tensor_v<AdjMatrix> && rank(AdjMatrix{}) == 2 &&
            cuda::std::is_same_v<typename AdjMatrix::value_type, floatPair>)
        __forceinline__ __host__
        bool operator()(const AdjMatrix& adjMatrix,
            const Worker* __restrict__ const& workers,
            const unsigned int& totalExpertCost,
            const unsigned int& totalExpertMem,
            uint* __restrict__ deviceToGroups) const {
            const auto world = cute::size<0>(adjMatrix);
            auto infeasibleGroups = std::unordered_set<unsigned int>{};
            for(uint i = 0; i < world; ++i){
                if(const auto w = workers[i]; w.memoryCapacity < totalExpertMem)
                    infeasibleGroups.insert(w.id);
            }
            DisjointSet groups(world);
            const uint edgeLen = world * (world - 1);
            uint current = 0U;
            std::vector<Edge> candidateEdges(edgeLen);
            std::priority_queue<Edge> externalEdges;
            auto groupInfo = std::unordered_map<unsigned int, Group>{};
            auto effectiveWorld = world - infeasibleGroups.size();
            for(uint i = 0; i < world; ++i) {
                auto dp = std::vector<floatPair>(world);
                for (uint j = 0; j < world; ++j) {
                    dp[j] = {0.0, 0.0};
                    if (i != j)[[likely]] {
                        const auto alpha = adjMatrix(i, j).alpha;
                        const auto beta = adjMatrix(i, j).beta;
                        candidateEdges[current++] = {i, j,
                                                 ObjArgs::p2pTransferTime(alpha, beta,
                                                                            ObjArgs::p2pBuffer)};
                        externalEdges.emplace(i, j, ARArgs::bottleneck(alpha, beta,
                                                                         ARArgs::gradBuffer, 2));
                        /// Invert the edge for the dp table
                        dp[j] = adjMatrix(j, i);
                    }
                }
                groupInfo.insert({i, Group(i,
                                           workers[i].memoryCapacity,
                                           workers[i].processingRate,
                                           world,
                                           ObjArgs(totalExpertCost, effectiveWorld, totalExpertMem),
                                           dp)});
            }
            auto extEdge = externalEdges.top();
            auto arArgs = ARArgs(adjMatrix(extEdge.node1, extEdge.node2).alpha,
                                   adjMatrix(extEdge.node1, extEdge.node2).beta,
                                   effectiveWorld);
            const auto art = allReduceT(arArgs);
            /// Second-pass group construction
            for(auto& g : std::views::values(groupInfo)){
                g.construct(art, effectiveWorld);
            }

            auto limbo = Edge::limboEdge();
            current = 0U;
            std::ranges::sort(candidateEdges.begin(), candidateEdges.end(), std::less{});
            while (current < edgeLen){
                const auto candidate = candidateEdges[current++];
                auto group1 = groups.find_set(candidate.node1);
                auto group2 = groups.find_set(candidate.node2);
                if (group1 == group2)[[likely]]{
                    continue;
                }
                extEdge = externalEdges.top();
                /// if number of groups is <= 2, then there is no need to find the edge when it coincides
                /// as the ar cost would be zero anyway for a single group
                auto extGroup1 = groups.find_set(extEdge.node1);
                auto extGroup2 = groups.find_set(extEdge.node2);
                if(groupInfo.size() > 2 && dualSetCompare(extGroup1, extGroup2, group1, group2))[[unlikely]]{
                    limbo = extEdge;
                }

                while(groupInfo.size() > 2 && ((groups.find_set(extEdge.node1) == groups.find_set(extEdge.node2))
                || dualSetCompare(groups.find_set(extEdge.node1), groups.find_set(extEdge.node2), group1, group2))){
                    externalEdges.pop();
                    extEdge = externalEdges.top();
                }
                const bool satisfiesConstraint = groupInfo.at(group1).memCapacity + groupInfo.at(group2).memCapacity >= totalExpertMem;
                arArgs.numGroups = groupInfo.size() - infeasibleGroups.size();
                if(infeasibleGroups.contains(group1) && infeasibleGroups.contains(group2)){
                    if(satisfiesConstraint){
                        arArgs.numGroups += 1;
                    }
                }
                else if(!infeasibleGroups.contains(group1) && !infeasibleGroups.contains(group2)){
                    arArgs.numGroups -= 1;
                }
                arArgs.refresh(adjMatrix(extEdge.node1, extEdge.node2).alpha, adjMatrix(extEdge.node1, extEdge.node2).beta);
                if(groupInfo.at(group1).shouldMerge(groupInfo.at(group2), allReduceT(arArgs), effectiveWorld)){
                    limbo = Edge::limboEdge();
                    groups.link(group1, group2);
                    auto parent = group1;
                    auto child = group2;
                    if(group1 != groups.find_set(group1)){
                        parent = group2;
                        child = group1;
                    }
                    if(satisfiesConstraint){
                        if(infeasibleGroups.contains(parent)){
                            infeasibleGroups.erase(parent);
                            effectiveWorld += groupInfo.at(parent).numNodes();
                        }
                        if(infeasibleGroups.contains(child)){
                            infeasibleGroups.erase(child);
                            effectiveWorld += groupInfo.at(child).numNodes();
                        }
                    }
                    else{
                        infeasibleGroups.erase(child);
                    }
                    groupInfo.at(parent).subsume(groupInfo.at(child));
                    groupInfo.erase(child);
                }
                if(!limbo.isLimboEdge()){
                    externalEdges.push(limbo);
                }
            }

            /// Post-processing
            for(const auto& i: std::views::keys(groupInfo)){
                if(infeasibleGroups.contains(i)){
                    groupInfo.erase(i);
                }
            }

            uint i = 0U;
            for (const auto& groupId: groups.parents()) {
                deviceToGroups[i++] = static_cast<uint>(groupId);
            }

            return infeasibleGroups.size() > 0; // provably, this can only be 0 or 1
        }
    };
    // No gradient all reduce modeling here
    template <>
    struct Decider<JobType::inference> {
        template<
            typename AdjMatrix
        >
        requires(cute::is_tensor_v<AdjMatrix> && rank(AdjMatrix{}) == 2 &&
            cuda::std::is_same_v<typename AdjMatrix::value_type, floatPair>)
        __forceinline__ __host__
        bool operator()(const AdjMatrix& adjMatrix,
            const Worker* __restrict__ const& workers,
            const unsigned int& totalExpertCost,
            const unsigned int& totalExpertMem,
            uint* __restrict__ deviceToGroups) const {
            const auto world = cute::size<0>(adjMatrix);
            auto infeasibleGroups = std::unordered_set<unsigned int>{};
            for(uint i = 0; i < world; ++i){
                if(const auto w = workers[i]; w.memoryCapacity < totalExpertMem)
                    infeasibleGroups.insert(w.id);
            }
            DisjointSet groups(world);
            const uint edgeLen = world * (world - 1);
            uint current = 0U;
            std::vector<Edge> candidateEdges(edgeLen);
            auto groupInfo = std::unordered_map<unsigned int, Group>{};
            auto effectiveWorld = world - infeasibleGroups.size();
            for(uint i = 0; i < world; ++i) {
                auto dp = std::vector<floatPair>(world);
                for (uint j = 0; j < world; ++j) {
                    dp[j] = {0.0, 0.0};
                    if (i != j)[[likely]] {
                        const auto alpha = adjMatrix(i, j).alpha;
                        const auto beta = adjMatrix(i, j).beta;
                        candidateEdges[current++] = {i, j,
                                                 ObjArgs::p2pTransferTime(alpha, beta,
                                                                          ObjArgs::p2pBuffer)};
                        /// Invert the edge for the dp table
                        dp[j] = adjMatrix(j, i);
                    }
                }
                groupInfo.insert({i, Group(i,
                                           workers[i].memoryCapacity,
                                           workers[i].processingRate,
                                           world,
                                           ObjArgs(totalExpertCost, effectiveWorld, totalExpertMem),
                                           dp)});
            }
            current = 0U;
            std::ranges::sort(candidateEdges.begin(), candidateEdges.end(), std::less{});
            while (current < edgeLen){
                const auto candidate = candidateEdges[current++];
                auto group1 = groups.find_set(candidate.node1);
                auto group2 = groups.find_set(candidate.node2);
                if (group1 == group2)[[likely]]{
                    continue;
                }
                const bool satisfiesConstraint = groupInfo.at(group1).memCapacity +
                    groupInfo.at(group2).memCapacity >= totalExpertMem;
                if(groupInfo.at(group1).shouldMerge(groupInfo.at(group2), 0.0f, effectiveWorld)){
                    groups.link(group1, group2);
                    auto parent = group1;
                    auto child = group2;
                    if(group1 != groups.find_set(group1)){
                        parent = group2;
                        child = group1;
                    }
                    if(satisfiesConstraint){
                        if(infeasibleGroups.contains(parent)){
                            infeasibleGroups.erase(parent);
                            effectiveWorld += groupInfo.at(parent).numNodes();
                        }
                        if(infeasibleGroups.contains(child)){
                            infeasibleGroups.erase(child);
                            effectiveWorld += groupInfo.at(child).numNodes();
                        }
                    }
                    else{
                        infeasibleGroups.erase(child);
                    }
                    groupInfo.at(parent).subsume(groupInfo.at(child));
                    groupInfo.erase(child);
                }
            }
            // If there is an infeasible group by this point,
            // then all nodes would be in that group, therefore return an empty vector
            uint i = 0U;
            for (const auto& groupId: groups.parents()) {
                deviceToGroups[i++] = static_cast<uint>(groupId);
            }

            return infeasibleGroups.size() == 0; // provably, this can only be 0 or 1
        }
    };

    /// Generates EP spec [E, D] -> Experts to Devices
    /// Assumes that the group satisfies memory constraints.
    /// Complexity 𝓞(|X|*log(|X|)), where |X| = |experts|
    __forceinline__
    void assign(Worker* __restrict__ const& wG,
        const uint& world,
        Expert* __restrict__ const& experts,
        const uint& numExperts,
        uint* __restrict__ const& assignment){
        using CostComparator = decltype([](const Expert& lhs, const Expert& rhs){
            return lhs.cost == rhs.cost? lhs.id > rhs.id : lhs.cost < rhs.cost;
        });
        std::set<Expert, CostComparator> t(experts, experts + numExperts);
        size_t totalCost = 0U, totalMem = 0U;
        for(uint i = 0; i < numExperts; ++i){
            const auto e = experts[i];
            totalCost += e.cost;
            totalMem += e.memoryDemand; // == experts.size()
        }
        auto wellDistributedCapacity = true;
        auto reqCap = totalMem / world;
        if(totalMem % world != 0){
            reqCap = static_cast<int>(std::ceil(static_cast<float>(totalMem) / static_cast<float>(world)));
        }
        auto totalRate = 0.0f;
        for(uint i = 0; i < world; ++i){
            const auto w = wG[i];
            wellDistributedCapacity = wellDistributedCapacity && w.memoryCapacity >= reqCap;
            totalRate += w.processingRate;
        }
        std::ranges::sort(wG, wG + world, std::greater<>());

        auto j = 0U;
        while(!t.empty()){
            auto budget = cute::max(static_cast<int>(std::floor(wG[j].processingRate * totalCost / totalRate)),
                1);
            const auto allocated = budget;
            while(budget > 0 && wG[j].memoryCapacity > 0 && !t.empty()){
                auto expertBudget = Expert(budget);
                auto lower = t.lower_bound(expertBudget);
                // Below is when lower == t.end() ==> budget is greater than any existing individual demand
                auto bestMatch = *std::prev(t.cend());
                if(lower->cost == budget || lower == t.cbegin()){
                    bestMatch = *lower;
                }
                else if (lower != t.cend()){
                    bestMatch = Expert::closest(*lower, *t.upper_bound(expertBudget), budget);
                }
                assignment[bestMatch.id] = wG[j].id;
                t.erase(bestMatch);
                wG[j].memoryCapacity -= 1;
                budget -= bestMatch.cost;
            }
            totalCost -= (allocated - budget);
            if(wG[j].memoryCapacity == 0 || wellDistributedCapacity){
                totalRate -= wG[j].processingRate;
            }
            j = (j + 1) % world;
        }
    }
}
#endif //CSRC_DECIDER_CUH
