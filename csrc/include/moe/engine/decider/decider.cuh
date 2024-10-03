/******************************************************************************
 * Copyright (c) 2024, Osayamen Jonathan Aimuyo.
 ******************************************************************************/

#ifndef CSRC_DECIDER_CUH
#define CSRC_DECIDER_CUH

#include <vector>
#include <algorithm>
#include <numeric>
#include <queue>
#include <set>
#include <boost/pending/disjoint_sets.hpp>
#include <cute/config.hpp>

#include "comps/edge.cuh"
#include "comps/niche.cuh"
#include "comps/expert.cuh"
#include "comps/group.cuh"
#include "comps/worker.cuh"

using AdjMatrix = std::vector<std::vector<std::pair<double, double>>>;
namespace aristos::decider{
    /// Necessary to use path halving to ensure amortized "practical constant" time
    using DisjointSet = boost::disjoint_sets_with_storage<boost::identity_property_map,
            boost::identity_property_map, boost::find_with_path_halving>;
    /// Generates DP-EP groups [D,G] -> Devices to Groups
    __forceinline__
    std::vector<size_t> decide(const AdjMatrix& adjMatrix,
                               const std::vector<Worker>& workers,
                               const unsigned long& totalExpertCost,
                               const unsigned int& totalExpertMem,
                               const ModelConfig& modelConfig,
                               bool doParetoSweep = false){
        auto infeasibleGroups = std::unordered_set<unsigned int>{};
        for(const auto& w: workers){
            if(w.memoryCapacity < totalExpertMem)
                infeasibleGroups.insert(w.id);
        }
        DisjointSet groups(workers.size());
        std::priority_queue<Edge, std::vector<Edge>, std::greater<>> candidateEdges;
        std::priority_queue<Edge> externalEdges;
        auto groupInfo = std::unordered_map<unsigned int, Group>{};
        auto effectiveWorld = workers.size() - infeasibleGroups.size();
        for(int i = 0; i < adjMatrix.size(); ++i) {
            auto dp = std::vector<std::pair<double, double>>(adjMatrix.size());
            for (int j = 0; j < adjMatrix.size(); ++j) {
                dp[j] = {0.0, 0.0};
                if (i != j)[[likely]] {
                    auto alpha = adjMatrix[i][j].first;
                    auto beta = adjMatrix[i][j].second;
                    candidateEdges.emplace(i, j,
                                             ObjArgs::p2pTransferTime(alpha, beta,
                                                                      modelConfig.p2pBuffer));
                    externalEdges.emplace(i, j, ARArgs::bottleneck(alpha, beta,
                                                                     modelConfig.gradBuffer, 2));
                    /// Invert the edge for the dp table
                    dp[j] = adjMatrix[j][i];
                }
            }
            groupInfo.insert({i, Group(i,
                                       workers[i].memoryCapacity,
                                       workers[i].processingRate,
                                       workers.size(),
                                       ObjArgs(totalExpertCost, effectiveWorld, totalExpertMem, modelConfig),
                                       dp)});
        }
        auto extEdge = externalEdges.top();
        auto arArgs = ARArgs(adjMatrix[extEdge.node1][extEdge.node2].first,
                               adjMatrix[extEdge.node1][extEdge.node2].second,
                               effectiveWorld, modelConfig.gradBuffer);
        const auto art = allReduceT(arArgs);
        /// Second-pass group construction
        for(auto& [i,g] : groupInfo){
            g.construct(art, effectiveWorld);
        }

        auto limbo = Edge::limboEdge();
        while (!candidateEdges.empty()){
            auto candidate = candidateEdges.top();
            candidateEdges.pop();
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
            bool satisfiesConstraint = groupInfo.at(group1).memCapacity + groupInfo.at(group2).memCapacity >= totalExpertMem;
            arArgs.numGroups = groupInfo.size() - infeasibleGroups.size();
            if(infeasibleGroups.contains(group1) && infeasibleGroups.contains(group2)){
                if(satisfiesConstraint){
                    arArgs.numGroups += 1;
                }
            }
            else if(!infeasibleGroups.contains(group1) && !infeasibleGroups.contains(group2)){
                arArgs.numGroups -= 1;
            }
            arArgs.refresh(adjMatrix[extEdge.node1][extEdge.node2].first, adjMatrix[extEdge.node1][extEdge.node2].second);
            if(groupInfo.at(group1).shouldMerge(groupInfo.at(group2), arArgs, effectiveWorld)){
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
        for(const auto& [i, _]: groupInfo){
            if(infeasibleGroups.contains(i)){
                groupInfo.erase(i);
            }
        }
        return groups.parents();
    }

    /// Generates EP spec [E,D] -> Experts to Devices
    /// Assumes that the group satisfies memory constraints.
    __forceinline__
    std::vector<unsigned int> assign(std::vector<Expert>& experts,
                            std::vector<Worker>& workerGroup){
        using CostComparator = decltype([](const Expert& lhs, const Expert& rhs){
            return lhs.cost == rhs.cost? lhs.id > rhs.id : lhs.cost < rhs.cost;
        });
        std::set<Expert, CostComparator> t(experts.cbegin(), experts.cend());
        std::vector<unsigned int> assignment(experts.size());
        size_t totalCost = 0U, totalMem = 0U;
        for(const auto& e: experts){
            totalCost += e.cost;
            totalMem += e.memoryDemand; // == experts.size()
        }
        auto wellDistributedCapacity = true;
        auto reqCap = totalMem / workerGroup.size();
        if(totalMem % workerGroup.size() != 0){
            reqCap = static_cast<int>(std::ceil(static_cast<double>(totalMem) / static_cast<double>(workerGroup.size())));
        }
        auto totalRate = 0U;
        for(const auto& w: workerGroup){
            wellDistributedCapacity = wellDistributedCapacity && w.memoryCapacity >= reqCap;
            totalRate += w.processingRate;
        }
        std::ranges::sort(workerGroup.begin(), workerGroup.end(), std::greater<>());

        auto j = 0U;
        while(!t.empty()){
            auto budget = static_cast<unsigned int>(std::ceil(static_cast<double>(workerGroup[j].processingRate * totalCost) / static_cast<double>(totalRate)));
            const auto allocated = budget;
            while(budget > 0 && workerGroup[j].memoryCapacity > 0 && !t.empty() > 0){
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
                assignment[bestMatch.id] = workerGroup[j].id;
                t.erase(bestMatch);
                workerGroup[j].memoryCapacity -= 1;
                budget -= bestMatch.cost;
            }
            j = (j + 1) % workerGroup.size();
            totalCost -= (allocated - budget);
            if(workerGroup[j].memoryCapacity == 0 || wellDistributedCapacity){
                totalRate -= workerGroup[j].processingRate;
            }
        }
        return assignment;
    }
}
#endif //CSRC_DECIDER_CUH
