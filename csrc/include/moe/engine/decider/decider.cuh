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
#include "comps/niche.cuh"
#include "comps/expert.cuh"
#include "comps/group.cuh"
#include "comps/worker.cuh"

namespace aristos::decider{
    /// Generates DP-EP groups [D,G] -> Devices to Groups
    using DisjointSet = boost::disjoint_sets_with_storage<boost::identity_property_map,
            boost::identity_property_map, boost::find_with_path_halving>;
    using AdjMatrix = std::vector<std::vector<std::pair<double, double>>>;
    __forceinline__
    std::vector<size_t> group(const AdjMatrix& adjMatrix,
                           const std::vector<Worker>& workers,
                           const std::vector<Expert>& experts,
                           bool doParetoSweep = false){
        auto totalCost = 0U, totalMem = 0U;
        for(const auto& e: experts){
            totalCost += e.cost;
            totalMem += e.memoryDemand;
        }

        auto infeasibleGroups = std::unordered_set<unsigned int>{};
        for(const auto& w: workers){
            if(w.memoryCapacity < totalMem)
                infeasibleGroups.insert(w.id);
        }
        DisjointSet groups(workers.size());
        std::priority_queue<Edge, std::vector<Edge>, std::greater<>> candidateEdges;
        std::priority_queue<Edge> externalEdges;
        auto groupInfo = std::unordered_map<unsigned int, Group>{};
        auto effectiveWorld = workers.size() - infeasibleGroups.size();
        for(int i = 0; i < adjMatrix.size(); ++i) {
            auto dp = std::vector<std::pair<double, double>>(adjMatrix.size());
            for (int j = 0; j < adjMatrix.size(); ++i) {
                auto alpha = adjMatrix[i][j].first;
                auto beta = adjMatrix[i][j].second;
                dp[j] = {0.0, 0.0};
                if (i != j)[[likely]] {
                    candidateEdges.push(Edge(i, j,
                                             ObjArgs::p2pTransferTime(alpha, beta,
                                                                      hostMoEConfig.p2pBuffer)));
                    externalEdges.push(Edge(i, j, ARArgs::bottleneck(alpha, beta,
                                                                     hostMoEConfig.gradBuffer, 2)));
                    /// Invert the edge for the dp table
                    dp[j] = adjMatrix[j][i];
                }
            }
            groupInfo.insert({i, Group(i,
                                       workers[i].memoryCapacity,
                                       workers[i].processingRate,
                                       workers.size(),
                                       ObjArgs(totalCost, effectiveWorld, totalMem),
                                       dp)});
        }
        auto extEdge = externalEdges.top();
        auto arArgs = ARArgs(adjMatrix[extEdge.node1][extEdge.node2].first,
                               adjMatrix[extEdge.node1][extEdge.node2].second,
                               effectiveWorld);
        auto art = allReduceT(arArgs);
        /// Second-pass group construction
        for(auto& i : groupInfo){
            i.second.construct(art, effectiveWorld);
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
            bool satisfiesConstraint = groupInfo.at(group1).memCapacity + groupInfo.at(group2).memCapacity >= totalMem;
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
        for(const auto& g: groupInfo){
            if(infeasibleGroups.contains(g.first)){
                groupInfo.erase(g.first);
            }
        }

        if(doParetoSweep && groupInfo.size() > 1)[[unlikely]]{

        }
        return groups.parents();
    }

    /// Generates EP spec [E,D] -> Experts to Devices
    __forceinline__
    std::vector<int> assign(){

    }
}
#endif //CSRC_DECIDER_CUH
