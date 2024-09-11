//
// Created by osayamen on 9/8/24.
//
/******************************************************************************
 * Copyright (c) 2024, Osayamen Jonathan Aimuyo.
 ******************************************************************************/

#ifndef CSRC_GROUP_CUH
#define CSRC_GROUP_CUH
#include <unordered_set>
#include "args.cuh"
#include "functions.cuh"
namespace aristos{
    __inline__ double clamp;
    template <typename F, typename A, typename P>
    requires std::regular_invocable<F, ObjArgs>
            && std::regular_invocable<A, ARArgs>
            && std::regular_invocable<P, double, double, double>
    struct Group{
        F objectiveFunction;
        A reduceTimeFunction;
        P policy;
        std::unordered_map<unsigned int, std::pair<unsigned int, unsigned int>> visited;
        std::vector<std::pair<unsigned int, unsigned int>> p2pTimes;
        std::unordered_set<unsigned int> internalNodes;
        unsigned int id;
        unsigned int memCapacity;
        unsigned int deviceRate;
        double allReduceTime;
        ObjArgs objArgs;
        double currentObjective;
        unsigned int worldSize;
        double cachedObjective;
        double cachedAllReduceTime;

        __forceinline__
        unsigned int numNodes(){
            return internalNodes.size();
        }

        __forceinline__
        double getCurrentObjective(){
            return (currentObjective - allReduceTime) + cachedAllReduceTime;
        }

        __forceinline__
        void updateVisited(const unsigned int& neighborID,
                           const unsigned int& myState,
                           const unsigned int& neighborState){
            // I dislike the unnecessary construction of a 'pair' object per insertion
            visited.try_emplace(neighborID, std::pair{myState, neighborState});
        }

        __forceinline__
        bool shouldMerge(Group& neighbor, const ARArgs& arArgs, const unsigned int& effectiveW){
            auto prevState = visited.at(neighbor.id);
            if(visited.contains(neighbor.id) &&
            prevState.first == numNodes() && prevState.second == neighbor.numNodes()){
                /// We have evaluated and rejected this group previously.
                /// Neither of our states has changed since our last encounter,
                /// thus we bypass the expensive evaluation procedure and proactively reject again.
                return false;
            }
            updateVisited(neighbor.id, numNodes(), neighbor.numNodes());
            /// Update from global state
            objArgs.effectiveWorld = effectiveW;
            /// Simulate the event of both groups merging and compute its objective
            auto cachedEffectiveWorld = objArgs.effectiveWorld;
            if(memCapacity + neighbor.memCapacity >= objArgs.numExperts){
                if(objArgs.effectiveWorld < worldSize && memCapacity < objArgs.numExperts){
                    objArgs.effectiveWorld += numNodes();
                }
                if(objArgs.effectiveWorld < worldSize && neighbor.memCapacity < objArgs.numExperts){
                    objArgs.effectiveWorld += numNodes();
                }
            }
            objArgs.totalDeviceRate = deviceRate + neighbor.deviceRate;
            objArgs.groupMemCapacity = memCapacity + neighbor.memCapacity;

            cachedAllReduceTime = reduceTimeFunction(arArgs);
            neighbor.cachedAllReduceTime = cachedAllReduceTime;
            objArgs.intraCommunicationCost = evalP2PTime(neighbor, numNodes() + neighbor.numNodes());

            cachedObjective = neighbor.cachedObjective = objectiveFunction(objArgs);
            objArgs.effectiveWorld = cachedEffectiveWorld;
            return policy(getCurrentObjective(), neighbor.getCurrentObjective(), cachedObjective);
        }

        /// Dynamic Programming magic
        __forceinline__
        double evalP2PTime(Group& neighbor, const unsigned int& numNodes){
            auto maxP2PTime = 0.0;
        }

        __forceinline__
        bool optimizingPolicy(const double& obj1, const double& obj2, const double& obj1_2) const{
            return (std::isinf(obj1) && std::isinf(obj2))? true :
            (obj1_2 * (1 / obj1 + 1/obj2)) <= (std::min((std::max(obj1, obj2) / std::min(obj1, obj2)) + 1, clamp));
        }
    };
}
#endif //CSRC_GROUP_CUH
