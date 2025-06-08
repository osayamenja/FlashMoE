/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Kleos Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 */
// Created 09/08/2024
#ifndef CSRC_GROUP_CUH
#define CSRC_GROUP_CUH
#include <unordered_set>
#include "args.cuh"
#include "functions.cuh"
namespace kleos{
    struct Group{
        std::unordered_map<unsigned int, std::pair<unsigned int, unsigned int>> visited{};
        /// Dynamic Programming State
        std::vector<floatPair> p2pTimes{};
        std::unordered_set<unsigned int> internalNodes{};
        unsigned int id;
        unsigned int memCapacity;
        float deviceRate;
        float allReduceTime{};
        ObjArgs objArgs;
        float currentObjective{};
        unsigned int worldSize;
        float cachedObjective{};
        float cachedAllReduceTime{};

        Group(const unsigned int& _id, const unsigned int& _mem, const float& _rate,
              const unsigned int& _world, const ObjArgs& _args,
              const std::vector<floatPair>& dp):
              id(_id), memCapacity(_mem), deviceRate(_rate),
              objArgs(_args),worldSize(_world){
            currentObjective = obj(objArgs);
            internalNodes.insert(id);
            p2pTimes = dp;
        }

        __forceinline__
        void construct(const float& art, const unsigned int& effective){
            objArgs.groupMemCapacity = memCapacity;
            objArgs.effectiveWorld = effective;
            objArgs.allReduceTime = allReduceTime = art;
            objArgs.totalDeviceRate = deviceRate;
            currentObjective = obj(objArgs);
        }

        __forceinline__
        bool shouldMerge(Group& neighbor, float const& aRt, const unsigned int& effectiveW){
            if(visited.contains(neighbor.id)){
                if(auto [myState, theirState] = visited.at(neighbor.id);
                    myState == numNodes() && theirState == neighbor.numNodes()){
                    /// We have evaluated and rejected this group previously.
                    /// Neither of our states has changed since our last encounter,
                    /// thus we bypass the expensive evaluation procedure and proactively reject again.
                    return false;
                }
            }
            updateVisited(neighbor.id, numNodes(), neighbor.numNodes());
            /// Update from global state
            objArgs.effectiveWorld = effectiveW;
            /// Simulate the event of both groups merging and compute its objective
            const auto cachedEffectiveWorld = objArgs.effectiveWorld;
            if(memCapacity + neighbor.memCapacity >= objArgs.totalExpertMemoryDemand){
                if(objArgs.effectiveWorld < worldSize && memCapacity < objArgs.totalExpertMemoryDemand){
                    objArgs.effectiveWorld += numNodes();
                }
                if(objArgs.effectiveWorld < worldSize && neighbor.memCapacity < objArgs.totalExpertMemoryDemand){
                    objArgs.effectiveWorld += numNodes();
                }
            }
            objArgs.totalDeviceRate = deviceRate + neighbor.deviceRate;
            objArgs.groupMemCapacity = memCapacity + neighbor.memCapacity;

            cachedAllReduceTime = aRt;
            neighbor.cachedAllReduceTime = cachedAllReduceTime;
            objArgs.intraCommunicationCost = evalP2PTime(neighbor, numNodes() + neighbor.numNodes());
            objArgs.allReduceTime = cachedAllReduceTime;

            cachedObjective = neighbor.cachedObjective = obj(objArgs);
            objArgs.effectiveWorld = cachedEffectiveWorld;
            return optimizingPolicy(getCurrentObjective(), neighbor.getCurrentObjective(), cachedObjective);
        }

        __forceinline__
        void subsume(const Group& neighbor){
            updateP2PTime(neighbor);
            internalNodes.insert(neighbor.internalNodes.cbegin(), neighbor.internalNodes.cend());
            currentObjective = cachedObjective;
            memCapacity += neighbor.memCapacity;
            deviceRate += neighbor.deviceRate;
            allReduceTime = cachedAllReduceTime;
        }

        __forceinline__
        unsigned int numNodes() const{
            return internalNodes.size();
        }

        __forceinline__
        bool operator>(const Group& other) const{
            if(floatEqual(getCurrentObjective(), other.getCurrentObjective())){
                return id > other.id;
            }
            return getCurrentObjective() > other.getCurrentObjective();
        }

        __forceinline__
        bool operator==(const Group& other) const{
            return id == other.id;
        }

        private:
            /// Complementary Dynamic Programming magic
            __forceinline__
            void updateP2PTime(const Group& neighbor){
                auto const len = p2pTimes.size();
                for(int i = 0; i < len; i++){
                    p2pTimes[i] = {p2pTimes[i].alpha + neighbor.p2pTimes[i].alpha,
                                            p2pTimes[i].beta + neighbor.p2pTimes[i].beta};
                }
            }
            /// Dynamic Programming magic yielding complexity O(|self| + |neighbor|)
            /// rather than O(|self| * |neighbor|).
            __forceinline__
            float evalP2PTime(const Group& neighbor, const unsigned int& numNodes) const{
                auto maxP2PTime = 0.0f;
                for(const auto& node: internalNodes){
                    maxP2PTime = std::max(maxP2PTime,
                                          ObjArgs::p2pTransferTime(p2pTimes[node].alpha + neighbor.p2pTimes[node].alpha,
                                                                   p2pTimes[node].beta + neighbor.p2pTimes[node].beta,
                                                                   objArgs.p2pBuffer / static_cast<float>(numNodes)));
                }
                for(const auto& node: neighbor.internalNodes){
                    maxP2PTime = std::max(maxP2PTime,
                                          ObjArgs::p2pTransferTime(p2pTimes[node].alpha + neighbor.p2pTimes[node].alpha,
                                                                   p2pTimes[node].beta + neighbor.p2pTimes[node].beta,
                                                                   objArgs.p2pBuffer / static_cast<float>(numNodes)));
                }
                return maxP2PTime;
            }

            __forceinline__
            float getCurrentObjective() const{
                return (currentObjective - allReduceTime) + cachedAllReduceTime;
            }

            __forceinline__
            void updateVisited(const unsigned int& neighborID,
                               const unsigned int& myState,
                               const unsigned int& neighborState){
                visited.try_emplace(neighborID, std::pair{myState, neighborState});
            }
    };
}
#endif //CSRC_GROUP_CUH
