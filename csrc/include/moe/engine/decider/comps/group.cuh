/******************************************************************************
 * Copyright (c) 09/08/2024, Osayamen Jonathan Aimuyo.
 ******************************************************************************/
#ifndef CSRC_GROUP_CUH
#define CSRC_GROUP_CUH
#include <unordered_set>
#include "args.cuh"
#include "functions.cuh"
namespace aristos{
    struct Group{
        std::unordered_map<unsigned int, std::pair<unsigned int, unsigned int>> visited{};
        /// Dynamic Programming State
        std::vector<std::pair<double, double>> p2pTimes{};
        std::unordered_set<unsigned int> internalNodes{};
        unsigned int id;
        unsigned int memCapacity;
        unsigned long deviceRate;
        double allReduceTime{};
        ObjArgs objArgs;
        double currentObjective{};
        unsigned int worldSize;
        double cachedObjective{};
        double cachedAllReduceTime{};

        Group(const unsigned int& _id, const unsigned int& _mem, const unsigned long& _rate,
              const unsigned int& _world, ObjArgs _args,
              const std::vector<std::pair<double, double>>& dp):
              id(_id), memCapacity(_mem), deviceRate(_rate),
              objArgs(_args),worldSize(_world){
            internalNodes.insert(id);
            p2pTimes = dp;
        }

        __forceinline__ void construct(const double& art, const unsigned int& effective){
            objArgs.groupMemCapacity = memCapacity;
            objArgs.effectiveWorld = effective;
            objArgs.allReduceTime = allReduceTime = art;
            objArgs.totalDeviceRate = deviceRate;
            currentObjective = obj(objArgs);
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

            cachedAllReduceTime = allReduceT(arArgs);
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
        bool operator>(const Group& other){
            if(floatEqual(getCurrentObjective(), other.getCurrentObjective())){
                return id > other.id;
            }
            return getCurrentObjective() > other.getCurrentObjective();
        }

        private:
            /// Complementary Dynamic Programming magic
            __forceinline__
            void updateP2PTime(const Group& neighbor){
                auto const len = p2pTimes.size();
                for(int i = 0; i < len; i++){
                    p2pTimes[i] = std::pair{p2pTimes[i].first + neighbor.p2pTimes[i].first,
                                            p2pTimes[i].second + neighbor.p2pTimes[i].second};
                }
            }
            /// Dynamic Programming magic yielding complexity O(|self| + |neighbor|)
            /// rather than O(|self| * |neighbor|).
            __forceinline__
            double evalP2PTime(Group& neighbor, const unsigned int& numNodes) const{
                auto maxP2PTime = 0.0;
                for(const auto& node: internalNodes){
                    maxP2PTime = std::max(maxP2PTime,
                                          ObjArgs::p2pTransferTime(p2pTimes[node].first + neighbor.p2pTimes[node].first,
                                                                   p2pTimes[node].second + neighbor.p2pTimes[node].second,
                                                                   objArgs.p2pBuffer / static_cast<double>(numNodes)));
                }
                for(const auto& node: neighbor.internalNodes){
                    maxP2PTime = std::max(maxP2PTime,
                                          ObjArgs::p2pTransferTime(p2pTimes[node].first + neighbor.p2pTimes[node].first,
                                                                   p2pTimes[node].second + neighbor.p2pTimes[node].second,
                                                                   objArgs.p2pBuffer / static_cast<double>(numNodes)));
                }
                return maxP2PTime;
            }

            __forceinline__
            double getCurrentObjective() const{
                return (currentObjective - allReduceTime) + cachedAllReduceTime;
            }

            __forceinline__
            void updateVisited(const unsigned int& neighborID,
                               const unsigned int& myState,
                               const unsigned int& neighborState){
                // I dislike the unnecessary construction of a 'pair' object per insertion
                visited.try_emplace(neighborID, std::pair{myState, neighborState});
            }
    };
}
#endif //CSRC_GROUP_CUH
