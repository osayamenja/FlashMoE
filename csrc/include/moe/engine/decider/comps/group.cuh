//
// Created by osayamen on 9/8/24.
//

#ifndef CSRC_GROUP_CUH
#define CSRC_GROUP_CUH
#include <unordered_set>
#include "args.cuh"
#include "functions.cuh"
namespace aristos{
    template <typename F, typename A, typename G>
    requires std::regular_invocable<F, ObjArgs>
            && std::regular_invocable<A, ARArgs>
    struct Group{
        F objectiveFunction;
        A reduceTimeFunction;
        std::unordered_map<unsigned int, std::pair<unsigned int, unsigned int>> visited;
        std::vector<std::pair<unsigned int, unsigned int>> p2pTimes;
        std::unordered_set<unsigned int> internalNodes;
        unsigned int id;
        unsigned int seedProcessingRate;
        unsigned int memCapacity;

        double allReduceTime;
    };
}
#endif //CSRC_GROUP_CUH
