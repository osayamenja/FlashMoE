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
    __forceinline__
    std::vector<int> group(){
    }

    /// Generates EP spec [E,D] -> Experts to Devices
    __forceinline__
    std::vector<int> assign(){

    }
}
#endif //CSRC_DECIDER_CUH
