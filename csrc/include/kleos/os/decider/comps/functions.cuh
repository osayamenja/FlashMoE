/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Kleos Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 */

//
// Created by osayamen on 9/10/24.
//

#ifndef CSRC_FUNCTIONS_CUH
#define CSRC_FUNCTIONS_CUH
#include "args.cuh"
#include <limits>
namespace kleos{
    __inline__ float clamp;
    __forceinline__
    float obj(const ObjArgs& a){
        return (a.groupMemCapacity < a.totalExpertMemoryDemand) ?
        std::numeric_limits<float>::infinity() :
               (ObjArgs::getGamma(a.globalMoEStages, a.effectiveWorld)
               *((static_cast<float>(a.totalExpertCost) / a.totalDeviceRate)
               + (a.commFreq * a.intraCommunicationCost))) + a.allReduceTime;
    }

    __forceinline__
    float allReduceT(const ARArgs& a){
        ///https://link.springer.com/content/pdf/10.1007/978-3-540-24685-5_1.pdf
        return 2.0 * (a.numGroups - 1) * a.bottleneckTime;
    }

    template<kleos::Stability s=STABLE>
    __forceinline__
    bool optimizingPolicy(const float& obj1, const float& obj2, const float& obj1_2){
        return (std::isinf(obj1) && std::isinf(obj2))? true :
               (obj1_2 <= std::max(obj1, obj2));
    }
    template<>
    __forceinline__
    bool optimizingPolicy<EXPERIMENTAL>(const float& obj1, const float& obj2, const float& obj1_2){
        return (std::isinf(obj1) && std::isinf(obj2))? true :
               (obj1_2 * (1 / obj1 + 1/obj2)) <= (std::min((std::max(obj1, obj2) / std::min(obj1, obj2)) + 1, clamp));
    }
}
#endif //CSRC_FUNCTIONS_CUH
