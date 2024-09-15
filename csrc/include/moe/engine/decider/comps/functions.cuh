//
// Created by osayamen on 9/10/24.
//

#ifndef CSRC_FUNCTIONS_CUH
#define CSRC_FUNCTIONS_CUH
#include "args.cuh"
#include <limits>
namespace aristos{
    __inline__ double clamp;
    __forceinline__
    double obj(const ObjArgs& a){
        return (a.groupMemCapacity < a.numExperts)?
        std::numeric_limits<double>::infinity() :
               (ObjArgs::getGamma(a.globalMoEStages, a.effectiveWorld)
               *((static_cast<double>(a.totalExpertCost) / static_cast<double>(a.totalDeviceRate))
               + (a.commFreq * a.intraCommunicationCost))) + a.allReduceTime;
    }

    __forceinline__
    double allReduceT(const ARArgs& a){
        ///https://link.springer.com/content/pdf/10.1007/978-3-540-24685-5_1.pdf
        return 2.0 * (a.numGroups - 1) * a.bottleneckTime;
    }

    __forceinline__
    bool optimizingPolicy(const double& obj1, const double& obj2, const double& obj1_2){
        return (std::isinf(obj1) && std::isinf(obj2))? true :
               (obj1_2 * (1 / obj1 + 1/obj2)) <= (std::min((std::max(obj1, obj2) / std::min(obj1, obj2)) + 1, clamp));
    }
}
#endif //CSRC_FUNCTIONS_CUH
