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

#ifndef CSRC_ARGS_CUH
#define CSRC_ARGS_CUH
#include "../../../types.cuh"
namespace kleos{
    struct ARArgs{
        // units is MB
        constexpr static unsigned int gradBuffer = ACC::GRB::value;
        /// 𝛼* from the paper
        float ringAlpha;
        /// β* from the paper
        float ringBeta;
        float bottleneckTime{};
        unsigned int numGroups;

        ARArgs(const float& _alpha, const float& _beta,
               const unsigned int& _n){
            ringAlpha = _alpha;
            ringBeta = _beta;
            numGroups = _n;
            setBottleneckTime();
        }

        __forceinline__
        void setBottleneckTime(){
            bottleneckTime = (numGroups == 0 )? 0 : ringAlpha + (ringBeta * (static_cast<float>(gradBuffer) / numGroups));
        }

        __forceinline__
        void refresh(const float& alpha, const float& beta){
            ringAlpha = alpha;
            ringBeta = beta;
            setBottleneckTime();
        }

        __forceinline__ static float bottleneck(const float& alpha,
                                                 const float& beta,
                                                 const unsigned int& buf,
                                                 const unsigned int& nG){
            return (nG == 0) ? 0 : (alpha + (beta * (static_cast<float>(buf) / nG)));
        }
    };

    struct ObjArgs{
        constexpr static unsigned int globalMoEStages = ACC::GMS::value;
        /// eta in the paper
        constexpr static unsigned int commFreq = 4U;
        /// Units is MB
        constexpr static unsigned int p2pBuffer = ACC::P2PB::value;
        float totalDeviceRate{};
        unsigned int totalExpertCost;
        unsigned int totalExpertMemoryDemand;
        float allReduceTime;
        unsigned int groupMemCapacity{};
        float intraCommunicationCost;
        unsigned int effectiveWorld;

        ObjArgs(const unsigned int& _totalCost,
                const unsigned int& _effW, const unsigned int& _totalMem) :
                totalExpertCost(_totalCost),
                totalExpertMemoryDemand(_totalMem), effectiveWorld(_effW){
            allReduceTime = 0.0f; // default, in case of inference where this is indeed 0
            intraCommunicationCost = 0.0;
        }

        __forceinline__ static float p2pTransferTime(const float& alpha,
                                                      const float& beta,
                                                      const float& bufferSize){
            return alpha + (beta * bufferSize);
        }

        /// 𝜸 from the paper
        __forceinline__
        static unsigned int getGamma(const unsigned int& globalMoEStages,
                                     const unsigned int& effectiveWorld){
            return globalMoEStages / effectiveWorld;
        }
    };
}
#endif //CSRC_ARGS_CUH
