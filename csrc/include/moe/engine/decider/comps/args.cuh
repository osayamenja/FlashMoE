//
// Created by osayamen on 9/10/24.
//

#ifndef CSRC_ARGS_CUH
#define CSRC_ARGS_CUH
namespace aristos{
    struct ARArgs{
        /// 𝜸 from the paper
        unsigned int moeObjWeight;
        /// 𝛼* from the paper
        float ringAlpha;
        /// β* from the paper
        float ringBeta;
        float bottleneckTime;
        unsigned int bufferSize;
    };

    struct ObjArgs{
        unsigned long totalDeviceRate;
        const unsigned long totalExpertCost;
        const unsigned int commFreq;
        /// ω in the paper
        unsigned int globalMoEStages;
        unsigned int numGroups;
        const unsigned int numExperts;
        unsigned int allReduceTime;
        unsigned int effectiveWorld;
        unsigned int groupMemCapacity;
        double intraCommunicationCost;

        __forceinline__
        static unsigned int getGlobalMoEStages(const unsigned int& globalBatch,
                                               const unsigned int& numLayers,
                                               const unsigned int& moeFreq,
                                               const unsigned int& redAmount,
                                               const unsigned int& miniBatch){
            return redAmount * (globalBatch/miniBatch) * (numLayers/moeFreq);
        }

        __forceinline__
        static unsigned int getGamma(const unsigned int& globalMoEStages,
                                     const unsigned int& effectiveWorld){
            return globalMoEStages / effectiveWorld;
        }
    };
}
#endif //CSRC_ARGS_CUH
