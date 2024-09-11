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
        unsigned int numGroups;

        [[nodiscard]] double getBottleneckTime() const{
            return (numGroups == 0 )? 0 : ringAlpha + (ringBeta * (static_cast<double>(bufferSize) / numGroups));
        }
    };

    struct ObjArgs{
        unsigned long totalDeviceRate;
        const unsigned long totalExpertCost;
        /// eta in the paper
        const unsigned int commFreq;
        unsigned int globalMoEStages;
        const unsigned int numExperts;
        unsigned int allReduceTime;
        unsigned int groupMemCapacity;
        double intraCommunicationCost;
        unsigned int effectiveWorld;
        unsigned int p2pBuffer;

        __forceinline__ static double p2pTransferTime(const double& alpha,
                                                      const double& beta,
                                                      const unsigned int& bufferSize){
            return alpha + (beta * bufferSize);
        }

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
