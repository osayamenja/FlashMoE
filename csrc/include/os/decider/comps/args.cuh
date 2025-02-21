//
// Created by osayamen on 9/10/24.
//

#ifndef CSRC_ARGS_CUH
#define CSRC_ARGS_CUH
namespace aristos{
    /// eta in the paper
    constexpr unsigned int p2pFreq = 4;
    struct ARArgs{
        /// 𝛼* from the paper
        float ringAlpha;
        /// β* from the paper
        float ringBeta;
        float bottleneckTime{};
        const unsigned int bufferSize;
        unsigned int numGroups;

        ARArgs(const float& _alpha, const float& _beta,
               const unsigned int& _n, const unsigned int& gradBuffer): bufferSize(gradBuffer){
            ringAlpha = _alpha;
            ringBeta = _beta;
            numGroups = _n;
            setBottleneckTime();
        }

        __forceinline__
        void setBottleneckTime(){
            bottleneckTime = (numGroups == 0 )? 0 : ringAlpha + (ringBeta * (static_cast<float>(bufferSize) / numGroups));
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
        float totalDeviceRate{};
        unsigned int totalExpertCost;
        unsigned int globalMoEStages;
        unsigned int totalExpertMemoryDemand;
        float allReduceTime{};
        unsigned int commFreq;
        unsigned int groupMemCapacity{};
        float intraCommunicationCost;
        unsigned int effectiveWorld;
        unsigned int p2pBuffer;

        ObjArgs(const unsigned int& _totalCost,
                const unsigned int& _effW, const unsigned int& _totalMem,
                const ModelConfig& m) :
                totalExpertCost(_totalCost),
                totalExpertMemoryDemand(_totalMem), effectiveWorld(_effW){
            globalMoEStages = getGlobalMoEStages(m);
            intraCommunicationCost = 0.0;
            p2pBuffer = m.p2pBuffer;
            commFreq = p2pFreq;
        }

        __forceinline__ static float p2pTransferTime(const float& alpha,
                                                      const float& beta,
                                                      const float& bufferSize){
            return alpha + (beta * bufferSize);
        }

        __forceinline__
        static unsigned int getGlobalMoEStages(const ModelConfig& m){
            return (2 + m.redAmount) *
            (m.globalBatch/m.miniBatch)
            * (m.numLayers/m.moeFreq);
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
