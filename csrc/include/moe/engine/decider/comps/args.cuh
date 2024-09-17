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
        double ringAlpha;
        /// β* from the paper
        double ringBeta;
        double bottleneckTime{};
        const unsigned int bufferSize;
        unsigned int numGroups;

        ARArgs(const double& _alpha, const double& _beta,
               const unsigned int& _n): bufferSize(hostMoEConfig.gradBuffer){
            ringAlpha = _alpha;
            ringBeta = _beta;
            numGroups = _n;
            setBottleneckTime();
        }

        __forceinline__
        void setBottleneckTime(){
            bottleneckTime = (numGroups == 0 )? 0 : ringAlpha + (ringBeta * (static_cast<double>(bufferSize) / numGroups));
        }

        __forceinline__
        void refresh(const double& alpha, const double& beta){
            ringAlpha = alpha;
            ringBeta = beta;
            setBottleneckTime();
        }

        __forceinline__ static double bottleneck(const double& alpha,
                                                 const double& beta,
                                                 const unsigned int& buf,
                                                 const unsigned int& nG){
            return (nG == 0) ? 0 : (alpha + (beta * (static_cast<double>(buf) / nG)));
        }
    };

    struct ObjArgs{
        unsigned long totalDeviceRate{};
        unsigned long totalExpertCost;
        unsigned int globalMoEStages;
        unsigned int totalExpertMemoryDemand;
        double allReduceTime{};
        unsigned int commFreq;
        unsigned int groupMemCapacity{};
        double intraCommunicationCost;
        unsigned int effectiveWorld;
        unsigned int p2pBuffer;

        ObjArgs(const unsigned long& _totalCost,
                const unsigned int& _effW, const unsigned int& _totalMem) :
                totalExpertCost(_totalCost),
                totalExpertMemoryDemand(_totalMem), effectiveWorld(_effW){
            globalMoEStages = getGlobalMoEStages();
            intraCommunicationCost = 0.0;
            p2pBuffer = hostMoEConfig.p2pBuffer;
            commFreq = p2pFreq;
        }

        __forceinline__ static double p2pTransferTime(const double& alpha,
                                                      const double& beta,
                                                      const double& bufferSize){
            return alpha + (beta * bufferSize);
        }

        __forceinline__
        static unsigned int getGlobalMoEStages(){
            return (2 + hostMoEConfig.redAmount) *
            (hostMoEConfig.globalBatch/hostMoEConfig.miniBatch)
            * (hostMoEConfig.numLayers/hostMoEConfig.moeFreq);
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
