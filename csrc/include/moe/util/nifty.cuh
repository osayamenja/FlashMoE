//
// Created by Jonathan on 8/17/24.
//

#ifndef CSRC_NIFTY_CUH
#define CSRC_NIFTY_CUH

#include <cuda/annotated_ptr>

namespace aristos{
    /// Annotate some hot global memory pointers to persist in cache
    CUTE_DEVICE
    void persistHotPointers(){
        /// Persist global stop flag
        cuda::associate_access_property(&stop, cuda::access_property::persisting{});
        /// Persist sequence number
        cuda::associate_access_property(&sequenceNumber, cuda::access_property::persisting{});
        /// Persist symmetric heap flags
        cuda::associate_access_property(moeConfig.flags,
                                        cuda::access_property(moeConfig.flags,
                                                              moeConfig.worldSize*sizeof(flagsType),
                                                              moeConfig.worldSize*sizeof(flagsType),
                                                              cuda::access_property::persisting{}));
        /// Persist book keeping state
        cuda::associate_access_property(moeConfig.bookKeeping,
                                        cuda::access_property(moeConfig.bookKeeping,
                                                              sizeof(specType)*(moeConfig.worldSize * (moeConfig.numLocalExperts + 1)),
                                                              sizeof(specType)*(moeConfig.worldSize * (moeConfig.numLocalExperts + 1)),
                                                              cuda::access_property::persisting{}));
    }
}
#endif //CSRC_NIFTY_CUH
