//
// Created by oja7 on 9/19/24.
//

#ifndef CONFIG_CUH
#define CONFIG_CUH

#define MIN_BLOCKS_PER_SM 16 // may be a bit too aggressive

#if __CUDA_ARCH__ == 700
    #define MIN_BLOCKS_PER_SM 12
#endif
#define ARISTOS_LAUNCH_BOUNDS __launch_bounds__(blockSize, MIN_BLOCKS_PER_SM)
#endif //CONFIG_CUH
