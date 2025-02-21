//
// Created by oja7 on 2/20/25.
//

#ifndef CONFIG_CUH
#define CONFIG_CUH
#include <cute/numeric/integral_constant.hpp>
#include "types.cuh"
namespace aristos {
    /// Aristos Compile-time Config
    struct ACC {
        using GRL = cute::C<NUM_EXPERTS <= BLOCK_N ? GateReductionLevel::singleBlock :
            GateReductionLevel::multiBlock>;
        using CM = cute::C<(E_TOP_K > 1) ? CombineMode::multithreaded : CombineMode::single>;
        using Element = VCT<CM::value, DType<DTYPE>::DT>::Element;
        using DTK = cute::C<DROP_TOKENS? DropTokens::yes : DropTokens::no>;
        using ActivationOp = AFunction<HIDDEN_ACT, GEA>::DT;
        using ActivationOpX = cute::identity;
        using PeakHardware = aristos::Hardware<ARISTOS_ARCH, 255>;
        using S = cute::C<SEQ_LEN>;
        using P = cute::C<I_SIZE>;
        using H = cute::C<HIDDEN_SIZE>;
        using E = cute::C<NUM_EXPERTS>;
    };
}
#endif //CONFIG_CUH
