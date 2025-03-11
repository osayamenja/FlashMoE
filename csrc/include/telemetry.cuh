//
// Created by oja7 on 3/11/25.
//

#ifndef TELEMETRY_CUH
#define TELEMETRY_CUH
#include <nvtx3/nvtx3.hpp>
namespace aristos {
    struct aristosDomain {
        static constexpr auto const* name{"Aristos"};
    };
    using aristosInitRange = nvtx3::scoped_range_in<aristosDomain>;
}
#endif //TELEMETRY_CUH
