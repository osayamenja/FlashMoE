//
// Created by oja7 on 3/11/25.
//

#ifndef TELEMETRY_CUH
#define TELEMETRY_CUH
#include <nvtx3/nvtx3.hpp>
namespace kleos {
    struct kleosDomain {
        static constexpr auto const* name{"Kleos"};
    };
    using kleosRange = nvtx3::scoped_range_in<kleosDomain>;
}
#endif //TELEMETRY_CUH
