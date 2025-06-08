/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Kleos Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 */

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
