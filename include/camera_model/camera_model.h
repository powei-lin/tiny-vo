#pragma once
#include "extended_unified_camera.hpp"
#include "unified_camera.hpp"
#include <variant>

namespace argus {
using camera_model_f =
    std::variant<UnifiedCamera<float>, ExtendedUnifiedCamera<float>>;

using camera_model_d =
    std::variant<UnifiedCamera<double>, ExtendedUnifiedCamera<double>>;

} // namespace argus
