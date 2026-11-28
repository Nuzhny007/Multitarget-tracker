#pragma once

///
/// \brief Umbrella header for the `mtracking` public API (2.0).
///
/// The public types are defined in `namespace mtracking`; global aliases keep
/// existing code compiling. This header also exposes `mtracking::tracking` for
/// the enums (the enums themselves stay in the global `tracking` namespace).
///
/// The detector factory is exposed as `mtracking::BaseDetector` by
/// `mdetection/BaseDetector.h`.
///

#include "defines.h"
#include "DetectorSettings.h"
#include "TrackerSettings.h"
#include "trajectory.h"
#include "BaseTracker.h"

namespace mtracking
{
namespace tracking = ::tracking;
}
