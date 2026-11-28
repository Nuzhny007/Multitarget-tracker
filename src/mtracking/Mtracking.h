#pragma once

///
/// \brief Umbrella header for the `mtracking` namespace facade.
///
/// The public types are still defined in the global namespace for source
/// compatibility; this header exposes them as `mtracking::<Type>` so new code
/// can use the namespace ahead of the 2.0 move of the definitions
/// (roadmap M2, phase 6).
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

using ::BaseTracker;
using ::CRegion;
using ::DetectorSettings;
using ::ParseTrackerSettings;
using ::TrackID;
using ::TrackerSettings;
using ::TrackingObject;
using ::Trace;
using ::TrajectoryPoint;
using ::TypeConverter;
using ::bad_type;
using ::objtype_t;
using ::track_id_t;
} // namespace mtracking
