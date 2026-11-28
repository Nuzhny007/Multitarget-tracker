# Changelog

## 1.2.0 (2026-11-28)

### Added

- `DetectorSettings`: detector configuration split from `TrackerSettings`
  (`TrackerSettings` inherits it, so `settings.m_nnWeights`-style access keeps
  working).
- `[meta] schema_version` (default 1) for the settings INI and the
  `kTrackerSettingsSchemaVersion` constant.
- Detector/tracker factories: `BaseDetector::RegisterDetector` /
  `BaseTracker::RegisterTracker` (registry instead of a hard-coded switch).
- `MT_API` decoration in `mtracking/Export.h` and a public/installed header
  boundary.
- `mtracking` namespace facade (`mtracking/Mtracking.h`): the public types are
  also available as `mtracking::<Type>`; `mdetection/BaseDetector.h` adds
  `mtracking::BaseDetector`.
- Shared helpers `src/util` (`mtutil::CreateLogger`, `mtutil::DefaultPalette`,
  `Frame`), used by the `example` and `async_detector` apps.

### Changed

- Only the public headers are installed; internal headers (`track.h`,
  `Kalman.h`, `byte_track/*`, `MotionDetector.h`, `BackgroundSubtract.h`,
  TensorRT parser headers, ...) are no longer installed.
- `BackgroundSubtract::Init` returns `false` when the requested algorithm is
  not available in the build (the fallback still sets up VIBE, but the error is
  now visible to the caller).
- `TrackerSettings` no longer defaults `m_detectorBackend` to TensorRT; the
  default is `DNN_OCV` (H21).

### Deprecated

- The global-namespace public types are kept for source compatibility; the
  definitions move into `namespace mtracking` in 2.0. New code should prefer
  `#include "mtracking/Mtracking.h"` and the `mtracking::` names.

### Fixed

- M6–M15 (except M3), M16–M19, M46, M47 — see `docs/roadmap-archive.md`.
