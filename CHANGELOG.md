# Changelog

## 2.0.0 (2026-11-28)

### Breaking

- The public types are now defined in `namespace mtracking`:
  `mtracking::BaseTracker`, `mtracking::BaseDetector`, `mtracking::TrackerSettings`,
  `mtracking::DetectorSettings`, `mtracking::TrackingObject`, `mtracking::Trace`,
  `mtracking::TrajectoryPoint`, `mtracking::CRegion`, `mtracking::TrackID`,
  `mtracking::track_id_t`, `mtracking::objtype_t`, `mtracking::bad_type`,
  `mtracking::TypeConverter`, `mtracking::ParseTrackerSettings`.
- Global aliases with the old names are kept for source compatibility and are
  deprecated; they will be removed in 3.0. New code should include
  `mtracking/Mtracking.h` (and `mdetection/BaseDetector.h`) and use the
  `mtracking::` names.
- Explicit `MT_API` exports (`dllexport`/`dllimport`); `MT_SHARED_EXPORTS` is
  defined by the library targets and `MT_SHARED` is propagated to consumers.
  `CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS` is no longer used.

### Added

- `DetectorSettings`: detector configuration split from `TrackerSettings`
  (`TrackerSettings` inherits it, so `settings.m_nnWeights`-style access keeps
  working).
- `[meta] schema_version` (default 1) for the settings INI and the
  `kTrackerSettingsSchemaVersion` constant.
- Detector/tracker factories: `BaseDetector::RegisterDetector` /
  `BaseTracker::RegisterTracker` (registry instead of a hard-coded switch).
- `mtracking::tracking` namespace alias for the enums.
- Shared helpers `src/util` (`mtutil::CreateLogger`, `mtutil::DefaultPalette`,
  `Frame`), used by the `example` and `async_detector` apps.
- ReID (roadmap M3): per-type model registry (`[reid]` / `[reid_<name>]`),
  person OSNet + vehicle ResNet50, per-model preprocessing and `modelId`,
  batched OpenCV DNN inference on CUDA with CPU fallback budget, and a bounded
  per-track appearance gallery with quality gating.

### Changed

- Only the public headers are installed; internal headers (`track.h`,
  `Kalman.h`, `byte_track/*`, `MotionDetector.h`, `BackgroundSubtract.h`,
  TensorRT parser headers, ...) are no longer installed.
- `BackgroundSubtract::Init` returns `false` when the requested algorithm is
  not available in the build (the fallback still sets up VIBE, but the error is
  now visible to the caller).
- `TrackerSettings` no longer defaults `m_detectorBackend` to TensorRT; the
  default is `DNN_OCV` (H21).

### Fixed

- M6–M15 (except M3), M16–M19, M46, M47 — see `docs/roadmap-archive.md`.
