"""Type stubs for the :mod:`pymtracking` pybind11 extension module.

Frames cross the C++/Python boundary as ``numpy.ndarray`` (``uint8`` BGR
``H x W x C`` or ``H x W``, C-contiguous). The module copies the array
data in both directions: a returned array never aliases C++-owned memory,
and an array passed to a method does not need to outlive the call.
"""

from typing import ClassVar, List, Sequence, Tuple

import numpy as np

__version__: str


class KeyVal:
    """Key/value multimap used to configure detectors and the tracker."""

    def __init__(self) -> None: ...
    def Add(self, key: str, value: str) -> None:
        """Add one key/value pair (INI-style names, e.g. 'history')."""
        ...


class MapStringString(dict): ...


class MapStringDouble(dict): ...


class CvRect:
    """Axis-aligned rectangle: x, y, width, height."""

    x: int
    y: int
    width: int
    height: int
    def __init__(self) -> None: ...


class CvRRect:
    """Rotated rectangle; use :meth:`brect` for its bounding box."""

    def __init__(self) -> None: ...
    def brect(self) -> CvRect:
        """Return the axis-aligned bounding rectangle."""
        ...


class CRegion:
    """A detection region: bounding rect, optional type and confidence."""

    brect: CvRect
    type: int
    confidence: float
    def __init__(self, rect: CvRect = ...) -> None: ...


class TrackingObject:
    """One tracked object: ID, geometry, velocity, type and state flags."""

    rrect: CvRRect
    isStatic: bool
    staticTime: int
    outOfTheFrame: bool
    type: int
    confidence: float
    speedMps: float

    def __init__(self) -> None: ...
    def IsRobust(
        self,
        minTraceSize: int,
        minRawRatio: float,
        sizeRatio: Tuple[float, float],
        lastDetectsCount: int = ...,
    ) -> bool:
        """Return True if the track is confirmed."""
        ...

    def GetTrajectory(self) -> List[Tuple[int, int]]:
        """Return the trajectory as a list of (x, y) integer points, oldest first."""
        ...

    def GetBoundingRect(self) -> CvRect:
        """Return the axis-aligned bounding rectangle of the object."""
        ...

    @property
    def ID(self) -> int:
        """Unique track ID."""
        ...

    @property
    def velocity(self) -> Tuple[float, float]:
        """Velocity as (vx, vy) in pixels per second."""
        ...


class TrackerSettings:
    """Tracker configuration; distances are weights that must sum to 1.0."""

    kalmanType: "MTracker.KalmanType"
    filterGoal: "MTracker.FilterGoal"
    lostTrackType: "MTracker.LostTrackType"
    matchType: "MTracker.MatchType"
    dt: float
    accelNoiseMag: float
    useAcceleration: bool
    distThres: float
    distThresLocalCartesian: float
    minAreaRadiusPix: float
    minAreaRadiusK: float
    maximumAllowedLostTime: float
    maxTraceLength: float
    useAbandonedDetection: bool
    minStaticTime: int
    maxStaticTime: int

    def __init__(self) -> None: ...
    def CheckDistance(self) -> bool:
        """Return True when the distance weights sum to 1.0."""
        ...

    def SetDistances(self, distances: Sequence[float]) -> bool:
        """Set all distance weights at once (DistsCount values, sum must be 1.0)."""
        ...

    def SetDistance(self, distType: "MTracker.DistType") -> bool:
        """Set one distance weight to 1.0 and the others to 0."""
        ...

    def AddNearTypes(self, type1: int, type2: int, sym: bool) -> None:
        """Mark two object types as compatible during matching."""
        ...

    def CheckType(self, type1: int, type2: int) -> bool:
        """Return True when the two object types may match."""
        ...


class MTracker:
    """Multiple-object tracker: Update() per frame, then GetTracks()."""

    class DistType:
        DistCenters: ClassVar["MTracker.DistType"]
        DistRects: ClassVar["MTracker.DistType"]
        DistJaccard: ClassVar["MTracker.DistType"]
        DistHist: ClassVar["MTracker.DistType"]
        DistFeatureCos: ClassVar["MTracker.DistType"]
        DistMahalanobis: ClassVar["MTracker.DistType"]
        DistLocalCartesian: ClassVar["MTracker.DistType"]

    class FilterGoal:
        FilterCenter: ClassVar["MTracker.FilterGoal"]
        FilterRect: ClassVar["MTracker.FilterGoal"]
        FilterRRect: ClassVar["MTracker.FilterGoal"]

    class KalmanType:
        KalmanLinear: ClassVar["MTracker.KalmanType"]
        KalmanUnscented: ClassVar["MTracker.KalmanType"]
        KalmanAugmentedUnscented: ClassVar["MTracker.KalmanType"]

    class MatchType:
        MatchHungrian: ClassVar["MTracker.MatchType"]
        MatchLAPJV: ClassVar["MTracker.MatchType"]

    class LostTrackType:
        TrackNone: ClassVar["MTracker.LostTrackType"]
        TrackKCF: ClassVar["MTracker.LostTrackType"]
        TrackCSRT: ClassVar["MTracker.LostTrackType"]
        TrackDaSiamRPN: ClassVar["MTracker.LostTrackType"]
        TrackNano: ClassVar["MTracker.LostTrackType"]
        TrackVit: ClassVar["MTracker.LostTrackType"]

    def __init__(self, settings: TrackerSettings, fps: float) -> None: ...
    def Update(
        self,
        regions: List[CRegion],
        frame: np.ndarray,
        timestamp: float = ...,
    ) -> None:
        """Update with the detections of one frame (timestamp in seconds)."""
        ...

    def GetTracks(self) -> List[TrackingObject]:
        """Return the list of current tracked objects (a copy)."""
        ...


class BaseDetector:
    """Base class for detectors; availability depends on the build flags."""

    class Detectors:
        VIBE: ClassVar["BaseDetector.Detectors"]
        MOG: ClassVar["BaseDetector.Detectors"]
        GMG: ClassVar["BaseDetector.Detectors"]
        CNT: ClassVar["BaseDetector.Detectors"]
        MOG2: ClassVar["BaseDetector.Detectors"]
        Yolo_TensorRT: ClassVar["BaseDetector.Detectors"]
        DNN_OCV: ClassVar["BaseDetector.Detectors"]

    def __init__(
        self, detectorType: "BaseDetector.Detectors", config: KeyVal, frame: np.ndarray
    ) -> None: ...
    def Init(self, config: KeyVal) -> bool:
        """Re-initialise the detector; returns True on success."""
        ...

    def Detect(self, frame: np.ndarray) -> None:
        """Run detection on one frame; read results with GetDetects()."""
        ...

    def ResetModel(self, frame: np.ndarray, roi: CvRect) -> None:
        """Reset the model inside the given region."""
        ...

    def CanGrayProcessing(self) -> bool: ...
    def SetMinObjectSize(self, minObjectSize: Tuple[int, int]) -> None: ...
    def GetDetects(self) -> List[CRegion]: ...
    def CalcMotionMap(self, frame: np.ndarray, drawOnlyMasks: bool = ...) -> None: ...


class MotionDetector(BaseDetector):
    """Background-subtraction motion detector."""

    class BGFG_ALGS:
        VIBE: ClassVar["MotionDetector.BGFG_ALGS"]
        MOG: ClassVar["MotionDetector.BGFG_ALGS"]
        GMG: ClassVar["MotionDetector.BGFG_ALGS"]
        CNT: ClassVar["MotionDetector.BGFG_ALGS"]
        MOG2: ClassVar["MotionDetector.BGFG_ALGS"]

    def __init__(self, algType: "MotionDetector.BGFG_ALGS", frame: np.ndarray) -> None: ...
    def Init(self, config: KeyVal) -> bool: ...
    def Detect(self, frame: np.ndarray) -> None: ...
    def SetMinObjectSize(self, minObjectSize: Tuple[int, int]) -> None: ...
    def GetDetects(self) -> List[CRegion]: ...
    def CalcMotionMap(self, frame: np.ndarray, drawOnlyMasks: bool = ...) -> None: ...


class TrackerTemplate:
    """Tracker implementation selected by MTracker."""

    UniversalTracker: ClassVar["TrackerTemplate"]
    ByteTrack: ClassVar["TrackerTemplate"]


# export_values() also exposes these names at module level.
UniversalTracker: TrackerTemplate
ByteTrack: TrackerTemplate


def read_image(image: str) -> np.ndarray:
    """Read an image from a file (BGR uint8)."""
    ...


def show_image(image: np.ndarray) -> None:
    """Show an image in a window (no-op when SILENT_WORK is set)."""
    ...


def passthru(image: np.ndarray) -> np.ndarray:
    """Return the same image (round-trip check)."""
    ...


def clone(image: np.ndarray) -> np.ndarray:
    """Return a deep copy of the image."""
    ...
