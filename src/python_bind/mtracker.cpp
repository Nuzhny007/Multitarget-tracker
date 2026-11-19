#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <pybind11/stl_bind.h>

// pybind11 2.5 expects a global ssize_t which MSVC does not define.
#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

#include <pybind11/numpy.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "../mtracking/defines.h"
#include "../Tracker/BaseTracker.h"
#include "../Detector/BaseDetector.h"
#include "../Detector/MotionDetector.h"

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::multimap<std::string, std::string>)
PYBIND11_MAKE_OPAQUE(std::map<std::string, std::string>)
PYBIND11_MAKE_OPAQUE(std::map<std::string, double>)

namespace pybind11 { namespace detail
{
template <>
struct type_caster<cv::Mat>
{
public:
    PYBIND11_TYPE_CASTER(cv::Mat, _("numpy.ndarray"));

    // numpy.ndarray -> cv::Mat (C-contiguous arrays only)
    bool load(handle obj, bool)
    {
        array b = reinterpret_borrow<array>(obj);
        buffer_info info = b.request();

        if ((info.ndim != 2 && info.ndim != 3) ||
            info.shape[0] < 0 || info.shape[1] < 0 ||
            (info.ndim == 3 && info.shape[2] < 0))
        {
            throw std::logic_error("Unsupported ndarray shape: expected 2D or 3D array");
        }
        if (!(b.flags() & array::c_style))
        {
            throw std::logic_error("Only C-contiguous numpy arrays are supported");
        }

        const int nh = static_cast<int>(info.shape[0]);
        const int nw = static_cast<int>(info.shape[1]);
        const int nc = (info.ndim == 3) ? static_cast<int>(info.shape[2]) : 1;
        if (nc <= 0 || nc > 4)
        {
            throw std::logic_error("Unsupported number of channels");
        }

        int dtype = -1;
        if (info.format == format_descriptor<unsigned char>::format())
            dtype = CV_8UC(nc);
        else if (info.format == format_descriptor<int>::format())
            dtype = CV_32SC(nc);
        else if (info.format == format_descriptor<float>::format())
            dtype = CV_32FC(nc);
        else
            throw std::logic_error("Unsupported ndarray type: only uint8, int32 and float32 are supported");

        // numpy owns the buffer, not the Mat: clone so the cv::Mat owns its
        // own data instead of wrapping an array that Python may release right
        // after the call (a Mat that only borrowed info.ptr used to dangle in
        // the trackers/detectors that keep a ROI alive). Roadmap M2.
        value = cv::Mat(nh, nw, dtype, info.ptr).clone();
        return true;
    }

    // cv::Mat -> numpy.ndarray (copy, so the array owns its data)
    static handle cast(const cv::Mat& mat, return_value_policy, handle)
    {
        if (mat.dims != 2)
        {
            throw std::logic_error("Only 2D cv::Mat can be converted to numpy.ndarray");
        }

        std::string format;
        switch (mat.depth())
        {
        case CV_8U:
            format = format_descriptor<unsigned char>::format();
            break;
        case CV_32S:
            format = format_descriptor<int>::format();
            break;
        case CV_32F:
            format = format_descriptor<float>::format();
            break;
        default:
            throw std::logic_error("Unsupported cv::Mat depth: only uint8, int32 and float32 are supported");
        }

        const int nc = mat.channels();
        std::vector<py::ssize_t> shape;
        if (nc == 1)
            shape = { mat.rows, mat.cols };
        else
            shape = { mat.rows, mat.cols, nc };

        py::array result(py::dtype(format), shape);
        if (mat.rows > 0 && mat.cols > 0)
        {
            char* dst = static_cast<char*>(result.mutable_data());
            const size_t rowBytes = static_cast<size_t>(mat.cols) * mat.elemSize();
            for (int y = 0; y < mat.rows; ++y)
            {
                std::memcpy(dst + static_cast<size_t>(y) * rowBytes, mat.ptr(y), rowBytes);
            }
        }
        return result.release();
    }
};

template <typename T>
struct type_caster<cv::Size_<T>>
{
    PYBIND11_TYPE_CASTER(cv::Size_<T>, _("tuple_wh"));

    bool load(handle obj, bool)
    {
        if (!py::isinstance<py::tuple>(obj))
        {
            throw std::logic_error("Size(w, h) should be a tuple");
        }

        py::tuple pt = reinterpret_borrow<py::tuple>(obj);
        if (pt.size() != 2)
        {
            throw std::logic_error("Size(w, h) tuple should be size of 2");
        }

        value = cv::Size_<T>(static_cast<T>(pt[0].cast<T>()), static_cast<T>(pt[1].cast<T>()));
        return true;
    }

    static handle cast(const cv::Size_<T>& sz, return_value_policy, handle)
    {
        return py::make_tuple(sz.width, sz.height).release();
    }
};

template <>
struct type_caster<cv::Point>
{
    PYBIND11_TYPE_CASTER(cv::Point, _("tuple_xy"));

    bool load(handle obj, bool)
    {
        if (!py::isinstance<py::tuple>(obj))
        {
            throw std::logic_error("Point(x, y) should be a tuple");
        }

        py::tuple pt = reinterpret_borrow<py::tuple>(obj);
        if (pt.size() != 2)
        {
            throw std::logic_error("Point(x, y) tuple should be size of 2");
        }

        value = cv::Point(pt[0].cast<int>(), pt[1].cast<int>());
        return true;
    }

    static handle cast(const cv::Point& pt, return_value_policy, handle)
    {
        return py::make_tuple(pt.x, pt.y).release();
    }
};
}} // namespace pybind11::detail

namespace
{
config_t ToConfig(const KeyVal& kv)
{
    config_t config;
    for (const auto& item : kv.m_config)
        config.emplace(item.first, item.second);
    return config;
}

time_point_t ToTimePoint(double seconds)
{
    return time_point_t(std::chrono::duration_cast<time_point_t::duration>(
        std::chrono::duration<double>(seconds)));
}
} // namespace

///
cv::Mat read_image(std::string image_name)
{
    return cv::imread(image_name, cv::IMREAD_COLOR);
}

///
void show_image(cv::Mat image)
{
#ifndef SILENT_WORK
    cv::imshow("image_from_Cpp", image);
    cv::waitKey(0);
#endif
}

///
cv::Mat passthru(cv::Mat image)
{
    return image;
}

///
cv::Mat cloneimg(cv::Mat image)
{
    return image.clone();
}

///
PYBIND11_MODULE(pymtracking, m)
{
    m.doc() = "Multitarget-tracker (pymtracking): multiple-object tracking and detection. "
              "Frames are numpy.ndarray (uint8 BGR HxWxC or HxW, C-contiguous). The module "
              "copies frame data on both the numpy -> C++ and C++ -> numpy boundaries, so no "
              "array aliases memory owned by the other side. See pymtracking.pyi for type hints.";

    py::class_<KeyVal>(m, "KeyVal",
            "Key/value multimap used to configure detectors and the tracker. "
            "Pass it to BaseDetector(...) or BaseDetector.Init(...).")
            .def(py::init<>(), "Create an empty key/value config.")
            .def("Add", &KeyVal::Add, py::arg("key"), py::arg("value"),
                 "Add one key/value pair (INI-style names, e.g. 'history', 'nmixtures').");

    py::bind_map<std::multimap<std::string, std::string>>(m, "MapStringString");
    py::bind_map<std::map<std::string, double>>(m, "MapStringDouble");

    py::class_<TrackerSettings>(m, "TrackerSettings",
            "Tracker configuration. Defaults are the library defaults; all fields are "
            "read/write. Distances are weights that must sum to 1.0 (see CheckDistance).")
            .def(py::init<>(), "Create settings filled with the library defaults.")
            .def("CheckDistance", &TrackerSettings::CheckDistance,
                 "Return True when the distance weights sum to 1.0.")
            .def("SetDistances", [](TrackerSettings& settings, const std::vector<track_t>& distances)
            {
                if (distances.size() != static_cast<size_t>(tracking::DistsCount))
                    throw std::runtime_error("SetDistances expects " + std::to_string(tracking::DistsCount) + " values");

                std::array<track_t, tracking::DistsCount> values{};
                std::copy(distances.begin(), distances.end(), values.begin());
                return settings.SetDistances(values);
            }, py::arg("distances"),
               "Set all distance weights at once (DistsCount values; the sum must be 1.0).")
            .def("SetDistance", &TrackerSettings::SetDistance, py::arg("distType"),
                 "Set one distance weight to 1.0 and all the others to 0.")
            .def("AddNearTypes", &TrackerSettings::AddNearTypes,
                 py::arg("type1"), py::arg("type2"), py::arg("sym"),
                 "Mark two object types as compatible during matching (sym - add both directions).")
            .def("CheckType", &TrackerSettings::CheckType, py::arg("type1"), py::arg("type2"),
                 "Return True when the two object types may match.")
            .def_readwrite("kalmanType", &TrackerSettings::m_kalmanType)
            .def_readwrite("filterGoal", &TrackerSettings::m_filterGoal)
            .def_readwrite("lostTrackType", &TrackerSettings::m_lostTrackType)
            .def_readwrite("matchType", &TrackerSettings::m_matchType)
            .def_readwrite("dt", &TrackerSettings::m_dt)
            .def_readwrite("accelNoiseMag", &TrackerSettings::m_accelNoiseMag)
            .def_readwrite("useAcceleration", &TrackerSettings::m_useAcceleration)
            .def_readwrite("distThres", &TrackerSettings::m_distThres)
            .def_readwrite("distThresLocalCartesian", &TrackerSettings::m_distThresLocalCartesian)
            .def_readwrite("minAreaRadiusPix", &TrackerSettings::m_minAreaRadiusPix)
            .def_readwrite("minAreaRadiusK", &TrackerSettings::m_minAreaRadiusK)
            .def_readwrite("maximumAllowedLostTime", &TrackerSettings::m_maximumAllowedLostTime)
            .def_readwrite("maxTraceLength", &TrackerSettings::m_maxTraceLength)
            .def_readwrite("useAbandonedDetection", &TrackerSettings::m_useAbandonedDetection)
            .def_readwrite("minStaticTime", &TrackerSettings::m_minStaticTime)
            .def_readwrite("maxStaticTime", &TrackerSettings::m_maxStaticTime);

    py::enum_<tracking::TrackerTemplate>(m, "TrackerTemplate",
            "Tracker implementation selected by MTracker; UniversalTracker is the default.")
            .value("UniversalTracker", tracking::TrackerTemplate::UniversalTracker, "Kalman-based universal tracker.")
            .value("ByteTrack", tracking::TrackerTemplate::ByteTrack, "BYTETracker implementation.")
            .export_values();

    py::class_<cv::Rect>(m, "CvRect",
            "Axis-aligned rectangle (cv::Rect): x, y, width, height.")
            .def(py::init<>(), "Create a zero rectangle.")
            .def_readwrite("x", &cv::Rect::x)
            .def_readwrite("y", &cv::Rect::y)
            .def_readwrite("width", &cv::Rect::width)
            .def_readwrite("height", &cv::Rect::height);

    py::class_<cv::RotatedRect>(m, "CvRRect",
            "Rotated rectangle (cv::RotatedRect). Use brect() for its bounding box.")
            .def(py::init<>(), "Create an empty rotated rectangle.")
            .def("brect", &cv::RotatedRect::boundingRect, "Return the axis-aligned bounding rectangle.");

    py::class_<CRegion>(m, "CRegion",
            "A detection region: bounding rect, optional type and confidence.")
            .def(py::init<>(), "Create an empty region.")
            .def(py::init<const cv::Rect&>(), py::arg("rect"), "Create a region from an axis-aligned rect.")
            .def_readwrite("brect", &CRegion::m_brect)
            .def_readwrite("type", &CRegion::m_type)
            .def_readwrite("confidence", &CRegion::m_confidence);

    py::class_<TrackingObject>(m, "TrackingObject",
            "One tracked object: ID, geometry, velocity, type and static/out-of-frame flags.")
            .def(py::init<>(), "Create an empty tracking object.")
            .def("IsRobust", &TrackingObject::IsRobust,
                 py::arg("minTraceSize"), py::arg("minRawRatio"), py::arg("sizeRatio"),
                 py::arg("lastDetectsCount") = 0,
                 "Return True if the track is confirmed (enough raw detections and a plausible size ratio).")
            .def("GetTrajectory", &TrackingObject::GetTrajectory,
                 "Return the trajectory as a list of (x, y) integer points, oldest first.")
            .def("GetBoundingRect", &TrackingObject::GetBoundingRect,
                 "Return the axis-aligned bounding rectangle of the object.")
            .def_property_readonly("ID", [](const TrackingObject& object) { return object.m_ID.m_val; },
                 "Unique track ID (int).")
            .def_readwrite("rrect", &TrackingObject::m_rrect)
            .def_readwrite("isStatic", &TrackingObject::m_isStatic)
            .def_readwrite("staticTime", &TrackingObject::m_staticTime)
            .def_readwrite("outOfTheFrame", &TrackingObject::m_outOfTheFrame)
            .def_readwrite("type", &TrackingObject::m_type)
            .def_readwrite("confidence", &TrackingObject::m_confidence)
            .def_property_readonly("velocity", [](const TrackingObject& object)
            {
                return py::make_tuple(object.m_velocity[0], object.m_velocity[1]);
            }, "Velocity as (vx, vy) in pixels per second.")
            .def_readwrite("speedMps", &TrackingObject::m_speedMps);

    py::class_<BaseTracker> mtracker(m, "MTracker",
            "Multiple-object tracker. Construct with settings and fps, call Update() once "
            "per frame and read GetTracks() afterwards.");
    mtracker.def(py::init(&BaseTracker::CreateTracker),
                 py::arg("settings"), py::arg("fps"),
                 "Create a tracker for the given settings and frame rate (fps).");
    mtracker.def("Update", [](BaseTracker& tracker, const regions_t& regions, cv::Mat frame, double timestamp)
    {
        tracker.UpdateMat(regions, frame, ToTimePoint(timestamp));
    }, py::arg("regions"), py::arg("frame"), py::arg("timestamp"),
       "Update the tracker with the detections of one frame at the given timestamp (seconds).");
    mtracker.def("Update", [](BaseTracker& tracker, const regions_t& regions, cv::Mat frame)
    {
        tracker.UpdateMat(regions, frame, std::chrono::system_clock::now());
    }, py::arg("regions"), py::arg("frame"),
       "Update the tracker with the detections of one frame; the wall clock is used as timestamp.");
    mtracker.def("GetTracks", &BaseTracker::GetTracksCopy,
                 "Return the list of current tracked objects (a copy).");

    py::enum_<tracking::DistType>(mtracker, "DistType",
            "Distance metric weights for matching. Weights must sum to 1.0.")
            .value("DistCenters", tracking::DistType::DistCenters, "Distance between region centers.")
            .value("DistRects", tracking::DistType::DistRects, "Distance between rectangles.")
            .value("DistJaccard", tracking::DistType::DistJaccard, "Jaccard distance between rectangles.")
            .value("DistHist", tracking::DistType::DistHist, "Histogram distance.")
            .value("DistFeatureCos", tracking::DistType::DistFeatureCos, "Cosine distance between embeddings.")
            .value("DistMahalanobis", tracking::DistType::DistMahalanobis, "Mahalanobis distance.")
            .value("DistLocalCartesian", tracking::DistType::DistLocalCartesian, "Distance in the local ENU plane.")
            .export_values();

    py::enum_<tracking::FilterGoal>(mtracker, "FilterGoal",
            "What the Kalman filter tracks: center point, rect or rotated rect.")
            .value("FilterCenter", tracking::FilterGoal::FilterCenter, "Track the center point.")
            .value("FilterRect", tracking::FilterGoal::FilterRect, "Track the axis-aligned rect.")
            .value("FilterRRect", tracking::FilterGoal::FilterRRect, "Track the rotated rect (linear Kalman only).")
            .export_values();

    py::enum_<tracking::KalmanType>(mtracker, "KalmanType",
            "Kalman filter flavour.")
            .value("KalmanLinear", tracking::KalmanType::KalmanLinear, "Linear Kalman filter.")
            .value("KalmanUnscented", tracking::KalmanType::KalmanUnscented, "Unscented Kalman filter.")
            .value("KalmanAugmentedUnscented", tracking::KalmanType::KalmanAugmentedUnscented, "Augmented unscented Kalman filter.")
            .export_values();

    py::enum_<tracking::MatchType>(mtracker, "MatchType",
            "Assignment problem solver used for matching.")
            .value("MatchHungrian", tracking::MatchType::MatchHungrian, "Hungarian algorithm.")
            .value("MatchLAPJV", tracking::MatchType::MatchLAPJV, "LAPJV algorithm.")
            .export_values();

    py::enum_<tracking::LostTrackType>(mtracker, "LostTrackType",
            "Visual tracker used to follow lost tracks.")
            .value("TrackNone", tracking::LostTrackType::TrackNone, "Do not track lost objects.")
            .value("TrackKCF", tracking::LostTrackType::TrackKCF, "KCF tracker.")
            .value("TrackCSRT", tracking::LostTrackType::TrackCSRT, "CSRT tracker.")
            .value("TrackDaSiamRPN", tracking::LostTrackType::TrackDaSiamRPN, "DaSiamRPN tracker.")
            .value("TrackNano", tracking::LostTrackType::TrackNano, "Nano tracker.")
            .value("TrackVit", tracking::LostTrackType::TrackVit, "ViT tracker.")
            .export_values();

    py::class_<BaseDetector> baseDetector(m, "BaseDetector",
            "Base class for detectors. Create with BaseDetector(BaseDetector.Detectors.<type>, "
            "config, frame); the detector type depends on the build flags.");
    baseDetector.def(py::init(&BaseDetector::CreateDetectorKV),
                     py::arg("detectorType"), py::arg("config"), py::arg("frame"),
                     "Create a detector of the given type; frame is used to size the internal state.");
    baseDetector.def("Init", [](BaseDetector& detector, const KeyVal& config)
    {
        return detector.Init(ToConfig(config));
    }, py::arg("config"), "Re-initialise the detector from a KeyVal config; returns True on success.");
    baseDetector.def("Detect", [](BaseDetector& detector, cv::Mat frame)
    {
        detector.DetectMat(frame);
    }, py::arg("frame"), "Run detection on one frame; results are available via GetDetects().");
    baseDetector.def("ResetModel", [](BaseDetector& detector, cv::Mat frame, cv::Rect roi)
    {
        detector.ResetModel(frame.getUMat(cv::ACCESS_READ), roi);
    }, py::arg("frame"), py::arg("roi"), "Reset the model inside the given region (new background).");
    baseDetector.def("CanGrayProcessing", &BaseDetector::CanGrayProcessing,
                     "Return True if the detector works on grayscale frames.");
    baseDetector.def("SetMinObjectSize", &BaseDetector::SetMinObjectSize, py::arg("minObjectSize"),
                     "Set the minimal object size as a (width, height) tuple.");
    baseDetector.def("GetDetects", &BaseDetector::GetDetects,
                     "Return the detections of the last Detect() call.");
    baseDetector.def("CalcMotionMap", [](BaseDetector& detector, cv::Mat frame, bool drawOnlyMasks)
    {
        detector.CalcMotionMap(frame, drawOnlyMasks);
    }, py::arg("frame"), py::arg("drawOnlyMasks") = true,
       "Overlay the current motion map on frame (in place).");

    py::class_<MotionDetector, BaseDetector> motionDetector(m, "MotionDetector",
            "Background-subtraction motion detector.");
    motionDetector.def(py::init<BackgroundSubtract::BGFG_ALGS, const cv::Mat&>(),
                       py::arg("algType"), py::arg("frame"),
                       "Create a motion detector with the given background algorithm.");
    motionDetector.def("Init", [](MotionDetector& detector, const KeyVal& config)
    {
        return detector.Init(ToConfig(config));
    }, py::arg("config"), "Re-initialise the motion detector; returns True on success.");
    motionDetector.def("Detect", [](MotionDetector& detector, cv::Mat frame)
    {
        detector.DetectMat(frame);
    }, py::arg("frame"), "Run detection on one frame; results are available via GetDetects().");
    motionDetector.def("SetMinObjectSize", &MotionDetector::SetMinObjectSize, py::arg("minObjectSize"),
                       "Set the minimal object size as a (width, height) tuple.");
    motionDetector.def("GetDetects", &MotionDetector::GetDetects,
                       "Return the detections of the last Detect() call.");
    motionDetector.def("CalcMotionMap", [](MotionDetector& detector, cv::Mat frame, bool drawOnlyMasks)
    {
        detector.CalcMotionMap(frame, drawOnlyMasks);
    }, py::arg("frame"), py::arg("drawOnlyMasks") = true,
       "Overlay the current motion map on frame (in place).");

    py::enum_<BackgroundSubtract::BGFG_ALGS>(motionDetector, "BGFG_ALGS",
            "Background-subtraction algorithms.")
            .value("VIBE", BackgroundSubtract::BGFG_ALGS::ALG_VIBE, "ViBe.")
            .value("MOG", BackgroundSubtract::BGFG_ALGS::ALG_MOG, "Mixture of Gaussians (legacy).")
            .value("GMG", BackgroundSubtract::BGFG_ALGS::ALG_GMG, "GMG.")
            .value("CNT", BackgroundSubtract::BGFG_ALGS::ALG_CNT, "Count-based.")
            .value("MOG2", BackgroundSubtract::BGFG_ALGS::ALG_MOG2, "Mixture of Gaussians (MOG2).")
            .export_values();

    py::enum_<tracking::Detectors>(baseDetector, "Detectors",
            "Detector types accepted by BaseDetector; availability depends on build flags.")
            .value("VIBE", tracking::Detectors::Motion_VIBE, "ViBe motion detector.")
            .value("MOG", tracking::Detectors::Motion_MOG, "MOG motion detector.")
            .value("GMG", tracking::Detectors::Motion_GMG, "GMG motion detector.")
            .value("CNT", tracking::Detectors::Motion_CNT, "CNT motion detector.")
            .value("MOG2", tracking::Detectors::Motion_MOG2, "MOG2 motion detector.")
            .value("Yolo_TensorRT", tracking::Detectors::ONNX_TensorRT, "YOLO via TensorRT (optional build).")
            .value("DNN_OCV", tracking::Detectors::DNN_OCV, "YOLO/SSD via OpenCV DNN.")
            .export_values();

    m.def("read_image", &read_image, py::arg("image"), "Read an image from a file (BGR uint8).");
    m.def("show_image", &show_image, py::arg("image"), "Show an image in a window (no-op when SILENT_WORK is set).");
    m.def("passthru", &passthru, py::arg("image"), "Return the same image (round-trip check).");
    m.def("clone", &cloneimg, py::arg("image"), "Return a deep copy of the image.");

    m.attr("__version__") = "1.1.0";
}
