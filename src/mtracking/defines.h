#pragma once

#include <vector>
#include <string>
#include <map>

#ifdef HAVE_EXPERIMENTAL_FILESYSTEM
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

#include <opencv2/opencv.hpp>
#include "object_types.h"
#include "Export.h"

// ---------------------------------------------------------------------------
//
// ---------------------------------------------------------------------------
typedef float track_t;
typedef cv::Point_<track_t> Point_t;
#define El_t CV_32F
#define Mat_t CV_32FC

typedef std::vector<int> assignments_t;
typedef std::vector<track_t> distMatrix_t;

typedef std::chrono::time_point<std::chrono::system_clock> time_point_t;

///
template<typename T>
class TrackID
{
public:
    typedef T value_type;

	TrackID() = default;
	TrackID(value_type val)
		: m_val(val)
	{
	}

    bool operator==(const TrackID& id) const
    {
        return m_val == id.m_val;
    }

    std::string ID2Str() const
    {
        return std::to_string(m_val);
    }
    static TrackID Str2ID(const std::string& id)
    {
        return TrackID(std::stoi(id));
    }
    TrackID NextID() const
    {
        return TrackID(m_val + 1);
    }
    size_t ID2Module(size_t module) const
    {
        return m_val % module;
    }

    value_type m_val{ 0 };
};

typedef TrackID<size_t> track_id_t;
namespace std
{
  template <>
  struct hash<track_id_t>
  {
    std::size_t operator()(const track_id_t& k) const
    {
      return std::hash<track_id_t::value_type>()(k.m_val);
    }
  };

}

///
/// \brief config_t
///
typedef std::multimap<std::string, std::string> config_t;

///
/// \brief The CRegion class
///
class CRegion
{
public:
    ///
    CRegion() = default;

    ///
    CRegion(const cv::Rect& rect) noexcept
        : m_brect(rect)
    {
        B2RRect();
    }

    ///
    CRegion(const cv::RotatedRect& rrect) noexcept
        : m_rrect(rrect)
    {
        if (m_rrect.size.width < 1)
            m_rrect.size.width = 1;
        if (m_rrect.size.height < 1)
            m_rrect.size.height = 1;
        R2BRect();
    }

    ///
    CRegion(const cv::RotatedRect& rrect, objtype_t type, float confidence) noexcept
        : m_type(type), m_rrect(rrect), m_confidence(confidence)
    {
        if (m_rrect.size.width < 1)
            m_rrect.size.width = 1;
        if (m_rrect.size.height < 1)
            m_rrect.size.height = 1;
        R2BRect();
    }

    ///
    CRegion(const cv::RotatedRect& rrect, const cv::Rect& brect, objtype_t type, float confidence, const cv::Mat& boxMask) noexcept
        : m_type(type), m_rrect(rrect), m_brect(brect), m_confidence(confidence)
    {
        m_boxMask = boxMask;

        if (m_rrect.size.width < 1)
            m_rrect.size.width = 1;
        if (m_rrect.size.height < 1)
            m_rrect.size.height = 1;

        if (!m_boxMask.empty() && m_boxMask.size() != m_brect.size())
        {
            m_brect.width = m_boxMask.cols;
            m_brect.height = m_boxMask.rows;
        }
    }

    ///
    CRegion(const cv::Rect& brect, objtype_t type, float confidence) noexcept
        : m_type(type), m_brect(brect), m_confidence(confidence)
    {
        B2RRect();
    }

    objtype_t m_type = bad_type;
    cv::RotatedRect m_rrect;
    cv::Rect m_brect;
    track_t m_confidence = -1;
    cv::Mat m_boxMask;

    ///
    /// \brief Local plane (ENU) coordinates of the region center in meters.
    /// x = East, y = North. Filled by the application (not by the detector).
    ///
    cv::Point2d m_localCartesian{ 0.0, 0.0 };
    bool m_hasLocalCartesian = false;

    ///
    void SetLocalCartesian(const cv::Point2d& pt)
    {
        m_localCartesian = pt;
        m_hasLocalCartesian = true;
    }

private:
    ///
    /// \brief R2BRect
    /// \return
    ///
    cv::Rect R2BRect() noexcept
    {
        m_brect = m_rrect.boundingRect();
        return m_brect;
    }
    ///
    /// \brief B2RRect
    /// \return
    ///
    cv::RotatedRect B2RRect() noexcept
    {
        m_rrect = cv::RotatedRect(m_brect.tl(), cv::Point2f(static_cast<float>(m_brect.x + m_brect.width), static_cast<float>(m_brect.y)), m_brect.br());
        if (m_rrect.size.width < 1)
            m_rrect.size.width = 1;
        if (m_rrect.size.height < 1)
            m_rrect.size.height = 1;
        return m_rrect;
    }
};

typedef std::vector<CRegion> regions_t;

///
/// \brief sqr
/// \param val
/// \return
///
template<class T> inline
T sqr(T val)
{
    return val * val;
}

///
/// \brief sqr: Euclid distance between two points
/// \param val
/// \return
///
template<class T, class POINT_TYPE> inline
T distance(const POINT_TYPE& p1, const POINT_TYPE& p2)
{
    return sqrt((T)(sqr(p2.x - p1.x) + sqr(p2.y - p1.y)));
}

///
/// \brief Clamp: Fit rectangle to frame
/// \param rect
/// \param size
/// \return
///
inline cv::Rect Clamp(cv::Rect rect, const cv::Size& size)
{
	if (rect.x < 0)
	{
		rect.width = std::min(rect.width, size.width - 1);
		rect.x = 0;
	}
	else if (rect.x + rect.width >= size.width)
	{
		rect.x = std::max(0, size.width - rect.width - 1);
		rect.width = std::min(rect.width, size.width - 1);
	}
	if (rect.y < 0)
	{
		rect.height = std::min(rect.height, size.height - 1);
		rect.y = 0;
	}
	else if (rect.y + rect.height >= size.height)
	{
		rect.y = std::max(0, size.height - rect.height - 1);
		rect.height = std::min(rect.height, size.height - 1);
	}
	return rect;
}

///
/// \brief SaveMat
/// \param m
/// \param name
/// \param path
///
inline bool SaveMat(const cv::Mat& m, std::string prefix, const std::string& ext, const std::string& savePath, bool compressToImage)
{
    bool res = true;

    std::map<int, std::string> depthDict;
    depthDict.emplace(CV_8U, "uint8");
    depthDict.emplace(CV_8S, "int8");
    depthDict.emplace(CV_16U, "uint16");
    depthDict.emplace(CV_16S, "int16");
    depthDict.emplace(CV_32S, "int32");
    depthDict.emplace(CV_32F, "float32");
    depthDict.emplace(CV_64F, "float64");
    depthDict.emplace(CV_16F, "float16");

    auto depth = depthDict.find(m.depth());
    if (depth == std::end(depthDict))
    {
        std::cout << "File " << prefix << " has a unknown depth: " << m.depth() << std::endl;
        res = false;
        return res;
    }
    assert(depth != std::end(depthDict));

    fs::path fullPath(savePath);
    fullPath.append(prefix + "_" + std::to_string(m.cols) + "x" + std::to_string(m.rows) + "_" + depth->second + "_C" + std::to_string(m.channels()) + ext);
    prefix = fullPath.generic_string();

    if (compressToImage)
    {
        res = cv::imwrite(prefix, m);
    }
    else
    {
        FILE* f = 0;
#ifdef _WIN32
        fopen_s(&f, prefix.c_str(), "wb");
#else
        f = fopen(prefix.c_str(), "wb");
#endif // _WIN32
        res = f != 0;
        if (res)
        {
            for (int y = 0; y < m.rows; ++y)
            {
                fwrite(m.ptr(y), 1, m.cols * m.elemSize(), f);
            }
            fclose(f);
            std::cout << "File " << prefix << " was writed" << std::endl;
        }
    }
    if (res)
        std::cout << "File " << prefix << " was writed" << std::endl;
    else
        std::cout << "File " << prefix << " can not be opened!" << std::endl;
    return res;
}

///
/// \brief DrawFilledRect
///
inline void DrawFilledRect(cv::Mat& frame, const cv::Rect& rect, cv::Scalar cl, int alpha)
{
    if (alpha)
    {
        const int alpha_1 = 255 - alpha;
        const int nchans = frame.channels();
        // cv::Scalar carries 4 components; a 4-channel (BGRA) frame reads
        // color[3]. The old int color[3] read out of bounds (roadmap M6).
        const int color[4] = { cv::saturate_cast<int>(cl[0]), cv::saturate_cast<int>(cl[1]),
                               cv::saturate_cast<int>(cl[2]), cv::saturate_cast<int>(cl[3]) };

        const int x0 = std::max(0, rect.x);
        const int y0 = std::max(0, rect.y);
        const int x1 = std::min(rect.x + rect.width, frame.cols);
        const int y1 = std::min(rect.y + rect.height, frame.rows);
        for (int y = y0; y < y1; ++y)
        {
            // Start at the clamped column: offsetting by a negative/off-frame
            // rect.x used to compute an out-of-range pointer (roadmap M6).
            uchar* ptr = frame.ptr(y) + nchans * x0;
            for (int x = x0; x < x1; ++x)
            {
                for (int i = 0; i < nchans; ++i)
                {
                    const int ci = (i < 4) ? i : 3;
                    ptr[i] = cv::saturate_cast<uchar>((alpha_1 * ptr[i] + alpha * color[ci]) / 255);
                }
                ptr += nchans;
            }
        }
    }
    else
    {
        cv::rectangle(frame, rect, cl, cv::FILLED);
    }
}

///
///
///
namespace tracking
{
///
/// \brief The Detectors enum
///
enum Detectors
{
    Motion_VIBE = 0,
    Motion_MOG = 1,
    Motion_GMG = 2,
    Motion_CNT = 3,
    Motion_MOG2 = 4,
    ONNX_TensorRT = 5,
    DNN_OCV = 6,
    DetectorsCount
};

///
/// \brief The TrackerTemplate enum
///
enum TrackerTemplate
{
    UniversalTracker = 0,
    ByteTrack = 1
};

///
/// \brief The DistType enum
///
enum DistType
{
    DistCenters,     // Euclidean distance between centers, [0, 1]
    DistRects,       // Euclidean distance between bounding rectangles, [0, 1]
    DistJaccard,     // Intersection over Union, IoU, [0, 1]
    DistHist,        // Bhatacharia distance between histograms, [0, 1]
    DistFeatureCos,  // Cosine distance between embeddings, [0, 1]
    DistMahalanobis, // Mahalanobis: https://ww2.mathworks.cn/help/vision/ug/motion-based-multiple-object-tracking.html
    DistLocalCartesian, // Euclidean distance in the local ENU plane, meters
    DistsCount
};

///
/// \brief The FilterGoal enum
///
enum FilterGoal
{
    FilterCenter,
    FilterRect,
    FilterRRect,
    FiltersCount
};

///
/// \brief The KalmanType enum
///
enum KalmanType
{
    KalmanLinear,
    KalmanUnscented,
    KalmanAugmentedUnscented,
    KalmanCount
};

///
/// \brief The MatchType enum
///
enum MatchType
{
    MatchHungrian,
    MatchLAPJV,
    MatchCount
};

///
/// \brief The LostTrackType enum
///
enum LostTrackType
{
    TrackNone,
    TrackKCF,
    TrackCSRT,
    TrackDaSiamRPN,
    TrackNano,
    TrackVit,
    SingleTracksCount
};
}
