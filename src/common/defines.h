#pragma once

#include <vector>
#include <string>
#include <map>
#include <opencv2/opencv.hpp>


// ---------------------------------------------------------------------------
//
// ---------------------------------------------------------------------------
typedef float track_t;
typedef cv::Point_<track_t> Point_t;
#define El_t CV_32F
#define Mat_t CV_32FC

typedef std::vector<int> assignments_t;
typedef std::vector<track_t> distMatrix_t;

typedef double geocoord_t;
typedef cv::Point_<geocoord_t> GeoPoint_t;
#define GeoEl_t CV_64F
#define GeoMat_t CV_64FC

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
    CRegion()
        : m_type(""), m_confidence(-1)
    {
    }

    CRegion(const cv::Rect& rect)
        : m_brect(rect)
    {
        B2RRect();
    }

    CRegion(const cv::RotatedRect& rrect)
        : m_rrect(rrect)
    {
        R2BRect();
    }

    CRegion(const cv::Rect& rect, const std::string& type, float confidence)
        : m_brect(rect), m_type(type), m_confidence(confidence)
    {
        B2RRect();
    }

	CRegion(const cv::Rect& rect, const std::string& type, float confidence,
		int globalType, int subType, cv::Point displayCoord)
		: m_brect(rect), m_type(type), m_confidence(confidence),
		m_globalType(globalType), m_subType(subType), m_displayCoord(displayCoord)
	{
		B2RRect();
	}

    cv::RotatedRect m_rrect;
    cv::Rect m_brect;

    std::string m_type;
    float m_confidence = -1;

	GeoPoint_t m_geoCoord;

	mutable cv::Mat m_hist;

	int m_globalType = 0;
	int m_subType = 0;
	cv::Point m_displayCoord;

private:
    ///
    /// \brief R2BRect
    /// \return
    ///
    cv::Rect R2BRect()
    {
        m_brect = m_rrect.boundingRect();
        return m_brect;
    }
    ///
    /// \brief B2RRect
    /// \return
    ///
    cv::RotatedRect B2RRect()
    {
        m_rrect = cv::RotatedRect(m_brect.tl(), cv::Point2f(static_cast<float>(m_brect.x + m_brect.width), static_cast<float>(m_brect.y)), m_brect.br());
        return m_rrect;
    }
};

typedef std::vector<CRegion> regions_t;


///
/// \brief The GeoParams class
///
template<typename T>
class GeoParams
{
public:
	///
	GeoParams() = default;

	///
	GeoParams(const std::vector<cv::Point>& framePoints, const std::vector<cv::Point_<T>>& geoPoints)
	{
		SetKeyPoints(framePoints, geoPoints);
	}

	///
	bool SetKeyPoints(const std::vector<cv::Point>& framePoints, const std::vector<cv::Point_<T>>& geoPoints)
	{
		m_framePoints = framePoints;
		m_geoPoints = geoPoints;

		assert(m_framePoints.size() == m_geoPoints.size());
		assert(m_framePoints.size() < 4);

		bool res = true;

		std::vector<cv::Point_<T>> tmpPix;
		tmpPix.reserve(m_framePoints.size());
		for (auto pix : m_framePoints)
		{
			tmpPix.emplace_back(static_cast<T>(pix.x), static_cast<T>(pix.y));
		}
		cv::perspectiveTransform(tmpPix, m_geoPoints, m_toGeo);
		cv::perspectiveTransform(m_geoPoints, tmpPix, m_toPix);

		return res;
	}

	///
	cv::Point Geo2Pix(const cv::Point_<T>& geo) const
	{
		cv::Vec<T, 3> g(geo.x, geo.y, 1);
		cv::Vec<T, 3> p = m_toPix * g;
		return cv::Point(cvRound(p[0]), cvRound(p[1]));
	}

	///
	cv::Point_<T> Pix2Geo(const cv::Point& pix) const
	{
		cv::Vec<T, 3> p(static_cast<T>(pix.x), static_cast<T>(pix.y), 1);
		cv::Vec<T, 3> g = m_toGeo * p;
		return cv::Point_<T>(g[0], g[1]);
	}

	std::vector<cv::Point> GetFramePoints() const
	{
		return m_framePoints;
	}

private:
	std::vector<cv::Point> m_framePoints;
	std::vector<cv::Point_<T>> m_geoPoints;

	cv::Matx<T, 3, 3> m_toGeo;
	cv::Matx<T, 3, 3> m_toPix;
};

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
    Motion_VIBE,
    Motion_MOG,
    Motion_GMG,
    Motion_CNT,
    Motion_SuBSENSE,
    Motion_LOBSTER,
    Motion_MOG2,
    Face_HAAR,
    Pedestrian_HOG,
    Pedestrian_C4,
    SSD_MobileNet,
    Yolo_OCV,
    Yolo_Darknet,
    Yolo_TensorRT,
    DetectorsCount
};

///
/// \brief The DistType enum
///
enum DistType
{
    DistCenters,   // Euclidean distance between centers, pixels
    DistRects,     // Euclidean distance between bounding rectangles, pixels
    DistJaccard,   // Intersection over Union, IoU, [0, 1]
    DistHist,      // Bhatacharia distance between histograms, [0, 1]
    DistGeo,       // Euclidean distance between centers in geographical coordinates, meters
    DistsCount
};

///
/// \brief The FilterGoal enum
///
enum FilterGoal
{
    FilterCenter,
    FilterRect,
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
    MatchBipart,
    MatchCount
};

///
/// \brief The LostTrackType enum
///
enum LostTrackType
{
    TrackNone,
    TrackKCF,
    TrackMIL,
    TrackMedianFlow,
    TrackGOTURN,
    TrackMOSSE,
    TrackCSRT,
    TrackDAT,
    TrackSTAPLE,
    TrackLDES,
    TrackCount
};
}
