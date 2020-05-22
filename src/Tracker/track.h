#pragma once
#include <iostream>
#include <vector>
#include <deque>
#include <memory>
#include <array>

#ifdef USE_OCV_KCF
#include <opencv2/tracking.hpp>
#endif

#include "defines.h"
#include "Kalman.h"
#include "VOTTracker.hpp"

// --------------------------------------------------------------------------
///
/// \brief The TrajectoryPoint struct
///
struct TrajectoryPoint
{
    ///
    /// \brief TrajectoryPoint
    ///
    TrajectoryPoint()
        : m_hasRaw(false)
    {
    }

    ///
    /// \brief TrajectoryPoint
    /// \param prediction
    ///
    TrajectoryPoint(const Point_t& prediction, const GeoPoint_t& geoPoint)
        :
          m_hasRaw(false),
          m_prediction(prediction),
		  m_geoPoint(geoPoint)
    {
    }

    ///
    /// \brief TrajectoryPoint
    /// \param prediction
    /// \param raw
    ///
    TrajectoryPoint(const Point_t& prediction, const Point_t& raw, const GeoPoint_t& geoPoint)
        :
          m_hasRaw(true),
          m_prediction(prediction),
          m_raw(raw),
		  m_geoPoint(geoPoint)
    {
    }

    bool m_hasRaw = false;
    Point_t m_prediction;
    Point_t m_raw;
	GeoPoint_t m_geoPoint;
};

// --------------------------------------------------------------------------
///
/// \brief The Trace class
///
class Trace
{
public:
    ///
    /// \brief operator []
    /// \param i
    /// \return
    ///
    const Point_t& operator[](size_t i) const
    {
        return m_trace[i].m_prediction;
    }

    ///
    /// \brief operator []
    /// \param i
    /// \return
    ///
    Point_t& operator[](size_t i)
    {
        return m_trace[i].m_prediction;
    }

    ///
    /// \brief at
    /// \param i
    /// \return
    ///
    const TrajectoryPoint& at(size_t i) const
    {
        return m_trace[i];
    }

    ///
    /// \brief size
    /// \return
    ///
    size_t size() const
    {
        return m_trace.size();
    }

    ///
    /// \brief push_back
    /// \param prediction
    ///
    void push_back(const Point_t& prediction, const GeoPoint_t& geoPoint)
    {
        m_trace.emplace_back(prediction, geoPoint);
    }
    void push_back(const Point_t& prediction, const Point_t& raw, const GeoPoint_t& geoPoint)
    {
        m_trace.emplace_back(prediction, raw, geoPoint);
    }

    ///
    /// \brief pop_front
    /// \param count
    ///
    void pop_front(size_t count)
    {
        if (count < size())
            m_trace.erase(m_trace.begin(), m_trace.begin() + count);
        else
            m_trace.clear();
    }

    ///
    /// \brief GetRawCount
    /// \param lastPeriod
    /// \return
    ///
    size_t GetRawCount(size_t lastPeriod) const
    {
        size_t res = 0;

        size_t i = 0;
        if (lastPeriod < m_trace.size())
        {
            i = m_trace.size() - lastPeriod;
        }
        for (; i < m_trace.size(); ++i)
        {
            if (m_trace[i].m_hasRaw)
            {
                ++res;
            }
        }

        return res;
    }

private:
    std::deque<TrajectoryPoint> m_trace;
};

// --------------------------------------------------------------------------
///
/// \brief The TrackingObject class
///
struct TrackingObject
{
    cv::RotatedRect m_rrect;           // Coordinates
	Trace m_trace;                     // Trajectory
	size_t m_ID = 0;                   // Objects ID
	bool m_isStatic = false;           // Object is abandoned
	bool m_outOfTheFrame = false;      // Is object out of freme
	std::string m_type;                // Objects type name or empty value
	float m_confidence = -1;           // From Detector with score (YOLO or SSD)
	cv::Vec<track_t, 2> m_velocity;    // pixels/sec

	///
    TrackingObject(const cv::RotatedRect& rrect, size_t ID, const Trace& trace,
		bool isStatic, bool outOfTheFrame, const std::string& type, float confidence, cv::Vec<track_t, 2> velocity)
		:
        m_rrect(rrect), m_ID(ID), m_isStatic(isStatic), m_outOfTheFrame(outOfTheFrame), m_type(type), m_confidence(confidence), m_velocity(velocity)
	{
		for (size_t i = 0; i < trace.size(); ++i)
		{
            auto tp = trace.at(i);
            if (tp.m_hasRaw)
                m_trace.push_back(tp.m_prediction, tp.m_raw, tp.m_geoPoint);
            else
                m_trace.push_back(tp.m_prediction, tp.m_geoPoint);
		}
	}

	///
	bool IsRobust(int minTraceSize, float minRawRatio, cv::Size2f sizeRatio) const
	{
		bool res = m_trace.size() > static_cast<size_t>(minTraceSize);
		res &= m_trace.GetRawCount(m_trace.size() - 1) / static_cast<float>(m_trace.size()) > minRawRatio;
		if (sizeRatio.width + sizeRatio.height > 0)
		{
            float sr = m_rrect.size.width / m_rrect.size.height;
			if (sizeRatio.width > 0)
				res &= (sr > sizeRatio.width);

			if (sizeRatio.height > 0)
				res &= (sr < sizeRatio.height);
		}
		if (m_outOfTheFrame)
			res = false;

		return res;
	}

	track_t Distance(size_t& period) const
	{
		if (period > m_trace.size())
			period = m_trace.size();

		const auto& from = m_trace.at(m_trace.size() - period);
		const auto& to = m_trace.at(m_trace.size() - 1);

		auto res = DistanceInMeters(from.m_geoPoint, to.m_geoPoint);
		return res;
	}
};

// --------------------------------------------------------------------------
///
/// \brief The CTrack class
///
class CTrack
{
public:
	CTrack() = delete;
    CTrack(const CRegion& region,
            tracking::KalmanType kalmanType,
            track_t deltaTime,
            track_t accelNoiseMag,
		    bool useAcceleration,
            size_t trackID,
            bool filterObjectSize,
            tracking::LostTrackType externalTrackerForLost,
		    bool useGeoCoords,
		    const GeoParams<geocoord_t>& geoParams);

    ///
    /// \brief CalcDistCenter
    /// Euclidean distance in pixels between objects centres on two N and N+1 frames
    /// \param reg
    /// \return
    ///
    track_t CalcDistCenter(const CRegion& reg) const;
    ///
    /// \brief CalcDistRect
    /// Euclidean distance in pixels between object contours on two N and N+1 frames
    /// \param reg
    /// \return
    ///
    track_t CalcDistRect(const CRegion& reg) const;
    ///
    /// \brief CalcDistJaccard
    /// Jaccard distance from 0 to 1 between object bounding rectangles on two N and N+1 frames
    /// \param reg
    /// \return
    ///
    track_t CalcDistJaccard(const CRegion& reg) const;
	///
	/// \brief CalcDistHist
	/// Distance from 0 to 1 between objects histogramms on two N and N+1 frames
	/// \param reg
	/// \param currFrame
	/// \return
	///
	track_t CalcDistHist(const CRegion& reg, cv::UMat currFrame) const;
	///
	/// \brief CalcDistGeo
	/// Euclidean distance in meters between objects centres in geographical coordinates
	/// \param reg
	/// \return
	///
	track_t CalcDistGeo(const CRegion& reg) const;


	cv::RotatedRect CalcPredictionEllipse(cv::Size_<track_t> minRadius) const;
	///
	/// \brief IsInsideArea
	/// Test point inside in prediction area: prediction area + object velocity
	/// \param pt
	/// \param minVal
	/// \return
	///
	track_t IsInsideArea(const Point_t& pt, const cv::RotatedRect& rrect) const;
    track_t WidthDist(const CRegion& reg) const;
    track_t HeightDist(const CRegion& reg) const;

    void Update(const CRegion& region, bool dataCorrect, size_t max_trace_length, cv::UMat prevFrame, cv::UMat currFrame, int trajLen);

    bool IsStatic() const;
    bool IsStaticTimeout(int framesTime) const;
	bool IsOutOfTheFrame() const;

    cv::RotatedRect GetLastRect() const;

    const Point_t& AveragePoint() const;
    Point_t& AveragePoint();
    const CRegion& LastRegion() const;
    size_t SkippedFrames() const;
    size_t& SkippedFrames();

    TrackingObject ConstructObject() const;

private:
    Trace m_trace;
    size_t m_trackID = 0;
    size_t m_skippedFrames = 0;
    CRegion m_lastRegion;

    Point_t m_predictionPoint;
    cv::RotatedRect m_predictionRect;
    TKalmanFilter m_kalman;
    bool m_filterObjectSize = false;
    bool m_outOfTheFrame = false;

	bool m_useGeoCoords = false;
	const GeoParams<geocoord_t>& m_geoParams;
	GeoPoint_t m_predictionGeoPoint;

    tracking::LostTrackType m_externalTrackerForLost;
#ifdef USE_OCV_KCF
    cv::Ptr<cv::Tracker> m_tracker;
#endif
    std::unique_ptr<VOTTracker> m_VOTTracker;

    void RectUpdate(const CRegion& region, bool dataCorrect, cv::UMat prevFrame, cv::UMat currFrame);

    void CreateExternalTracker(int channels);

    void PointUpdate(const Point_t& pt, const cv::Size& newObjSize, bool dataCorrect, const cv::Size& frameSize);

    bool CheckStatic(int trajLen, cv::UMat currFrame, const CRegion& region);
    bool m_isStatic = false;
    int m_staticFrames = 0;
    cv::UMat m_staticFrame;
    cv::Rect m_staticRect;
};

typedef std::vector<std::unique_ptr<CTrack>> tracks_t;
