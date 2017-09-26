#pragma once
#include <iostream>
#include <vector>
#include <memory>
#include <array>
#include <chrono>

#include "defines.h"
#include "track.h"
#include "LocalTracker.h"

// ----------------------------------------------------------------------
#define SAVE_TRAJECTORIES 1

#if SAVE_TRAJECTORIES
#include <fstream>

class SaveTrajectories
{
public:
    SaveTrajectories()
        : m_sizeWrited(false), m_delim(",")
    {

    }
    ~SaveTrajectories()
    {
    }

    bool Open(std::string fileName)
    {
        if (fileName.length() > 0)
        {
            m_file.open(fileName.c_str(), std::ios_base::out | std::ios_base::trunc);
        }
        if (m_file.is_open())
        {
            m_sizeWrited = false;
        }
        return m_file.is_open();
    }

    bool WriteFrameSize(int frame_width, int frame_height)
    {
        if (m_file.is_open() && !m_sizeWrited)
        {
            m_file << frame_width << m_delim << frame_height << std::endl;
            m_sizeWrited = true;
        }
        return m_sizeWrited;
    }

    bool NewTrack(const CTrack& track)
    {
        if (m_file.is_open())
        {
            int type = 2;

            if (track.IsRobust(25, 0.7, cv::Size2f(0.9f, 4.0f)))
            {
                for (size_t j = 0; j < track.m_trace.size(); ++j)
                {
                    const TrajectoryPoint& pt = track.m_trace.at(j);

                    m_file << pt.m_frameInd << m_delim
                           << track.m_trackID << m_delim
                           << type << m_delim
                           << pt.m_prediction.x << m_delim
                           << pt.m_prediction.y << m_delim
                           << (static_cast<track_t>(pt.m_size.width) / static_cast<track_t>(pt.m_size.height)) << m_delim
                           << pt.m_size.width << m_delim
                           << pt.m_size.height << m_delim
                           << pt.m_time << m_delim
                           << (j + 1) << std::endl;
                }
            }
            return true;
        }
        return false;
    }

private:
    std::ofstream m_file;
    bool m_sizeWrited;
    std::string m_delim;
};
#endif

// --------------------------------------------------------------------------
class CTracker
{
public:
    CTracker(bool useLocalTracking,
             tracking::DistType distType,
             tracking::KalmanType kalmanType,
             tracking::FilterGoal filterGoal,
             tracking::LostTrackType useExternalTrackerForLostObjects,
             tracking::MatchType matchType,
             track_t dt_,
             track_t accelNoiseMag_,
             track_t dist_thres_ = 60,
             size_t maximum_allowed_skipped_frames_ = 10,
             size_t max_trace_length_ = 10);
	~CTracker(void);

    tracks_t tracks;
    void Update(const std::vector<Point_t>& detections, const regions_t& regions, cv::Mat grayFrame, int frameInd = 0);

#if SAVE_TRAJECTORIES
    void WriteAllTracks();
#endif

private:
    // Use local tracking for regions between two frames
    bool m_useLocalTracking;

    tracking::DistType m_distType;
    tracking::KalmanType m_kalmanType;
    tracking::FilterGoal m_filterGoal;
    tracking::LostTrackType m_useExternalTrackerForLostObjects;
    tracking::MatchType m_matchType;

	// Шаг времени опроса фильтра
	track_t dt;

	track_t accelNoiseMag;

	// Порог расстояния. Если точки находятся дуг от друга на расстоянии,
	// превышающем этот порог, то эта пара не рассматривается в задаче о назначениях.
	track_t dist_thres;
	// Максимальное количество кадров которое трек сохраняется не получая данных о измерений.
    size_t maximum_allowed_skipped_frames;
	// Максимальная длина следа
    size_t max_trace_length;

	size_t NextTrackID;

    LocalTracker m_localTracker;

    cv::Mat m_prevFrame;

#if SAVE_TRAJECTORIES
    SaveTrajectories m_saveTraj;
#endif
};
