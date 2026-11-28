#pragma once

#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>
#include <condition_variable>
#include <atomic>

#include "BaseDetector.h"
#include "BaseTracker.h"
#include "FileLogger.h"
#include "cvatAnnotationsGenerator.h"
#include "motResultsWriter.h"
#include "util/Frame.h"

#ifdef USE_VSR
#include "TrackPostProcessor.h"
#endif

#include "spdlog/spdlog.h"
#include "spdlog/async.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/rotating_file_sink.h"


///
/// \brief The FrameInfo struct
///
struct FrameInfo
{
    ///
    FrameInfo()
    {
        m_frames.reserve(m_batchSize);
        m_regions.reserve(m_batchSize);
        m_frameInds.reserve(m_batchSize);
    }
    ///
    FrameInfo(size_t batchSize)
        : m_batchSize(batchSize)
    {
        m_frames.reserve(m_batchSize);
        m_regions.reserve(m_batchSize);
        m_frameInds.reserve(m_batchSize);
    }

    ///
    void SetBatchSize(size_t batchSize)
    {
        m_batchSize = batchSize;
        m_frames.reserve(m_batchSize);
        m_regions.reserve(m_batchSize);
        m_frameInds.reserve(m_batchSize);
    }

    ///
    void CleanRegions()
    {
        if (m_regions.size() != m_batchSize)
            m_regions.resize(m_batchSize);
        for (auto& regions : m_regions)
        {
            regions.clear();
        }
    }

    ///
    void CleanTracks()
    {
        if (m_tracks.size() != m_batchSize)
            m_tracks.resize(m_batchSize);
        for (auto& tracks : m_tracks)
        {
            tracks.clear();
        }
    }

    std::vector<Frame> m_frames;
    std::vector<regions_t> m_regions;
    std::vector<std::vector<TrackingObject>> m_tracks;
    std::vector<int> m_frameInds;
    std::vector<time_point_t> m_frameTimeStamps;

    size_t m_batchSize = 1;

    int64 m_dt = 0;

    std::condition_variable m_cond;
    std::mutex m_mutex;
    std::atomic<bool> m_captured { false };
};

///
/// \brief The VideoExample class
///
class VideoExample
{
public:
    VideoExample(const cv::CommandLineParser& parser);
    VideoExample(const VideoExample&) = delete;
    VideoExample(VideoExample&&) = delete;
    VideoExample& operator=(const VideoExample&) = delete;
    VideoExample& operator=(VideoExample&&) = delete;

    virtual ~VideoExample() = default;

    void AsyncProcess();
    void SyncProcess();

protected:
    std::unique_ptr<BaseDetector> m_detector;
    std::unique_ptr<BaseTracker> m_tracker;

    std::string m_showLogsLevel = "debug";
    float m_fps = 25;
	cv::Size m_frameSize;
	int m_framesCount = 0;

    bool m_useContrastAdjustment = false;

	size_t m_batchSize = 1;

    int m_captureTimeOut = 60000;
    int m_trackingTimeOut = 60000;

    ResultsLog m_resultsLog;
	CVATAnnotationsGenerator m_cvatAnnotationsGenerator;
	MOTResultsWriter m_motResultsWriter;

    bool m_noShow = false;

    static void CaptureAndDetect(VideoExample* thisPtr, std::atomic<bool>& stopCapture);

    virtual bool InitDetector(cv::UMat frame) = 0;
    virtual bool InitTracker(cv::UMat frame) = 0;

    virtual void Detection(FrameInfo& frame);
    void Tracking(FrameInfo& frame);

    virtual void DrawData(cv::Mat frame, const std::vector<TrackingObject>& tracks, int framesCounter, int currTime) = 0;
    virtual void DrawTrack(cv::Mat frame, const TrackingObject& track, bool drawTrajectory, int framesCounter, const std::string& userLabel = "");

    TrackerSettings m_trackerSettings;
    bool m_trackerSettingsLoaded = false;

#ifdef USE_VSR
    std::unique_ptr<TrackPostProcessor> m_postProcessor;
#endif

    std::vector<cv::Scalar> m_colors;

    std::shared_ptr<spdlog::logger> m_logger;

private:
	std::vector<TrackingObject> m_tracks;

    bool m_isTrackerInitialized = false;
    bool m_isDetectorInitialized = false;
    std::string m_inFile;
    std::string m_outFile;
#if 0
    int m_fourcc = cv::VideoWriter::fourcc('h', '2', '6', '4');
#else
    int m_fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
#endif
    int m_startFrame = 0;
    int m_endFrame = 0;
    int m_finishDelay = 0;
    
    time_point_t m_startTimeStamp;
    bool m_useArchieveTime = true;

    FrameInfo m_frameInfo[2];
    
    time_point_t GetNextTimeStamp(int framesCounter) const;

    bool OpenCapture(cv::VideoCapture& capture);
    bool WriteFrame(cv::VideoWriter& writer, const cv::Mat& frame);
};
