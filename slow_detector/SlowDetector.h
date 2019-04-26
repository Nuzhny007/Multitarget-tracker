#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <thread>
#include <mutex>
#include <chrono>
#include <condition_variable>

#include "BaseDetector.h"
#include "Ctracker.h"
#include "Queue.h"

// ----------------------------------------------------------------------

///
/// \brief The Gate struct
///
struct Gate
{
	bool m_gateOpen = false;
	mutable std::condition_variable m_cond;
	mutable std::mutex m_mutex;

	void Lock()
	{
		m_mutex.lock();
		m_gateOpen = false;
	}
	void Unlock()
	{
		m_gateOpen = true;
		m_mutex.unlock();
	}

	void OpenGate()
	{
		{
			std::unique_lock<std::mutex> lock(m_mutex);
			m_gateOpen = true;
		}
		m_cond.notify_all();
	}

	void WaitAtGate()
	{
		std::unique_lock<std::mutex> lock(m_mutex);
#if 1
		m_cond.wait(lock);
#else
		m_cond.wait(lock, [this] { return m_gateOpen; });
#endif
		m_gateOpen = false;
	}

	bool WaitAtGateUntil(int timeOut)
	{
		std::unique_lock<std::mutex> lock(m_mutex);
		auto now = std::chrono::system_clock::now();
#if 1
		bool res = m_cond.wait_until(lock, now + std::chrono::milliseconds(timeOut)) != std::cv_status::timeout;
#else
		bool res = m_cond.wait_until(lock, now + std::chrono::milliseconds(timeOut), [this] { return m_gateOpen; });
#endif
		m_gateOpen = false;
		return res;
	}
};

// ----------------------------------------------------------------------

///
/// \brief The SlowDetector class
///
class SlowDetector
{
public:
    SlowDetector(const cv::CommandLineParser& parser);
    ~SlowDetector();

    void Process();

private:
    bool m_showLogs = false;
    float m_fps = 0;

	struct FrameInfo
	{
		cv::Mat m_frame;
		cv::UMat m_gray;
		regions_t m_regions;
		std::vector<TrackingObject> m_tracks;
		int64 m_dt = 0;
		float m_fps = 0;

		int m_inDetector = 0; // 0 - not in Detector, 1 - detector start processing, 2 - objects was detected
	};

    std::string m_inFile;
    std::string m_outFile;
    int m_startFrame = 0;
    int m_endFrame = 0;
    int m_finishDelay = 0;
    std::vector<cv::Scalar> m_colors;

	SafeQueue<FrameInfo> m_framesQue;

	void DrawData(FrameInfo* frameInfo, int framesCounter, int currTime);

	void DrawTrack(cv::Mat frame,
		int resizeCoeff,
		const TrackingObject& track,
		bool drawTrajectory = true);

	static void DetectThread(const config_t& config, cv::UMat firstGray, SafeQueue<FrameInfo>* framesQue, bool* stopFlag, Gate* frameLock);
	static void TrackingThread(const TrackerSettings& settings, SafeQueue<FrameInfo>* framesQue, bool* stopFlag, Gate* frameLock);
};
