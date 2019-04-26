#include "SlowDetector.h"

///
/// \brief SlowDetector::SlowDetector
/// \param parser
///
SlowDetector::SlowDetector(const cv::CommandLineParser& parser)
    :
      m_showLogs(true),
      m_fps(25),
      m_startFrame(0),
      m_endFrame(0),
      m_finishDelay(0)
{
    m_inFile = parser.get<std::string>(0);
    m_outFile = parser.get<std::string>("out");
    m_showLogs = parser.get<int>("show_logs") != 0;
    m_startFrame = parser.get<int>("start_frame");
    m_endFrame = parser.get<int>("end_frame");
    m_finishDelay = parser.get<int>("end_delay");

    m_colors.push_back(cv::Scalar(255, 0, 0));
    m_colors.push_back(cv::Scalar(0, 255, 0));
    m_colors.push_back(cv::Scalar(0, 0, 255));
    m_colors.push_back(cv::Scalar(255, 255, 0));
    m_colors.push_back(cv::Scalar(0, 255, 255));
    m_colors.push_back(cv::Scalar(255, 0, 255));
    m_colors.push_back(cv::Scalar(255, 127, 255));
    m_colors.push_back(cv::Scalar(127, 0, 255));
    m_colors.push_back(cv::Scalar(127, 0, 127));
}

///
/// \brief SlowDetector::~SlowDetector
///
SlowDetector::~SlowDetector()
{

}

///
/// \brief SlowDetector::Process
///
void SlowDetector::Process()
{
    cv::VideoWriter writer;

    int k = 0;

    double freq = cv::getTickFrequency();

    int64 allTime = 0;

    bool manualMode = false;
    int framesCounter = m_startFrame + 1;

    cv::VideoCapture capture;
    if (m_inFile.size() == 1)
    {
        capture.open(atoi(m_inFile.c_str()));
    }
    else
    {
        capture.open(m_inFile);
    }
    if (!capture.isOpened())
    {
        std::cerr << "Can't open " << m_inFile << std::endl;
        return;
    }
    capture.set(cv::CAP_PROP_POS_FRAMES, m_startFrame);

    m_fps = std::max(1.f, (float)capture.get(cv::CAP_PROP_FPS));
	int frameWidth = cvRound(capture.get(cv::CAP_PROP_FRAME_WIDTH));
	int frameHeight = cvRound(capture.get(cv::CAP_PROP_FRAME_HEIGHT));

	// Detector
	config_t detectorConfig;
	const int yoloModel = 0;

#ifdef _WIN32
	std::string pathToModel = "../../data/";
#else
	std::string pathToModel = "../data/";
#endif

	switch (yoloModel)
	{
	case 0:
		detectorConfig["modelConfiguration"] = pathToModel + "tiny-yolo.cfg";
		detectorConfig["modelBinary"] = pathToModel + "tiny-yolo.weights";
		break;

	case 1:
		detectorConfig["modelConfiguration"] = pathToModel + "yolov3-tiny.cfg";
		detectorConfig["modelBinary"] = pathToModel + "yolov3-tiny.weights";
		detectorConfig["classNames"] = pathToModel + "coco.names";
		break;
	}

	detectorConfig["confidenceThreshold"] = "0.1";
	detectorConfig["maxCropRatio"] = "2.0";
	detectorConfig["dnnTarget"] = "DNN_TARGET_CPU";
	detectorConfig["dnnBackend"] = "DNN_BACKEND_INFERENCE_ENGINE";

	// Tracker
	const int minStaticTime = 5;

	TrackerSettings trackerSettings;
	trackerSettings.m_useLocalTracking = false;
	trackerSettings.m_distType = tracking::DistCenters;
	trackerSettings.m_kalmanType = tracking::KalmanLinear;
	trackerSettings.m_filterGoal = tracking::FilterRect;
	trackerSettings.m_lostTrackType = tracking::TrackCSRT; // Use KCF tracker for collisions resolving
	trackerSettings.m_matchType = tracking::MatchHungrian;
	trackerSettings.m_dt = 0.5f;                             // Delta time for Kalman filter
	trackerSettings.m_accelNoiseMag = 0.5f;                  // Accel noise magnitude for Kalman filter
	trackerSettings.m_distThres = frameHeight / 15.f;         // Distance threshold between region and object on two frames

	trackerSettings.m_useAbandonedDetection = false;
	if (trackerSettings.m_useAbandonedDetection)
	{
		trackerSettings.m_minStaticTime = minStaticTime;
		trackerSettings.m_maxStaticTime = 60;
		trackerSettings.m_maximumAllowedSkippedFrames = cvRound(trackerSettings.m_minStaticTime * m_fps); // Maximum allowed skipped frames
		trackerSettings.m_maxTraceLength = 2 * trackerSettings.m_maximumAllowedSkippedFrames;        // Maximum trace length
	}
	else
	{
		trackerSettings.m_maximumAllowedSkippedFrames = cvRound(2 * m_fps); // Maximum allowed skipped frames
		trackerSettings.m_maxTraceLength = cvRound(4 * m_fps);              // Maximum trace length
	}

	// Capture the first frame
	cv::Mat firstFrame;
	cv::UMat firstGray;
	capture >> firstFrame;
	cv::cvtColor(firstFrame, firstGray, cv::COLOR_BGR2GRAY);
	
	bool stopFlag = false;
	Gate frameLock;
	std::thread thDetection(DetectThread, detectorConfig, firstGray, &m_framesQue, &stopFlag, &frameLock);
	thDetection.detach();
	std::thread thTracking(TrackingThread, trackerSettings, &m_framesQue, &stopFlag, &frameLock);
	thTracking.detach();

	cv::namedWindow("Video", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
	cv::imshow("Video", firstFrame);
	cv::waitKey(1);

    for (;;)
    {
		FrameInfo frameInfo;
        capture >> frameInfo.m_frame;
        if (frameInfo.m_frame.empty())
        {
            std::cerr << "Frame is empty!" << std::endl;
            break;
        }
        cv::cvtColor(frameInfo.m_frame, frameInfo.m_gray, cv::COLOR_BGR2GRAY);
		
		m_framesQue.enqueue(frameInfo);

        if (!writer.isOpened())
        {
            writer.open(m_outFile, cv::VideoWriter::fourcc('H', 'F', 'Y', 'U'), m_fps, frameInfo.m_frame.size(), true);
			writer << firstFrame;
        }

        int64 t1 = cv::getTickCount();

        

        int64 t2 = cv::getTickCount();

        allTime += t2 - t1;
        int currTime = cvRound(1000 * (t2 - t1) / freq);

        DrawData(&frameInfo, framesCounter, currTime);

        cv::imshow("Video", frameInfo.m_frame);

        int waitTime = manualMode ? 0 : std::max<int>(1, cvRound(1000 / m_fps - currTime));
        k = cv::waitKey(waitTime);
        if (k == 'm' || k == 'M')
        {
            manualMode = !manualMode;
        }
        else if (k == 27)
        {
            break;
        }

        if (writer.isOpened())
        {
            writer << frameInfo.m_frame;
        }

        ++framesCounter;
        if (m_endFrame && framesCounter > m_endFrame)
        {
            std::cout << "Process: riched last " << m_endFrame << " frame" << std::endl;
            break;
        }
    }

	std::cout << "Stopping threads..." << std::endl;
	stopFlag = true;
	if (thDetection.joinable())
	{
		thDetection.join();
	}
	if (thTracking.joinable())
	{
		thTracking.join();
	}

    std::cout << "work time = " << (allTime / freq) << std::endl;
    cv::waitKey(m_finishDelay);
}

///
/// \brief SlowDetector::DrawTrack
/// \param frame
/// \param resizeCoeff
/// \param track
/// \param drawTrajectory
/// \param isStatic
///
void SlowDetector::DrawTrack(cv::Mat frame,
                             int resizeCoeff,
                             const TrackingObject& track,
                             bool drawTrajectory
        )
{
    auto ResizeRect = [&](const cv::Rect& r) -> cv::Rect
    {
        return cv::Rect(resizeCoeff * r.x, resizeCoeff * r.y, resizeCoeff * r.width, resizeCoeff * r.height);
    };
    auto ResizePoint = [&](const cv::Point& pt) -> cv::Point
    {
        return cv::Point(resizeCoeff * pt.x, resizeCoeff * pt.y);
    };

    if (track.m_isStatic)
    {
#if (CV_VERSION_MAJOR >= 4)
        cv::rectangle(frame, ResizeRect(track.m_rect), cv::Scalar(255, 0, 255), 2, cv::LINE_AA);
#else
		cv::rectangle(frame, ResizeRect(track.m_rect), cv::Scalar(255, 0, 255), 2, CV_AA);
#endif
    }
    else
    {
#if (CV_VERSION_MAJOR >= 4)
        cv::rectangle(frame, ResizeRect(track.m_rect), cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
#else
		cv::rectangle(frame, ResizeRect(track.m_rect), cv::Scalar(0, 255, 0), 1, CV_AA);
#endif
    }

    if (drawTrajectory)
    {
        cv::Scalar cl = m_colors[track.m_ID % m_colors.size()];

        for (size_t j = 0; j < track.m_trace.size() - 1; ++j)
        {
            const TrajectoryPoint& pt1 = track.m_trace.at(j);
            const TrajectoryPoint& pt2 = track.m_trace.at(j + 1);
#if (CV_VERSION_MAJOR >= 4)
            cv::line(frame, ResizePoint(pt1.m_prediction), ResizePoint(pt2.m_prediction), cl, 1, cv::LINE_AA);
#else
			cv::line(frame, ResizePoint(pt1.m_prediction), ResizePoint(pt2.m_prediction), cl, 1, CV_AA);
#endif
            if (!pt2.m_hasRaw)
            {
#if (CV_VERSION_MAJOR >= 4)
                cv::circle(frame, ResizePoint(pt2.m_prediction), 4, cl, 1, cv::LINE_AA);
#else
				cv::circle(frame, ResizePoint(pt2.m_prediction), 4, cl, 1, CV_AA);
#endif
            }
        }
    }
}

///
/// \brief SlowDetector::DrawData
/// \param frameinfo
///
void SlowDetector::DrawData(FrameInfo* frameInfo, int framesCounter, int currTime)
{
    if (m_showLogs)
    {
        std::cout << "Frame " << framesCounter << ": tracks = " << frameInfo->m_tracks.size() << ", time = " << currTime << std::endl;
    }

	
    for (const auto& track : frameInfo->m_tracks)
    {
        if (track.m_isStatic)
        {
            DrawTrack(frameInfo->m_frame, 1, track, true);
        }
        else
        {
            if (track.IsRobust(cvRound(m_fps / 4),          // Minimal trajectory size
                                0.7f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
                                cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
                    )
            {
                DrawTrack(frameInfo->m_frame, 1, track, true);
            }
        }
    }
}

///
/// \brief SlowDetector::DetectThread
/// \param
///
void SlowDetector::DetectThread(const config_t& config, cv::UMat firstGray, SafeQueue<FrameInfo>* framesQue, bool* stopFlag, Gate* frameLock)
{
	std::unique_ptr<BaseDetector> detector = std::unique_ptr<BaseDetector>(CreateDetector(tracking::Detectors::Yolo_OCV, config, false, firstGray));
	detector->SetMinObjectSize(cv::Size(firstGray.cols / 50, firstGray.cols / 50));

	for (; !(*stopFlag);)
	{
		FrameInfo frameInfo = framesQue->dequeue();

		frameInfo.m_inDetector = 1;
		cv::UMat clFrame = frameInfo.m_frame.getUMat(cv::ACCESS_READ);
		detector->Detect(clFrame);
		frameInfo.m_inDetector = 2;

		const regions_t& regions = detector->GetDetects();
		frameInfo.m_regions.assign(regions.begin(), regions.end());
	}
}

///
/// \brief SlowDetector::TrackingThread
/// \param
///
void SlowDetector::TrackingThread(const TrackerSettings& settings, SafeQueue<FrameInfo>* framesQue, bool* stopFlag, Gate* frameLock)
{
	std::unique_ptr<CTracker> tracker = std::make_unique<CTracker>(settings);

	for (; !(*stopFlag);)
	{
		FrameInfo frameInfo = framesQue->dequeue();
		if (tracker->GrayFrameToTrack())
		{
			tracker->Update(frameInfo.m_regions, frameInfo.m_gray, frameInfo.m_fps);
		}
		else
		{
			cv::UMat clFrame = frameInfo.m_frame.getUMat(cv::ACCESS_READ);
			tracker->Update(frameInfo.m_regions, clFrame, frameInfo.m_fps);
		}
		
		frameInfo.m_tracks = tracker->GetTracks();
	}
}
