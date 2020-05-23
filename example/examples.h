#pragma once

#include <iostream>
#include <vector>
#include <map>

#include "VideoExample.h"

///
/// \brief DrawFilledRect
///
void DrawFilledRect(cv::Mat& frame, const cv::Rect& rect, cv::Scalar cl, int alpha)
{
	if (alpha)
	{
		const int alpha_1 = 255 - alpha;
		const int nchans = frame.channels();
		int color[3] = { cv::saturate_cast<int>(cl[0]), cv::saturate_cast<int>(cl[1]), cv::saturate_cast<int>(cl[2]) };
		for (int y = rect.y; y < rect.y + rect.height; ++y)
		{
			uchar* ptr = frame.ptr(y) + nchans * rect.x;
			for (int x = rect.x; x < rect.x + rect.width; ++x)
			{
				for (int i = 0; i < nchans; ++i)
				{
					ptr[i] = cv::saturate_cast<uchar>((alpha_1 * ptr[i] + alpha * color[i]) / 255);
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
/// \brief The MotionDetectorExample class
///
class MotionDetectorExample : public VideoExample
{
public:
    MotionDetectorExample(const cv::CommandLineParser& parser)
        :
          VideoExample(parser),
          m_minObjWidth(10)
    {
    }

protected:
    ///
    /// \brief InitDetector
    /// \param frame
    /// \return
    ///
    bool InitDetector(cv::UMat frame)
    {
        m_minObjWidth = frame.cols / 20;

        config_t config;
		config.emplace("useRotatedRect", "0");

		tracking::Detectors detectorType = tracking::Detectors::Motion_VIBE;

		switch (detectorType)
		{
		case tracking::Detectors::Motion_VIBE:
			config.emplace("samples", "20");
			config.emplace("pixelNeighbor", "1");
			config.emplace("distanceThreshold", "20");
			config.emplace("matchingThreshold", "3");
			config.emplace("updateFactor", "16");
			break;
		case tracking::Detectors::Motion_MOG:
			config.emplace("history", std::to_string(cvRound(50 * m_minStaticTime * m_fps)));
			config.emplace("nmixtures", "3");
			config.emplace("backgroundRatio", "0.7");
			config.emplace("noiseSigma", "0");
			break;
		case tracking::Detectors::Motion_GMG:
			config.emplace("initializationFrames", "50");
			config.emplace("decisionThreshold", "0.7");
			break;
		case tracking::Detectors::Motion_CNT:
			config.emplace("minPixelStability", "15");
			config.emplace("maxPixelStability", std::to_string(cvRound(20 * m_minStaticTime * m_fps)));
			config.emplace("useHistory", "1");
			config.emplace("isParallel", "1");
			break;
		case tracking::Detectors::Motion_SuBSENSE:
			break;
		case tracking::Detectors::Motion_LOBSTER:
			break;
		case tracking::Detectors::Motion_MOG2:
			config.emplace("history", std::to_string(cvRound(20 * m_minStaticTime * m_fps)));
			config.emplace("varThreshold", "10");
			config.emplace("detectShadows", "1");
			break;
		}
        m_detector = std::unique_ptr<BaseDetector>(CreateDetector(detectorType, config, frame));

        if (m_detector.get())
        {
            m_detector->SetMinObjectSize(cv::Size(m_minObjWidth, m_minObjWidth));
            return true;
        }
        return false;
    }
    ///
    /// \brief InitTracker
    /// \param frame
    /// \return
    ///
    bool InitTracker(cv::UMat frame)
    {
		m_trackerSettings.SetDistance(tracking::DistRects);
        m_trackerSettings.m_kalmanType = tracking::KalmanLinear;
        m_trackerSettings.m_filterGoal = tracking::FilterCenter;
        m_trackerSettings.m_lostTrackType = tracking::TrackCSRT;       // Use visual objects tracker for collisions resolving
        m_trackerSettings.m_matchType = tracking::MatchHungrian;
		m_trackerSettings.m_useAcceleration = false;                   // Use constant acceleration motion model
		m_trackerSettings.m_dt = m_trackerSettings.m_useAcceleration ? 0.05f : 0.2f; // Delta time for Kalman filter
		m_trackerSettings.m_accelNoiseMag = 0.2f;                  // Accel noise magnitude for Kalman filter
        m_trackerSettings.m_distThres = 0.95f;                    // Distance threshold between region and object on two frames
#if 0
		m_trackerSettings.m_minAreaRadiusPix = frame.rows / 20.f;
#else
		m_trackerSettings.m_minAreaRadiusPix = -1.f;
#endif
		m_trackerSettings.m_minAreaRadiusK = 0.8f;

        m_trackerSettings.m_useAbandonedDetection = true;
        if (m_trackerSettings.m_useAbandonedDetection)
        {
            m_trackerSettings.m_minStaticTime = m_minStaticTime;
            m_trackerSettings.m_maxStaticTime = 10;
            m_trackerSettings.m_maximumAllowedSkippedFrames = cvRound(m_trackerSettings.m_minStaticTime * m_fps); // Maximum allowed skipped frames
            m_trackerSettings.m_maxTraceLength = 2 * m_trackerSettings.m_maximumAllowedSkippedFrames;        // Maximum trace length
        }
        else
        {
            m_trackerSettings.m_maximumAllowedSkippedFrames = cvRound(2 * m_fps); // Maximum allowed skipped frames
            m_trackerSettings.m_maxTraceLength = cvRound(4 * m_fps);              // Maximum trace length
        }

        m_tracker = std::make_unique<CTracker>(m_trackerSettings);

        return true;
    }

    ///
    /// \brief DrawData
    /// \param frame
    /// \param framesCounter
    /// \param currTime
    ///
    void DrawData(cv::Mat frame, int framesCounter, int currTime)
    {
		m_tracks = m_tracker->GetTracks();

        if (m_showLogs)
        {
            std::cout << "Frame " << framesCounter << ": tracks = " << m_tracks.size() << ", time = " << currTime << std::endl;
        }

        for (const auto& track : m_tracks)
        {
            if (track.m_isStatic)
            {
                DrawTrack(frame, 1, track, false);

				std::string label = "abandoned " + std::to_string(track.m_ID);
				int baseLine = 0;
				cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

				cv::Rect brect = track.m_rrect.boundingRect();
				if (brect.x < 0)
				{
					brect.width = std::min(brect.width, frame.cols - 1);
					brect.x = 0;
				}
				else if (brect.x + brect.width >= frame.cols)
				{
					brect.x = std::max(0, frame.cols - brect.width - 1);
					brect.width = std::min(brect.width, frame.cols - 1);
				}
				if (brect.y - labelSize.height < 0)
				{
					brect.height = std::min(brect.height, frame.rows - 1);
					brect.y = labelSize.height;
				}
				else if (brect.y + brect.height >= frame.rows)
				{
					brect.y = std::max(0, frame.rows - brect.height - 1);
					brect.height = std::min(brect.height, frame.rows - 1);
				}
				DrawFilledRect(frame, cv::Rect(cv::Point(brect.x, brect.y - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)), cv::Scalar(255, 0, 255), 150);
				cv::putText(frame, label, brect.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
            }
            else
            {
                if (track.IsRobust(cvRound(m_fps / 4),          // Minimal trajectory size
                                    0.7f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
                                    cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
                        )
                {
                    DrawTrack(frame, 1, track, true);
                }
            }
        }

        m_detector->CalcMotionMap(frame);
    }

private:
    int m_minObjWidth = 8;
    int m_minStaticTime = 5;
};

// ----------------------------------------------------------------------

///
/// \brief The FaceDetectorExample class
///
class FaceDetectorExample : public VideoExample
{
public:
    FaceDetectorExample(const cv::CommandLineParser& parser)
        :
          VideoExample(parser)
    {
    }

protected:
    ///
    /// \brief InitDetector
    /// \param frame
    /// \return
    ///
    bool InitDetector(cv::UMat frame)
    {
#ifdef _WIN32
        std::string pathToModel = "../../data/";
#else
        std::string pathToModel = "../data/";
#endif

        config_t config;
        config.emplace("cascadeFileName", pathToModel + "haarcascade_frontalface_alt2.xml");
        m_detector = std::unique_ptr<BaseDetector>(CreateDetector(tracking::Detectors::Face_HAAR, config, frame));
        if (m_detector.get())
        {
            m_detector->SetMinObjectSize(cv::Size(frame.cols / 20, frame.rows / 20));
            return true;
        }
        return false;
    }
    ///
    /// \brief InitTracker
    /// \param frame
    /// \return
    ///
    bool InitTracker(cv::UMat frame)
    {
		m_trackerSettings.SetDistance(tracking::DistJaccard);
        m_trackerSettings.m_kalmanType = tracking::KalmanUnscented;
        m_trackerSettings.m_filterGoal = tracking::FilterRect;
        m_trackerSettings.m_lostTrackType = tracking::TrackCSRT;      // Use visual objects tracker for collisions resolving
        m_trackerSettings.m_matchType = tracking::MatchHungrian;
        m_trackerSettings.m_dt = 0.3f;                                // Delta time for Kalman filter
        m_trackerSettings.m_accelNoiseMag = 0.1f;                     // Accel noise magnitude for Kalman filter
        m_trackerSettings.m_distThres = 0.8f;                         // Distance threshold between region and object on two frames
        m_trackerSettings.m_minAreaRadiusPix = frame.rows / 20.f;
        m_trackerSettings.m_maximumAllowedSkippedFrames = cvRound(m_fps / 2);   // Maximum allowed skipped frames
        m_trackerSettings.m_maxTraceLength = cvRound(5 * m_fps);            // Maximum trace length

        m_tracker = std::make_unique<CTracker>(m_trackerSettings);

        return true;
    }

    ///
    /// \brief DrawData
    /// \param frame
    /// \param framesCounter
    /// \param currTime
    ///
    void DrawData(cv::Mat frame, int framesCounter, int currTime)
    {
		m_tracks = m_tracker->GetTracks();

        if (m_showLogs)
        {
            std::cout << "Frame " << framesCounter << ": tracks = " << m_tracks.size() << ", time = " << currTime << std::endl;
        }

        for (const auto& track : m_tracks)
        {
            if (track.IsRobust(8,                           // Minimal trajectory size
                                0.4f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
                                cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
                    )
            {
                DrawTrack(frame, 1, track);
            }
        }

        m_detector->CalcMotionMap(frame);
    }
};

// ----------------------------------------------------------------------

///
/// \brief The PedestrianDetectorExample class
///
class PedestrianDetectorExample : public VideoExample
{
public:
    PedestrianDetectorExample(const cv::CommandLineParser& parser)
        :
          VideoExample(parser)
    {
    }

protected:
    ///
    /// \brief InitDetector
    /// \param frame
    /// \return
    ///
    bool InitDetector(cv::UMat frame)
    {
        tracking::Detectors detectorType = tracking::Detectors::Pedestrian_C4; // tracking::Detectors::Pedestrian_HOG;

#ifdef _WIN32
        std::string pathToModel = "../../data/";
#else
        std::string pathToModel = "../data/";
#endif

        config_t config;
        config.emplace("detectorType", (detectorType == tracking::Pedestrian_HOG) ? "HOG" : "C4");
        config.emplace("cascadeFileName1", pathToModel + "combined.txt.model");
        config.emplace("cascadeFileName2", pathToModel + "combined.txt.model_");
        m_detector = std::unique_ptr<BaseDetector>(CreateDetector(detectorType, config, frame));
        if (m_detector.get())
        {
            m_detector->SetMinObjectSize(cv::Size(frame.cols / 20, frame.rows / 20));
            return true;
        }
        return false;
    }
    ///
    /// \brief InitTracker
    /// \param frame
    /// \return
    ///
    bool InitTracker(cv::UMat frame)
    {
		m_trackerSettings.SetDistance(tracking::DistRects);
        m_trackerSettings.m_kalmanType = tracking::KalmanLinear;
        m_trackerSettings.m_filterGoal = tracking::FilterRect;
        m_trackerSettings.m_lostTrackType = tracking::TrackCSRT;   // Use visual objects tracker for collisions resolving
        m_trackerSettings.m_matchType = tracking::MatchHungrian;
        m_trackerSettings.m_dt = 0.3f;                             // Delta time for Kalman filter
        m_trackerSettings.m_accelNoiseMag = 0.1f;                  // Accel noise magnitude for Kalman filter
        m_trackerSettings.m_distThres = 0.8f;                      // Distance threshold between region and object on two frames
        m_trackerSettings.m_minAreaRadiusPix = frame.rows / 20.f;
        m_trackerSettings.m_maximumAllowedSkippedFrames = cvRound(m_fps);   // Maximum allowed skipped frames
        m_trackerSettings.m_maxTraceLength = cvRound(5 * m_fps);   // Maximum trace length

        m_tracker = std::make_unique<CTracker>(m_trackerSettings);

        return true;
    }

    ///
    /// \brief DrawData
    /// \param frame
    /// \param framesCounter
    /// \param currTime
    ///
    void DrawData(cv::Mat frame, int framesCounter, int currTime)
    {
		m_tracks = m_tracker->GetTracks();

        if (m_showLogs)
        {
            std::cout << "Frame " << framesCounter << ": tracks = " << m_tracks.size() << ", time = " << currTime << std::endl;
        }

        for (const auto& track : m_tracks)
        {
			if (track.IsRobust(cvRound(m_fps / 2),          // Minimal trajectory size
                                0.4f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
                                cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
                    )
            {
                DrawTrack(frame, 1, track);
            }
        }

        m_detector->CalcMotionMap(frame);
    }
};

// ----------------------------------------------------------------------

///
/// \brief The SSDMobileNetExample class
///
class SSDMobileNetExample : public VideoExample
{
public:
    SSDMobileNetExample(const cv::CommandLineParser& parser)
        :
          VideoExample(parser)
    {
    }

protected:
    ///
    /// \brief InitDetector(
    /// \param frame
    /// \return
    ///
    bool InitDetector(cv::UMat frame)
    {
#ifdef _WIN32
        std::string pathToModel = "../../data/";
#else
        std::string pathToModel = "../data/";
#endif
        config_t config;
        config.emplace("modelConfiguration", pathToModel + "MobileNetSSD_deploy.prototxt");
        config.emplace("modelBinary", pathToModel + "MobileNetSSD_deploy.caffemodel");
        config.emplace("confidenceThreshold", "0.5");
        config.emplace("maxCropRatio", "3.0");
        config.emplace("dnnTarget", "DNN_TARGET_CPU");
        config.emplace("dnnBackend", "DNN_BACKEND_INFERENCE_ENGINE");

        m_detector = std::unique_ptr<BaseDetector>(CreateDetector(tracking::Detectors::SSD_MobileNet, config, frame));
        if (m_detector.get())
        {
            m_detector->SetMinObjectSize(cv::Size(frame.cols / 20, frame.rows / 20));
            return true;
        }
        return false;
    }
    ///
    /// \brief InitTracker
    /// \param frame
    /// \return
    ///
    bool InitTracker(cv::UMat frame)
    {
		m_trackerSettings.SetDistance(tracking::DistRects);
        m_trackerSettings.m_kalmanType = tracking::KalmanLinear;
        m_trackerSettings.m_filterGoal = tracking::FilterRect;
        m_trackerSettings.m_lostTrackType = tracking::TrackCSRT;      // Use visual objects tracker for collisions resolving
        m_trackerSettings.m_matchType = tracking::MatchHungrian;
        m_trackerSettings.m_dt = 0.3f;                                // Delta time for Kalman filter
        m_trackerSettings.m_accelNoiseMag = 0.1f;                     // Accel noise magnitude for Kalman filter
        m_trackerSettings.m_distThres = 0.8f;                         // Distance threshold between region and object on two frames
        m_trackerSettings.m_minAreaRadiusPix = frame.rows / 20.f;
        m_trackerSettings.m_maximumAllowedSkippedFrames = cvRound(2 * m_fps); // Maximum allowed skipped frames
        m_trackerSettings.m_maxTraceLength = cvRound(5 * m_fps);      // Maximum trace length

        m_tracker = std::make_unique<CTracker>(m_trackerSettings);

        return true;
    }

    ///
    /// \brief DrawData
    /// \param frame
    /// \param framesCounter
    /// \param currTime
    ///
    void DrawData(cv::Mat frame, int framesCounter, int currTime)
    {
		m_tracks = m_tracker->GetTracks();

        if (m_showLogs)
        {
            std::cout << "Frame " << framesCounter << ": tracks = " << m_tracks.size() << ", time = " << currTime << std::endl;
        }

        for (const auto& track : m_tracks)
        {
            if (track.IsRobust(5,                           // Minimal trajectory size
                                0.2f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
                                cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
                    )
            {
                DrawTrack(frame, 1, track);

                std::string label = track.m_type + ": " + std::to_string(track.m_confidence);
                int baseLine = 0;
                cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

                cv::Rect brect = track.m_rrect.boundingRect();
				DrawFilledRect(frame, cv::Rect(cv::Point(brect.x, brect.y - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)), cv::Scalar(200, 200, 200), 150);
                cv::putText(frame, label, brect.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
            }
        }

        m_detector->CalcMotionMap(frame);
    }
};

// ----------------------------------------------------------------------

///
/// \brief The YoloExample class
///
class YoloExample : public VideoExample
{
public:
    YoloExample(const cv::CommandLineParser& parser)
        :
          VideoExample(parser)
    {
    }

protected:
    ///
    /// \brief InitDetector(
    /// \param frame
    /// \return
    ///
    bool InitDetector(cv::UMat frame)
    {
        config_t config;
        const int yoloTest = 1;

#ifdef _WIN32
        std::string pathToModel = "../../data/";
#else
        std::string pathToModel = "../data/";
#endif

        switch (yoloTest)
        {
        case 0:
            config.emplace("modelConfiguration", pathToModel + "tiny-yolo.cfg");
            config.emplace("modelBinary", pathToModel + "tiny-yolo.weights");
            break;

        case 1:
            config.emplace("modelConfiguration", pathToModel + "yolov3-tiny.cfg");
            config.emplace("modelBinary", pathToModel + "yolov3-tiny.weights");
            config.emplace("classNames", pathToModel + "coco.names");
            break;
        }

        config.emplace("confidenceThreshold", "0.1");
        config.emplace("maxCropRatio", "2.0");
        config.emplace("dnnTarget", "DNN_TARGET_CPU");
        config.emplace("dnnBackend", "DNN_BACKEND_OPENCV");

        m_detector = std::unique_ptr<BaseDetector>(CreateDetector(tracking::Detectors::Yolo_OCV, config, frame));
        if (m_detector.get())
        {
            m_detector->SetMinObjectSize(cv::Size(frame.cols / 40, frame.rows / 40));
            return true;
        }
        return false;
    }
    ///
    /// \brief InitTracker
    /// \param frame
    /// \return
    ///
    bool InitTracker(cv::UMat frame)
    {
		m_trackerSettings.SetDistance(tracking::DistRects);
        m_trackerSettings.m_kalmanType = tracking::KalmanLinear;
        m_trackerSettings.m_filterGoal = tracking::FilterRect;
        m_trackerSettings.m_lostTrackType = tracking::TrackCSRT;      // Use visual objects tracker for collisions resolving
        m_trackerSettings.m_matchType = tracking::MatchHungrian;
        m_trackerSettings.m_dt = 0.3f;                                // Delta time for Kalman filter
        m_trackerSettings.m_accelNoiseMag = 0.2f;                     // Accel noise magnitude for Kalman filter
        m_trackerSettings.m_distThres = 0.8f;                         // Distance threshold between region and object on two frames
        m_trackerSettings.m_minAreaRadiusPix = frame.rows / 20.f;
        m_trackerSettings.m_maximumAllowedSkippedFrames = cvRound(2 * m_fps); // Maximum allowed skipped frames
        m_trackerSettings.m_maxTraceLength = cvRound(5 * m_fps);      // Maximum trace length

        m_tracker = std::make_unique<CTracker>(m_trackerSettings);

        return true;
    }

    ///
    /// \brief DrawData
    /// \param frame
    /// \param framesCounter
    /// \param currTime
    ///
    void DrawData(cv::Mat frame, int framesCounter, int currTime)
    {
		m_tracks = m_tracker->GetTracks();

        if (m_showLogs)
        {
            std::cout << "Frame " << framesCounter << ": tracks = " << m_tracks.size() << ", time = " << currTime << std::endl;
        }

        for (const auto& track : m_tracks)
        {
            if (track.IsRobust(1,                           // Minimal trajectory size
                                0.1f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
                                cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
                    )
            {
                DrawTrack(frame, 1, track);

                std::string label = track.m_type + ": " + std::to_string(track.m_confidence);
                int baseLine = 0;
                cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

                cv::Rect brect = track.m_rrect.boundingRect();
				DrawFilledRect(frame, cv::Rect(cv::Point(brect.x, brect.y - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)), cv::Scalar(200, 200, 200), 150);
                cv::putText(frame, label, brect.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
            }
        }

        m_detector->CalcMotionMap(frame);
    }
};

#ifdef BUILD_YOLO_LIB
// ----------------------------------------------------------------------

///
/// \brief The YoloDarknetExample class
///
class YoloDarknetExample : public VideoExample
{
public:
	YoloDarknetExample(const cv::CommandLineParser& parser)
		:
		VideoExample(parser)
	{
	}

protected:

    ///
    /// \brief InitDetector
    /// \param frame
    /// \return
    ///
	bool InitDetector(cv::UMat frame)
	{
		config_t config;

#ifdef _WIN32
		std::string pathToModel = "../../data/";
#else
		std::string pathToModel = "../data/";
#endif
		enum class YOLOModels
		{
			TinyYOLOv3 = 0,
			YOLOv3,
			YOLOv4
		};
		YOLOModels usedModel = YOLOModels::YOLOv4;
		switch (usedModel)
		{
		case YOLOModels::TinyYOLOv3:
			config.emplace("modelConfiguration", pathToModel + "yolov3-tiny.cfg");
			config.emplace("modelBinary", pathToModel + "yolov3-tiny.weights");
			config.emplace("confidenceThreshold", "0.5");
			break;

		case YOLOModels::YOLOv3:
			config.emplace("modelConfiguration", pathToModel + "yolov3.cfg");
			config.emplace("modelBinary", pathToModel + "yolov3.weights");
			config.emplace("confidenceThreshold", "0.7");
			break;

		case YOLOModels::YOLOv4:
			config.emplace("modelConfiguration", pathToModel + "yolov4.cfg");
			config.emplace("modelBinary", pathToModel + "yolov4.weights");
			config.emplace("confidenceThreshold", "0.5");
			break;
		}
        config.emplace("classNames", pathToModel + "coco.names");
        config.emplace("maxCropRatio", "-1");

        config.emplace("white_list", "person");
        config.emplace("white_list", "car");
        config.emplace("white_list", "bicycle");
        config.emplace("white_list", "motorbike");
        config.emplace("white_list", "bus");
        config.emplace("white_list", "truck");
        //config.emplace("white_list", "traffic light");
        //config.emplace("white_list", "stop sign");

        m_detector = std::unique_ptr<BaseDetector>(CreateDetector(tracking::Detectors::Yolo_Darknet, config, frame));
        if (m_detector.get())
        {
            m_detector->SetMinObjectSize(cv::Size(frame.cols / 40, frame.rows / 40));
            return true;
        }
        return false;
    }

    ///
    /// \brief InitTracker
    /// \param frame
    /// \return
    ///
    bool InitTracker(cv::UMat frame)
	{
        m_trackerSettings.SetDistance(tracking::DistRects);
		m_trackerSettings.m_kalmanType = tracking::KalmanLinear;
        m_trackerSettings.m_filterGoal = tracking::FilterCenter;
        m_trackerSettings.m_lostTrackType = tracking::TrackCSRT;      // Use visual objects tracker for collisions resolving
		m_trackerSettings.m_matchType = tracking::MatchHungrian;
		m_trackerSettings.m_useAcceleration = false;                   // Use constant acceleration motion model
		m_trackerSettings.m_dt = m_trackerSettings.m_useAcceleration ? 0.05f : 0.4f; // Delta time for Kalman filter
		m_trackerSettings.m_accelNoiseMag = 0.2f;                     // Accel noise magnitude for Kalman filter
        m_trackerSettings.m_distThres = 0.8f;                         // Distance threshold between region and object on two frames
#if 0
		m_trackerSettings.m_minAreaRadiusPix = frame.rows / 20.f;
#else
		m_trackerSettings.m_minAreaRadiusPix = -1.f;
#endif
		m_trackerSettings.m_minAreaRadiusK = 0.8f;
		m_trackerSettings.m_maximumAllowedSkippedFrames = cvRound(2 * m_fps); // Maximum allowed skipped frames
		m_trackerSettings.m_maxTraceLength = cvRound(2 * m_fps);      // Maximum trace length

#if 1
		m_trackerSettings.m_useGeoCoords = true;
#if 1
		std::vector<cv::Point> framePoints{ cv::Point(420, 348), cv::Point(509, 283), cv::Point(731, 281), cv::Point(840, 343) };
		std::vector<GeoPoint_t> geoPoints{ GeoPoint_t(45.526646, 5.974535), GeoPoint_t(45.527566, 5.973849), GeoPoint_t(45.527904, 5.974135), GeoPoint_t(45.526867, 5.974826) };
#else
		std::vector<cv::Point> framePoints{ cv::Point(1110, 497), cv::Point(164, 347), cv::Point(380, 188), cv::Point(1182, 252) };
		std::vector<GeoPoint_t> geoPoints{ GeoPoint_t(60.006536, 30.258855), GeoPoint_t(60.006855, 30.258051), GeoPoint_t(60.007414, 30.258080), GeoPoint_t(60.007064, 30.259066) };
#endif
		m_trackerSettings.m_geoParams.SetKeyPoints(framePoints, geoPoints);
		m_trackerSettings.SetDistance(tracking::DistGeo);
		m_trackerSettings.m_distThres = 4.0f;  // For Geo distance this value mean meters
		m_trackerSettings.m_minAreaRadiusPix = -1.f;
#else
		m_trackerSettings.m_useGeoCoords = false;
#endif


		m_trackerSettings.AddNearTypes("car", "bus", true);
		m_trackerSettings.AddNearTypes("car", "truck", true);
		m_trackerSettings.AddNearTypes("bus", "truck", true);
		m_trackerSettings.AddNearTypes("person", "bicycle", true);
		m_trackerSettings.AddNearTypes("person", "motorbike", true);

		m_tracker = std::make_unique<CTracker>(m_trackerSettings);

		return true;
	}

    ///
    /// \brief DrawData
    /// \param frame
    /// \param framesCounter
    /// \param currTime
    ///
	void DrawData(cv::Mat frame, int framesCounter, int currTime)
	{
		m_tracks = m_tracker->GetTracks();

		if (m_showLogs)
		{
			std::cout << "Frame " << framesCounter << ": tracks = " << m_tracks.size() << ", time = " << currTime << std::endl;
		}

		if (m_trackerSettings.m_useGeoCoords)
		{
			std::vector<cv::Point> points = m_trackerSettings.m_geoParams.GetFramePoints();
			for (size_t i = 0; i < points.size(); ++i)
			{
				cv::line(frame, points[i % points.size()], points[(i + 1) % points.size()], cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
			}
		}

		for (const auto& track : m_tracks)
		{
            if (track.IsRobust(3,                           // Minimal trajectory size
                0.7f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
				cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
				)
			{
				DrawTrack(frame, 1, track, false);

				std::stringstream label;
#if 0
				label << track.m_type << std::setprecision(2) << ": " << track.m_confidence;
#else
				if (m_trackerSettings.m_useGeoCoords)
				{
					size_t period = 2 * cvRound(m_fps);
					auto dist = track.Distance(period);
					if (period >= cvRound(m_fps) / 4)
					{
						auto velocity = (3.6f * dist * m_fps) / period;
						//std::cout << track.m_type << ": distance " << std::fixed << std::setw(2) << std::setprecision(2) << dist << " on time " << (period / m_fps) << " with velocity " << velocity << " km/h: " << track.m_confidence << std::endl;
						if (velocity < 1.f || std::isnan(velocity))
							velocity = 0;
						label << track.m_type << " " << std::fixed << std::setw(2) << std::setprecision(2) << velocity << " km/h";// << track.m_confidence;
					}
					else
					{
						label << track.m_type << std::setprecision(2) << ": " << track.m_confidence;
					}
				}
				else
					label << track.m_type << " " << std::setprecision(2) << track.m_velocity << ": " << track.m_confidence;
#endif
				int baseLine = 0;
				cv::Size labelSize = cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

                cv::Rect brect = track.m_rrect.boundingRect();
				if (brect.x < 0)
				{
					brect.width = std::min(brect.width, frame.cols - 1);
					brect.x = 0;
				}
				else if (brect.x + brect.width >= frame.cols)
				{
					brect.x = std::max(0, frame.cols - brect.width - 1);
					brect.width = std::min(brect.width, frame.cols - 1);
				}
				if (brect.y - labelSize.height < 0)
				{
					brect.height = std::min(brect.height, frame.rows - 1);
					brect.y = labelSize.height;
				}
				else if (brect.y + brect.height >= frame.rows)
				{
					brect.y = std::max(0, frame.rows - brect.height - 1);
					brect.height = std::min(brect.height, frame.rows - 1);
				}
				DrawFilledRect(frame, cv::Rect(cv::Point(brect.x, brect.y - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)), cv::Scalar(200, 200, 200), 200);
                cv::putText(frame, label.str(), brect.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
			}
		}

        //m_detector->CalcMotionMap(frame);
	}
};

#endif

#ifdef BUILD_YOLO_TENSORRT
// ----------------------------------------------------------------------

///
/// \brief The YoloTensorRTExample class
///
class YoloTensorRTExample : public VideoExample
{
public:
	YoloTensorRTExample(const cv::CommandLineParser& parser)
		:
		VideoExample(parser)
	{
	}

protected:
	///
	/// \brief InitDetector
	/// \param frame
	/// \return
	///
	bool InitDetector(cv::UMat frame)
	{
		config_t config;

#ifdef _WIN32
		std::string pathToModel = "../../data/";
#else
		std::string pathToModel = "../data/";
#endif
#if 0
		config.emplace("modelConfiguration", pathToModel + "yolov3-tiny.cfg");
		config.emplace("modelBinary", pathToModel + "yolov3-tiny.weights");
		config.emplace("confidenceThreshold", "0.5");
#else
		config.emplace("modelConfiguration", pathToModel + "yolov3.cfg");
		config.emplace("modelBinary", pathToModel + "yolov3.weights");
		config.emplace("confidenceThreshold", "0.7");
#endif
		config.emplace("classNames", pathToModel + "coco.names");
		config.emplace("maxCropRatio", "-1");

		config.emplace("white_list", "person");
		config.emplace("white_list", "car");
		config.emplace("white_list", "bicycle");
		config.emplace("white_list", "motorbike");
		config.emplace("white_list", "bus");
		config.emplace("white_list", "truck");
		//config.emplace("white_list", "traffic light");
		//config.emplace("white_list", "stop sign");

		m_detector = std::unique_ptr<BaseDetector>(CreateDetector(tracking::Detectors::Yolo_TensorRT, config, frame));
		if (m_detector.get())
		{
			m_detector->SetMinObjectSize(cv::Size(frame.cols / 40, frame.rows / 40));
			return true;
		}
		return false;
	}

	///
	/// \brief InitTracker
	/// \param frame
	/// \return
	///
	bool InitTracker(cv::UMat frame)
	{
		m_trackerSettings.SetDistance(tracking::DistCenters);
		m_trackerSettings.m_kalmanType = tracking::KalmanLinear;
		m_trackerSettings.m_filterGoal = tracking::FilterCenter;
		m_trackerSettings.m_lostTrackType = tracking::TrackKCF;      // Use visual objects tracker for collisions resolving
		m_trackerSettings.m_matchType = tracking::MatchHungrian;
		m_trackerSettings.m_dt = 0.3f;                                // Delta time for Kalman filter
		m_trackerSettings.m_accelNoiseMag = 0.2f;                     // Accel noise magnitude for Kalman filter
		m_trackerSettings.m_distThres = 0.8f;                         // Distance threshold between region and object on two frames
		m_trackerSettings.m_minAreaRadiusPix = frame.rows / 20.f;
		m_trackerSettings.m_maximumAllowedSkippedFrames = cvRound(2 * m_fps); // Maximum allowed skipped frames
		m_trackerSettings.m_maxTraceLength = cvRound(5 * m_fps);      // Maximum trace length

		m_trackerSettings.AddNearTypes("car", "bus", false);
		m_trackerSettings.AddNearTypes("car", "truck", false);
		m_trackerSettings.AddNearTypes("person", "bicycle", true);
		m_trackerSettings.AddNearTypes("person", "motorbike", true);

		m_tracker = std::make_unique<CTracker>(m_trackerSettings);

		return true;
	}

	///
	/// \brief DrawData
	/// \param frame
	/// \param framesCounter
	/// \param currTime
	///
	void DrawData(cv::Mat frame, int framesCounter, int currTime)
	{
		m_tracks = m_tracker->GetTracks();

		if (m_showLogs)
		{
			std::cout << "Frame " << framesCounter << ": tracks = " << m_tracks.size() << ", time = " << currTime << std::endl;
		}

		for (const auto& track : m_tracks)
		{
			if (track.IsRobust(2,                           // Minimal trajectory size
				0.5f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
				cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
				)
			{
				DrawTrack(frame, 1, track);


				std::stringstream label;
				label << track.m_type << " " << std::setprecision(2) << track.m_velocity << ": " << track.m_confidence;
				int baseLine = 0;
				cv::Size labelSize = cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

				cv::Rect brect = track.m_rrect.boundingRect();
				if (brect.x < 0)
				{
					brect.width = std::min(brect.width, frame.cols - 1);
					brect.x = 0;
				}
				else if (brect.x + brect.width >= frame.cols)
				{
					brect.x = std::max(0, frame.cols - brect.width - 1);
					brect.width = std::min(brect.width, frame.cols - 1);
				}
				if (brect.y - labelSize.height < 0)
				{
					brect.height = std::min(brect.height, frame.rows - 1);
					brect.y = labelSize.height;
				}
				else if (brect.y + brect.height >= frame.rows)
				{
					brect.y = std::max(0, frame.rows - brect.height - 1);
					brect.height = std::min(brect.height, frame.rows - 1);
				}
				DrawFilledRect(frame, cv::Rect(cv::Point(brect.x, brect.y - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)), cv::Scalar(200, 200, 200), 150);
				cv::putText(frame, label.str(), brect.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
			}
		}

		//m_detector->CalcMotionMap(frame);
	}
};

#endif
