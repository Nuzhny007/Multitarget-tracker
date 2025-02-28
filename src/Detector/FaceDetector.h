#pragma once

#include "BaseDetector.h"

#if (CV_VERSION_MAJOR >= 5)
#include <opencv2/xobjdetect.hpp>
#endif //(CV_VERSION_MAJOR < 5)

///
/// \brief The FaceDetector class
///
class FaceDetector final : public BaseDetector
{
public:
    FaceDetector(const cv::UMat& gray);
    FaceDetector(const cv::Mat& gray);
    ~FaceDetector(void) = default;

    bool Init(const config_t& config) override;

    void Detect(const cv::UMat& gray) override;

	bool CanGrayProcessing() const override
	{
		return true;
	}

private:
    cv::CascadeClassifier m_cascade;
};
