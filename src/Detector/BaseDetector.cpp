#include "BaseDetector.h"
#include "MotionDetector.h"
#include "OCVDNNDetector.h"

#include <map>

#ifdef BUILD_ONNX_TENSORRT
#include "ONNXTensorRTDetector.h"
#endif

namespace
{
using DetectorRegistryMap = std::map<tracking::Detectors, BaseDetector::CreateFn>;

///
/// \brief DetectorRegistry
/// The detector factory table. Registration happens once, on the first
/// CreateDetector() call, so the built-in detectors are always available while
/// applications can add their own via BaseDetector::RegisterDetector().
///
DetectorRegistryMap& DetectorRegistry()
{
    static DetectorRegistryMap registry;
    return registry;
}

///
/// \brief RegisterBuiltinDetectors
/// Thread-safe one-time registration of the detectors compiled into this build.
///
void RegisterBuiltinDetectors()
{
    static const bool registered = []()
    {
        BaseDetector::RegisterDetector(tracking::Motion_VIBE,
            [](const cv::UMat& f) -> std::unique_ptr<BaseDetector>
            { return std::make_unique<MotionDetector>(BackgroundSubtract::BGFG_ALGS::ALG_VIBE, f); });
        BaseDetector::RegisterDetector(tracking::Motion_MOG,
            [](const cv::UMat& f) -> std::unique_ptr<BaseDetector>
            { return std::make_unique<MotionDetector>(BackgroundSubtract::BGFG_ALGS::ALG_MOG, f); });
        BaseDetector::RegisterDetector(tracking::Motion_GMG,
            [](const cv::UMat& f) -> std::unique_ptr<BaseDetector>
            { return std::make_unique<MotionDetector>(BackgroundSubtract::BGFG_ALGS::ALG_GMG, f); });
        BaseDetector::RegisterDetector(tracking::Motion_CNT,
            [](const cv::UMat& f) -> std::unique_ptr<BaseDetector>
            { return std::make_unique<MotionDetector>(BackgroundSubtract::BGFG_ALGS::ALG_CNT, f); });
        BaseDetector::RegisterDetector(tracking::Motion_MOG2,
            [](const cv::UMat& f) -> std::unique_ptr<BaseDetector>
            { return std::make_unique<MotionDetector>(BackgroundSubtract::BGFG_ALGS::ALG_MOG2, f); });
#ifdef USE_OCV_DNN
        BaseDetector::RegisterDetector(tracking::DNN_OCV,
            [](const cv::UMat& f) -> std::unique_ptr<BaseDetector>
            { return std::make_unique<OCVDNNDetector>(f); });
#endif
#ifdef BUILD_ONNX_TENSORRT
        BaseDetector::RegisterDetector(tracking::ONNX_TensorRT,
            [](const cv::UMat& f) -> std::unique_ptr<BaseDetector>
            { return std::make_unique<ONNXTensorRTDetector>(f); });
#endif
        return true;
    }();
    (void)registered;
}
} // namespace

namespace mtracking
{

///
/// \brief RegisterDetector
/// \param detectorType
/// \param fn
///
void BaseDetector::RegisterDetector(tracking::Detectors detectorType, CreateFn fn)
{
    DetectorRegistry()[detectorType] = fn;
}

///
/// \brief CreateDetector
/// \param detectorType
/// \param gray
/// \return
///
std::unique_ptr<BaseDetector> BaseDetector::CreateDetector(tracking::Detectors detectorType,
                                                           const config_t& config,
                                                           const cv::UMat& frame)
{
    RegisterBuiltinDetectors();

    std::unique_ptr<BaseDetector> detector;
    auto it = DetectorRegistry().find(detectorType);
    if (it != DetectorRegistry().end())
        detector = it->second(frame);
    else if (detectorType == tracking::ONNX_TensorRT)
        std::cerr << "TensorRT inference engine was not configured in CMake" << std::endl;

    if (!detector)
    {
        std::cerr << "Detector type " << static_cast<int>(detectorType) << " is not available in this build" << std::endl;
        return nullptr;
    }
    if (!detector->Init(config))
    {
        detector.reset();
        std::cerr << "Detector type " << static_cast<int>(detectorType) << " init failed" << std::endl;
        return nullptr;
    }
    return detector;
}

///
std::unique_ptr<BaseDetector> BaseDetector::CreateDetectorKV(tracking::Detectors detectorType, const KeyVal& config, const cv::Mat& gray)
{
    config_t mconfig;
    for (auto kv : config.m_config)
    {
        mconfig.emplace(kv.first, kv.second);
    }
    cv::UMat uframe = gray.getUMat(cv::ACCESS_READ);
    return CreateDetector(detectorType, mconfig, uframe);
}

} // namespace mtracking
