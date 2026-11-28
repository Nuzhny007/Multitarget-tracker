#pragma once

#include <deque>
#include <string>

#include "defines.h"

///
/// \brief Detector-specific settings.
///
/// Split from TrackerSettings (roadmap M2): the model/backend/threshold fields
/// consumed by the detector factory and the detector implementations live here.
/// TrackerSettings inherits this struct, so `settings.m_nnWeights`-style access
/// keeps working for existing code and examples.
///
struct DetectorSettings
{
    ///
    std::string m_nnWeights = "data/yolov4-tiny_best.weights";

    ///
    std::string m_nnConfig = "data/yolov4-tiny.cfg";

    ///
    std::string m_classNames = "data/traffic.names";

    ///
    std::deque<std::string> m_whiteList;

    ///
    float m_confidenceThreshold = 0.5f;

    ///
    float m_maxCropRatio = -1.f;

    ///
    int m_maxBatch = 1;

    ///
    int m_gpuId = 0;

    ///
    /// \brief Neural network input size
    cv::Size m_inputSize{ 0, 0 };

    ///
    /// YOLOV2
    /// YOLOV3
    /// YOLOV4
    /// YOLOV4_TINY
    /// YOLOV5
    std::string m_netType = "YOLOV4";

    ///
    /// INT8
    /// FP16
    /// FP32
    std::string m_inferencePrecision = "FP16";

    // The detector backend for a settings struct that is not filled from an
    // ini file (ParseTrackerSettings reads "detector_backend" and otherwise
    // also falls back to DNN_OCV). ONNX_TensorRT used to be the default: a
    // raw struct then required a TensorRT-enabled build (H21).
    int m_detectorBackend = static_cast<int>(tracking::Detectors::DNN_OCV);

    // DNN_TARGET_CPU
    // DNN_TARGET_OPENCL
    // DNN_TARGET_OPENCL_FP16
    // DNN_TARGET_MYRIAD
    // DNN_TARGET_CUDA
    // DNN_TARGET_CUDA_FP16
    std::string m_dnnTarget = "DNN_TARGET_CPU";

    // DNN_BACKEND_DEFAULT
    // DNN_BACKEND_HALIDE
    // DNN_BACKEND_INFERENCE_ENGINE
    // DNN_BACKEND_OPENCV
    // DNN_BACKEND_VKCOM
    // DNN_BACKEND_CUDA
    // DNN_BACKEND_INFERENCE_ENGINE_NGRAPH
    // DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019
    std::string m_dnnBackend = "DNN_BACKEND_OPENCV";

    // For TensorRT optimization, bytes
    size_t m_maxVideoMemory = 0;
};
