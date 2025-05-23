#include "class_detector.h"
#include "class_yolo_detector.hpp"
#include "YoloONNX.hpp"

#include "YoloONNXv5_bb.hpp"
#include "YoloONNXv6_bb.hpp"
#include "YoloONNXv7_bb.hpp"
#include "YoloONNXv7_instance.hpp"
#include "YoloONNXv8_bb.hpp"
#include "YoloONNXv8_obb.hpp"
#include "YoloONNXv8_instance.hpp"
#include "YoloONNXv9_bb.hpp"
#include "YoloONNXv10_bb.hpp"
#include "YoloONNXv11_bb.hpp"
#include "YoloONNXv11_obb.hpp"
#include "YoloONNXv11_instance.hpp"
#include "YoloONNXv12_bb.hpp"
#include "RFDETR_bb.hpp"
#include "DFINE_bb.hpp"

namespace tensor_rt
{
    ///
    /// \brief The Detector::Impl class
    ///
    class Detector::Impl
	{
	public:
		Impl() = default;
        virtual ~Impl() = default;

        virtual bool Init(const Config& config) = 0;
        virtual void Detect(const std::vector<cv::Mat>& mat_image, std::vector<BatchResult>& vec_batch_result) = 0;
        virtual cv::Size GetInputSize() const = 0;
    };

    ///
    /// \brief The YoloDectectorImpl class
    ///
    class YoloDectectorImpl final : public Detector::Impl
    {
    public:
        virtual bool Init(const Config& config) override
        {
            m_detector.init(config);
            return true;
        }
        virtual void Detect(const std::vector<cv::Mat>& mat_image, std::vector<BatchResult>& vec_batch_result) override
        {
            m_detector.detect(mat_image, vec_batch_result);
        }
        virtual cv::Size GetInputSize() const override
        {
            return m_detector.get_input_size();
        }

    private:
        YoloDectector m_detector;
    };

    ///
    /// \brief The YoloDectectorImpl class
    ///
    class YoloONNXImpl final : public Detector::Impl
    {
    public:
        bool Init(const Config& config) override
        {
            // The onnx file to load
            m_params.onnxFileName = config.file_model_cfg; //"yolov6s.onnx"

            switch (config.net_type)
            {
            case ModelType::YOLOV5:
                m_detector = std::make_unique<YOLOv5_bb_onnx>(m_params.inputTensorNames, m_params.outputTensorNames);
                break;
            case ModelType::YOLOV6:
                m_detector = std::make_unique<YOLOv6_bb_onnx>(m_params.inputTensorNames, m_params.outputTensorNames);
                break;
            case ModelType::YOLOV7:
                m_detector = std::make_unique<YOLOv7_bb_onnx>(m_params.inputTensorNames, m_params.outputTensorNames);
                break;
            case ModelType::YOLOV7Mask:
                m_detector = std::make_unique<YOLOv7_instance_onnx>(m_params.inputTensorNames, m_params.outputTensorNames);
                break;
            case ModelType::YOLOV8:
                m_detector = std::make_unique<YOLOv8_bb_onnx>(m_params.inputTensorNames, m_params.outputTensorNames);
                break;
            case ModelType::YOLOV8_OBB:
                m_detector = std::make_unique<YOLOv8_obb_onnx>(m_params.inputTensorNames, m_params.outputTensorNames);
                break;
            case ModelType::YOLOV8Mask:
                m_detector = std::make_unique<YOLOv8_instance_onnx>(m_params.inputTensorNames, m_params.outputTensorNames);
                break;
            case ModelType::YOLOV9:
                m_detector = std::make_unique<YOLOv9_bb_onnx>(m_params.inputTensorNames, m_params.outputTensorNames);
                break;
            case ModelType::YOLOV10:
                m_detector = std::make_unique<YOLOv10_bb_onnx>(m_params.inputTensorNames, m_params.outputTensorNames);
                break;
            case ModelType::YOLOV11:
                m_detector = std::make_unique<YOLOv11_bb_onnx>(m_params.inputTensorNames, m_params.outputTensorNames);
                break;
            case ModelType::YOLOV11_OBB:
                m_detector = std::make_unique<YOLOv11_obb_onnx>(m_params.inputTensorNames, m_params.outputTensorNames);
                break;
            case ModelType::YOLOV11Mask:
                m_detector = std::make_unique<YOLOv11_instance_onnx>(m_params.inputTensorNames, m_params.outputTensorNames);
                break;
            case ModelType::YOLOV12:
                m_detector = std::make_unique<YOLOv12_bb_onnx>(m_params.inputTensorNames, m_params.outputTensorNames);
                break;
            case ModelType::RFDETR:
                m_detector = std::make_unique<RFDETR_bb_onnx>(m_params.inputTensorNames, m_params.outputTensorNames);
                break;
            case ModelType::DFINE:
                m_detector = std::make_unique<DFINE_bb_onnx>(m_params.inputTensorNames, m_params.outputTensorNames);
                break;
            }                

            // Threshold values
            m_params.confThreshold = config.detect_thresh;
            m_params.nmsThreshold = 0.5;

            m_params.videoMemory = config.video_memory;

            // Batch size, you can modify to other batch size values if needed
            m_params.explicitBatchSize = config.batch_size;

            m_params.m_precision = config.inference_precision;
            m_params.m_netType = config.net_type;

            std::string precisionStr;
            std::map<tensor_rt::Precision, std::string> dictprecision;
            dictprecision[tensor_rt::INT8] = "kINT8";
            dictprecision[tensor_rt::FP16] = "kHALF";
            dictprecision[tensor_rt::FP32] = "kFLOAT";
            auto precision = dictprecision.find(m_params.m_precision);
            if (precision != dictprecision.end())
                precisionStr = precision->second;
            m_params.engineFileName = config.file_model_cfg + "-" + precisionStr + "-batch" + std::to_string(config.batch_size) + ".engine";

            return m_detector->Init(m_params);
        }

        void Detect(const std::vector<cv::Mat>& mat_image, std::vector<BatchResult>& vec_batch_result) override
        {
            vec_batch_result.clear();
            if (vec_batch_result.capacity() < mat_image.size())
                vec_batch_result.reserve(mat_image.size());

            m_detector->Detect(mat_image, vec_batch_result);
        }

        cv::Size GetInputSize() const override
        {
            return m_detector->GetInputSize();
        }

    private:
        std::unique_ptr<YoloONNX> m_detector;
        SampleYoloParams m_params;
    };

	///
	/// \brief Detector::Detector
	///
	Detector::Detector() noexcept
	{
	}

    ///
    /// \brief Detector::~Detector
    ///
    Detector::~Detector()
    {
        if (m_impl)
            delete m_impl;
    }

    ///
    /// \brief Detector::Init
    /// \param config
    ///
    bool Detector::Init(const Config& config)
	{
        if (m_impl)
            delete m_impl;

        if (fs::path(config.file_model_cfg).extension() == ".onnx")
            m_impl = new YoloONNXImpl();
        else
            m_impl = new YoloDectectorImpl();

        bool res = m_impl->Init(config);
        assert(res);
        return res;
	}

    ///
    /// \brief Detector::Detect
    /// \param mat_image
    /// \param vec_batch_result
    ///
    void Detector::Detect(const std::vector<cv::Mat>& mat_image, std::vector<BatchResult>& vec_batch_result)
	{
        m_impl->Detect(mat_image, vec_batch_result);
	}

    ///
    /// \brief Detector::GetInputSize
    /// \return
    ///
    cv::Size Detector::GetInputSize() const
	{
        return m_impl->GetInputSize();
	}
}
