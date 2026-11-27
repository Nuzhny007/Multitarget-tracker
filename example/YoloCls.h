#pragma once

#include <string>
#include <vector>
#include <utility>

#include <opencv2/core.hpp>

///
/// \brief The YoloCls class
///
/// Minimal TensorRT runner for a closed-set image classifier exported from
/// ultralytics (e.g. yolo11s-cls): fixed input 1x3xSxS, output 1xN logits.
/// Preprocessing matches ultralytics classify_transforms for an SxS input:
/// resize to SxS (bilinear), RGB, /255. No ImageNet mean/std.
///
class YoloCls
{
public:
    YoloCls() = default;
    ~YoloCls();
    YoloCls(const YoloCls&) = delete;
    YoloCls& operator=(const YoloCls&) = delete;

    ///
    /// \brief Init loads a serialized TensorRT engine and the class-name list.
    /// \param enginePath  path to a .engine built from the classifier ONNX
    /// \param namesPath   one class name per line (N lines, N = logits count)
    /// \param imageSize   network input side (224 for yolo11s-cls)
    ///
    bool Init(const std::string& enginePath, const std::string& namesPath,
              int imageSize, std::string* error);

    bool Valid() const { return m_engine != nullptr; }

    ///
    /// \brief Classify one crop.
    /// \param img     8UC3 image of any size
    /// \param swapRB  true if img is BGR (the network expects RGB)
    /// \param top1    output: predicted class index
    /// \param msp     output: max softmax probability (MSP score)
    /// \param topk    optional output: (class index, probability) pairs, best first
    ///
    bool Classify(const cv::Mat& img, bool swapRB, int* top1, float* msp,
                  std::vector<std::pair<int, float>>* topk = nullptr);

    const std::string& ClassName(int index) const;

private:
    void Destroy();

    struct TrtDeleter;

    void* m_runtime = nullptr;
    void* m_engine = nullptr;
    void* m_context = nullptr;
    void* m_dIn = nullptr;
    void* m_dOut = nullptr;

    std::vector<float> m_hIn;
    std::vector<float> m_hOut;

    int m_inElems = 0;
    int m_outElems = 0;
    int m_imageSize = 224;
    int m_numClasses = 0;

    std::vector<std::string> m_names;
};
