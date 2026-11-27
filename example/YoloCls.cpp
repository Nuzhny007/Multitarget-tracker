#include "YoloCls.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <opencv2/imgproc.hpp>

namespace
{

class TrtLogger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
            std::cerr << "[YoloCls/TRT] " << msg << std::endl;
    }
};

TrtLogger gLogger;

int64_t Volume(const nvinfer1::Dims& dims)
{
    int64_t volume = 1;
    for (int i = 0; i < dims.nbDims; ++i)
        volume *= static_cast<int64_t>(dims.d[i]);
    return volume;
}

}  // namespace

YoloCls::~YoloCls()
{
    Destroy();
}

void YoloCls::Destroy()
{
    if (m_dIn)
    {
        cudaFree(m_dIn);
        m_dIn = nullptr;
    }
    if (m_dOut)
    {
        cudaFree(m_dOut);
        m_dOut = nullptr;
    }
    if (m_context)
    {
        delete static_cast<nvinfer1::IExecutionContext*>(m_context);
        m_context = nullptr;
    }
    if (m_engine)
    {
        delete static_cast<nvinfer1::ICudaEngine*>(m_engine);
        m_engine = nullptr;
    }
    if (m_runtime)
    {
        delete static_cast<nvinfer1::IRuntime*>(m_runtime);
        m_runtime = nullptr;
    }
}

bool YoloCls::Init(const std::string& enginePath, const std::string& namesPath,
                   int imageSize, std::string* error)
{
    Destroy();

    std::ifstream file(enginePath, std::ios::binary);
    if (!file.good())
    {
        if (error)
            *error = "can not open classifier engine: " + enginePath;
        return false;
    }
    file.seekg(0, file.end);
    const std::streamsize size = file.tellg();
    file.seekg(0, file.beg);
    if (size <= 0)
    {
        if (error)
            *error = "classifier engine file is empty: " + enginePath;
        return false;
    }
    std::vector<char> trtModel(static_cast<size_t>(size));
    if (!file.read(trtModel.data(), size))
    {
        if (error)
            *error = "can not read classifier engine: " + enginePath;
        return false;
    }
    file.close();

    m_imageSize = imageSize;

    m_runtime = nvinfer1::createInferRuntime(gLogger);
    auto* runtime = static_cast<nvinfer1::IRuntime*>(m_runtime);
    if (!runtime)
    {
        if (error)
            *error = "can not create TensorRT runtime";
        return false;
    }

    m_engine = runtime->deserializeCudaEngine(trtModel.data(), trtModel.size());
    auto* engine = static_cast<nvinfer1::ICudaEngine*>(m_engine);
    if (!engine)
    {
        if (error)
            *error = "can not deserialize classifier engine: " + enginePath;
        return false;
    }

    if (engine->getNbIOTensors() != 2)
    {
        if (error)
            *error = "classifier engine must have exactly 2 I/O tensors";
        return false;
    }

    const char* inName = engine->getIOTensorName(0);
    const char* outName = engine->getIOTensorName(1);
    if (!inName || !outName)
    {
        if (error)
            *error = "classifier engine has no I/O tensor names";
        return false;
    }

    const nvinfer1::Dims inDims = engine->getTensorShape(inName);
    const nvinfer1::Dims outDims = engine->getTensorShape(outName);
    if (inDims.nbDims != 4 || inDims.d[1] != 3 || inDims.d[2] != m_imageSize || inDims.d[3] != m_imageSize)
    {
        if (error)
            *error = "classifier engine input is not 1x3x" + std::to_string(m_imageSize) +
                     "x" + std::to_string(m_imageSize);
        return false;
    }

    m_inElems = static_cast<int>(Volume(inDims));
    m_outElems = static_cast<int>(Volume(outDims));
    m_numClasses = m_outElems;

    m_context = engine->createExecutionContext();
    if (!m_context)
    {
        if (error)
            *error = "can not create TensorRT execution context";
        return false;
    }

    if (cudaMalloc(&m_dIn, static_cast<size_t>(m_inElems) * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&m_dOut, static_cast<size_t>(m_outElems) * sizeof(float)) != cudaSuccess)
    {
        if (error)
            *error = "can not allocate CUDA buffers";
        return false;
    }
    m_hIn.assign(static_cast<size_t>(m_inElems), 0.f);
    m_hOut.assign(static_cast<size_t>(m_outElems), 0.f);

    m_names.clear();
    std::ifstream namesFile(namesPath);
    if (namesFile.is_open())
    {
        std::string line;
        while (std::getline(namesFile, line))
        {
            if (!line.empty() && line.back() == '\r')
                line.pop_back();
            m_names.push_back(line);
        }
    }
    if (static_cast<int>(m_names.size()) != m_numClasses)
        std::cerr << "YoloCls: warning: " << m_names.size() << " class names for "
                  << m_numClasses << " logits (" << namesPath << ")" << std::endl;

    std::cout << "YoloCls: engine loaded: " << enginePath << ", input "
              << inDims.d[0] << "x" << inDims.d[1] << "x" << inDims.d[2] << "x" << inDims.d[3]
              << ", classes " << m_numClasses << std::endl;
    return true;
}

const std::string& YoloCls::ClassName(int index) const
{
    static const std::string kEmpty;
    if (index < 0 || index >= static_cast<int>(m_names.size()))
        return kEmpty;
    return m_names[static_cast<size_t>(index)];
}

bool YoloCls::Classify(const cv::Mat& img, bool swapRB, int* top1, float* msp,
                       std::vector<std::pair<int, float>>* topk)
{
    if (!m_engine || !m_context || img.empty())
        return false;

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(m_imageSize, m_imageSize), 0, 0, cv::INTER_LINEAR);

    cv::Mat rgb;
    if (swapRB)
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    else
        rgb = resized;
    if (rgb.channels() == 1)
        cv::cvtColor(rgb, rgb, cv::COLOR_GRAY2RGB);

    cv::Mat f;
    rgb.convertTo(f, CV_32FC3, 1.0 / 255.0);

    std::vector<cv::Mat> ch;
    cv::split(f, ch);
    const int plane = m_imageSize * m_imageSize;
    std::memcpy(m_hIn.data(), ch[0].ptr<float>(), static_cast<size_t>(plane) * sizeof(float));
    std::memcpy(m_hIn.data() + plane, ch[1].ptr<float>(), static_cast<size_t>(plane) * sizeof(float));
    std::memcpy(m_hIn.data() + 2 * plane, ch[2].ptr<float>(), static_cast<size_t>(plane) * sizeof(float));

    cudaMemcpy(m_dIn, m_hIn.data(), static_cast<size_t>(m_inElems) * sizeof(float),
               cudaMemcpyHostToDevice);

    void* bindings[2] = { m_dIn, m_dOut };
    if (!static_cast<nvinfer1::IExecutionContext*>(m_context)->executeV2(bindings))
        return false;

    cudaMemcpy(m_hOut.data(), m_dOut, static_cast<size_t>(m_outElems) * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Softmax (MSP = max probability)
    float maxLogit = -1e30f;
    for (int i = 0; i < m_outElems; ++i)
        maxLogit = std::max(maxLogit, m_hOut[static_cast<size_t>(i)]);
    float sum = 0.f;
    for (int i = 0; i < m_outElems; ++i)
    {
        const float e = std::exp(m_hOut[static_cast<size_t>(i)] - maxLogit);
        m_hOut[static_cast<size_t>(i)] = e;
        sum += e;
    }
    if (sum <= 0.f)
        return false;

    int best = 0;
    float bestP = -1.f;
    for (int i = 0; i < m_outElems; ++i)
    {
        const float p = m_hOut[static_cast<size_t>(i)] / sum;
        m_hOut[static_cast<size_t>(i)] = p;
        if (p > bestP)
        {
            bestP = p;
            best = i;
        }
    }

    if (top1)
        *top1 = best;
    if (msp)
        *msp = bestP;

    if (topk)
    {
        topk->clear();
        const int k = std::min<int>(5, m_outElems);
        std::vector<int> idx(static_cast<size_t>(m_outElems));
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                          [this](int a, int b)
                          {
                              return m_hOut[static_cast<size_t>(a)] > m_hOut[static_cast<size_t>(b)];
                          });
        for (int i = 0; i < k; ++i)
            topk->emplace_back(idx[static_cast<size_t>(i)], m_hOut[static_cast<size_t>(idx[static_cast<size_t>(i)])]);
    }

    return true;
}
