#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

///
/// \brief Description of one ReID model.
///
/// All preprocessing parameters are part of the model description: the
/// OpenCV DNN graph itself does not know which mean/std and channel order the
/// model was trained with.
///
struct EmbeddingModelDesc
{
    std::string m_name;
    std::string m_path;
    cv::Size m_inputSize{ 128, 256 };
    cv::Scalar m_mean{ 0, 0, 0 };
    cv::Scalar m_std{ 1, 1, 1 };
    bool m_swapRB = false;
    int m_maxBatch = 16;
    std::string m_backend = "cuda_fp16";
    bool m_warmup = true;
    int m_minCropSide = 0;
};

///
/// \brief The EmbeddingsCalculator class
///
/// One instance owns one cv::dnn::Net. Inference is always called through a
/// mutex because OpenCV DNN Net is not thread-safe. Crops are resized and
/// normalized according to EmbeddingModelDesc, then processed in batches
/// (chunked by m_maxBatch) to amortize the per-call overhead on GPU.
///
class EmbeddingsCalculator
{
public:
    EmbeddingsCalculator() = default;
    virtual ~EmbeddingsCalculator() = default;

    EmbeddingsCalculator(const EmbeddingsCalculator&) = delete;
    EmbeddingsCalculator& operator=(const EmbeddingsCalculator&) = delete;

    ///
    /// \brief Initialize the model from a full description.
    /// \param desc model path, input size, mean/std, channel order, batch limit,
    ///        preferred backend
    /// \param modelId stable id used by tracks to avoid comparing embeddings
    ///        produced by different models
    /// \return true when the model is loaded and a warm-up inference succeeded
    ///
    bool Initialize(const EmbeddingModelDesc& desc, int modelId)
    {
#ifdef USE_OCV_EMBEDDINGS
        m_desc = desc;
        m_modelId = modelId;
        m_maxBatch = std::max(1, desc.m_maxBatch);
        m_batchSupported = true;

        if (m_desc.m_path.empty() || m_desc.m_inputSize.width <= 0 || m_desc.m_inputSize.height <= 0)
        {
            std::cerr << "EmbeddingsCalculator: invalid model description (path/input size)" << std::endl;
            return false;
        }

        try
        {
            m_net = cv::dnn::readNet(m_desc.m_path);
        }
        catch (const cv::Exception& e)
        {
            std::cerr << "EmbeddingsCalculator: readNet failed: " << e.what() << std::endl;
            return false;
        }

        if (m_net.empty())
        {
            std::cerr << "EmbeddingsCalculator: model is empty: " << m_desc.m_path << std::endl;
            return false;
        }

        ConfigureBackend(m_desc.m_backend);

        // A single one-image call also determines the output dimensionality.
        cv::Mat dummy(m_desc.m_inputSize.height, m_desc.m_inputSize.width, CV_8UC3, cv::Scalar(127, 128, 129));
        cv::Mat probe;
        if (!Forward(std::vector<cv::Mat>{ dummy }, probe) || probe.empty())
        {
            std::cerr << "EmbeddingsCalculator: probe inference failed: " << m_desc.m_path << std::endl;
            return false;
        }
        m_outputDim = probe.cols;

        if (m_desc.m_warmup)
        {
            // Trigger CUDA/cuDNN algorithm selection for every batch size we
            // may see in a frame. cuDNN autotuning is per exact shape, so a
            // missing size (e.g. 25) can otherwise pick a very slow algorithm.
            const int warmupLimit = m_isCpu ? std::min(m_maxBatch, 4) : std::min(m_maxBatch, 32);
            for (int warmupBatch = 1; warmupBatch <= warmupLimit; ++warmupBatch)
            {
                if (!m_batchSupported && warmupBatch > 1)
                    break;
                std::vector<cv::Mat> warmup;
                warmup.reserve(static_cast<size_t>(warmupBatch));
                for (int i = 0; i < warmupBatch; ++i)
                    warmup.push_back(dummy);
                cv::Mat warmupResult;
                Forward(warmup, warmupResult);
            }
        }

        std::cout << "ReID model '" << m_desc.m_name << "' loaded: " << m_desc.m_path
                  << ", input " << m_desc.m_inputSize.width << "x" << m_desc.m_inputSize.height
                  << ", output " << m_outputDim
                  << ", backend " << (m_isCpu ? "CPU" : "GPU")
                  << ", max batch " << m_maxBatch << std::endl;
        return true;
#else
        (void)desc;
        (void)modelId;
        std::cerr << "EmbeddingsCalculator was disabled in CMAKE! Check USE_OCV_EMBEDDINGS." << std::endl;
        return false;
#endif
    }

    ///
    /// \brief Backward-compatible initialization (legacy OpenCV DNN path).
    /// Uses CPU, no mean/std normalization and BGR input, matching the old
    /// behavior. New code should use the EmbeddingModelDesc overload.
    ///
    bool Initialize(const std::string& cfgName, const std::string& weightsName, const cv::Size& inputLayer)
    {
        EmbeddingModelDesc desc;
        desc.m_name = "legacy";
        desc.m_path = weightsName;
        desc.m_inputSize = inputLayer;
        desc.m_backend = "cpu";
        desc.m_warmup = false;
        (void)cfgName;
        return Initialize(desc, 0);
    }

    ///
    bool IsInitialized() const
    {
#ifdef USE_OCV_EMBEDDINGS
        return !m_net.empty();
#else
        return false;
#endif
    }

    ///
    /// \brief True when inference runs on CPU (used by the CPU fallback budget).
    ///
    bool IsCpu() const
    {
        return m_isCpu;
    }

    ///
    int GetModelId() const
    {
        return m_modelId;
    }

    ///
    int GetOutputDim() const
    {
        return m_outputDim;
    }

    ///
    const cv::Size& GetInputSize() const
    {
        return m_desc.m_inputSize;
    }

    ///
    int GetMinCropSide() const
    {
        return m_desc.m_minCropSide;
    }

    ///
    /// \brief Extract L2-normalized embeddings for a batch of crops.
    /// \param crops source crops (any size/channels, resized internally)
    /// \param embeddings output [N, D] CV_32F matrix, each row normalized
    /// \return false when the model is not initialized or inference failed
    ///
    bool Extract(const std::vector<cv::Mat>& crops, cv::Mat& embeddings)
    {
#ifdef USE_OCV_EMBEDDINGS
        if (crops.empty())
        {
            embeddings.release();
            return true;
        }
        if (!IsInitialized())
        {
            embeddings.release();
            return false;
        }

        std::lock_guard<std::mutex> lock(m_mutex);
        return Forward(crops, embeddings);
#else
        (void)crops;
        embeddings.release();
        return false;
#endif
    }

private:
#ifdef USE_OCV_EMBEDDINGS
    ///
    /// \brief Configure the preferred backend and fall back to CPU when the
    /// current OpenCV build has no CUDA support.
    ///
    void ConfigureBackend(const std::string& requested)
    {
        m_isCpu = true;
        std::string backend = requested;
        std::transform(backend.begin(), backend.end(), backend.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

        // Always start from a safe CPU configuration.
        m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        if (backend == "cuda_fp16" || backend == "cuda")
        {
            const cv::dnn::Target target = (backend == "cuda_fp16")
                ? cv::dnn::DNN_TARGET_CUDA_FP16
                : cv::dnn::DNN_TARGET_CUDA;

            bool available = true;
#if (CV_VERSION_MAJOR > 4) || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 2)
            try
            {
                available = false;
                const auto pairs = cv::dnn::getAvailableBackends();
                for (const auto& pair : pairs)
                {
                    if (pair.first == cv::dnn::DNN_BACKEND_CUDA && pair.second == target)
                    {
                        available = true;
                        break;
                    }
                }
            }
            catch (const cv::Exception&)
            {
                available = false;
            }
#endif
            if (!available)
            {
                std::cerr << "EmbeddingsCalculator: CUDA backend/target is unavailable, using CPU" << std::endl;
                return;
            }

            try
            {
                m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                m_net.setPreferableTarget(target);
                m_isCpu = false;
            }
            catch (const cv::Exception& e)
            {
                std::cerr << "EmbeddingsCalculator: cannot enable CUDA (" << e.what()
                          << "), using CPU" << std::endl;
                m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
                m_isCpu = true;
            }
        }
        else if (backend == "opencl")
        {
            try
            {
                m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                m_net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
                m_isCpu = false;
            }
            catch (const cv::Exception& e)
            {
                std::cerr << "EmbeddingsCalculator: cannot enable OpenCL (" << e.what()
                          << "), using CPU" << std::endl;
                m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
                m_isCpu = true;
            }
        }
        // "cpu" and any unknown value keep the safe CPU configuration above.
    }

    ///
    /// \brief Convert a crop to 3-channel 8-bit for blobFromImages.
    ///
    static bool PrepareCrop(const cv::Mat& crop, cv::Mat& converted)
    {
        if (crop.empty())
            return false;

        cv::Mat src = crop;
        if (src.depth() != CV_8U)
            src.convertTo(src, CV_8U);

        if (src.channels() == 1)
            cv::cvtColor(src, converted, cv::COLOR_GRAY2BGR);
        else if (src.channels() == 4)
            cv::cvtColor(src, converted, cv::COLOR_BGRA2BGR);
        else if (src.channels() == 3)
            converted = src;
        else
            return false;
        return true;
    }

    ///
    /// \brief Round a batch size up to the next power of two when the padding
    /// overhead is small. Stable batch buckets avoid a new cuDNN autotune for
    /// every slightly different number of objects in a frame.
    ///
    int TargetBatchSize(int requested) const
    {
        if (requested <= 0)
            return 0;

        int target = 1;
        while (target < requested)
            target *= 2;
        target = std::min(target, m_maxBatch);
        if (target < requested)
            target = requested;
        // On GPU, always use a power-of-two batch bucket. Some OpenCV CUDA
        // builds pick a very slow algorithm for "odd" batch sizes (e.g. 25),
        // so a small amount of padding is cheaper than a 5-10x slowdown.
        // CPU keeps the exact size unless the padding overhead is tiny.
        if (!m_isCpu)
            return target;
        if (target > requested && target * 4 > requested * 5)
            return requested;
        return target;
    }

    ///
    /// \brief Run one actual batch through the network.
    ///
    bool ForwardBatch(const std::vector<cv::Mat>& crops, cv::Mat& embeddings)
    {
        if (crops.empty())
        {
            embeddings.release();
            return true;
        }

        const int requested = static_cast<int>(crops.size());
        const int target = TargetBatchSize(requested);
        std::vector<cv::Mat> padded = crops;
        if (target > requested)
            padded.resize(static_cast<size_t>(target), crops.back());

        std::vector<cv::Mat> resized;
        resized.reserve(padded.size());
        for (const cv::Mat& crop : padded)
        {
            cv::Mat converted;
            if (!PrepareCrop(crop, converted))
                return false;

            cv::Mat resizedCrop;
            cv::resize(converted, resizedCrop, m_desc.m_inputSize, 0, 0, cv::INTER_LINEAR);
            resized.push_back(resizedCrop);
        }

        cv::Mat blob = cv::dnn::blobFromImages(
            resized, 1.0 / 255.0, m_desc.m_inputSize, cv::Scalar(), m_desc.m_swapRB, false, CV_32F);
        if (blob.empty())
            return false;

        ApplyMeanStd(blob, static_cast<int>(resized.size()));

        m_net.setInput(blob);
        cv::Mat out = m_net.forward();
        if (!NormalizeOutput(out, static_cast<int>(resized.size()), embeddings))
            return false;

        if (embeddings.rows > requested)
            embeddings = embeddings.rowRange(0, requested).clone();
        return true;
    }

    ///
    /// \brief Apply per-channel (x - mean) / std after blobFromImages scaling.
    ///
    void ApplyMeanStd(cv::Mat& blob, int batch)
    {
        const bool hasMean = m_desc.m_mean != cv::Scalar(0, 0, 0);
        const bool hasStd = m_desc.m_std != cv::Scalar(1, 1, 1);
        if (!hasMean && !hasStd)
            return;

        if (blob.dims != 4 || blob.size[1] != 3 || batch <= 0)
            return;

        cv::Mat flat = blob.reshape(1, batch * 3);
        for (int n = 0; n < batch; ++n)
        {
            for (int c = 0; c < 3; ++c)
            {
                const double mean = m_desc.m_mean[c];
                const double std = std::fabs(m_desc.m_std[c]) > 1e-12 ? m_desc.m_std[c] : 1.0;
                cv::Mat row = flat.row(n * 3 + c);
                row.convertTo(row, CV_32F, 1.0 / std, -mean / std);
            }
        }
    }

    ///
    /// \brief Flatten the network output to [N, D] and L2-normalize each row.
    ///
    static bool NormalizeOutput(const cv::Mat& raw, int batch, cv::Mat& embeddings)
    {
        if (raw.empty() || batch <= 0)
            return false;

        const size_t total = raw.total();
        if (total == 0 || total % static_cast<size_t>(batch) != 0)
            return false;

        const int dim = static_cast<int>(total / static_cast<size_t>(batch));
        cv::Mat flat = raw.isContinuous() ? raw : raw.clone();
        embeddings = flat.reshape(1, batch);
        if (embeddings.cols != dim)
            return false;

        for (int i = 0; i < batch; ++i)
        {
            cv::Mat row = embeddings.row(i);
            const double norm = cv::norm(row, cv::NORM_L2);
            if (norm > 1e-12)
                row /= norm;
        }
        return true;
    }

    ///
    /// \brief Run a batch, falling back to one-image calls if the model was
    /// exported with a fixed batch (e.g. the old OSNet ONNX with Reshape [1,-1]).
    ///
    bool Forward(const std::vector<cv::Mat>& crops, cv::Mat& embeddings)
    {
        if (crops.empty())
        {
            embeddings.release();
            return true;
        }

        if (m_batchSupported)
        {
            try
            {
                bool ok = true;
                std::vector<cv::Mat> parts;
                for (size_t offset = 0; offset < crops.size(); offset += static_cast<size_t>(m_maxBatch))
                {
                    const size_t count = std::min<size_t>(static_cast<size_t>(m_maxBatch), crops.size() - offset);
                    std::vector<cv::Mat> chunk(crops.begin() + static_cast<ptrdiff_t>(offset),
                                               crops.begin() + static_cast<ptrdiff_t>(offset + count));
                    cv::Mat part;
                    if (!ForwardBatch(chunk, part))
                    {
                        ok = false;
                        break;
                    }
                    parts.push_back(part);
                }
                if (ok && !parts.empty())
                {
                    cv::vconcat(parts, embeddings);
                    return true;
                }
                if (!ok && crops.size() > 1)
                {
                    std::cerr << "EmbeddingsCalculator: batch inference failed, falling back to single crops" << std::endl;
                    m_batchSupported = false;
                }
            }
            catch (const cv::Exception& e)
            {
                if (crops.size() > 1)
                {
                    std::cerr << "EmbeddingsCalculator: batch inference failed, falling back to single crops: "
                              << e.what() << std::endl;
                    m_batchSupported = false;
                }
                else
                {
                    std::cerr << "EmbeddingsCalculator: inference failed: " << e.what() << std::endl;
                    return false;
                }
            }
        }

        // Fixed-batch model or a failed batch: run one crop at a time.
        std::vector<cv::Mat> parts;
        parts.reserve(crops.size());
        for (const cv::Mat& crop : crops)
        {
            try
            {
                cv::Mat part;
                if (!ForwardBatch(std::vector<cv::Mat>{ crop }, part))
                    return false;
                parts.push_back(part);
            }
            catch (const cv::Exception& e)
            {
                std::cerr << "EmbeddingsCalculator: single-crop inference failed: " << e.what() << std::endl;
                return false;
            }
        }
        if (parts.empty())
            return false;
        cv::vconcat(parts, embeddings);
        return true;
    }

#endif // USE_OCV_EMBEDDINGS (private methods)

#ifdef USE_OCV_EMBEDDINGS
    cv::dnn::Net m_net;
#endif
    EmbeddingModelDesc m_desc;
    int m_modelId = -1;
    int m_outputDim = 0;
    int m_maxBatch = 1;
    bool m_batchSupported = true;
    bool m_isCpu = true;
    std::mutex m_mutex;
};
