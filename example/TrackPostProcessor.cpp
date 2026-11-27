#include "TrackPostProcessor.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

#include <opencv2/imgproc.hpp>

#include "vsr_inference/vsr_inference.h"

TrackPostProcessor::~TrackPostProcessor() = default;

bool TrackPostProcessor::Init(const Settings& settings, std::string* error)
{
    m_settings = settings;
    if (!m_settings.m_enable)
        return true;

    m_cls = std::make_unique<YoloCls>();
    if (!m_cls->Init(m_settings.m_clsEngine, m_settings.m_clsNames, m_settings.m_clsSize, error))
        return false;

    if (m_settings.m_variant == 1)
    {
        vsr::Config cfg;
        cfg.model_path = m_settings.m_vsrEngine;
        cfg.frames = m_settings.m_vsrK;
        cfg.lr_size = m_settings.m_vsrLrSize;
        cfg.scale = 4;
        cfg.verbose = false;

        m_vsr = vsr::MakeBackend(vsr::Backend::kTensorRT, cfg, error);
        if (!m_vsr)
            return false;
    }

    if (!m_settings.m_dumpPath.empty())
    {
        m_dump.open(m_settings.m_dumpPath);
        if (!m_dump.is_open())
        {
            if (error)
                *error = "can not open dump file: " + m_settings.m_dumpPath;
            return false;
        }
        m_dump << "frame,id,x,y,w,h,msp,label,cls_index,rejected\n";
    }

    std::cout << "TrackPostProcessor: enabled, variant=" << m_settings.m_variant
              << ", cls=" << m_settings.m_clsEngine
              << ", msp_threshold=" << m_settings.m_mspThreshold
              << ", recheck=" << m_settings.m_recheckInterval;
    if (m_settings.m_variant == 1)
        std::cout << ", vsr=" << m_settings.m_vsrEngine << ", K=" << m_settings.m_vsrK;
    std::cout << std::endl;
    return true;
}

void TrackPostProcessor::Process(const cv::Mat& frameBGR, int frameInd,
                                 std::vector<TrackingObject>& tracks)
{
    if (!m_settings.m_enable || tracks.empty() || frameBGR.empty())
        return;

    const int recheck = std::max(1, m_settings.m_recheckInterval);
    const int lr = m_settings.m_vsrLrSize;

    std::vector<TrackingObject> kept;
    kept.reserve(tracks.size());

    for (const auto& track : tracks)
    {
        const cv::Rect box = track.m_rrect.boundingRect();
        const track_id_t::value_type key = track.m_ID.m_val;
        TrackState& st = m_tracks[key];

        bool updated = false;
        if (box.width >= m_settings.m_minObjectSide &&
            box.height >= m_settings.m_minObjectSide &&
            box.x >= 0 && box.y >= 0 &&
            box.x + box.width <= frameBGR.cols &&
            box.y + box.height <= frameBGR.rows)
        {
            const bool due = (frameInd - st.m_lastFrame >= recheck) || (st.m_lastFrame < -100000);
            const cv::Mat window = MakeWindow(frameBGR, box);
            if (!window.empty())
            {
                if (m_settings.m_variant == 1)
                {
                    cv::Mat lrCrop = window;
                    if (window.cols != lr || window.rows != lr)
                        cv::resize(window, lrCrop, cv::Size(lr, lr), 0, 0, cv::INTER_LINEAR);
                    st.m_crops.push_back(lrCrop.clone());
                    while (static_cast<int>(st.m_crops.size()) > m_settings.m_vsrK)
                        st.m_crops.pop_front();

                    if (due && static_cast<int>(st.m_crops.size()) == m_settings.m_vsrK)
                        updated = ClassifyVsr(st);
                }
                else if (due)
                {
                    updated = ClassifyRaw(window, st);
                }
            }
        }
        if (updated)
            st.m_lastFrame = frameInd;

        const bool rejected = (m_settings.m_mspThreshold > 0.f) &&
                              (st.m_msp >= 0.f) &&
                              (st.m_msp < m_settings.m_mspThreshold);
        st.m_rejected = rejected;

        if (m_dump.is_open())
        {
            m_dump << frameInd << ',' << key << ','
                   << box.x << ',' << box.y << ',' << box.width << ',' << box.height << ','
                   << st.m_msp << ',' << (st.m_label.empty() ? "?" : st.m_label) << ','
                   << st.m_clsIndex << ',' << (rejected ? 1 : 0) << '\n';
        }

        if (!rejected)
            kept.push_back(track);
    }

    tracks.swap(kept);

    // Forget the state of tracks that have been gone for a long time.
    if (frameInd % 100 == 0)
    {
        for (auto it = m_tracks.begin(); it != m_tracks.end();)
        {
            if (frameInd - it->second.m_lastFrame > 300)
                it = m_tracks.erase(it);
            else
                ++it;
        }
    }
}

cv::Mat TrackPostProcessor::MakeWindow(const cv::Mat& frame, const cv::Rect& box) const
{
    const int lr = m_settings.m_vsrLrSize;
    int side = std::max(lr, cvRound(1.25f * static_cast<float>(std::max(box.width, box.height))));
    side = std::min(side, std::min(frame.cols, frame.rows));
    if (side <= 0)
        return cv::Mat();

    const int cx = box.x + box.width / 2;
    const int cy = box.y + box.height / 2;
    int x0 = cx - side / 2;
    int y0 = cy - side / 2;
    x0 = std::max(0, std::min(x0, frame.cols - side));
    y0 = std::max(0, std::min(y0, frame.rows - side));
    return frame(cv::Rect(x0, y0, side, side)).clone();
}

bool TrackPostProcessor::ClassifyRaw(const cv::Mat& window, TrackState& state)
{
    if (!m_cls)
        return false;

    int top1 = -1;
    float msp = -1.f;
    if (!m_cls->Classify(window, true, &top1, &msp))
        return false;

    state.m_clsIndex = top1;
    state.m_msp = msp;
    state.m_label = m_cls->ClassName(top1);
    return true;
}

bool TrackPostProcessor::ClassifyVsr(TrackState& state)
{
    if (!m_vsr || !m_cls)
        return false;

    // The TensorRT VSR graph is exported as [1,K,3,H,W] / [1,K,3,4H,4W],
    // i.e. per-frame planar CHW (not HWC as the vsr_inference header claims).
    const int lr = m_settings.m_vsrLrSize;
    const int inPlane = lr * lr;
    const int inFrameElems = inPlane * 3;
    m_lrScratch.resize(static_cast<size_t>(m_settings.m_vsrK) * inFrameElems);

    for (int i = 0; i < m_settings.m_vsrK; ++i)
    {
        cv::Mat rgb;
        cv::cvtColor(state.m_crops[static_cast<size_t>(i)], rgb, cv::COLOR_BGR2RGB);
        cv::Mat f;
        rgb.convertTo(f, CV_32FC3, 1.0 / 255.0);
        std::vector<cv::Mat> ch;
        cv::split(f, ch);  // R, G, B planes
        float* dst = m_lrScratch.data() + static_cast<size_t>(i) * inFrameElems;
        for (int c = 0; c < 3; ++c)
            std::memcpy(dst + c * inPlane, ch[static_cast<size_t>(c)].ptr<float>(),
                        static_cast<size_t>(inPlane) * sizeof(float));
    }

    std::vector<float> sr;
    std::string error;
    if (!m_vsr->ProcessSequence(m_lrScratch.data(), &sr, &error))
        return false;

    const int outSide = lr * 4;
    const int outPlane = outSide * outSide;
    int target = m_settings.m_vsrTargetFrame;
    if (target < 0 || target >= m_settings.m_vsrK)
        target = m_settings.m_vsrK / 2;

    const float* p = sr.data() + static_cast<size_t>(target) * outPlane * 3;
    cv::Mat r32(outSide, outSide, CV_32FC1, const_cast<float*>(p));
    cv::Mat g32(outSide, outSide, CV_32FC1, const_cast<float*>(p + outPlane));
    cv::Mat b32(outSide, outSide, CV_32FC1, const_cast<float*>(p + 2 * outPlane));
    cv::Mat r8, g8, b8;
    r32.convertTo(r8, CV_8UC1, 255.0);
    g32.convertTo(g8, CV_8UC1, 255.0);
    b32.convertTo(b8, CV_8UC1, 255.0);
    cv::Mat rgb8;
    cv::merge(std::vector<cv::Mat>{r8, g8, b8}, rgb8);

    int top1 = -1;
    float msp = -1.f;
    if (!m_cls->Classify(rgb8, false, &top1, &msp))
        return false;

    state.m_clsIndex = top1;
    state.m_msp = msp;
    state.m_label = m_cls->ClassName(top1);
    return true;
}

const std::string& TrackPostProcessor::Label(track_id_t::value_type id) const
{
    static const std::string kEmpty;
    const auto it = m_tracks.find(id);
    return it == m_tracks.end() ? kEmpty : it->second.m_label;
}

float TrackPostProcessor::Msp(track_id_t::value_type id) const
{
    const auto it = m_tracks.find(id);
    return it == m_tracks.end() ? -1.f : it->second.m_msp;
}
