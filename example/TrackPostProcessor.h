#pragma once

#include <deque>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>

#include "trajectory.h"
#include "YoloCls.h"
#include "vsr_inference/vsr_inference.h"

///
/// \brief The TrackPostProcessor class
///
/// Runs after ByteTrack and implements the demo variants:
///   variant 0 — yolo11s-cls on the detector crop (raw low-resolution window);
///   variant 1 — the same classifier on the crop restored by a K-frame VSR
///               engine (per-track sliding buffer of the last K crops).
/// With m_mspThreshold > 0 the MSP score acts as an OOD gate: tracks whose
/// cached confidence is below the threshold are removed from the output.
///
class TrackPostProcessor
{
public:
    struct Settings
    {
        bool m_enable = false;
        int m_variant = 0;            // 0 = raw classifier, 1 = VSR + classifier
        std::string m_clsEngine;
        std::string m_clsNames;
        int m_clsSize = 224;
        std::string m_vsrEngine;
        int m_vsrK = 5;
        int m_vsrLrSize = 64;
        int m_vsrTargetFrame = -1;    // -1 => K/2 (middle frame, best VSR quality)
        float m_mspThreshold = 0.0f;  // 0 => no OOD filtering
        int m_recheckInterval = 1;
        int m_minObjectSide = 8;      // skip objects smaller than this
        std::string m_dumpPath;       // optional per-frame CSV dump
    };

    TrackPostProcessor() = default;
    ~TrackPostProcessor();
    TrackPostProcessor(const TrackPostProcessor&) = delete;
    TrackPostProcessor& operator=(const TrackPostProcessor&) = delete;

    bool Init(const Settings& settings, std::string* error);

    ///
    /// \brief Process annotates the tracks with the classifier result and
    ///        filters out the tracks rejected by the OOD (MSP) threshold.
    ///
    void Process(const cv::Mat& frameBGR, int frameInd, std::vector<TrackingObject>& tracks);

    bool Enabled() const { return m_settings.m_enable; }

    const std::string& Label(track_id_t::value_type id) const;
    float Msp(track_id_t::value_type id) const;

private:
    struct TrackState
    {
        std::deque<cv::Mat> m_crops;   // last K LR crops (BGR, lr_size x lr_size)
        std::string m_label;
        int m_clsIndex = -1;
        float m_msp = -1.f;
        int m_lastFrame = -1000000;    // last frame a classification was attempted
        int m_lastSeen = -1;           // last frame the track was observed (GC)
        bool m_rejected = false;
    };

    cv::Mat MakeWindow(const cv::Mat& frame, const cv::Rect& box) const;
    bool ClassifyRaw(const cv::Mat& window, TrackState& state);
    bool ClassifyVsr(TrackState& state);

    Settings m_settings;
    std::unique_ptr<YoloCls> m_cls;
    std::unique_ptr<vsr::IVsrBackend> m_vsr;
    std::unordered_map<track_id_t::value_type, TrackState> m_tracks;
    std::ofstream m_dump;
    std::vector<float> m_lrScratch;
};
