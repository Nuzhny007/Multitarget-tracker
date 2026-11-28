#pragma once
#include <vector>
#include <array>
#include <numeric>

#include "defines.h"
#include "DetectorSettings.h"

namespace mtracking
{

///
/// \brief Current INI schema version understood by ParseTrackerSettings.
/// Bump when the ini keys/sections change incompatibly.
///
inline constexpr int kTrackerSettingsSchemaVersion = 1;

///
/// \brief The TrackerSettings struct
/// Inherits the detector-specific fields from DetectorSettings (roadmap M2).
///
struct TrackerSettings : public DetectorSettings
{
    ///
    /// Tracker settings
    ///
    /// 

    tracking::TrackerTemplate m_tracker = tracking::UniversalTracker;

    ///
    /// \brief INI schema version this struct was parsed from.
    ///
    int m_schemaVersion = kTrackerSettingsSchemaVersion;

    tracking::KalmanType m_kalmanType = tracking::KalmanLinear;
    tracking::FilterGoal m_filterGoal = tracking::FilterCenter;
    tracking::LostTrackType m_lostTrackType = tracking::TrackKCF; // Used if m_filterGoal == tracking::FilterRect
    tracking::MatchType m_matchType = tracking::MatchLAPJV;

    std::array<track_t, tracking::DistsCount> m_distType;

    ///
    /// \brief m_dt
    /// Time step for Kalman
    ///
    track_t m_dt = 1.0f;

    ///
    /// \brief m_accelNoiseMag
    /// Noise magnitude for Kalman
    ///
    track_t m_accelNoiseMag = 0.1f;

    ///
    /// \brief m_useAcceleration
    /// Constant velocity or constant acceleration motion model
    ///
    bool m_useAcceleration = false;

    ///
    /// \brief m_distThres
    /// Distance threshold for Assignment problem: from 0 to 1
    ///
    track_t m_distThres = 0.8f;

    ///
    /// \brief m_distThresLocalCartesian
    /// Distance threshold for Assignment problem in the local ENU plane, meters.
    /// Used only when tracking::DistLocalCartesian is selected.
    ///
    track_t m_distThresLocalCartesian = 10.0f;

    ///
    /// \brief m_minAreaRadius
    /// Minimal area radius in pixels for objects centers
    ///
    track_t m_minAreaRadiusPix = 20.f;

    ///
    /// \brief m_minAreaRadius
    /// Minimal area radius in ration for object size. Used if m_minAreaRadiusPix < 0
    ///
    track_t m_minAreaRadiusK = 0.5f;

    ///
    /// \brief m_maximumAllowedLostTime
    /// If the object don't assignment more than this time in seconds then it will be removed
    ///
    double m_maximumAllowedLostTime = 1.;

    ///
    /// \brief m_maxTraceLength
    /// The maximum trajectory length in seconds
    ///
    double m_maxTraceLength = 2.f;

    ///
    /// \brief m_useAbandonedDetection
    /// Detection abandoned objects
    ///
    bool m_useAbandonedDetection = false;

    ///
    /// \brief m_minStaticTime
    /// After this time (in seconds) the object is considered abandoned
    ///
    int m_minStaticTime = 5;
    ///
    /// \brief m_maxStaticTime
    /// After this time (in seconds) the abandoned object will be removed
    ///
    int m_maxStaticTime = 25;
    ///
    /// \brief m_maxSpeedForStatic
    /// Speed in m/s in the local ENU plane (requires metric coordinates in
    /// detections). If the metric speed of an object is known and exceeds
    /// this value, the object is not static. Without metric data the speed
    /// is unknown and does not restrict the static detection.
    ///
    track_t m_maxSpeedForStatic = 0.5f;

    ///
    /// \brief m_nearTypes
    /// Object types that can be matched while tracking
    ///
    std::map<objtype_t, std::set<objtype_t>> m_nearTypes;

	///
    /// \brief struct ByteTrackSettings
    /// Settings only for m_tracker = tracking::ByteTrack
    ///
	struct ByteTrackSettings
	{
		int m_trackBuffer = 30;
		float m_trackThresh = 0.5f;
		float m_highThresh = 0.5f;
		float m_matchThresh = 0.8f;
	};
    ByteTrackSettings m_byteTrackSettings;

    ///
    /// \brief ClipSettings
    /// Optional ruCLIP zero-shot classification of detections. Filled from the
    /// [clip] section of the settings ini file; used only when the project is
    /// built with -DUSE_CLIP=ON.
    ///
    struct ClipSettings
    {
        ///
        /// \brief m_modelDir
        /// Directory with the ruCLIP model (config.json, bpe.model and the
        /// clip_visual.onnx / clip_textual.onnx files for the TensorRT backend).
        /// Empty disables classification.
        ///
        std::string m_modelDir;

        ///
        /// \brief m_labels
        /// Positive class prompts (UTF-8) to compare the crops with.
        ///
        std::vector<std::string> m_labels;

        ///
        /// \brief m_rejectLabel
        /// Catch-all "unknown" prompt appended to m_labels. A prediction of
        /// this label (or a confidence below m_threshold) rejects the object.
        ///
        std::string m_rejectLabel = "другое";

        ///
        /// \brief m_backend
        /// ruCLIP backend: "trt" (TensorRT), "torch" (LibTorch) or "dnn".
        ///
        std::string m_backend = "trt";

        ///
        /// \brief m_useFp16
        /// FP16 inference (TensorRT engine / LibTorch CUDA).
        ///
        bool m_useFp16 = true;

        ///
        /// \brief m_maxBatch
        /// Upper bound of the batch in one inference call (TensorRT profile).
        ///
        int m_maxBatch = 32;

        ///
        /// \brief m_deviceIndex
        /// CUDA device index.
        ///
        int m_deviceIndex = 0;

        ///
        /// \brief m_threshold
        /// Minimum confidence to accept a label; lower values are rejected.
        ///
        float m_threshold = 0.5f;

        ///
        /// \brief m_recheckInterval
        /// Number of frames a track keeps its classification before re-checking.
        ///
        int m_recheckInterval = 30;

        ///
        /// \brief m_minObjectWidth, m_minObjectHeight
        /// Minimal crop size in pixels; smaller crops are not classified.
        ///
        int m_minObjectWidth = 16;
        int m_minObjectHeight = 16;
    };
    ClipSettings m_clipSettings;

    ///
    /// \brief PostProcessSettings
    /// Optional classifier / VSR post-processing of tracks (demo pipeline:
    /// yolo11s-cls on raw crops or on VSR-restored crops, plus an MSP-based
    /// OOD gate). Filled from the [postprocess] section of the settings ini.
    ///
    struct PostProcessSettings
    {
        bool m_enable = false;

        /// 0 = classifier on raw crops, 1 = VSR + classifier
        int m_variant = 0;

        /// Closed-set classifier engine (TensorRT) and its class-name list.
        std::string m_clsEngine;
        std::string m_clsNames;
        int m_clsSize = 224;

        /// VSR engine (fixed K), crop size and the frame used for the result.
        std::string m_vsrEngine;
        int m_vsrK = 5;
        int m_vsrLrSize = 64;
        int m_vsrTargetFrame = -1; // -1 => K/2

        /// MSP threshold: tracks below it are treated as false positives.
        /// 0 disables the OOD gate.
        float m_mspThreshold = 0.0f;

        /// How often a track is re-classified (frames).
        int m_recheckInterval = 1;

        /// Objects smaller than this (both sides, px) are not classified.
        int m_minObjectSide = 8;

        /// Optional per-frame CSV dump of classifier decisions.
        std::string m_dumpPath;
    };

    PostProcessSettings m_postProcessSettings;

    ///
    struct EmbeddingParams
    {
        ///
        /// \brief m_embeddingCfgName
        /// Neural network config file for embeddings
        ///
        std::string m_embeddingCfgName;
        ///
        /// \brief m_embeddingWeightsName
        /// Neural network weights file for embeddings
        ///
        std::string m_embeddingWeightsName;

		///
		cv::Size m_inputLayer{128, 256};

		///
        std::vector<objtype_t> m_objectTypes;

		EmbeddingParams(const std::string& embeddingCfgName, const std::string& embeddingWeightsName,
            const cv::Size& inputLayer, const std::vector<objtype_t>& objectTypes)
			: m_embeddingCfgName(embeddingCfgName),
			m_embeddingWeightsName(embeddingWeightsName),
			m_inputLayer(inputLayer),
			m_objectTypes(objectTypes)
		{
			assert(!m_objectTypes.empty());
		}
    };
	///
	std::vector<EmbeddingParams> m_embeddings;

	///
	TrackerSettings()
	{
		// The default matching metric is IoU only: histogram (and embedding)
		// distances require a color frame and a valid ROI, and silently add
		// a constant penalty to every pair when they are unavailable.
		// ParseTrackerSettings() always overrides these weights from the
		// ini file. H21: DistHist was 0.5 by default.
		m_distType[tracking::DistCenters] = static_cast<track_t>(0.0);
		m_distType[tracking::DistRects] = static_cast<track_t>(0.0);
		m_distType[tracking::DistJaccard] = static_cast<track_t>(1.0);
		m_distType[tracking::DistHist] = static_cast<track_t>(0.0);
		m_distType[tracking::DistFeatureCos] = static_cast<track_t>(0.0);
		m_distType[tracking::DistMahalanobis] = static_cast<track_t>(0.0);
		m_distType[tracking::DistLocalCartesian] = static_cast<track_t>(0.0);

		assert(CheckDistance());
	}

	///
	/// \brief CheckFilterModel
	/// Validates the Kalman motion model against the filter goal.
	/// All motion models (linear constant velocity/acceleration, UKF and
	/// augmented UKF) are implemented for every filter goal, including the
	/// rotated-rect filter, so there is nothing left to reject. The flag is
	/// ignored by the unscented models (they always estimate acceleration),
	/// exactly as for the point and rect filters.
	/// \param reason Unused, kept for API compatibility.
	/// \return always true
	///
	bool CheckFilterModel(std::string* reason = nullptr) const
	{
		(void)reason;
		return true;
	}

	///
	bool CheckDistance() const
	{
		track_t sum = std::accumulate(m_distType.begin(), m_distType.end(), static_cast<track_t>(0.0));
		track_t maxOne = std::max(static_cast<track_t>(1.0), std::fabs(sum));
		//std::cout << "CheckDistance: " << sum << " - " << (std::numeric_limits<track_t>::epsilon() * maxOne) << ", " << std::fabs(sum - 1.0f) << std::endl;
		return std::fabs(sum - static_cast<track_t>(1.0)) <= std::numeric_limits<track_t>::epsilon() * maxOne;
	}

	///
	bool SetDistances(std::array<track_t, tracking::DistsCount> distType)
	{
		bool res = true;
		auto oldDists = m_distType;
		m_distType = distType;
		if (!CheckDistance())
		{
			m_distType = oldDists;
			res = false;
		}
		return res;
	}

	///
	bool SetDistance(tracking::DistType distType)
	{
		std::fill(m_distType.begin(), m_distType.end(), 0.0f);
		m_distType[distType] = 1.f;
		return true;
	}

	///
	void AddNearTypes(objtype_t type1, objtype_t type2, bool sym)
	{
		auto AddOne = [&](objtype_t type1, objtype_t type2)
		{
			auto it = m_nearTypes.find(type1);
			if (it == std::end(m_nearTypes))
				m_nearTypes[type1] = std::set<objtype_t>{ type2 };
			else
				it->second.insert(type2);
		};
		AddOne(type1, type2);
		if (sym)
			AddOne(type2, type1);
	}

	///
	bool CheckType(objtype_t type1, objtype_t type2) const
	{
		bool res = (type1 == bad_type) || (type2 == bad_type) || (type1 == type2);
		if (!res)
		{
			auto it = m_nearTypes.find(type1);
			if (it != std::end(m_nearTypes))
				res = it->second.find(type2) != std::end(it->second);
		}
		return res;
	}
};

///
MT_API bool ParseTrackerSettings(const std::string& settingsFile, TrackerSettings& trackerSettings);

} // namespace mtracking

// Backward-compatible global names (roadmap M2, 2.0).
using mtracking::kTrackerSettingsSchemaVersion;
using mtracking::ParseTrackerSettings;
using mtracking::TrackerSettings;
