#include <algorithm>
#include <cctype>
#include <sstream>
#include "TrackerSettings.h"
#include <inih/INIReader.h>

namespace mtracking
{

namespace
{
std::string ToLower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::string Trim(const std::string& value)
{
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
        return {};
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

///
/// \brief SplitList
/// Splits a ';' or ',' separated ini value, trimming whitespace and dropping
/// empty items. Used for model names and object type lists.
///
std::vector<std::string> SplitList(const std::string& value)
{
    std::vector<std::string> result;
    std::string item;
    for (char c : value)
    {
        if (c == ';' || c == ',')
        {
            item = Trim(item);
            if (!item.empty())
                result.push_back(item);
            item.clear();
        }
        else
        {
            item.push_back(c);
        }
    }
    item = Trim(item);
    if (!item.empty())
        result.push_back(item);
    return result;
}

///
/// \brief ParseScalar
/// Parses "0.485;0.456;0.406" (or comma separated) into a cv::Scalar.
/// A single value is replicated to all three channels.
///
bool ParseScalar(const std::string& value, cv::Scalar& result)
{
    const std::vector<std::string> parts = SplitList(value);
    if (parts.empty())
        return false;

    std::vector<double> numbers;
    numbers.reserve(parts.size());
    for (const std::string& part : parts)
    {
        try
        {
            size_t consumed = 0;
            const double number = std::stod(part, &consumed);
            if (consumed != part.size())
                return false;
            numbers.push_back(number);
        }
        catch (const std::exception&)
        {
            return false;
        }
    }

    if (numbers.size() == 1)
        result = cv::Scalar(numbers[0], numbers[0], numbers[0]);
    else if (numbers.size() >= 3)
        result = cv::Scalar(numbers[0], numbers[1], numbers[2]);
    else
        return false;
    return true;
}

///
/// \brief ParseDistances
/// Parses the 7 metric weights from `distance_weights` (semicolon/comma
/// separated, sum must be 1).
///
bool ParseDistances(const std::string& value, std::array<track_t, tracking::DistsCount>& result)
{
    const std::vector<std::string> parts = SplitList(value);
    if (parts.size() != tracking::DistsCount)
        return false;

    for (size_t i = 0; i < parts.size(); ++i)
    {
        try
        {
            size_t consumed = 0;
            const double number = std::stod(parts[i], &consumed);
            if (consumed != parts[i].size() || number < 0.0 || number > 1.0)
                return false;
            result[i] = static_cast<track_t>(number);
        }
        catch (const std::exception&)
        {
            return false;
        }
    }
    return true;
}

///
/// \brief ParseDetectorSettings
/// Fills the detector half of the settings (the DetectorSettings base).
/// Kept separate from the tracker keys so the two configs evolve independently
/// (roadmap M2).
///
void ParseDetectorSettings(const INIReader& reader, DetectorSettings& settings)
{
    settings.m_nnWeights = reader.GetString("detection", "nn_weights", "data/yolov4-tiny_best.weights");
    settings.m_nnConfig = reader.GetString("detection", "nn_config", "data/yolov4-tiny.cfg");
    settings.m_classNames = reader.GetString("detection", "class_names", "data/traffic.names");
    settings.m_confidenceThreshold = static_cast<track_t>(reader.GetReal("detection", "confidence_threshold", 0.5));
    settings.m_maxCropRatio = static_cast<track_t>(reader.GetReal("detection", "max_crop_ratio", -1));
    settings.m_maxBatch = reader.GetInteger("detection", "max_batch", 1);
    settings.m_gpuId = reader.GetInteger("detection", "gpu_id", 0);
    settings.m_netType = reader.GetString("detection", "net_type", "YOLOV4");
    settings.m_inferencePrecision = reader.GetString("detection", "inference_precision", "FP16");
    settings.m_detectorBackend = reader.GetInteger("detection", "detector_backend", (int)tracking::Detectors::DNN_OCV);
    settings.m_dnnTarget = reader.GetString("detection", "ocv_dnn_target", "DNN_TARGET_CPU");
    settings.m_dnnBackend = reader.GetString("detection", "ocv_dnn_backend", "DNN_BACKEND_OPENCV");
    const int videoMemory = reader.GetInteger("detection", "video_memory", 0);
    settings.m_maxVideoMemory = videoMemory > 0 ? static_cast<size_t>(videoMemory) : 0;
    settings.m_inputSize.width = reader.GetInteger("detection", "input_width", 0);
    settings.m_inputSize.height = reader.GetInteger("detection", "input_height", 0);
}
} // namespace

///
/// \brief CarsCounting::ParseTrackerSettings
///
bool ParseTrackerSettings(const std::string& settingsFile, TrackerSettings& trackerSettings)
{
    bool res = false;

    std::cout << "ParseTrackerSettings: " << settingsFile << " ..." << std::endl;

    INIReader reader(settingsFile);

    // A file with a syntax error must not be treated as loaded: ini_parse
    // stops at the first error, so only the values before it would be read
    // and the rest would silently fall back to defaults (roadmap H17).
    const int parseError = reader.ParseError();
    if (parseError != 0)
    {
        if (parseError > 0)
            std::cout << "ParseTrackerSettings - error in the ini file at line: " << parseError << std::endl;
        else
            std::cout << "ParseTrackerSettings - can't open the file" << std::endl;

        std::cout << "ParseTrackerSettings: " << false << std::endl;
        return false;
    }

    std::cout << "ParseTrackerSettings - readed" << std::endl;

    trackerSettings = TrackerSettings();

    // INI schema version (roadmap M2). Absent => legacy schema v1.
    const int schemaVersion = reader.GetInteger("meta", "schema_version", kTrackerSettingsSchemaVersion);
    trackerSettings.m_schemaVersion = (schemaVersion >= 1 && schemaVersion <= kTrackerSettingsSchemaVersion)
        ? schemaVersion : kTrackerSettingsSchemaVersion;
    if (schemaVersion != trackerSettings.m_schemaVersion)
    {
        std::cout << "ParseTrackerSettings: unsupported schema_version " << schemaVersion
                  << " (supported 1.." << kTrackerSettingsSchemaVersion
                  << "), unknown keys are ignored" << std::endl;
    }

        // Read tracking settings
        auto trackerType = reader.GetInteger("tracking", "tracker_type", -1);
        if (trackerType == (int)tracking::ByteTrack)
            trackerSettings.m_tracker = tracking::ByteTrack;
        else
            trackerSettings.m_tracker = tracking::UniversalTracker;

        auto distType = reader.GetInteger("tracking", "distance_type", -1);
        if (distType >= 0 && distType < (int)tracking::DistsCount)
            trackerSettings.SetDistance((tracking::DistType)distType);

        // Optional explicit metric weights, e.g. for ReID:
        //   distance_weights = 0;0;0.5;0;0.5;0;0
        // Order: DistCenters, DistRects, DistJaccard, DistHist,
        //        DistFeatureCos, DistMahalanobis, DistLocalCartesian.
        const std::string distanceWeights = reader.GetString("tracking", "distance_weights", "");
        if (!distanceWeights.empty())
        {
            std::array<track_t, tracking::DistsCount> weights;
            if (!ParseDistances(distanceWeights, weights))
            {
                std::cout << "ParseTrackerSettings: invalid tracking.distance_weights (expected "
                          << tracking::DistsCount << " values)" << std::endl;
            }
            else if (!trackerSettings.SetDistances(weights))
            {
                std::cout << "ParseTrackerSettings: tracking.distance_weights must sum to 1"
                          << std::endl;
            }
        }

        auto kalmanType = reader.GetInteger("tracking", "kalman_type", -1);
        if (kalmanType >= 0 && kalmanType < (int)tracking::KalmanCount)
            trackerSettings.m_kalmanType = (tracking::KalmanType)kalmanType;

        auto filterGoal = reader.GetInteger("tracking", "filter_goal", -1);
        if (filterGoal >= 0 && filterGoal < (int)tracking::FiltersCount)
            trackerSettings.m_filterGoal = (tracking::FilterGoal)filterGoal;

        auto lostTrackType = reader.GetInteger("tracking", "lost_track_type", -1);
        if (lostTrackType >= 0 && lostTrackType < (int)tracking::SingleTracksCount)
            trackerSettings.m_lostTrackType = (tracking::LostTrackType)lostTrackType;

        auto matchType = reader.GetInteger("tracking", "match_type", -1);
        if (matchType >= 0 && matchType < (int)tracking::MatchCount)
            trackerSettings.m_matchType = (tracking::MatchType)matchType;

        trackerSettings.m_useAcceleration = reader.GetInteger("tracking", "use_aceleration", 0) != 0; // Use constant acceleration motion model
        trackerSettings.m_dt = static_cast<track_t>(reader.GetReal("tracking", "delta_time", 0.4));  // Delta time for Kalman filter
        trackerSettings.m_accelNoiseMag = static_cast<track_t>(reader.GetReal("tracking", "accel_noise", 0.2)); // Accel noise magnitude for Kalman filter
        trackerSettings.m_distThres = static_cast<track_t>(reader.GetReal("tracking", "dist_thresh", 0.8));     // Distance threshold between region and object on two frames
        trackerSettings.m_distThresLocalCartesian = static_cast<track_t>(reader.GetReal("tracking", "local_cartesian_dist_thresh", 10.0)); // Same threshold in meters for DistLocalCartesian
        trackerSettings.m_minAreaRadiusPix = static_cast<track_t>(reader.GetReal("tracking", "min_area_radius_pix", -1.));
        trackerSettings.m_minAreaRadiusK = static_cast<track_t>(reader.GetReal("tracking", "min_area_radius_k", 0.8));
        trackerSettings.m_maximumAllowedLostTime = reader.GetReal("tracking", "max_lost_time", 1.);         // Maximum lost time in seconds
        trackerSettings.m_maxTraceLength = reader.GetReal("tracking", "max_trace_len", 2.);                 // Maximum trace length in seconds
        trackerSettings.m_useAbandonedDetection = reader.GetInteger("tracking", "detect_abandoned", 0) != 0;
        trackerSettings.m_minStaticTime = reader.GetInteger("tracking", "min_static_time", 5);
        trackerSettings.m_maxStaticTime = reader.GetInteger("tracking", "max_static_time", 25);
        trackerSettings.m_maxSpeedForStatic = static_cast<track_t>(reader.GetReal("tracking", "max_speed_for_static", 0.5));

        trackerSettings.m_byteTrackSettings.m_trackBuffer = reader.GetInteger("tracking", "bytetrack_track_buffer", 30);
        trackerSettings.m_byteTrackSettings.m_trackThresh = static_cast<float>(reader.GetReal("tracking", "bytetrack_track_thresh", 0.5));
        trackerSettings.m_byteTrackSettings.m_highThresh = static_cast<float>(reader.GetReal("tracking", "bytetrack_high_thresh", 0.5));
        trackerSettings.m_byteTrackSettings.m_matchThresh = static_cast<float>(reader.GetReal("tracking", "bytetrack_match_thresh", 0.8));

        // Read detection settings (DetectorSettings base, roadmap M2)
        ParseDetectorSettings(reader, trackerSettings);

		std::stringstream whiteList{ reader.GetString("detection", "white_list", "") };
		trackerSettings.m_whiteList.clear();
		std::string wname;
		while (std::getline(whiteList, wname, ';'))
		{
			trackerSettings.m_whiteList.push_back(wname);
		}

        // Read clip (ruCLIP) settings
        trackerSettings.m_clipSettings.m_modelDir = reader.GetString("clip", "model_dir", "");
        trackerSettings.m_clipSettings.m_rejectLabel = reader.GetString("clip", "reject_label", "другое");
        trackerSettings.m_clipSettings.m_backend = reader.GetString("clip", "backend", "trt");
        trackerSettings.m_clipSettings.m_useFp16 = reader.GetInteger("clip", "use_fp16", 1) != 0;
        trackerSettings.m_clipSettings.m_maxBatch = reader.GetInteger("clip", "max_batch", 32);
        trackerSettings.m_clipSettings.m_deviceIndex = reader.GetInteger("clip", "device_index", 0);
        trackerSettings.m_clipSettings.m_threshold = static_cast<float>(reader.GetReal("clip", "threshold", 0.5));
        trackerSettings.m_clipSettings.m_recheckInterval = reader.GetInteger("clip", "recheck_interval", 30);
        trackerSettings.m_clipSettings.m_minObjectWidth = reader.GetInteger("clip", "min_object_width", 16);
        trackerSettings.m_clipSettings.m_minObjectHeight = reader.GetInteger("clip", "min_object_height", 16);

        std::stringstream clipLabels{ reader.GetString("clip", "labels", "человек;машина;грузовик;автобус;велосипед") };
        trackerSettings.m_clipSettings.m_labels.clear();
        std::string clipLabel;
        while (std::getline(clipLabels, clipLabel, ';'))
        {
            if (!clipLabel.empty())
                trackerSettings.m_clipSettings.m_labels.push_back(clipLabel);
        }

        // Read post-processing (classifier / VSR) settings
        trackerSettings.m_postProcessSettings.m_enable = reader.GetInteger("postprocess", "enable", 0) != 0;
        trackerSettings.m_postProcessSettings.m_variant = reader.GetInteger("postprocess", "variant", 0);
        trackerSettings.m_postProcessSettings.m_clsEngine = reader.GetString("postprocess", "cls_engine", "");
        trackerSettings.m_postProcessSettings.m_clsNames = reader.GetString("postprocess", "cls_names", "");
        trackerSettings.m_postProcessSettings.m_clsSize = reader.GetInteger("postprocess", "cls_size", 224);
        trackerSettings.m_postProcessSettings.m_vsrEngine = reader.GetString("postprocess", "vsr_engine", "");
        trackerSettings.m_postProcessSettings.m_vsrK = reader.GetInteger("postprocess", "vsr_k", 5);
        trackerSettings.m_postProcessSettings.m_vsrLrSize = reader.GetInteger("postprocess", "vsr_lr_size", 64);
        trackerSettings.m_postProcessSettings.m_vsrTargetFrame = reader.GetInteger("postprocess", "vsr_target_frame", -1);
        trackerSettings.m_postProcessSettings.m_mspThreshold = static_cast<float>(reader.GetReal("postprocess", "msp_threshold", 0.0));
        trackerSettings.m_postProcessSettings.m_recheckInterval = reader.GetInteger("postprocess", "recheck_interval", 1);
        trackerSettings.m_postProcessSettings.m_minObjectSide = reader.GetInteger("postprocess", "min_object_side", 8);
        trackerSettings.m_postProcessSettings.m_dumpPath = reader.GetString("postprocess", "dump", "");

        // Read ReID settings (roadmap M3): a registry of models bound to object
        // types plus a global GPU/CPU and appearance-gallery policy.
        const std::string modelsList = reader.GetString("reid", "models", "");
        const std::vector<std::string> modelNames = SplitList(modelsList);
        const bool reIdEnabled = reader.GetBoolean("reid", "enable", !modelNames.empty());

        trackerSettings.m_reidSettings.m_enable = reIdEnabled;
        trackerSettings.m_reidSettings.m_backend = ToLower(reader.GetString("reid", "backend", "cuda_fp16"));
        trackerSettings.m_reidSettings.m_maxBatch = std::max(1,
            static_cast<int>(reader.GetInteger("reid", "max_batch", 32)));
        trackerSettings.m_reidSettings.m_gallerySize = std::max(1,
            static_cast<int>(reader.GetInteger("reid", "gallery_size", 15)));
        trackerSettings.m_reidSettings.m_qualityGate = reader.GetBoolean("reid", "quality_gate", true);
        trackerSettings.m_reidSettings.m_minQuality = static_cast<float>(
            std::clamp(reader.GetReal("reid", "min_quality", 0.35), 0.0, 1.0));
        trackerSettings.m_reidSettings.m_minCropSide = std::max(1,
            static_cast<int>(reader.GetInteger("reid", "min_crop_side", 16)));
        trackerSettings.m_reidSettings.m_cpuMaxObjects = std::max(0,
            static_cast<int>(reader.GetInteger("reid", "cpu_max_objects", 8)));
        trackerSettings.m_reidSettings.m_cpuInterval = std::max(1,
            static_cast<int>(reader.GetInteger("reid", "cpu_interval", 5)));
        trackerSettings.m_reidSettings.m_warmup = reader.GetBoolean("reid", "warmup", true);

        trackerSettings.m_embeddings.clear();
        if (reIdEnabled)
        {
            if (modelNames.empty())
            {
                std::cout << "ParseTrackerSettings: [reid] enable=1 but models list is empty" << std::endl;
            }

            for (const std::string& modelName : modelNames)
            {
                const std::string section = "reid_" + modelName;
                if (!reader.HasSection(section))
                {
                    std::cout << "ParseTrackerSettings: missing [" << section << "] section" << std::endl;
                    continue;
                }

                const std::string modelPath = reader.GetString(section, "model", "");
                if (modelPath.empty())
                {
                    std::cout << "ParseTrackerSettings: [" << section << "] model is empty" << std::endl;
                    continue;
                }

                TrackerSettings::EmbeddingParams params;
                params.m_name = modelName;
                params.m_embeddingCfgName = modelPath;
                params.m_embeddingWeightsName = modelPath;
                params.m_inputLayer.width = reader.GetInteger(section, "input_width", 0);
                params.m_inputLayer.height = reader.GetInteger(section, "input_height", 0);
                if (params.m_inputLayer.width <= 0 || params.m_inputLayer.height <= 0)
                {
                    std::cout << "ParseTrackerSettings: [" << section
                              << "] input_width/input_height must be positive" << std::endl;
                    continue;
                }

                if (!ParseScalar(reader.GetString(section, "mean", "0.485;0.456;0.406"), params.m_mean))
                {
                    std::cout << "ParseTrackerSettings: [" << section << "] invalid mean; using 0" << std::endl;
                    params.m_mean = cv::Scalar(0, 0, 0);
                }
                if (!ParseScalar(reader.GetString(section, "std", "0.229;0.224;0.225"), params.m_std))
                {
                    std::cout << "ParseTrackerSettings: [" << section << "] invalid std; using 1" << std::endl;
                    params.m_std = cv::Scalar(1, 1, 1);
                }
                params.m_swapRB = reader.GetBoolean(section, "swap_rb", true);
                params.m_maxBatch = static_cast<int>(reader.GetInteger(section, "max_batch",
                    trackerSettings.m_reidSettings.m_maxBatch));
                params.m_backend = ToLower(reader.GetString(section, "backend",
                    trackerSettings.m_reidSettings.m_backend));
                params.m_minCropSide = static_cast<int>(reader.GetInteger(section, "min_crop_side", 0));

                const std::vector<std::string> typeNames = SplitList(reader.GetString(section, "types", modelName));
                for (const std::string& typeName : typeNames)
                {
                    objtype_t type = TypeConverter::Str2Type(typeName);
                    if (type == bad_type)
                    {
                        // Detector class names (e.g. VisDrone pedestrian/van)
                        // may be registered later by BaseDetector::FillTypesMap.
                        // Register them now so a [reid_*] section can refer to
                        // them before the detector is initialized.
                        TypeConverter::AddNewType(typeName);
                        type = TypeConverter::Str2Type(typeName);
                    }
                    if (type == bad_type)
                    {
                        std::cout << "ParseTrackerSettings: [" << section
                                  << "] unknown object type '" << typeName << "'" << std::endl;
                        continue;
                    }
                    params.m_objectTypes.push_back(type);
                }

                if (params.m_objectTypes.empty())
                {
                    std::cout << "ParseTrackerSettings: [" << section
                              << "] no valid object types; model skipped" << std::endl;
                    continue;
                }

                trackerSettings.m_embeddings.push_back(params);
            }
        }

    res = true;
    std::cout << "ParseTrackerSettings: " << res << std::endl;
    return res;
}

} // namespace mtracking
