#include <fstream>
#include <set>
#include <stdexcept>
#include "BaseTracker.h"
#include "byte_track/BYTETracker.h"
#include "ShortPathCalculator.h"
#include "EmbeddingsCalculator.hpp"
#include "track.h"

///
/// \brief The CTracker class
///
class CTracker final : public BaseTracker
{
public:
    CTracker(const TrackerSettings& settings);
	CTracker(const CTracker&) = delete;
	CTracker(CTracker&&) = delete;
	CTracker& operator=(const CTracker&) = delete;
	CTracker& operator=(CTracker&&) = delete;

	~CTracker(void) = default;

    void Update(const regions_t& regions, cv::UMat currFrame, time_point_t frameTime) override;
    void Update(const regions_t& regions, cv::UMat currFrame, time_point_t frameTime, tracking::DistType distType) override;

    void GetTracks(std::vector<TrackingObject>& tracks) const override;
    void GetRemovedTracks(std::vector<track_id_t>& trackIDs) const override;

private:
    TrackerSettings m_settings;

	tracks_t m_tracks;
    time_point_t m_lastFrameTime;

    track_id_t m_nextTrackID;
    std::vector<track_id_t> m_removedObjects;

    std::unique_ptr<ShortPathCalculator> m_SPCalculator;
    std::map<objtype_t, std::shared_ptr<EmbeddingsCalculator>> m_embCalculators;

    // ReID model registry. Models are deduplicated by their full description,
    // so one network file shared by several object types is loaded once.
    std::map<int, std::shared_ptr<EmbeddingsCalculator>> m_reidModels;
    bool m_reidAllCpu = true;
    size_t m_reidFrameCounter = 0;
    std::set<int> m_reidCpuSkippedModels;

    // Reused across frames to avoid per-frame reallocations (roadmap M3).
    std::vector<RegionEmbedding> m_regionEmbeddings;

    void InitReIdModels();
    void CreateDistaceMatrix(const regions_t& regions, const std::vector<RegionEmbedding>& regionEmbeddings, distMatrix_t& costMatrix, track_t maxPossibleCost, track_t& maxCost, bool useLocalCartesian);
    void UpdateTrackingState(const regions_t& regions, cv::UMat currFrame, time_point_t frameTime, bool useLocalCartesian);
    void CalcEmbeddins(std::vector<RegionEmbedding>& regionEmbeddings, const regions_t& regions, cv::UMat currFrame);

    track_t GetEllipseDist(const CTrack& trackRef, const CRegion& reg);
};
// ----------------------------------------------------------------------

///
/// \brief CTracker::CTracker
/// Manage tracks: create, remove, update.
/// \param settings
///
CTracker::CTracker(const TrackerSettings& settings)
    : m_settings(settings)
{
    std::string reason;
    if (!settings.CheckFilterModel(&reason))
        throw std::invalid_argument("CTracker: " + reason);

    m_SPCalculator.reset();
    SPSettings spSettings = { settings.m_distThres, 12 };
    switch (m_settings.m_matchType)
    {
    case tracking::MatchHungrian:
        m_SPCalculator = std::make_unique<SPHungrian>(spSettings);
        break;
    case tracking::MatchLAPJV:
        m_SPCalculator = std::make_unique<SPLAPJV>(spSettings);
        break;
    }
    assert(m_SPCalculator);

    InitReIdModels();
}

///
/// \brief CTracker::InitReIdModels
/// Builds the object-type -> model registry from settings. Models with the
/// same path/preprocessing are shared; a type may be bound to only one model.
///
void CTracker::InitReIdModels()
{
    m_embCalculators.clear();
    m_reidModels.clear();
    m_reidCpuSkippedModels.clear();
    m_reidAllCpu = true;

    if (m_settings.m_embeddings.empty())
    {
        if (m_settings.m_distType[tracking::DistFeatureCos] > 0.0f)
        {
            std::cerr << "CTracker: DistFeatureCos is enabled but no ReID models are configured" << std::endl;
        }
        return;
    }

    std::map<std::string, std::shared_ptr<EmbeddingsCalculator>> byDescription;
    int nextModelId = 0;
    int initializedModels = 0;

    for (const auto& params : m_settings.m_embeddings)
    {
        if (params.m_objectTypes.empty())
        {
            std::cerr << "CTracker: ReID model without object types is skipped: "
                      << params.m_embeddingWeightsName << std::endl;
            continue;
        }

        EmbeddingModelDesc desc;
        desc.m_name = params.m_name.empty() ? params.m_embeddingWeightsName : params.m_name;
        desc.m_path = params.m_embeddingWeightsName;
        desc.m_inputSize = params.m_inputLayer;
        desc.m_mean = params.m_mean;
        desc.m_std = params.m_std;
        desc.m_swapRB = params.m_swapRB;
        desc.m_maxBatch = params.m_maxBatch > 0
            ? params.m_maxBatch
            : m_settings.m_reidSettings.m_maxBatch;
        desc.m_backend = !params.m_backend.empty()
            ? params.m_backend
            : m_settings.m_reidSettings.m_backend;
        desc.m_warmup = m_settings.m_reidSettings.m_warmup;
        desc.m_minCropSide = params.m_minCropSide > 0
            ? params.m_minCropSide
            : m_settings.m_reidSettings.m_minCropSide;

        // Key includes every property that changes the produced embedding:
        // path, input size, preprocessing and backend.
        const std::string key =
            desc.m_path + "|" +
            std::to_string(desc.m_inputSize.width) + "x" + std::to_string(desc.m_inputSize.height) + "|" +
            cv::format("%.6f,%.6f,%.6f", desc.m_mean[0], desc.m_mean[1], desc.m_mean[2]) + "|" +
            cv::format("%.6f,%.6f,%.6f", desc.m_std[0], desc.m_std[1], desc.m_std[2]) + "|" +
            (desc.m_swapRB ? "rgb" : "bgr") + "|" + desc.m_backend;

        std::shared_ptr<EmbeddingsCalculator> calculator;
        const auto existingModel = byDescription.find(key);
        if (existingModel != std::end(byDescription))
        {
            calculator = existingModel->second;
        }
        else
        {
            calculator = std::make_shared<EmbeddingsCalculator>();
            const int modelId = nextModelId++;
            if (!calculator->Initialize(desc, modelId))
            {
                std::cerr << "CTracker: ReID model initialization error: " << desc.m_path << std::endl;
                continue;
            }
            byDescription.emplace(key, calculator);
            m_reidModels.emplace(modelId, calculator);
            ++initializedModels;
        }

        for (const objtype_t objType : params.m_objectTypes)
        {
            const auto existingType = m_embCalculators.find(objType);
            if (existingType != std::end(m_embCalculators))
            {
                if (existingType->second != calculator)
                {
                    throw std::invalid_argument(
                        "CTracker: object type '" + TypeConverter::Type2Str(objType) +
                        "' is bound to more than one ReID model");
                }
                continue;
            }
            m_embCalculators.emplace(objType, calculator);
        }
    }

    m_reidAllCpu = true;
    for (const auto& item : m_reidModels)
    {
        if (!item.second->IsCpu())
        {
            m_reidAllCpu = false;
            break;
        }
    }

    if (m_settings.m_distType[tracking::DistFeatureCos] > 0.0f && initializedModels == 0)
    {
        throw std::invalid_argument(
            "CTracker: DistFeatureCos is enabled but no ReID model could be initialized");
    }
}

///
/// \brief GetTracks
/// \return
///
void CTracker::GetTracks(std::vector<TrackingObject>& tracks) const
{
    tracks.clear();

    if (m_tracks.size() > tracks.capacity())
        tracks.reserve(m_tracks.size());
    for (const auto& track : m_tracks)
    {
        tracks.emplace_back(track->ConstructObject(m_lastFrameTime));
    }
}

///
/// \brief GetRemovedTracks
/// \return
///
void CTracker::GetRemovedTracks(std::vector<track_id_t>& trackIDs) const
{
    trackIDs.assign(std::begin(m_removedObjects), std::end(m_removedObjects));
}

///
/// \brief CTracker::Update
/// \param regions
/// \param currFrame
/// \param fps
///
void CTracker::Update(const regions_t& regions, cv::UMat currFrame, time_point_t frameTime)
{
    const bool useLocalCartesian = m_settings.m_distType[tracking::DistLocalCartesian] > 0.0f;
    Update(regions, currFrame, frameTime, useLocalCartesian ? tracking::DistLocalCartesian : tracking::DistJaccard);
}

///
/// \brief CTracker::Update
/// \param regions
/// \param currFrame
/// \param frameTime
/// \param distType
///
void CTracker::Update(const regions_t& regions, cv::UMat currFrame, time_point_t frameTime, tracking::DistType distType)
{
    m_lastFrameTime = frameTime;
    m_removedObjects.clear();

    // distType is only the ENU-mode flag (see BaseTracker.h): the metric weights always come from m_settings.m_distType.
    UpdateTrackingState(regions, currFrame, frameTime, distType == tracking::DistLocalCartesian);
}

#define DRAW_DBG_ASSIGNMENT 0

///
/// \brief CTracker::UpdateTrackingState
/// \param regions
/// \param currFrame
/// \param fps
///
void CTracker::UpdateTrackingState(const regions_t& regions,
                                   cv::UMat currFrame,
                                   time_point_t frameTime,
                                   bool useLocalCartesian)
{
    const size_t colsTracks = m_tracks.size();	// Tracking objects
    const size_t rowsRegions = regions.size();	// Detections or regions
    const track_t distThres = useLocalCartesian ? m_settings.m_distThresLocalCartesian : m_settings.m_distThres;

    assignments_t assignmentT2R(colsTracks, -1); // Assignments: index - track, value - region

    // Reuse the per-tracker buffer instead of allocating a new vector every
    // frame; CalcEmbeddins leaves it empty when no embedding metric is active.
    ++m_reidFrameCounter;
    CalcEmbeddins(m_regionEmbeddings, regions, currFrame);
    const std::vector<RegionEmbedding>& regionEmbeddings = m_regionEmbeddings;

#if DRAW_DBG_ASSIGNMENT
    std::cout << "CTracker::UpdateTrackingState: m_tracks = " << colsTracks << ", regions = " << rowsRegions << std::endl;

    int fontType = cv::FONT_HERSHEY_TRIPLEX;
    double fontSize = (currFrame.cols < 1000) ? 0.4 : 0.6;
    cv::Scalar colorRegionEllow(100, 100, 100);
    cv::Scalar colorMatchedAboveThreshRed(0, 0, 255);
    cv::Scalar colorMatchedGreen(0, 255, 0);
    cv::Scalar colorMatchedNearMargenta(255, 0, 255);
    cv::Scalar colorNotMatchedNearWhite(255, 255, 255);
    cv::Scalar colorUnknownBlue(255, 0, 0);

	auto DrawRRect = [](cv::Mat& img, const cv::RotatedRect& rr, const cv::Scalar& cl, int thikness)
	{
		cv::Point2f rectPoints[4];
		rr.points(rectPoints);
		for (int i = 0; i < 4; ++i)
		{
			cv::line(img, rectPoints[i], rectPoints[(i + 1) % 4], cl, thikness, cv::LINE_4);
		}
	};

    cv::Mat dbgAssignment = currFrame.getMat(cv::ACCESS_READ).clone();
    {
#if 0
        cv::Mat foreground(dbgAssignment.size(), CV_8UC1, cv::Scalar(0, 0, 100));
        for (const auto& track : m_tracks)
        {
            cv::ellipse(foreground, track->GetLastRect(), cv::Scalar(255, 255, 255), cv::FILLED);
        }

        const int chans = dbgAssignment.channels();
        const int height = dbgAssignment.rows;
#pragma omp parallel for
        for (int y = 0; y < height; ++y)
        {
            uchar* imgPtr = dbgAssignment.ptr(y);
            const uchar* frgrndPtr = foreground.ptr(y);
            for (int x = 0; x < dbgAssignment.cols; ++x)
            {
                for (int ci = chans - 1; ci < chans; ++ci)
                {
                    imgPtr[ci] = cv::saturate_cast<uchar>(imgPtr[ci] + frgrndPtr[0]);
                }
                imgPtr += chans;
                ++frgrndPtr;
            }
        }
#endif
        for (const auto& reg : regions)
        {
            DrawRRect(dbgAssignment, reg.m_rrect, colorRegionEllow, 2);
        }
    }
#endif

    if (!m_tracks.empty())
    {
        // Distance matrix between all tracks to all regions
#if DRAW_DBG_ASSIGNMENT
        std::cout << "CTracker::UpdateTrackingState: Distance matrix between all tracks to all regions" << std::endl;
#endif
        distMatrix_t costMatrix(colsTracks * rowsRegions);
        // In the local Cartesian mode the cost matrix is measured in meters, so the
        // pixel based upper bound is not applicable.
        const track_t maxPossibleCost = useLocalCartesian
                ? static_cast<track_t>(1000000.0) // 1000 km
                : std::max(static_cast<track_t>(1.), static_cast<track_t>(currFrame.cols * currFrame.rows));
        track_t maxCost = 0;
        CreateDistaceMatrix(regions, regionEmbeddings, costMatrix, maxPossibleCost, maxCost, useLocalCartesian);
#if DRAW_DBG_ASSIGNMENT
        std::cout << "CTracker::UpdateTrackingState: maxPossibleCost = " << maxPossibleCost << ", maxCost = " << maxCost << std::endl;
        std::cout << "costMatrix: " << cv::Mat_<track_t>(rowsRegions, colsTracks, costMatrix.data()) << std::endl;
#endif

        // Solving assignment problem (shortest paths)
#if DRAW_DBG_ASSIGNMENT
        std::cout << "CTracker::UpdateTrackingState: Solving assignment problem (shortest paths)" << std::endl;
#endif
        m_SPCalculator->Solve(costMatrix, colsTracks, rowsRegions, assignmentT2R, maxCost);

#if 0
        {
            static size_t saveSolveNum = 0;
            std::ofstream resCSV("mt_example" + std::to_string(saveSolveNum) + ".csv");
            for (size_t r = 0; r < rowsRegions; ++r)
            {
                for (size_t c = 0; c < colsTracks; ++c)
                {
                    if (c == colsTracks - 1)
                        resCSV << std::fixed << std::setw(2) << std::setprecision(2) << costMatrix[c + r * colsTracks] << std::endl;
                    else
                        resCSV << std::fixed << std::setw(2) << std::setprecision(2) << costMatrix[c + r * colsTracks] << ", ";
                }
            }
            std::ofstream resSol("mt_example" + std::to_string(saveSolveNum) + ".sol");
            resSol << maxCost << std::endl;
            for (size_t r = 0; r < assignmentT2R.size(); ++r)
			{
				resSol << assignmentT2R[r] << std::endl;
			}
            ++saveSolveNum;
        }
#endif

        // Clean assignment from pairs with large distance
#if DRAW_DBG_ASSIGNMENT
        std::cout << "CTracker::UpdateTrackingState: Clean assignment from pairs with large distance" << std::endl;
#endif
        for (size_t i = 0; i < assignmentT2R.size(); ++i)
        {
#if DRAW_DBG_ASSIGNMENT
            std::stringstream ss;
            if (assignmentT2R[i] != -1)
            {
                ss << m_tracks[i]->GetID().ID2Str() << "-" << assignmentT2R[i] << ": " << std::fixed << std::setprecision(2) << costMatrix[i + assignmentT2R[i] * colsTracks];

				if (costMatrix[i + assignmentT2R[i] * colsTracks] > distThres)
                {
                    ss << ">" << distThres;
                    cv::line(dbgAssignment, m_tracks[i]->GetLastRect().center, regions[assignmentT2R[i]].m_rrect.center, colorMatchedAboveThreshRed, 2);
					DrawRRect(dbgAssignment, m_tracks[i]->LastRegion().m_rrect, colorMatchedAboveThreshRed, 1);
                }
                else
                {
                    ss << "<" << distThres;
                    cv::line(dbgAssignment, m_tracks[i]->GetLastRect().center, regions[assignmentT2R[i]].m_rrect.center, colorMatchedGreen, 1);
					DrawRRect(dbgAssignment, m_tracks[i]->LastRegion().m_rrect, colorMatchedGreen, 1);
                }

                for (size_t ri = 0; ri < regions.size(); ++ri)
                {
                    if (static_cast<int>(ri) != assignmentT2R[i] && costMatrix[i + ri * colsTracks] < 1)
                    {
                        std::stringstream liness;
                        liness << std::fixed << std::setprecision(2) << costMatrix[i + ri * colsTracks];
                        auto p1 = m_tracks[i]->GetLastRect().center;
                        auto p2 = regions[ri].m_rrect.center;
                        cv::line(dbgAssignment, p1, p2, colorMatchedNearMargenta, 1);
                        cv::putText(dbgAssignment, liness.str(), cv::Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2), fontType, fontSize, colorMatchedNearMargenta, 1, 8);
                    }
                }
            }
            else
            {
                // If track have no assigned detect, then increment skipped frames counter.
				DrawRRect(dbgAssignment, m_tracks[i]->LastRegion().m_rrect, colorNotMatchedNearWhite, 1);
                for (size_t ri = 0; ri < regions.size(); ++ri)
                {
                    if (costMatrix[i + ri * colsTracks] < 1)
                    {
                        std::stringstream liness;
                        liness << std::fixed << std::setprecision(2) << costMatrix[i + ri * colsTracks];
                        auto p1 = m_tracks[i]->GetLastRect().center;
                        auto p2 = regions[ri].m_rrect.center;
                        cv::line(dbgAssignment, p1, p2, colorNotMatchedNearWhite, 1);
                        cv::putText(dbgAssignment, liness.str(), cv::Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2), fontType, fontSize, colorNotMatchedNearWhite, 1, 8);
                    }
                }
            }
            if (ss.str().length() > 0)
            {
                auto brect = m_tracks[i]->LastRegion().m_brect;
                cv::putText(dbgAssignment, ss.str(), cv::Point(brect.x, brect.y), fontType, fontSize, colorUnknownBlue, 1, 8);
            }
#endif

            if (assignmentT2R[i] != -1 && costMatrix[i + assignmentT2R[i] * colsTracks] > distThres)
                assignmentT2R[i] = -1;
        }

        // If track didn't get detects long time, remove it
#if DRAW_DBG_ASSIGNMENT
        std::cout << "CTracker::UpdateTrackingState: If track did not get detects long time, remove it" << std::endl;
#endif
        for (size_t i = 0; i < m_tracks.size();)
        {
            if (m_tracks[i]->GetLostPeriod(frameTime) > m_settings.m_maximumAllowedLostTime ||
                    m_tracks[i]->IsOutOfTheFrame() ||
                    m_tracks[i]->IsStaticTimeout(frameTime, m_settings.m_maxStaticTime - m_settings.m_minStaticTime))
            {
                m_removedObjects.push_back(m_tracks[i]->GetID());
#if DRAW_DBG_ASSIGNMENT
                std::cout << "Remove: " << m_tracks[i]->GetID().ID2Str() << ": lost = " << m_tracks[i]->GetLostPeriod(frameTime) << ", maximumAllowedLostTime = " << m_settings.m_maximumAllowedLostTime << ", out of frame " << m_tracks[i]->IsOutOfTheFrame() << std::endl;
#endif
                m_tracks.erase(m_tracks.begin() + i);
                assignmentT2R.erase(assignmentT2R.begin() + i);
            }
            else
            {
                ++i;
            }
        }
    }

    // Search for unassigned detects and start new tracks for them. Mark the
    // assigned regions once (O(tracks)) instead of scanning assignmentT2R for
    // every region, which made this O(tracks * regions) per frame (roadmap M4).
    std::vector<bool> regionAssigned(regions.size(), false);
    for (int assignedRegion : assignmentT2R)
    {
        if (assignedRegion >= 0 && static_cast<size_t>(assignedRegion) < regions.size())
            regionAssigned[static_cast<size_t>(assignedRegion)] = true;
    }

#if DRAW_DBG_ASSIGNMENT
    std::cout << "CTracker::UpdateTrackingState: Search for unassigned detects and start new tracks for them" << std::endl;
#endif
    for (size_t i = 0; i < regions.size(); ++i)
    {
#if DRAW_DBG_ASSIGNMENT
		std::cout << "CTracker::update: regions[" << i << "].m_rrect: " << regions[i].m_rrect.center << ", " << regions[i].m_rrect.angle << ", " << regions[i].m_rrect.size << std::endl;
#endif
        if (!regionAssigned[i])
        {
            std::unique_ptr<CTrack> newTrack;
            if (regionEmbeddings.empty())
            {
                newTrack = std::make_unique<CTrack>(regions[i],
                                                    m_settings.m_kalmanType,
                                                    m_settings.m_dt,
                                                    m_settings.m_accelNoiseMag,
                                                    m_settings.m_useAcceleration,
                                                    m_nextTrackID,
                                                    m_settings.m_filterGoal,
                                                    m_settings.m_lostTrackType,
                                                    frameTime);
            }
            else
            {
                newTrack = std::make_unique<CTrack>(regions[i],
                                                    regionEmbeddings[i],
                                                    m_settings.m_kalmanType,
                                                    m_settings.m_dt,
                                                    m_settings.m_accelNoiseMag,
                                                    m_settings.m_useAcceleration,
                                                    m_nextTrackID,
                                                    m_settings.m_filterGoal,
                                                    m_settings.m_lostTrackType,
                                                    frameTime);
            }
            newTrack->SetAppearanceParams(
                static_cast<size_t>(std::max(1, m_settings.m_reidSettings.m_gallerySize)),
                m_settings.m_reidSettings.m_qualityGate,
                m_settings.m_reidSettings.m_minQuality);
            m_tracks.push_back(std::move(newTrack));
            m_nextTrackID = m_nextTrackID.NextID();
        }
    }

    // Update Kalman Filters state
#if DRAW_DBG_ASSIGNMENT
    std::cout << "CTracker::UpdateTrackingState: Update Kalman Filters state" << std::endl;
#endif

    const ptrdiff_t stop_i = static_cast<ptrdiff_t>(assignmentT2R.size());
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < stop_i; ++i)
    {
        // If track updated less than one time, than filter state is not correct.
        if (assignmentT2R[i] != -1) // If we have assigned detect, then update using its coordinates,
        {
            m_tracks[i]->ResetLostTime(frameTime);
#if DRAW_DBG_ASSIGNMENT
            std::cout << "Update track " << i << " for " << assignmentT2R[i] << " region, regionEmbeddings.size = " << regionEmbeddings.size() << std::endl;
#endif
            if (regionEmbeddings.empty())
                m_tracks[i]->Update(regions[assignmentT2R[i]],
                        true, m_settings.m_maxTraceLength,
                        currFrame,
                        m_settings.m_useAbandonedDetection ? m_settings.m_minStaticTime : 0, m_settings.m_maxSpeedForStatic,
                        frameTime);
            else
                m_tracks[i]->Update(regions[assignmentT2R[i]], regionEmbeddings[assignmentT2R[i]],
                        true, m_settings.m_maxTraceLength,
                        currFrame,
                        m_settings.m_useAbandonedDetection ? m_settings.m_minStaticTime : 0, m_settings.m_maxSpeedForStatic,
                        frameTime);
        }
        else				     // if not continue using predictions
        {
            m_tracks[i]->Update(CRegion(), false, m_settings.m_maxTraceLength, currFrame, 0, m_settings.m_maxSpeedForStatic, frameTime);
        }
    }

#if DRAW_DBG_ASSIGNMENT
    std::cout << "CTracker::UpdateTrackingState: show results" << std::endl;
#ifndef SILENT_WORK
    cv::namedWindow("dbgAssignment", cv::WINDOW_NORMAL);
    cv::imshow("dbgAssignment", dbgAssignment);
    //cv::waitKey(1);
#endif
#endif

}

///
/// \brief CTracker::CreateDistaceMatrix
/// \param regions
/// \param costMatrix
/// \param maxPossibleCost
/// \param maxCost
///
void CTracker::CreateDistaceMatrix(const regions_t& regions,
                                   const std::vector<RegionEmbedding>& regionEmbeddings,
                                   distMatrix_t& costMatrix,
                                   track_t maxPossibleCost,
                                   track_t& maxCost,
                                   bool useLocalCartesian)
{
    const size_t colsTracks = m_tracks.size();	// Tracking objects
    maxCost = 0;

    for (size_t i = 0; i < colsTracks; ++i)
    {
        const auto& track = m_tracks[i];

        // call kalman prediction fist
		if (track->GetFilterGoal() == tracking::FilterGoal::FilterRect ||
			track->GetFilterGoal() == tracking::FilterGoal::FilterRRect)
			track->KalmanPredictRect();
		else
			track->KalmanPredictPoint();

        constexpr bool DIST_LOGS = false;

        // Calc distance between track and regions
        for (size_t j = 0; j < regions.size(); ++j)
        {
            const auto& reg = regions[j];

            auto dist = maxPossibleCost;
            if (m_settings.CheckType(m_tracks[i]->LastRegion().m_type, reg.m_type))
            {
                dist = 0;
                size_t ind = 0;
                // Euclidean distance between centers
                if (m_settings.m_distType[ind] > 0.0f && ind == tracking::DistCenters)
                {
#if 1
                    track_t ellipseDist = GetEllipseDist(*track, reg);
                    if (ellipseDist > 1)
                        dist += m_settings.m_distType[ind];
                    else
                        dist += ellipseDist * m_settings.m_distType[ind];
#else
                    dist += m_settings.m_distType[ind] * track->CalcDistCenter(reg);
#endif
                    if constexpr (DIST_LOGS)
                    {
                        std::cout << "DistCenters : " << m_settings.m_distType[ind] << ", dist = " << dist << "\n";
                        //std::cout << "dist = " << dist << ", ed = " << ellipseDist << ", reg.m_rrect.center = " << reg.m_rrect.center << ", predictedArea: center = " << predictedArea.center << ", size = " << predictedArea.size << ", angle = " << predictedArea.angle << "\n";
                        std::cout << "track id = " << m_tracks[i]->GetID().ID2Str() << " type = " << TypeConverter::Type2Str(m_tracks[i]->LastRegion().m_type) << " (" << m_tracks[i]->LastRegion().m_type << "), region id = " << j << ", type = " << TypeConverter::Type2Str(reg.m_type) << " (" << reg.m_type << ")" << std::endl;
                    }
				}
                ++ind;

                // Euclidean distance between bounding rectangles
                if (m_settings.m_distType[ind] > 0.0f && ind == tracking::DistRects)
                {
#if 1
                    track_t ellipseDist = GetEllipseDist(*track, reg);
					if (ellipseDist < 1)
					{
						track_t dw = track->WidthDist(reg);
						track_t dh = track->HeightDist(reg);
						dist += m_settings.m_distType[ind] * (1 - (1 - ellipseDist) * (dw + dh) * 0.5f);
					}
					else
					{
						dist += m_settings.m_distType[ind];
					}

                    if constexpr (DIST_LOGS)
                    {
                        std::cout << "DistRects : " << m_settings.m_distType[ind] << ", dist = " << dist << "\n";
                        track_t dw = track->WidthDist(reg);
                        track_t dh = track->HeightDist(reg);
                        std::cout << "dist = " << dist << ", ed = " << ellipseDist << ", dw = " << dw << ", dh = " << dh << "\n";
                        std::cout << "track type = " << TypeConverter::Type2Str(m_tracks[i]->LastRegion().m_type) << " (" << m_tracks[i]->LastRegion().m_type << "), region type = " << TypeConverter::Type2Str(reg.m_type) << " (" << reg.m_type << ")\n";
                        std::cout << "track = " << m_tracks[i]->LastRegion().m_brect << ", reg = " << reg.m_brect << ", rrect = [" << reg.m_rrect.size << " from " << reg.m_rrect.center << ", " << reg.m_rrect.angle << "]" << std::endl;
                    }
#else
					dist += m_settings.m_distType[ind] * track->CalcDistRect(reg);
#endif
				}
				++ind;

				// Intersection over Union, IoU
				if (m_settings.m_distType[ind] > 0.0f && ind == tracking::DistJaccard)
                {
					dist += m_settings.m_distType[ind] * track->CalcDistJaccard(reg);
                    if constexpr (DIST_LOGS)
                        std::cout << "DistJaccard : " << m_settings.m_distType[ind] << ", dist = " << dist << std::endl;
                }
				++ind;

				// Bhatacharia distance between histograms
				if (m_settings.m_distType[ind] > 0.0f && ind == tracking::DistHist)
                {
                    dist += m_settings.m_distType[ind] * track->CalcDistHist(regionEmbeddings[j]);
                    if constexpr (DIST_LOGS)
                        std::cout << "DistHist : " << m_settings.m_distType[ind] << ", dist = " << dist << std::endl;
                }
				++ind;

				// Cosine distance between embeddings. The track itself checks
                // that the observation and its gallery come from the same
                // model, so near types (car/truck/bus) can share one model.
                if (m_settings.m_distType[ind] > 0.0f && ind == tracking::DistFeatureCos)
                {
                    auto resCos = track->CalcCosine(regionEmbeddings[j]);
                    if (resCos.second)
                    {
                        dist += m_settings.m_distType[ind] * resCos.first;
                    }
                    else
                    {
                        // Embeddings are unavailable (no model for this type or an empty embedding): the cosine metric is excluded from the sum.
                        // Rescale the accumulated distance by the weights of the remaining metrics so the overall scale (and the m_distThres semantics) stays consistent.
                        const track_t restWeight = 1.f - m_settings.m_distType[ind];
                        if (restWeight > 1e-6f)
                            dist /= restWeight;
                        else
                            dist = maxPossibleCost; // No usable metric for this pair
                    }
                    if constexpr (DIST_LOGS)
                        std::cout << "DistFeatureCos : " << m_settings.m_distType[ind] << ", dist = " << dist << std::endl;
                }
				++ind;

                // Mahalanobis
				if (m_settings.m_distType[ind] > 0.0f && ind == tracking::DistMahalanobis)
					dist += m_settings.m_distType[ind] * track->CalcMahalanobisDist(reg.m_rrect);
				++ind;

				// Euclidean distance in the local ENU plane, meters
				if (useLocalCartesian && ind == tracking::DistLocalCartesian)
				{
					const track_t distMeters = track->CalcDistLocalCartesian(reg);
					if (distMeters < 0.0f)
					{
						dist = maxPossibleCost;
					}
					else
					{
						const track_t weight = m_settings.m_distType[ind] > 0.0f ? m_settings.m_distType[ind] : static_cast<track_t>(1.0);
						dist += weight * distMeters;
					}

					if constexpr (DIST_LOGS)
						std::cout << "DistLocalCartesian : " << m_settings.m_distType[ind] << ", dist = " << dist << " m" << std::endl;
				}
				++ind;

				assert(ind == tracking::DistsCount);
			}

			costMatrix[i + j * colsTracks] = dist;
            if constexpr (DIST_LOGS)
                    std::cout << "costMatrix[" << j << "][" << i << "] (or " << (i + j * colsTracks) << ") = " << dist << std::endl;

            if (!std::isfinite(dist) || dist < 0)
                dist = maxPossibleCost;
            dist = std::min(dist, maxPossibleCost);

			if (dist > maxCost)
				maxCost = dist;
		}
	}
}

///
/// \brief CTracker::CalcEmbeddins
/// \param regionEmbeddings
/// \param regions
/// \param currFrame
///
void CTracker::CalcEmbeddins(std::vector<RegionEmbedding>& regionEmbeddings, const regions_t& regions, cv::UMat currFrame)
{
    regionEmbeddings.clear();

    if (regions.empty())
        return;

    // Only histogram/cosine distances need per-region buffers. In the default
    // IoU-only mode (neither metric enabled) there is nothing to compute, so
    // leave the vector empty: the callers already take the "no embeddings"
    // path (roadmap M3, 4.1). This also keeps the reused buffer capacity.
    const bool needHist = m_settings.m_distType[tracking::DistHist] > 0.0f;
    const bool needFeature = m_settings.m_distType[tracking::DistFeatureCos] > 0.0f;
    if (!needHist && !needFeature)
        return;

    regionEmbeddings.resize(regions.size());

    const cv::Rect frameRect(0, 0, currFrame.cols, currFrame.rows);
    const bool hasFrame = !currFrame.empty() && currFrame.cols > 1 && currFrame.rows > 1;

    // Bhatacharia distance between histograms
    if (hasFrame && needHist)
    {
        const int bins = 64;
        std::vector<int> histSize;
        std::vector<float> ranges;
        std::vector<int> channels;

        for (int i = 0, stop = currFrame.channels(); i < stop; ++i)
        {
            histSize.push_back(bins);
            ranges.push_back(0);
            ranges.push_back(255);
            channels.push_back(i);
        }

        for (size_t j = 0; j < regions.size(); ++j)
        {
            // A detector may return a box partially outside the frame (or an
            // empty/degenerate one). calcHist() must not receive such an ROI.
            const cv::Rect roi = regions[j].m_brect & frameRect;
            if (roi.width < 2 || roi.height < 2)
                continue;

            std::vector<cv::UMat> regROI = { currFrame(roi) };
            cv::calcHist(regROI, channels, cv::Mat(), regionEmbeddings[j].m_hist, histSize, ranges, false);
            cv::normalize(regionEmbeddings[j].m_hist, regionEmbeddings[j].m_hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
        }
    }

    // Cosine distance between embeddings: group crops by model and run one
    // batched inference per model. CPU fallback is throttled/capped so a
    // 30-object frame does not turn into a multi-second stall.
    if (!hasFrame || !needFeature)
        return;

    const bool cpuMode = m_reidAllCpu && !m_reidModels.empty();
    if (cpuMode && m_settings.m_reidSettings.m_cpuInterval > 1 &&
        ((m_reidFrameCounter > 0 ? m_reidFrameCounter - 1 : 0) %
            static_cast<size_t>(m_settings.m_reidSettings.m_cpuInterval)) != 0)
    {
        return;
    }

    struct ReIdCrop
    {
        size_t m_regionIndex = 0;
        cv::Mat m_crop;
        float m_quality = 0.0f;
    };
    std::map<int, std::vector<ReIdCrop>> groups;
    std::map<int, std::shared_ptr<EmbeddingsCalculator>> groupModels;

    const cv::Mat frame = currFrame.getMat(cv::ACCESS_READ);

    auto Clamp01 = [](float value) -> float
    {
        return std::max(0.0f, std::min(1.0f, value));
    };

    // Cheap crop-quality heuristic: size, border truncation, sharpness and
    // detector confidence. It is used to decide whether a crop may update the
    // track gallery; matching still uses the current observation.
    auto CropQuality = [&](const cv::Mat& crop, const cv::Rect& roi, float confidence) -> float
    {
        const int minSide = std::min(roi.width, roi.height);
        const float sizeScore = Clamp01(static_cast<float>(minSide) / 48.0f);

        float borderScore = 1.0f;
        if (roi.x <= 0 || roi.y <= 0 ||
            roi.x + roi.width >= frameRect.width || roi.y + roi.height >= frameRect.height)
        {
            borderScore = 0.5f;
        }

        float sharpnessScore = 0.5f;
        try
        {
            cv::Mat gray;
            if (crop.channels() == 3)
                cv::cvtColor(crop, gray, cv::COLOR_BGR2GRAY);
            else if (crop.channels() == 4)
                cv::cvtColor(crop, gray, cv::COLOR_BGRA2GRAY);
            else
                gray = crop;
            cv::Mat laplacian;
            cv::Laplacian(gray, laplacian, CV_64F);
            cv::Scalar mean;
            cv::Scalar stddev;
            cv::meanStdDev(laplacian, mean, stddev);
            sharpnessScore = Clamp01(static_cast<float>(stddev[0] * stddev[0]) / 500.0f);
        }
        catch (const cv::Exception&)
        {
            sharpnessScore = 0.5f;
        }

        const float confidenceScore = confidence >= 0.0f ? Clamp01(confidence) : 0.5f;
        return Clamp01(0.45f * sizeScore + 0.25f * sharpnessScore +
                       0.20f * confidenceScore + 0.10f * borderScore);
    };

    for (size_t j = 0; j < regions.size(); ++j)
    {
        const auto embCalc = m_embCalculators.find(regions[j].m_type);
        if (embCalc == std::end(m_embCalculators))
            continue;

        const cv::Rect roi = regions[j].m_brect & frameRect;
        const int minSide = std::min(roi.width, roi.height);
        if (minSide < std::max(2, embCalc->second->GetMinCropSide()))
            continue;

        ReIdCrop item;
        item.m_regionIndex = j;
        item.m_crop = frame(roi);
        item.m_quality = m_settings.m_reidSettings.m_qualityGate
            ? CropQuality(item.m_crop, roi, regions[j].m_confidence)
            : 1.0f;

        const int modelId = embCalc->second->GetModelId();
        groups[modelId].push_back(item);
        groupModels[modelId] = embCalc->second;
    }

    for (auto& group : groups)
    {
        const int modelId = group.first;
        const auto modelIt = groupModels.find(modelId);
        if (modelIt == std::end(groupModels))
            continue;
        const auto& calculator = modelIt->second;
        auto& items = group.second;

        if (calculator->IsCpu() && m_settings.m_reidSettings.m_cpuMaxObjects > 0 &&
            items.size() > static_cast<size_t>(m_settings.m_reidSettings.m_cpuMaxObjects))
        {
            if (m_reidCpuSkippedModels.insert(modelId).second)
            {
                std::cerr << "CTracker: CPU ReID budget exceeded (" << items.size()
                          << " > " << m_settings.m_reidSettings.m_cpuMaxObjects
                          << "), ReID is skipped for model " << modelId << std::endl;
            }
            continue;
        }

        std::vector<cv::Mat> crops;
        crops.reserve(items.size());
        for (const ReIdCrop& item : items)
            crops.push_back(item.m_crop);

        cv::Mat embeddings;
        if (!calculator->Extract(crops, embeddings) || embeddings.rows != static_cast<int>(items.size()))
        {
            std::cerr << "CTracker: ReID batch inference failed for model " << modelId << std::endl;
            continue;
        }

        for (size_t k = 0; k < items.size(); ++k)
        {
            RegionEmbedding& regionEmbedding = regionEmbeddings[items[k].m_regionIndex];
            regionEmbedding.m_embedding = embeddings.row(static_cast<int>(k)).clone();
            regionEmbedding.m_modelId = modelId;
            regionEmbedding.m_quality = items[k].m_quality;
        }
    }
}

///
/// \brief CTracker::GetEllipseDist
/// \param trackRef
/// \param reg
/// \return
///
track_t CTracker::GetEllipseDist(const CTrack& trackRef, const CRegion& reg)
{
	cv::Size_<track_t> minRadius;

	if (m_settings.m_minAreaRadiusPix <= 0)
	{
		minRadius.width = m_settings.m_minAreaRadiusK * trackRef.LastRegion().m_rrect.size.width;
		minRadius.height = m_settings.m_minAreaRadiusK * trackRef.LastRegion().m_rrect.size.height;
	}
	else
	{
		minRadius.width = m_settings.m_minAreaRadiusPix;
		minRadius.height = m_settings.m_minAreaRadiusPix;
	}

	// Calc predicted area for track
	cv::RotatedRect predictedArea = trackRef.CalcPredictionEllipse(minRadius);

	return trackRef.IsInsideArea(reg.m_rrect.center, predictedArea);
}

namespace
{
using TrackerRegistryMap = std::map<tracking::TrackerTemplate, BaseTracker::CreateFn>;

///
/// \brief TrackerRegistry
/// The tracker factory table. Registration happens once, on the first
/// CreateTracker() call, so the built-in trackers are always available while
/// applications can add their own via BaseTracker::RegisterTracker().
///
TrackerRegistryMap& TrackerRegistry()
{
    static TrackerRegistryMap registry;
    return registry;
}

///
/// \brief RegisterBuiltinTrackers
/// Thread-safe one-time registration of the trackers compiled into this build.
///
void RegisterBuiltinTrackers()
{
    static const bool registered = []()
    {
        BaseTracker::RegisterTracker(tracking::UniversalTracker,
            [](const TrackerSettings& settings, float) -> std::unique_ptr<BaseTracker>
            {
                std::string reason;
                if (!settings.CheckFilterModel(&reason))
                {
                    std::cerr << "BaseTracker::CreateTracker: " << reason << std::endl;
                    return nullptr;
                }
                return std::make_unique<CTracker>(settings);
            });
        BaseTracker::RegisterTracker(tracking::ByteTrack,
            [](const TrackerSettings& settings, float fps) -> std::unique_ptr<BaseTracker>
            {
                return std::make_unique<byte_track::BYTETracker>(
                    (fps > 1.f) ? cvRound(fps) : 30,
                    settings.m_byteTrackSettings.m_trackBuffer,
                    settings.m_byteTrackSettings.m_trackThresh,
                    settings.m_byteTrackSettings.m_highThresh,
                    settings.m_byteTrackSettings.m_matchThresh);
            });
        return true;
    }();
    (void)registered;
}
} // namespace

namespace mtracking
{

///
/// \brief RegisterTracker
/// \param trackerType
/// \param fn
///
void BaseTracker::RegisterTracker(tracking::TrackerTemplate trackerType, CreateFn fn)
{
    TrackerRegistry()[trackerType] = fn;
}

///
/// BaseTracker::CreateTracker
///
std::unique_ptr<BaseTracker> BaseTracker::CreateTracker(const TrackerSettings& settings, float fps)
{
    RegisterBuiltinTrackers();

    auto it = TrackerRegistry().find(settings.m_tracker);
    if (it == TrackerRegistry().end())
    {
        std::cerr << "BaseTracker::CreateTracker: unknown tracker type "
                  << static_cast<int>(settings.m_tracker) << std::endl;
        return nullptr;
    }
    return it->second(settings, fps);
}

} // namespace mtracking
