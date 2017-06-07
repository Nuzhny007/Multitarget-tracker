#pragma once
#include <iostream>
#include <vector>
#include <memory>
#include <array>

#include "defines.h"
#include "track.h"
#include "LocalTracker.h"

#include "HungarianAlg.h"

#include <GTL/GTL.h>
#include "mygraph.h"
#include "mwbmatching.h"
#include "tokenise.h"

namespace TrackTypes
{
enum DistType
{
    DistCenters = 0,
    DistRects = 1,
    DistJaccard = 2
};
enum FilterGoal
{
    FilterCenter = 0,
    FilterRect = 1
};
enum KalmanType
{
    KalmanLinear = 0,
    KalmanUnscented = 1
};
enum MatchType
{
    MatchHungrian = 0,
    MatchBipart = 1
};
enum LostTrackType
{
    TrackNone = 0,
    TrackKCF = 1
};
}

///
/// \brief The CTracker class
/// Tracker. Manage tracks. Create, remove, update.
///
template<typename REGION_T>
class CTracker
{
public:
    typedef std::vector<std::unique_ptr<CTrack<REGION_T>>> tracks_t;

    CTracker(bool useLocalTracking,
             TrackTypes::DistType distType,
             TrackTypes::KalmanType kalmanType,
             TrackTypes::FilterGoal filterGoal,
             TrackTypes::LostTrackType useExternalTrackerForLostObjects,
             TrackTypes::MatchType matchType,
             track_t dt_,
             track_t accelNoiseMag_,
             track_t dist_thres_ = 60,
             size_t maximum_allowed_skipped_frames_ = 10,
             size_t max_trace_length_ = 10)
        :
          m_useLocalTracking(useLocalTracking),
          m_distType(distType),
          m_kalmanType(kalmanType),
          m_filterGoal(filterGoal),
          m_useExternalTrackerForLostObjects(useExternalTrackerForLostObjects),
          m_matchType(matchType),
          dt(dt_),
          accelNoiseMag(accelNoiseMag_),
          dist_thres(dist_thres_),
          maximum_allowed_skipped_frames(maximum_allowed_skipped_frames_),
          max_trace_length(max_trace_length_),
          NextTrackID(0)
    {

    }

    ~CTracker(void)
    {

    }

    tracks_t tracks;

    ///
    /// \brief Update
    /// \param regions
    /// \param grayFrame
    ///
    template<typename REGIONS_T>
    void Update(const REGIONS_T& regions,
                cv::Mat grayFrame)
    {
        TKalmanFilter::KalmanType kalmanType = (m_kalmanType == TrackTypes::KalmanLinear) ? TKalmanFilter::TypeLinear : TKalmanFilter::TypeUnscented;

        if (m_prevFrame.size() == grayFrame.size())
        {
            if (m_useLocalTracking)
            {
                m_localTracker.Update(tracks, m_prevFrame, grayFrame);
            }
        }

        // -----------------------------------
        // If there is no tracks yet, then every cv::Point begins its own track.
        // -----------------------------------
        if (tracks.size() == 0)
        {
            // If no tracks yet
            for (size_t i = 0; i < regions.size(); ++i)
            {
                tracks.push_back(std::make_unique<CTrack<REGION_T>>(regions[i],
                                                                    kalmanType,
                                                                    dt,
                                                                    accelNoiseMag,
                                                                    NextTrackID++,
                                                                    m_filterGoal == TrackTypes::FilterRect,
                                                                    m_useExternalTrackerForLostObjects == TrackTypes::TrackKCF));
            }
        }

        size_t N = tracks.size();		// треки
        size_t M = regions.size();	// детекты

        assignments_t assignment(N, -1); // назначения

        if (!tracks.empty())
        {
            // Матрица расстояний от N-ного трека до M-ного детекта.
            distMatrix_t Cost(N * M);

            // -----------------------------------
            // Треки уже есть, составим матрицу расстояний
            // -----------------------------------
            track_t maxCost = 0;
            switch (m_distType)
            {
            case TrackTypes::DistCenters:
                for (size_t i = 0; i < tracks.size(); i++)
                {
                    for (size_t j = 0; j < regions.size(); j++)
                    {
                        auto dist = tracks[i]->CalcDist(regions[j].m_obj.Center());
                        Cost[i + j * N] = dist;
                        if (dist > maxCost)
                        {
                            maxCost = dist;
                        }
                    }
                }
                break;

            case TrackTypes::DistRects:
                for (size_t i = 0; i < tracks.size(); i++)
                {
                    for (size_t j = 0; j < regions.size(); j++)
                    {
                        auto dist = tracks[i]->CalcDist(regions[j].m_obj.BoundingRect());
                        Cost[i + j * N] = dist;
                        if (dist > maxCost)
                        {
                            maxCost = dist;
                        }
                    }
                }
                break;

            case TrackTypes::DistJaccard:
                for (size_t i = 0; i < tracks.size(); i++)
                {
                    for (size_t j = 0; j < regions.size(); j++)
                    {
                        auto dist = tracks[i]->CalcDistJaccard(regions[j].m_obj.BoundingRect());
                        Cost[i + j * N] = dist;
                        if (dist > maxCost)
                        {
                            maxCost = dist;
                        }
                    }
                }
                break;
            }
            // -----------------------------------
            // Solving assignment problem (tracks and predictions of Kalman filter)
            // -----------------------------------
            if (m_matchType == TrackTypes::MatchHungrian)
            {
                AssignmentProblemSolver APS;
                APS.Solve(Cost, N, M, assignment, AssignmentProblemSolver::optimal);
            }
            else
            {
                MyGraph G;
                G.make_directed();

                std::vector<node> nodes(N + M);

                for (size_t i = 0; i < nodes.size(); ++i)
                {
                    nodes[i] = G.new_node();
                }

                edge_map<int> weights(G, 100);
                for (size_t i = 0; i < tracks.size(); i++)
                {
                    bool hasZeroEdge = false;

                    for (size_t j = 0; j < regions.size(); j++)
                    {
                        track_t currCost = Cost[i + j * N];

                        edge e = G.new_edge(nodes[i], nodes[N + j]);

                        if (currCost < dist_thres)
                        {
                            int weight = maxCost - currCost + 1;
                            G.set_edge_weight(e, weight);
                            weights[e] = weight;
                        }
                        else
                        {
                            if (!hasZeroEdge)
                            {
                                G.set_edge_weight(e, 0);
                                weights[e] = 0;
                            }
                            hasZeroEdge = true;
                        }
                    }
                }

                edges_t L = MAX_WEIGHT_BIPARTITE_MATCHING(G, weights);
                for (edges_t::iterator it = L.begin(); it != L.end(); ++it)
                {
                    node a = it->source();
                    node b = it->target();
                    assignment[b.id()] = a.id() - N;
                }
            }

            // -----------------------------------
            // clean assignment from pairs with large distance
            // -----------------------------------
            for (size_t i = 0; i < assignment.size(); i++)
            {
                if (assignment[i] != -1)
                {
                    if (Cost[i + assignment[i] * N] > dist_thres)
                    {
                        assignment[i] = -1;
                        tracks[i]->m_skippedFrames++;
                    }
                }
                else
                {
                    // If track have no assigned detect, then increment skipped frames counter.
                    tracks[i]->m_skippedFrames++;
                }
            }

            // -----------------------------------
            // If track didn't get detects long time, remove it.
            // -----------------------------------
            for (int i = 0; i < static_cast<int>(tracks.size()); i++)
            {
                if (tracks[i]->m_skippedFrames > maximum_allowed_skipped_frames)
                {
                    tracks.erase(tracks.begin() + i);
                    assignment.erase(assignment.begin() + i);
                    i--;
                }
            }
        }

        // -----------------------------------
        // Search for unassigned detects and start new tracks for them.
        // -----------------------------------
        for (size_t i = 0; i < regions.size(); ++i)
        {
            if (find(assignment.begin(), assignment.end(), i) == assignment.end())
            {
                tracks.push_back(std::make_unique<CTrack<REGION_T>>(regions[i],
                                                                    kalmanType,
                                                                    dt,
                                                                    accelNoiseMag,
                                                                    NextTrackID++,
                                                                    m_filterGoal == TrackTypes::FilterRect,
                                                                    m_useExternalTrackerForLostObjects == TrackTypes::TrackKCF));
            }
        }

        // Update Kalman Filters state

        for (size_t i = 0; i < assignment.size(); i++)
        {
            // If track updated less than one time, than filter state is not correct.

            if (assignment[i] != -1) // If we have assigned detect, then update using its coordinates,
            {
                tracks[i]->m_skippedFrames = 0;
                tracks[i]->Update(regions[assignment[i]], true, max_trace_length, m_prevFrame, grayFrame);
            }
            else				     // if not continue using predictions
            {
                tracks[i]->Update(REGION_T(), false, max_trace_length, m_prevFrame, grayFrame);
            }
        }

        grayFrame.copyTo(m_prevFrame);
    }

private:
    // Use local tracking for regions between two frames
    bool m_useLocalTracking;

    TrackTypes::DistType m_distType;
    TrackTypes::KalmanType m_kalmanType;
    TrackTypes::FilterGoal m_filterGoal;
    TrackTypes::LostTrackType m_useExternalTrackerForLostObjects;
    TrackTypes::MatchType m_matchType;

    // Шаг времени опроса фильтра
    track_t dt;

    track_t accelNoiseMag;

    // Порог расстояния. Если точки находятся дуг от друга на расстоянии,
    // превышающем этот порог, то эта пара не рассматривается в задаче о назначениях.
    track_t dist_thres;
    // Максимальное количество кадров которое трек сохраняется не получая данных о измерений.
    size_t maximum_allowed_skipped_frames;
    // Максимальная длина следа
    size_t max_trace_length;

    size_t NextTrackID;

    LocalTracker m_localTracker;

    cv::Mat m_prevFrame;
};
