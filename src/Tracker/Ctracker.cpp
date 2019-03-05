#include "Ctracker.h"

#include <GTL/GTL.h>
#include "mygraph.h"
#include "mwbmatching.h"
#include "tokenise.h"

///
/// \brief CTracker::CTracker
/// Tracker. Manage tracks. Create, remove, update.
/// \param settings
///
CTracker::CTracker(const TrackerSettings& settings)
    :
      m_settings(settings),
      m_nextTrackID(0)
{
#ifdef BUILD_LBM
    m_LBMfilter = std::unique_ptr<lmbTracker>(new lmbTracker());
#endif
}

///
/// \brief CTracker::~CTracker
///
CTracker::~CTracker(void)
{
}

///
/// \brief CTracker::Update
/// \param regions
/// \param grayFrame
/// \param fps
///
void CTracker::Update(
        const regions_t& regions,
        cv::UMat grayFrame,
        float fps
        )
{
    if (m_prevFrame.size() == grayFrame.size())
    {
        if (m_settings.m_useLocalTracking)
        {
            m_localTracker.Update(tracks, m_prevFrame, grayFrame);
        }
    }

    UpdateTrackingState(regions, grayFrame, fps);

    grayFrame.copyTo(m_prevFrame);
}

///
/// \brief CTracker::UpdateTrackingState
/// \param regions
/// \param grayFrame
/// \param fps
///
void CTracker::UpdateTrackingState(
        const regions_t& regions,
        cv::UMat grayFrame,
        float fps
        )
{
    const size_t N = tracks.size();	// Tracking objects
    const size_t M = regions.size();	// Detections or regions

    assignments_t assignment(N, -1); // Assignments regions -> tracks

    if (!tracks.empty())
    {
        // Distance matrix between all tracks to all regions
        distMatrix_t costMatrix(N * M);
        const track_t maxPossibleCost = static_cast<track_t>(grayFrame.cols * grayFrame.rows);
        track_t maxCost = 0;
        CreateDistaceMatrix(regions, costMatrix, maxPossibleCost, maxCost);

        // Solving assignment problem (tracks and predictions of Kalman filter)
        switch (m_settings.m_matchType)
        {
        case tracking::MatchHungrian:
            SolveHungrian(costMatrix, N, M, assignment);
            break;

        case tracking::MatchBipart:
            SolveBipartiteGraphs(costMatrix, N, M, assignment, maxCost);
            break;

        case tracking::MatchLBM:
            SolveLBM(regions, assignment);
            break;
        }

        // clean assignment from pairs with large distance
        for (size_t i = 0; i < assignment.size(); i++)
        {
            if (assignment[i] != -1)
            {
                if (costMatrix[i + assignment[i] * N] > m_settings.m_distThres)
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

        // If track didn't get detects long time, remove it.
        for (int i = 0; i < static_cast<int>(tracks.size()); i++)
        {
            if (tracks[i]->m_skippedFrames > m_settings.m_maximumAllowedSkippedFrames ||
                    tracks[i]->IsStaticTimeout(cvRound(fps * (m_settings.m_maxStaticTime - m_settings.m_minStaticTime))))
            {
                tracks.erase(tracks.begin() + i);
                assignment.erase(assignment.begin() + i);
                i--;
            }
        }
    }

    // Search for unassigned detects and start new tracks for them.
    for (size_t i = 0; i < regions.size(); ++i)
    {
        if (find(assignment.begin(), assignment.end(), i) == assignment.end())
        {
            tracks.push_back(std::make_unique<CTrack>(regions[i],
                                                      m_settings.m_kalmanType,
                                                      m_settings.m_dt,
                                                      m_settings.m_accelNoiseMag,
                                                      m_nextTrackID++,
                                                      m_settings.m_filterGoal == tracking::FilterRect,
                                                      m_settings.m_lostTrackType));
        }
    }

    // Update Kalman Filters state
    const ptrdiff_t stop_i = static_cast<int>(assignment.size());
//#pragma omp parallel for
    for (int i = 0; i < stop_i; ++i)
    {
        // If track updated less than one time, than filter state is not correct.
        if (assignment[i] != -1) // If we have assigned detect, then update using its coordinates,
        {
            tracks[i]->m_skippedFrames = 0;
            tracks[i]->Update(
                        regions[assignment[i]], true,
                    m_settings.m_maxTraceLength,
                    m_prevFrame, grayFrame,
                    m_settings.m_useAbandonedDetection ? cvRound(m_settings.m_minStaticTime * fps) : 0);
        }
        else				     // if not continue using predictions
        {
            tracks[i]->Update(CRegion(), false, m_settings.m_maxTraceLength, m_prevFrame, grayFrame, 0);
        }
    }
}

///
/// \brief CTracker::CreateDistaceMatrix
/// \param regions
/// \param costMatrix
/// \param maxPossibleCost
/// \param maxCost
///
void CTracker::CreateDistaceMatrix(const regions_t& regions, distMatrix_t& costMatrix, track_t maxPossibleCost, track_t& maxCost)
{
    const size_t N = tracks.size();	// Tracking objects
    maxCost = 0;
    switch (m_settings.m_distType)
    {
    case tracking::DistCenters:
        for (size_t i = 0; i < tracks.size(); i++)
        {
            for (size_t j = 0; j < regions.size(); j++)
            {
                auto dist = tracks[i]->CheckType(regions[j].m_type) ? tracks[i]->CalcDist((regions[j].m_rect.tl() + regions[j].m_rect.br()) / 2) : maxPossibleCost;
                costMatrix[i + j * N] = dist;
                if (dist > maxCost)
                {
                    maxCost = dist;
                }
            }
        }
        break;

    case tracking::DistRects:
        for (size_t i = 0; i < tracks.size(); i++)
        {
            for (size_t j = 0; j < regions.size(); j++)
            {
                auto dist = tracks[i]->CheckType(regions[j].m_type) ? tracks[i]->CalcDist(regions[j].m_rect) : maxPossibleCost;
                costMatrix[i + j * N] = dist;
                if (dist > maxCost)
                {
                    maxCost = dist;
                }
            }
        }
        break;

    case tracking::DistJaccard:
        for (size_t i = 0; i < tracks.size(); i++)
        {
            for (size_t j = 0; j < regions.size(); j++)
            {
                auto dist = tracks[i]->CheckType(regions[j].m_type) ? tracks[i]->CalcDistJaccard(regions[j].m_rect) : 1;
                costMatrix[i + j * N] = dist;
                if (dist > maxCost)
                {
                    maxCost = dist;
                }
            }
        }
        break;
    }
}

///
/// \brief CTracker::SolveHungrian
/// \param costMatrix
/// \param N
/// \param M
/// \param assignment
///
void CTracker::SolveHungrian(const distMatrix_t& costMatrix, size_t N, size_t M, assignments_t& assignment)
{
    AssignmentProblemSolver APS;
    APS.Solve(costMatrix, N, M, assignment, AssignmentProblemSolver::optimal);
}

///
/// \brief CTracker::SolveBipartiteGraphs
/// \param costMatrix
/// \param N
/// \param M
/// \param assignment
/// \param maxCost
///
void CTracker::SolveBipartiteGraphs(const distMatrix_t& costMatrix, size_t N, size_t M, assignments_t& assignment, track_t maxCost)
{
    MyGraph G;
    G.make_directed();

    std::vector<node> nodes(N + M);

    for (size_t i = 0; i < nodes.size(); ++i)
    {
        nodes[i] = G.new_node();
    }

    edge_map<int> weights(G, 100);
    for (size_t i = 0; i < N; i++)
    {
        bool hasZeroEdge = false;

        for (size_t j = 0; j < M; j++)
        {
            track_t currCost = costMatrix[i + j * N];

            edge e = G.new_edge(nodes[i], nodes[N + j]);

            if (currCost < m_settings.m_distThres)
            {
                int weight = static_cast<int>(maxCost - currCost + 1);
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
        assignment[b.id()] = static_cast<assignments_t::value_type>(a.id() - N);
    }
}

///
/// \brief CTracker::SolveLBM
/// \param costMatrix
/// \param N
/// \param M
/// \param assignment
///
void CTracker::SolveLBM(const regions_t& regions, assignments_t& assignment)
{
#ifdef BUILD_LBM
    std::vector<arma::fvec> Zk;
    Zk.reserve(regions.size());

    for (const CRegion& region : regions)
    {
        Zk.emplace_back(arma::fvec({region.m_rect.x + region.m_rect.width / 2.f, region.m_rect.y + region.m_rect.height / 2.f}));
    }

    std::vector<std::pair<std::pair<unsigned, unsigned>, arma::fvec>> Xk = m_LBMfilter->runFilter(Zk);

    for (auto &x : Xk)
    {
        //      std::cout << "<" << x.first.first << "," << x.first.second << ">" << std::endl;
        //      x.second.print();
        cv::Rect rect;
        rect.height = 10;
        rect.width = 10;
        rect.x = std::max(0.0, x.second(0) - rect.width / 2.0);
        rect.y = std::max(0.0, x.second(2) - rect.height / 2.0);

        //assignment[x.first.second] = 1;
    }
#else
    std::cerr << "Project was compiled without LBM tracking! Set BUILD_LBM=ON in CMake " << std::endl;
#endif
}
