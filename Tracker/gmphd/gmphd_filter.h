#ifndef GMPHD_FILTER_H
#define GMPHD_FILTER_H

// Author : Benjamin Lefaudeux (blefaudeux@github)


#include "gaussian_mixture.h"
#include <iostream>
#include <memory>

/*!
 * \brief The spawning_model struct
 */
template<int DIM>
struct SpawningModel
{
    SpawningModel()
    {
        m_trans = Eigen::MatrixXf::Ones(2 * DIM, 2 * DIM);
        m_cov = Eigen::MatrixXf::Ones(2 * DIM, 2 * DIM);
        m_offset = Eigen::MatrixXf::Zero(2 * DIM, 1);
        m_weight = 0.1f;
    }

    float m_weight;

    Eigen::MatrixXf m_trans;
    Eigen::MatrixXf m_cov;
    Eigen::MatrixXf m_offset;
};

/*!
 * \brief The gmphd_filter class
 */
template<int DIM>
class GMPHD
{
public:
    typedef GaussianModel<2 * DIM> GModel;
    typedef GaussianMixture<2 * DIM> GMixture;
    typedef SpawningModel<DIM> SModel;

    /*!
     * \brief GMPHD
     * \param max_gaussians
     * \param verbose
     */
    GMPHD(bool verbose = false)
        :
          m_bVerbose(verbose)
    {
        m_pruneTruncThld = 0.f;
        m_pDetection = 0.f;
        m_pSurvival = 0.f;

        // Initialize all gaussian mixtures, we know the dimension now
        m_measTargets.reset(new GMixture);
        m_birthTargets.reset(new GMixture);
        m_currTargets.reset(new GMixture);
        m_expTargets.reset(new GMixture);
        m_extractedTargets.reset(new GMixture);
        m_spawnTargets.reset(new GMixture);
    }

    /*!
     * \brief isInitialized
     * \return
     */
    bool isInitialized()
    {
        if( m_tgtDynTrans.cols() != 2 * DIM)
        {
            printf("[GMPHD] - Motion model not set\n");
            return false;
        }

        if( m_pruneTruncThld <= 0.f)
        {
            printf("[GMPHD] - Pruning parameters not set\n");
            return false;
        }

        if( m_pDetection <= 0.f || m_pSurvival <= 0.f )
        {
            printf("[GMPHD] - Observation model not set\n");
            return false;
        }

        return true;
    }

    /*!
     * \brief setNewReferential
     * Input: raw measurements and possible ref change
     * \param transform
     */
    void setNewReferential( Eigen::Matrix4f const & transform)
    {
        // Change referential for every gaussian in the gaussian mixture
        m_currTargets->changeReferential(transform);
    }

    /*!
     * \brief setNewMeasurements
     * \param position
     * \param speed
     */
    void setNewMeasurements( std::vector<float> const & position, std::vector<float> const & speed)
    {
        // Clear the gaussian mixture
        m_measTargets->m_gaussians.clear();

        unsigned int iTarget = 0;

        while(iTarget < position.size()/DIM)
        {
            GModel new_obs;

            for (unsigned int i=0; i< DIM; ++i) {
                // Create new gaussian model according to measurement
                new_obs.m_mean(i) = position[iTarget*DIM + i];
                new_obs.m_mean(i+DIM) = speed[iTarget*DIM + i];
            }

            new_obs.m_cov = m_obsCov;
            new_obs.m_weight = 1.f;

            m_measTargets->m_gaussians.push_back(std::move(new_obs));

            iTarget++;
        }
    }

    /*!
     * \brief getTrackedTargets
     * Output
     * \param position
     * \param speed
     * \param weight
     * \param extract_thld
     */
    void  getTrackedTargets( std::vector<float> & position, std::vector<float> & speed, std::vector<float> & weight,
                             float const & extract_thld )
    {
        // Fill in "extracted_targets" from the "current_targets"
        extractTargets(extract_thld);

        position.clear();
        speed.clear();
        weight.clear();

        for (auto const & gaussian : m_extractedTargets->m_gaussians)
        {
            for (unsigned int j=0; j<DIM; ++j)
            {
                position.push_back(gaussian.m_mean(j,0));
                speed.push_back(gaussian.m_mean(DIM + j,0));
            }

            weight.push_back(gaussian.m_weight);
        }
    }

    /*!
     * \brief setDynamicsModel
     * Parameters to set before use
     * \param sampling
     * \param processNoise
     */
    void setDynamicsModel(float sampling, float processNoise)
    {

        m_samplingPeriod  = sampling;
        m_processNoise    = processNoise;

        // Fill in propagation matrix :
        m_tgtDynTrans = Eigen::MatrixXf::Identity(2 * DIM, 2 * DIM);

        for (unsigned int i = 0; i < DIM; ++i)
        {
            m_tgtDynTrans(i, DIM + i) = m_samplingPeriod;
        }

        // Fill in covariance matrix
        // Extra covariance added by the dynamics. Could be 0.
        m_tgtDynCov = processNoise * processNoise * Eigen::MatrixXf::Identity(2 * DIM, 2 * DIM);
    }

    /*!
     * \brief setDynamicsModel
     * \param tgt_dyn_transitions
     * \param tgt_dyn_covariance
     */
    void  setDynamicsModel( Eigen::MatrixXf const & tgt_dyn_transitions, Eigen::MatrixXf const & tgt_dyn_covariance)
    {
        m_tgtDynTrans = tgt_dyn_transitions;
        m_tgtDynCov = tgt_dyn_covariance;
    }

    /*!
     * \brief setSurvivalProbability
     * \param _prob_survival
     */
    void  setSurvivalProbability(float _prob_survival)
    {
        m_pSurvival = _prob_survival;
    }

    /*!
     * \brief setObservationModel
     * \param probDetectionOverall
     * \param m_measNoisePose
     * \param m_measNoiseSpeed
     * \param m_measNoiseBackground
     */
    void  setObservationModel(float probDetectionOverall, float measurement_noise_pose,
                              float measurement_noise_speed, float measurement_background )
    {
        m_pDetection      = probDetectionOverall;
        m_measNoisePose   = measurement_noise_pose;
        m_measNoiseSpeed  = measurement_noise_speed;
        m_measNoiseBackground   = measurement_background; // False detection probability

        // Set model matrices
        m_obsMat  = Eigen::MatrixXf::Identity(2 * DIM, 2 * DIM);
        m_obsMatT = m_obsMat.transpose();
        m_obsCov  = Eigen::MatrixXf::Identity(2 * DIM, 2 * DIM);

        // FIXME: deal with the _motion_model parameter !
        m_obsCov.block(0,0,DIM, DIM) *= m_measNoisePose * m_measNoisePose;
        m_obsCov.block(DIM,DIM,DIM, DIM) *= m_measNoiseSpeed * m_measNoiseSpeed;
    }

    /*!
     * \brief setPruningParameters
     * \param prune_trunc_thld
     * \param prune_merge_thld
     * \param prune_max_nb
     */
    void  setPruningParameters(float prune_trunc_thld, float prune_merge_thld,
                               int   prune_max_nb)
    {

        m_pruneTruncThld = prune_trunc_thld;
        m_pruneMergeThld = prune_merge_thld;
        m_nMaxPrune     = prune_max_nb;
    }

    /*!
     * \brief setBirthModel
     * \param m_birthModel
     */
    void  setBirthModel(std::vector<GModel> & birth_model)
    {
        m_birthModel.reset( new GMixture( birth_model) );
    }

    /*!
     * \brief setSpawnModel
     * \param spawnModels
     */
    void  setSpawnModel(std::vector<SModel> & spawnModels)
    {
        // Stupid implementation, maybe to be improved..
        for (auto const & model : spawnModels)
        {
            m_spawnModels.push_back( model);
        }
    }

    // Auxiliary functions
    /*!
     * \brief print
     */
    void  print() const
    {
        printf("Current gaussian mixture : \n");

        int i = 0;
        for (auto const & gauss : m_currTargets->m_gaussians )
        {
            if (check_val(gauss.m_mean(0, 0)))
            {
                printf("print filter Error!!!!");
            }

            printf("Gaussian %d - pos %.1f  %.1f %.1f - cov %.1f  %.1f %.1f - weight %.3f\n",
                   i++,
                   gauss.m_mean(0,0), gauss.m_mean(1,0), gauss.m_mean(2,0),
                   gauss.m_cov(0,0), gauss.m_cov(1,1), gauss.m_cov(2,2),
                   gauss.m_weight);
        }
        printf("\n");
    }

    /*!
     * \brief propagate
     */
    void  propagate()
    {
        m_nPredTargets = 0;

        // Predict new targets (spawns):
        predictBirth();

        // Predict propagation of expected targets :
        predictTargets();

        // Build the update components
        buildUpdate();

        if (m_bVerbose)
        {
            printf("\nGMPHD_propagate :--- Expected targets : %d ---\n", m_nPredTargets);
            m_expTargets->print();
        }

        // Update GMPHD
        update();

        if (m_bVerbose)
        {
            printf("\nGMPHD_propagate :--- \n");
            m_currTargets->print();
        }

        // Prune gaussians (remove weakest, merge close enough gaussians)
        pruneGaussians ();

        if (m_bVerbose)
        {
            printf("\nGMPHD_propagate :--- Pruned targets : ---\n");
            m_currTargets->print();
        }

        // Clean vectors :
        m_expMeasure.clear ();
        m_expDisp.clear ();
        m_uncertainty.clear ();
        m_covariance.clear ();
    }

    void  reset()
    {
        m_currTargets->m_gaussians.clear ();
        m_extractedTargets->m_gaussians.clear ();
    }

private:
    /*!
     * \brief The spawning models (how gaussians spawn from existing targets)
     * Example : how airplanes take off from a carrier..
     */
    std::vector <SModel, Eigen::aligned_allocator <SModel> > m_spawnModels;

    /*!
     * \brief buildUpdate
     */
    void buildUpdate()
    {
        Eigen::MatrixXf temp_matrix(2 * DIM, 2 * DIM);

        // Concatenate all the wannabe targets :
        // - birth targets
        m_iBirthTargets.clear();

        if(m_birthTargets->m_gaussians.size () > 0)
        {
            for (unsigned int i = 0; i < m_birthTargets->m_gaussians.size(); ++i)
            {
                m_iBirthTargets.push_back(m_expTargets->m_gaussians.size() + i);
            }

            m_expTargets->m_gaussians.insert(m_expTargets->m_gaussians.end(), m_birthTargets->m_gaussians.begin(),
                                             m_birthTargets->m_gaussians.begin() + m_birthTargets->m_gaussians.size());
        }

        // - spawned targets
        if (m_spawnTargets->m_gaussians.size () > 0)
        {
            m_expTargets->m_gaussians.insert(m_expTargets->m_gaussians.end(), m_spawnTargets->m_gaussians.begin(),
                                              m_spawnTargets->m_gaussians.begin() + m_spawnTargets->m_gaussians.size());
        }

        if (m_bVerbose)
        {
            printf("GMPHD : inserted %zu birth targets, now %zu expected\n", m_birthTargets->m_gaussians.size (), m_expTargets->m_gaussians.size());

            m_birthTargets->print();

            printf("GMPHD : inserted %zu spawned targets, now %zu expected\n", m_spawnTargets->m_gaussians.size (), m_expTargets->m_gaussians.size());

            m_spawnTargets->print();
        }

        // Compute PHD update components (for every expected target)
        m_nPredTargets = m_expTargets->m_gaussians.size ();

        m_expMeasure.clear();
        m_expMeasure.reserve (m_nPredTargets);

        m_expDisp.clear();
        m_expDisp.reserve (m_nPredTargets);

        m_uncertainty.clear();
        m_uncertainty.reserve (m_nPredTargets);

        m_covariance.clear();
        m_covariance.reserve (m_nPredTargets);

        for (auto const & tgt : m_expTargets->m_gaussians)
        {
            // Compute the expected measurement
            m_expMeasure.push_back( m_obsMat * tgt.m_mean );
            m_expDisp.push_back( m_obsCov + m_obsMat * tgt.m_cov * m_obsMatT );

            temp_matrix = m_expDisp.back().inverse();

            m_uncertainty.push_back( tgt.m_cov * m_obsMatT * temp_matrix );

            m_covariance.push_back( (Eigen::MatrixXf::Identity(2 * DIM, 2 * DIM) - m_uncertainty.back()*m_obsMat) * tgt.m_cov );
        }
    }

    /*!
     * \brief extractTargets
     * \param threshold
     */
    void  extractTargets(float threshold)
    {
        float const thld = std::max(threshold, 0.f);

        // Get trough every target, keep the ones whose weight is above threshold
        m_extractedTargets->m_gaussians.clear();

        for ( auto const & current_target : m_currTargets->m_gaussians)
        {
            if (current_target.m_weight >= thld)
            {
                m_extractedTargets->m_gaussians.push_back(current_target);
            }
        }
    }

    /*!
     * \brief predictBirth
     */
    void  predictBirth()
    {
        m_spawnTargets->m_gaussians.clear();
        m_birthTargets->m_gaussians.clear();

        // -----------------------------------------
        // Compute spontaneous births
        m_birthTargets->m_gaussians = m_birthModel->m_gaussians;
        m_nPredTargets += m_birthTargets->m_gaussians.size ();

        // -----------------------------------------
        // Compute spawned targets
        for( auto const & curr : m_currTargets->m_gaussians )
        {
            for( auto const & spawn : m_spawnModels )
            {
                GModel new_spawn;

                // Define a gaussian model from the existing target
                // and spawning properties
                new_spawn.m_weight = curr.m_weight * spawn.m_weight;

                new_spawn.m_mean = spawn.m_offset + spawn.m_trans * curr.m_mean;

                new_spawn.m_cov = spawn.m_cov + spawn.m_trans * curr.m_cov * spawn.m_trans.transpose();

                if (!check_val(new_spawn.m_mean(0, 0)))
                {
                    // Add this new gaussian to the list of expected targets
                    m_spawnTargets->m_gaussians.push_back(std::move(new_spawn));

                    // Update the number of expected targets
                    ++m_nPredTargets;
                }
            }
        }
    }

    /*!
     * \brief predictTargets
     */
    void  predictTargets()
    {
        GModel new_target;

        m_expTargets->m_gaussians.clear();
        m_expTargets->m_gaussians.reserve( m_currTargets->m_gaussians.size () );

        for (auto const & curr : m_currTargets->m_gaussians)
        {
            // Compute the new shape of the target
            new_target.m_weight = m_pSurvival * curr.m_weight;

            new_target.m_mean = m_tgtDynTrans * curr.m_mean;

            new_target.m_cov = m_tgtDynCov + m_tgtDynTrans * curr.m_cov * m_tgtDynTrans.transpose();

            // Push back to the expected targets
            if (!check_val(new_target.m_mean(0, 0)))
            {
                m_expTargets->m_gaussians.push_back(new_target);
                ++m_nPredTargets;
            }
        }
    }

    /*!
     * \brief pruneGaussians
     */
    void pruneGaussians()
    {
        m_currTargets->prune( m_pruneTruncThld, m_pruneMergeThld, m_nMaxPrune );
    }

    /*!
     * \brief update
     */
    void update()
    {
        m_currTargets->m_gaussians.clear();

        // We'll consider every possible association : vector size is (expected targets)*(measured targets)
        m_currTargets->m_gaussians.resize((m_measTargets->m_gaussians.size() + 1) * m_expTargets->m_gaussians.size());

        // First set of gaussians : mere propagation of existing ones
        // \warning : don't propagate the "birth" targets...
        // we set their weight to 0

        m_nPredTargets =  m_expTargets->m_gaussians.size ();
        int i_birth_current = 0;

        for (unsigned int i = 0; i < m_nPredTargets; ++i)
        {
            if (i != m_iBirthTargets[i_birth_current])
            {
                m_currTargets->m_gaussians[i].m_weight = (1.f - m_pDetection) * m_expTargets->m_gaussians[i].m_weight;
            }
            else
            {
                i_birth_current = std::min(i_birth_current + 1, (int)m_iBirthTargets.size());
                m_currTargets->m_gaussians[i].m_weight = 0.f;
            }

            m_currTargets->m_gaussians[i].m_mean = m_expTargets->m_gaussians[i].m_mean;
            m_currTargets->m_gaussians[i].m_cov  = m_expTargets->m_gaussians[i].m_cov;
        }

        // Second set of gaussians : match observations and previsions
        if (m_measTargets->m_gaussians.size () == 0)
        {
            return;
        }

        for (unsigned int n_meas = 1; n_meas <= m_measTargets->m_gaussians.size(); ++n_meas)
        {
            for (unsigned int n_targt = 0; n_targt < m_nPredTargets; ++n_targt)
            {
                unsigned int index = n_meas * m_nPredTargets + n_targt;

                if (index >= m_currTargets->m_gaussians.size())
                {
                    continue;
                }

                // Compute matching factor between predictions and measures.
                m_currTargets->m_gaussians[index].m_weight =  m_pDetection * m_expTargets->m_gaussians[n_targt].m_weight /
                        mahalanobis( m_measTargets->m_gaussians[n_meas -1].m_mean.block(0,0,DIM,1),
                        m_expMeasure[n_targt].block(0,0,DIM,1),
                        m_expDisp[n_targt].block(0,0, DIM, DIM));

                m_currTargets->m_gaussians[index].m_mean = m_expTargets->m_gaussians[n_targt].m_mean +
                        m_uncertainty[n_targt] * (m_measTargets->m_gaussians[n_meas -1].m_mean - m_expMeasure[n_targt]);

                m_currTargets->m_gaussians[index].m_cov = m_covariance[n_targt];

                if (check_val(m_currTargets->m_gaussians[index].m_mean(0, 0)))
                {
                    printf("update Error!!!!");
                    m_currTargets->m_gaussians.erase(m_currTargets->m_gaussians.begin() + index);
                }
            }

            // Normalize weights in the same predicted set,
            // taking clutter into account
            m_currTargets->normalize(m_measNoiseBackground, n_meas * m_nPredTargets, (n_meas + 1) * m_nPredTargets, 1);
        }
    }


private:
    bool  m_bVerbose;

    uint   m_nPredTargets;
    uint   m_nCurrentTargets;
    uint   m_nMaxPrune;

    float m_pSurvival;
    float m_pDetection;

    float m_samplingPeriod;
    float m_processNoise;

    float m_pruneMergeThld;
    float m_pruneTruncThld;

    float m_measNoisePose;
    float m_measNoiseSpeed;
    float m_measNoiseBackground; // Background detection "noise", other models are possible..

    std::vector<uint> m_iBirthTargets;

    Eigen::MatrixXf  m_tgtDynTrans;
    Eigen::MatrixXf  m_tgtDynCov;

    Eigen::MatrixXf  m_obsMat;
    Eigen::MatrixXf  m_obsMatT;
    Eigen::MatrixXf  m_obsCov;

    // Temporary matrices, used for the update process
    std::vector <Eigen::MatrixXf, Eigen::aligned_allocator <Eigen::MatrixXf> > m_covariance;
    std::vector <Eigen::MatrixXf, Eigen::aligned_allocator <Eigen::MatrixXf> > m_expMeasure;
    std::vector <Eigen::MatrixXf, Eigen::aligned_allocator <Eigen::MatrixXf> > m_expDisp;
    std::vector <Eigen::MatrixXf, Eigen::aligned_allocator <Eigen::MatrixXf> > m_uncertainty;

    std::unique_ptr<GMixture> m_birthModel;

    std::unique_ptr<GMixture> m_birthTargets;
    std::unique_ptr<GMixture> m_currTargets;
    std::unique_ptr<GMixture> m_expTargets;
    std::unique_ptr<GMixture> m_extractedTargets;
    std::unique_ptr<GMixture> m_measTargets;
    std::unique_ptr<GMixture> m_spawnTargets;

private:
    /*!
     * \brief mahalanobis
     * \param point
     * \param mean
     * \param cov
     * \return
     */
    float mahalanobis(const Eigen::Matrix <float, DIM, 1> &point,
                      const Eigen::Matrix <float, DIM, 1> &mean,
                      const Eigen::Matrix <float, DIM, DIM> &cov)
    {
        int ps = point.rows();
        Eigen::MatrixXf x_cen = point-mean;
        Eigen::MatrixXf b = Eigen::MatrixXf::Identity(ps,ps);

        // TODO: Ben - cov needs to be normalized !
        cov.ldlt().solveInPlace(b);
        x_cen = b*x_cen;
        Eigen::MatrixXf res = x_cen.transpose() * x_cen;
        return res.sum();
    }

    /*!
     * \brief gaussDensity
     * \param point
     * \param mean
     * \param cov
     * \return
     */
    float   gaussDensity(const Eigen::Matrix <float, DIM, 1> &point,
                         const Eigen::Matrix <float, DIM, 1> &mean,
                         const Eigen::Matrix <float, DIM, DIM> &cov) const
    {
        float det, res;

        Eigen::Matrix <float, DIM, DIM> cov_inverse;
        Eigen::Matrix <float, DIM, 1> mismatch;

        det = cov.determinant();
        cov_inverse = cov.inverse();

        mismatch = point - mean;

        Eigen::Matrix <float, 1, 1> distance = mismatch.transpose() * cov_inverse * mismatch;

        distance /= -2.f;

        // Deal with faulty determinant case
        if (det == 0.f)
        {
            return 0.f;
        }

        res = 1.f/sqrt(pow(2*M_PI, DIM) * fabs(det)) * exp(distance.coeff (0,0));

        if (std::isinf(det))
        {
            printf("Problem in multivariate gaussian\n distance : %f - det %f\n", distance.coeff (0,0), det);
            std::cout << "Cov \n" << cov << std::endl << "Cov inverse \n" << cov_inverse << std::endl;
            return 0.f;
        }

        return res;
    }
};

#endif // GMPHD_FILTER_H
