#include "gmphd_filter.h"


// Author : Benjamin Lefaudeux (blefaudeux@github)


GMPHD::GMPHD(int max_gaussians, int dimension, bool motion_model, bool verbose):
    m_motionModel(motion_model),
    m_bVerbose(verbose),
    m_maxGaussians(max_gaussians),
    m_dimMeasures(dimension)
{
    m_dimState = motion_model ? 2 * m_dimMeasures : m_dimMeasures;
    m_pruneTruncThld = 0.f;
    m_pDetection = 0.f;
    m_pSurvival = 0.f;

    // Initialize all gaussian mixtures, we know the dimension now
    m_measTargets.reset( new GaussianMixture(m_dimState) );
    m_birthTargets.reset( new GaussianMixture(m_dimState) );
    m_currTargets.reset( new GaussianMixture(m_dimState) );
    m_expTargets.reset( new GaussianMixture(m_dimState) );
    m_extractedTargets.reset( new GaussianMixture(m_dimState) );
    m_spawnTargets.reset( new GaussianMixture(m_dimState) );
}

void  GMPHD::buildUpdate ()
{
    Eigen::MatrixXf temp_matrix(m_dimState, m_dimState);

    // Concatenate all the wannabe targets :
    // - birth targets
    m_iBirthTargets.clear();

    if(m_birthTargets->m_gaussians.size () > 0)
    {
        for (unsigned int i=0; i<m_birthTargets->m_gaussians.size (); ++i)
        {
            m_iBirthTargets.push_back( m_expTargets->m_gaussians.size () + i );
        }

        m_expTargets->m_gaussians.insert(m_expTargets->m_gaussians.end (), m_birthTargets->m_gaussians.begin (),
                                         m_birthTargets->m_gaussians.begin () + m_birthTargets->m_gaussians.size ());
    }

    // - spawned targets
    if (m_spawnTargets->m_gaussians.size () > 0)
    {
        m_expTargets->m_gaussians.insert( m_expTargets->m_gaussians.end (), m_spawnTargets->m_gaussians.begin (),
                                          m_spawnTargets->m_gaussians.begin () + m_spawnTargets->m_gaussians.size ());
    }

    if (m_bVerbose)
    {
        printf("GMPHD : inserted %zu birth targets, now %zu expected\n",
               m_birthTargets->m_gaussians.size (), m_expTargets->m_gaussians.size());

        m_birthTargets->print ();

        printf("GMPHD : inserted %zu spawned targets, now %zu expected\n",
               m_spawnTargets->m_gaussians.size (), m_expTargets->m_gaussians.size());

        m_spawnTargets->print ();
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

        m_covariance.push_back( (Eigen::MatrixXf::Identity(m_dimState, m_dimState) - m_uncertainty.back()*m_obsMat)
                                * tgt.m_cov );
    }
}

bool GMPHD::isInitialized()
{
    if( m_tgtDynTrans.cols() != m_dimState)
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

void    GMPHD::extractTargets(float threshold)
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

void GMPHD::getTrackedTargets(std::vector<float> & position,
                              std::vector<float> & speed,
                              std::vector<float> & weight,
                              float const & extract_thld)
{
    // Fill in "extracted_targets" from the "current_targets"
    extractTargets(extract_thld);

    position.clear();
    speed.clear();
    weight.clear();

    for (auto const & gaussian : m_extractedTargets->m_gaussians)
    {
        for (unsigned int j=0; j<m_dimMeasures; ++j)
        {
            position.push_back(gaussian.m_mean(j,0));
            speed.push_back(gaussian.m_mean(m_dimMeasures + j,0));
        }

        weight.push_back(gaussian.m_weight);
    }
}

void  GMPHD::predictBirth()
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
            GaussianModel new_spawn(m_dimState);

            // Define a gaussian model from the existing target
            // and spawning properties
            new_spawn.m_weight = curr.m_weight * spawn.m_weight;

            new_spawn.m_mean = spawn.m_offset + spawn.m_trans * curr.m_mean;

            new_spawn.m_cov = spawn.m_cov + spawn.m_trans * curr.m_cov
                    * spawn.m_trans.transpose();

            // Add this new gaussian to the list of expected targets
            m_spawnTargets->m_gaussians.push_back ( std::move(new_spawn) );

            // Update the number of expected targets
            ++m_nPredTargets;
        }
    }
}

void  GMPHD::predictTargets () {
    GaussianModel new_target(m_dimState);

    m_expTargets->m_gaussians.clear();
    m_expTargets->m_gaussians.reserve( m_currTargets->m_gaussians.size () );

    for (auto const & curr : m_currTargets->m_gaussians)
    {
        // Compute the new shape of the target
        new_target.m_weight = m_pSurvival * curr.m_weight;

        new_target.m_mean = m_tgtDynTrans * curr.m_mean;

        new_target.m_cov = m_tgtDynCov + m_tgtDynTrans
                * curr.m_cov * m_tgtDynTrans.transpose();

        // Push back to the expected targets
        m_expTargets->m_gaussians.push_back( new_target );
        ++m_nPredTargets;
    }
}

void GMPHD::print() const
{
    printf("Current gaussian mixture : \n");

    int i = 0;
    for (auto const & gauss : m_currTargets->m_gaussians )
    {
        printf("Gaussian %d - pos %.1f  %.1f %.1f - cov %.1f  %.1f %.1f - weight %.3f\n",
               i++,
               gauss.m_mean(0,0), gauss.m_mean(1,0), gauss.m_mean(2,0),
               gauss.m_cov(0,0), gauss.m_cov(1,1), gauss.m_cov(2,2),
               gauss.m_weight);
    }
    printf("\n");
}

void  GMPHD::propagate ()
{
    m_nPredTargets = 0;

    // Predict new targets (spawns):
    predictBirth();

    // Predict propagation of expected targets :
    predictTargets();

    // Build the update components
    buildUpdate ();

    if( m_bVerbose )
    {
        printf("\nGMPHD_propagate :--- Expected targets : %d ---\n", m_nPredTargets);
        m_expTargets->print();
    }

    // Update GMPHD
    update();

    if (m_bVerbose)
    {
        printf("\nGMPHD_propagate :--- \n");
        m_currTargets->print ();
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

void  GMPHD::pruneGaussians()
{
    m_currTargets->prune( m_pruneTruncThld, m_pruneMergeThld, m_nMaxPrune );
}

void GMPHD::reset()
{
    m_currTargets->m_gaussians.clear ();
    m_extractedTargets->m_gaussians.clear ();
}


void  GMPHD::setBirthModel(std::vector<GaussianModel> &birth_model)
{
    m_birthModel.reset( new GaussianMixture( birth_model) );
}

void  GMPHD::setDynamicsModel(float sampling, float processNoise)
{

    m_samplingPeriod  = sampling;
    m_processNoise    = processNoise;

    // Fill in propagation matrix :
    m_tgtDynTrans = Eigen::MatrixXf::Identity(m_dimState, m_dimState);

    for (unsigned int i = 0; i<m_dimMeasures; ++i)
    {
        m_tgtDynTrans(i,m_dimMeasures+i) = m_samplingPeriod;
    }

    // Fill in covariance matrix
    // Extra covariance added by the dynamics. Could be 0.
    m_tgtDynCov = processNoise * processNoise *
            Eigen::MatrixXf::Identity(m_dimState, m_dimState);
}

void GMPHD::setDynamicsModel( Eigen::MatrixXf const & tgt_dyn_transitions,
                              Eigen::MatrixXf const & tgt_dyn_covariance)
{
    m_tgtDynTrans = tgt_dyn_transitions;
    m_tgtDynCov = tgt_dyn_covariance;
}

void  GMPHD::setNewMeasurements(std::vector<float> const & position,
                                std::vector<float> const & speed)
{
    // Clear the gaussian mixture
    m_measTargets->m_gaussians.clear();

    unsigned int iTarget = 0;

    while(iTarget < position.size()/m_dimMeasures)
    {
        GaussianModel new_obs(m_dimState);

        for (unsigned int i=0; i< m_dimMeasures; ++i) {
            // Create new gaussian model according to measurement
            new_obs.m_mean(i) = position[iTarget*m_dimMeasures + i];
            new_obs.m_mean(i+m_dimMeasures) = speed[iTarget*m_dimMeasures + i];
        }

        new_obs.m_cov = m_obsCov;
        new_obs.m_weight = 1.f;

        m_measTargets->m_gaussians.push_back(std::move(new_obs));

        iTarget++;
    }
}

void  GMPHD::setNewReferential(const Eigen::Matrix4f & transform)
{
    // Change referential for every gaussian in the gaussian mixture
    m_currTargets->changeReferential(transform);
}

void  GMPHD::setPruningParameters (float  prune_trunc_thld,
                                   float  prune_merge_thld,
                                   int    prune_max_nb)
{

    m_pruneTruncThld = prune_trunc_thld;
    m_pruneMergeThld = prune_merge_thld;
    m_nMaxPrune     = prune_max_nb;
}


void  GMPHD::setObservationModel(float probDetectionOverall,
                                 float measurement_noise_pose,
                                 float measurement_noise_speed,
                                 float measurement_background )
{
    m_pDetection      = probDetectionOverall;
    m_measNoisePose   = measurement_noise_pose;
    m_measNoiseSpeed  = measurement_noise_speed;
    m_measNoiseBackground   = measurement_background; // False detection probability

    // Set model matrices
    m_obsMat  = Eigen::MatrixXf::Identity(m_dimState, m_dimState);
    m_obsMatT = m_obsMat.transpose();
    m_obsCov  = Eigen::MatrixXf::Identity(m_dimState,m_dimState);

    // FIXME: deal with the _motion_model parameter !
    m_obsCov.block(0,0,m_dimMeasures, m_dimMeasures) *= m_measNoisePose * m_measNoisePose;
    m_obsCov.block(m_dimMeasures,m_dimMeasures,m_dimMeasures, m_dimMeasures) *= m_measNoiseSpeed * m_measNoiseSpeed;
}

void  GMPHD::setSpawnModel(std::vector <SpawningModel> & spawnModels)
{
    // Stupid implementation, maybe to be improved..
    for (auto const & model : spawnModels)
    {
        m_spawnModels.push_back( model);
    }
}

void  GMPHD::setSurvivalProbability(float _prob_survival)
{
    m_pSurvival = _prob_survival;
}

void  GMPHD::update()
{
    unsigned int n_meas, n_targt, index;
    m_currTargets->m_gaussians.clear();

    // We'll consider every possible association : vector size is (expected targets)*(measured targets)
    m_currTargets->m_gaussians.resize((m_measTargets->m_gaussians.size () + 1) *
                                      m_expTargets->m_gaussians.size ());

    // First set of gaussians : mere propagation of existing ones
    // \warning : don't propagate the "birth" targets...
    // we set their weight to 0

    m_nPredTargets =  m_expTargets->m_gaussians.size ();
    int i_birth_current = 0;

    for (unsigned int i=0; i<m_nPredTargets; ++i)
    {
        if (i != m_iBirthTargets[i_birth_current])
        {
            m_currTargets->m_gaussians[i].m_weight = (1.f - m_pDetection) *
                    m_expTargets->m_gaussians[i].m_weight;
        }
        else
        {
            i_birth_current = std::min(i_birth_current+1, (int) m_iBirthTargets.size ());
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

    for (n_meas=1; n_meas <= m_measTargets->m_gaussians.size (); ++n_meas)
    {
        for (n_targt = 0; n_targt < m_nPredTargets; ++n_targt)
        {
            index = n_meas * m_nPredTargets + n_targt;

            // Compute matching factor between predictions and measures.
            m_currTargets->m_gaussians[index].m_weight =  m_pDetection * m_expTargets->m_gaussians[n_targt].m_weight /
                    mahalanobis<2>( m_measTargets->m_gaussians[n_meas -1].m_mean.block(0,0,m_dimMeasures,1),
                    m_expMeasure[n_targt].block(0,0,m_dimMeasures,1),
                    m_expDisp[n_targt].block(0,0, m_dimMeasures, m_dimMeasures));


            m_currTargets->m_gaussians[index].m_mean =  m_expTargets->m_gaussians[n_targt].m_mean +
                    m_uncertainty[n_targt] * (m_measTargets->m_gaussians[n_meas -1].m_mean - m_expMeasure[n_targt]);

            m_currTargets->m_gaussians[index].m_cov = m_covariance[n_targt];
        }

        // Normalize weights in the same predicted set,
        // taking clutter into account
        m_currTargets->normalize (m_measNoiseBackground, n_meas * m_nPredTargets,
                                  (n_meas + 1) * m_nPredTargets, 1);
    }
}
