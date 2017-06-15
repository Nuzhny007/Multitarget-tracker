#ifndef GMPHD_FILTER_H
#define GMPHD_FILTER_H

// Author : Benjamin Lefaudeux (blefaudeux@github)


#include "gaussian_mixture.h"
#include <iostream>
#include <memory>

/*!
 * \brief The spawning_model struct
 */
struct SpawningModel {

  SpawningModel(int dim = 2):
    m_dim(dim)
  {
    m_state = m_dim * 2;
    m_trans = Eigen::MatrixXf::Ones(m_state, m_state);
    m_cov = Eigen::MatrixXf::Ones(m_state, m_state);
    m_offset = Eigen::MatrixXf::Zero(m_state,1);
    m_weight = 0.1f;
  }

  int m_dim;
  int m_state;

  float m_weight;

  Eigen::MatrixXf m_trans;
  Eigen::MatrixXf m_cov;
  Eigen::MatrixXf m_offset;
};

/*!
 * \brief The gmphd_filter class
 */
class GMPHD
{
public:
  GMPHD(int max_gaussians, int dimension, bool motion_model = false, bool verbose = false);

  bool isInitialized();

  // Input: raw measurements and possible ref change
  void  setNewReferential( Eigen::Matrix4f const & transform);

  void  setNewMeasurements( std::vector<float> const & position, std::vector<float> const & speed);

  // Output
  void  getTrackedTargets( std::vector<float> & position, std::vector<float> & speed, std::vector<float> & weight,
                           float const & extract_thld );

  // Parameters to set before use
  void  setDynamicsModel( float sampling, float processNoise );

  void  setDynamicsModel( Eigen::MatrixXf const & tgt_dyn_transitions, Eigen::MatrixXf const & tgt_dyn_covariance);

  void  setSurvivalProbability(float _prob_survival);

  void  setObservationModel(float probDetectionOverall, float m_measNoisePose,
                            float m_measNoiseSpeed, float m_measNoiseBackground );

  void  setPruningParameters(float  prune_trunc_thld, float  prune_merge_thld,
                             int    prune_max_nb);

  void  setBirthModel(std::vector<GaussianModel> & m_birthModel);

  void  setSpawnModel(std::vector<SpawningModel> & spawnModels);

  // Auxiliary functions
  void  print() const;

  void  propagate();

  void  reset();

private:
  /*!
     * \brief The spawning models (how gaussians spawn from existing targets)
     * Example : how airplanes take off from a carrier..
     */
  std::vector <SpawningModel, Eigen::aligned_allocator <SpawningModel> > m_spawnModels;

  void  buildUpdate();

  void  extractTargets(float threshold);

  void  predictBirth();

  void  predictTargets();

  void  pruneGaussians();

  void  update();


private:
  bool  m_motionModel;
  bool  m_bVerbose;

  uint   m_maxGaussians;
  uint   m_dimMeasures;
  uint   m_dimState;
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

  std::unique_ptr<GaussianMixture> m_birthModel;

  std::unique_ptr<GaussianMixture> m_birthTargets;
  std::unique_ptr<GaussianMixture> m_currTargets;
  std::unique_ptr<GaussianMixture> m_expTargets;
  std::unique_ptr<GaussianMixture> m_extractedTargets;
  std::unique_ptr<GaussianMixture> m_measTargets;
  std::unique_ptr<GaussianMixture> m_spawnTargets;

private:

  template <size_t D>
  float mahalanobis(const Eigen::Matrix <float, D,1> &point,
                    const Eigen::Matrix <float, D,1> &mean,
                    const Eigen::Matrix <float, D,D> &cov)
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

  template <size_t D>
  float   gaussDensity(const Eigen::Matrix <float, D,1> &point,
                       const Eigen::Matrix <float, D,1> &mean,
                       const Eigen::Matrix <float, D,D> &cov) const
  {
    float det, res;

    Eigen::Matrix <float, D, D> cov_inverse;
    Eigen::Matrix <float, D, 1> mismatch;

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

    res = 1.f/sqrt(pow(2*M_PI, D) * fabs(det)) * exp(distance.coeff (0,0));

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
