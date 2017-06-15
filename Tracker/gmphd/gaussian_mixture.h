#ifndef GAUSSIAN_MIXTURE_H
#define GAUSSIAN_MIXTURE_H

// Author : Benjamin Lefaudeux (blefaudeux@github)


#include "eigen_tools.h"
#include <list>
#include <algorithm>

/*!
 * \brief Stupid index structure to help sort gaussian mixtures
 */
struct index_w {
        float m_weight;
        int   m_index;
};

struct GaussianModel
{
        GaussianModel(int dim=4): //Ben - fixme, define this as a template argument !
            m_dim(dim)
        {
            clear();
        }

        GaussianModel & operator=(const GaussianModel & rhs)
        {
            if( this != &rhs )
            {
                m_mean = rhs.m_mean;
                m_cov = rhs.m_cov;
                m_dim = rhs.m_dim;
                m_weight = rhs.m_weight;
            }

            return *this;
        }

        void clear()
        {
            m_mean = Eigen::MatrixXf::Zero(m_dim,1);
            m_cov  = Eigen::MatrixXf::Identity( m_dim, m_dim);
            m_weight = 0.f;
        }

        int m_dim;
        float m_weight;

        Eigen::MatrixXf m_mean;
        Eigen::MatrixXf m_cov;
};

/*!
 * \brief The gaussian_mixture is a sum of gaussian models,
 *  with according weights. Everything is public, no need to get/set...
 */
class GaussianMixture {
    public :
        GaussianMixture(int dim);

        GaussianMixture( GaussianMixture const & source);

        GaussianMixture( std::vector<GaussianModel> const & source );

        GaussianMixture operator=(const GaussianMixture &source);

        GaussianModel mergeGaussians(std::vector<int> &i_gaussians_to_merge, bool b_remove_from_mixture);


        void  normalize(float linear_offset);
        void  normalize(float linear_offset, int start_pos, int stop_pos, int step);

        void print();

        void prune(float  trunc_threshold, float  merge_threshold, unsigned int max_gaussians);

        void sort();

        void selectCloseGaussians(int i_ref, float threshold, std::vector<int> & close_gaussians);

        int selectBestGaussian();

        void changeReferential(const Eigen::Matrix4f & transform);

    public:
        std::vector <GaussianModel> m_gaussians;
        int m_dim;

};

#endif // GAUSSIAN_MIXTURE_H
