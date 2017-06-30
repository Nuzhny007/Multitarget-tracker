#ifndef GAUSSIAN_MIXTURE_H
#define GAUSSIAN_MIXTURE_H

// Author : Benjamin Lefaudeux (blefaudeux@github)


#include "eigen_tools.h"
#include <list>
#include <algorithm>

/*!
 * \brief check_val
 * \param val
 * \return
 */
template<typename T>
bool check_val(T val)
{
    bool n = std::isnan(val) || std::isnan(-val);
    bool i = std::isinf(val);
    bool f = !std::isfinite(val);
    bool neg = val < 0;
    bool big = val > 10000;

    bool res = n || i || f || neg || big;

    if (res)
    {
        //std::cout << "Val " << val << " error!!!" << std::endl;
        //assert(0);
    }

    return res;
}

/*!
 * \brief Stupid index structure to help sort gaussian mixtures
 */
struct index_w
{
    float m_weight;
    int   m_index;
};

/*!
 * \brief The GaussianModel struct
 */
template<int DIM>
struct GaussianModel
{
    /*!
     * \brief GaussianModel
     */
    GaussianModel()
    {
        clear();
    }

    /*!
     * \brief operator =
     * \param rhs
     * \return
     */
    GaussianModel & operator=(const GaussianModel & rhs)
    {
        if( this != &rhs )
        {
            m_mean = rhs.m_mean;
            m_cov = rhs.m_cov;
            m_weight = rhs.m_weight;
        }

        return *this;
    }

    /*!
     * \brief clear
     */
    void clear()
    {
        m_mean = Eigen::MatrixXf::Zero(DIM, 1);
        m_cov  = Eigen::MatrixXf::Identity(DIM, DIM);
        m_weight = 0.f;
    }

    float m_weight;

    Eigen::MatrixXf m_mean;
    Eigen::MatrixXf m_cov;
};

/*!
 * \brief The gaussian_mixture is a sum of gaussian models,
 *  with according weights. Everything is public, no need to get/set...
 */
template<int DIM>
class GaussianMixture
{
public:
    typedef GaussianModel<DIM> GModel;
    typedef GaussianMixture<DIM> GMixture;

    /*!
     * \brief GaussianMixture
     */
    GaussianMixture()
    {

    }

    /*!
     * \brief GaussianMixture
     * \param source
     */
    GaussianMixture(GMixture const & source)
    {
        m_gaussians = source.m_gaussians;
    }

    /*!
     * \brief GaussianMixture
     * \param source
     */
    GaussianMixture( std::vector<GModel> const & source )
    {
        m_gaussians = source;
    }

    /*!
     * \brief operator =
     * \param source
     * \return
     */
    GaussianMixture operator=(const GMixture &source)
    {
        // Skip assignment if same object
        if (this == &source)
            return *this;

        // Else, use vectors & Eigen "=" operator
        m_gaussians = source.m_gaussians;
        return *this;
    }

    /*!
     * \brief mergeGaussians
     * \param i_gaussians_to_merge
     * \param b_remove_from_mixture
     * \return
     */
    GModel mergeGaussians(std::vector<int> &i_gaussians_to_merge, bool b_remove_from_mixture)
    {
        // TODO: Ben - rewrite this crap, could be half way long

        GModel merged_model;

        Eigen::MatrixXf diff(DIM, 1);

        if (i_gaussians_to_merge.size() > 1)
        {
            // Reset the destination
            merged_model.clear ();

            // Build merged gaussian :
            // - weight is the sum of all weights
            for ( auto const & i_g : i_gaussians_to_merge)
            {
                merged_model.m_weight += m_gaussians[i_g].m_weight;
            }

            // - gaussian center is the weighted m_mean of all centers
            for (auto const & i_g : i_gaussians_to_merge)
            {
                merged_model.m_mean += m_gaussians[i_g].m_mean * m_gaussians[i_g].m_weight;
            }

            if (merged_model.m_weight != 0.f) {
                merged_model.m_mean /= merged_model.m_weight;
            }

            // - covariance is related to initial gaussian model cov and the discrepancy
            // from merged m_mean position and every merged gaussian pose
            merged_model.m_cov.setZero(DIM, DIM);
            for (auto const & i_g : i_gaussians_to_merge)
            {
                diff = merged_model.m_mean - m_gaussians[i_g].m_mean;

                merged_model.m_cov += m_gaussians[i_g].m_weight * (m_gaussians[i_g].m_cov + diff * diff.transpose());
            }

            if (merged_model.m_weight != 0.f)
            {
                merged_model.m_cov /= merged_model.m_weight;
            }
        }
        else
        {
            // Just return the initial single gaussian model :
            merged_model = m_gaussians[i_gaussians_to_merge[0]];
        }

        if (b_remove_from_mixture)
        {
            // Remove input gaussians from the mixture
            // - sort the index vector
            std::sort(i_gaussians_to_merge.begin (),
                      i_gaussians_to_merge.end ());

            // - pop out the corresponding gaussians, in reverse
            auto it = m_gaussians.begin ();

            for (int i = i_gaussians_to_merge.size () -1; i > -1; ++i)
            {
                m_gaussians.erase(it + i);
            }
        }

        return merged_model;
    }

    /*!
     * \brief normalize
     * \param linear_offset
     */
    void  normalize(float linear_offset)
    {
        float sum = 0.f;

        for ( auto const & gaussian : m_gaussians)
        {
            sum += gaussian.m_weight;
        }

        if ((linear_offset + sum) != 0.f)
        {
            for (auto & gaussian : m_gaussians)
            {
                gaussian.m_weight /= (linear_offset + sum);
            }
        }
    }

    /*!
     * \brief normalize
     * \param linear_offset
     * \param start_pos
     * \param stop_pos
     * \param step
     */
    void  normalize(float linear_offset, int start_pos, int stop_pos, int step)
    {
        float sum = 0.f;

        for (int i = start_pos; i< stop_pos; ++i)
        {
            sum += m_gaussians[i * step].m_weight;
        }

        if ((linear_offset + sum) != 0.f)
        {
            for (int i = start_pos; i< stop_pos; ++i)
            {
                m_gaussians[i * step].m_weight /= (linear_offset + sum);
            }
        }
    }

    /*!
     * \brief print
     */
    void print()
    {
        if (m_gaussians.size () > 0)
        {
            printf("Gaussian mixture : \n");

            int i = 0;
            for ( auto const & gaussian : m_gaussians)
            {
                if (check_val(gaussian.m_mean(0, 0)))
                {
                    printf("print gm Error!!!!");
                }
#if 0
                printf("%2d - pos %3.1f | %3.1f | %3.1f - cov %3.1f | %3.1f | %3.1f - spd %3.2f | %3.2f | %3.2f - weight %3.3f\n",
                       i++,
                       gaussian.m_mean(0,0),
                       gaussian.m_mean(1,0),
                       gaussian.m_mean(2,0),
                       gaussian.m_cov(0,0),
                       gaussian.m_cov(1,1),
                       gaussian.m_cov(2,2),
                       gaussian.m_mean(3,0),
                       gaussian.m_mean(4,0),
                       gaussian.m_mean(5,0),
                       gaussian.m_weight) ;
#else
                std::cout << i++ << " - pos " <<
                        gaussian.m_mean(0,0) << " | " <<
                        gaussian.m_mean(1,0) << " | " <<
                        gaussian.m_mean(2,0) << " | " <<
                        gaussian.m_cov(0,0) << " | " <<
                        gaussian.m_cov(1,1) << " | " <<
                        gaussian.m_cov(2,2) << " | " <<
                        gaussian.m_mean(3,0) << " | " <<
                        gaussian.m_mean(4,0) << " | " <<
                        gaussian.m_mean(5,0) << " | " <<
                        gaussian.m_weight << std::endl;

#endif
            }
            printf("\n");
        }
    }

    /*!
     * \brief prune
     * \param trunc_threshold
     * \param merge_threshold
     * \param max_gaussians
     */
    void prune(float trunc_threshold, float merge_threshold, unsigned int max_gaussians)
    {
        // Sort the gaussians mixture, ascending order
        sort();

        std::vector<int> i_close_to_best;

        GMixture pruned_targets;
        GModel merged_gaussian;

        merged_gaussian.clear();
        pruned_targets.m_gaussians.reserve(max_gaussians);

        while ( !m_gaussians.empty()
                && pruned_targets.m_gaussians.size () < max_gaussians )
        {
            // - Pick the bigger gaussian (based on weight)
            int i_best = selectBestGaussian();

            if ( i_best == -1 || m_gaussians[i_best].m_weight < trunc_threshold)
            {
                break;
            }
            else
            {
                // - Select all the gaussians close enough, to merge if needed
                i_close_to_best.clear();
                selectCloseGaussians(i_best, merge_threshold, i_close_to_best);

                // - Build a new merged gaussian
                i_close_to_best.push_back(i_best); // Add the initial gaussian

                if (i_close_to_best.size() > 1)
                {
                    merged_gaussian = mergeGaussians(i_close_to_best, false);
                }
                else
                {
                    merged_gaussian = m_gaussians[i_close_to_best[0]];
                }

                // - Append merged gaussian to the pruned_targets gaussian mixture
                pruned_targets.m_gaussians.push_back(merged_gaussian);

                // - Remove all the merged gaussians from current_targets :
                // -- Sort the indexes
                std::sort(i_close_to_best.begin(), i_close_to_best.end());

                // -- Remove from the last one (to keep previous indexes unchanged)
                while (!i_close_to_best.empty())
                {
                    int index = i_close_to_best.back();
                    i_close_to_best.pop_back();

                    m_gaussians.erase(m_gaussians.begin() + index);
                }
            }
        }

        m_gaussians = pruned_targets.m_gaussians;
    }

    /*!
     * \brief sort
     */
    void sort()
    {
        std::sort(m_gaussians.begin(), m_gaussians.end(), [](GModel const & lhs, GModel const & rhs)
        {
            return lhs.m_weight > rhs.m_weight;
        });
    }

    /*!
     * \brief selectCloseGaussians
     * \param i_ref
     * \param threshold
     * \param close_gaussians
     */
    void selectCloseGaussians(int i_ref, float threshold, std::vector<int> & close_gaussians)
    {
        close_gaussians.clear();

        Eigen::Matrix<float, 3, 1> diff_vec;
        Eigen::Matrix<float, 3, 3> cov_inverse;

        // We only take positions into account there
        int i= 0;
        for (auto const & gaussian : m_gaussians)
        {
            if (i != i_ref)
            {
                // Compute distance
                diff_vec = m_gaussians[i_ref].m_mean.block(0,0,3,1) - gaussian.m_mean.block(0,0,3,1);

                cov_inverse = (m_gaussians[i_ref].m_cov.block(0,0,3,3)).inverse();

                float gauss_distance = diff_vec.transpose() * cov_inverse.block(0,0,3,3) * diff_vec;

                // Add to the set of close gaussians, if below threshold
                if ((gauss_distance < threshold) && (gaussian.m_weight != 0.f))
                {
                    close_gaussians.push_back(i);
                }
            }
            ++i;
        }
    }

    /*!
     * \brief selectBestGaussian
     * \return
     */
    int selectBestGaussian()
    {
        // TODO: Ben - move this to a lambda and std::for_each

        float best_weight = 0.f;
        int   best_index = -1;
        int i = 0;

        std::for_each(m_gaussians.begin(), m_gaussians.end(), [&](GModel const & gaussian)
        {
            if( gaussian.m_weight > best_weight )
            {
                best_weight = gaussian.m_weight;
                best_index = i;
            }
            ++i;
        });

        return best_index;
    }

    /*!
     * \brief changeReferential
     * \param transform
     */
    void changeReferential(const Eigen::Matrix4f & transform)
    {
        Eigen::Matrix<float, 4,1> temp_vec;
        Eigen::Matrix<float, 4,1> temp_vec_new;

        temp_vec(3,0) = 1.f;

        // Gaussian model :
        // - [x, y, z, dx/dt, dy/dt, dz/dt] m_mean values
        // - 6x6 covariance

        // For every gaussian model, change referential
        for ( auto & gaussian : m_gaussians)
        {
            // Change positions
            temp_vec.block(0,0, 3,1) = gaussian.m_mean.block(0,0,3,1);

            temp_vec_new = transform * temp_vec;

            gaussian.m_mean.block(0,0,3,1) = temp_vec_new.block(0,0,3,1);

            // Change speeds referential
            temp_vec.block(0,0, 3,1) = gaussian.m_mean.block(3,0,3,1);

            temp_vec_new = transform * temp_vec;

            gaussian.m_mean.block(3,0,3,1) = temp_vec_new.block(0,0,3,1);

            // Change covariance referential
            //  (only take the rotation into account)
            // TODO
        }
    }

public:
    std::vector <GModel> m_gaussians;
};

#endif // GAUSSIAN_MIXTURE_H