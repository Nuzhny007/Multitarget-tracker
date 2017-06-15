#include "gaussian_mixture.h"
#include <algorithm>
// Author : Benjamin Lefaudeux (blefaudeux@github)

GaussianMixture::GaussianMixture( int dim)
{
    m_dim = dim;
    m_gaussians.clear ();
}

GaussianMixture::GaussianMixture( std::vector<GaussianModel> const & source )
{
    m_gaussians = source;
    m_dim = source[0].m_dim;
}

GaussianMixture::GaussianMixture( GaussianMixture const & source)
{
    m_gaussians = source.m_gaussians;
}

GaussianMixture GaussianMixture::operator = ( GaussianMixture const &source)
{
    // Skip assignment if same object
    if (this == &source)
        return *this;

    // Else, use vectors & Eigen "=" operator
    m_gaussians = source.m_gaussians;
    return *this;
}

void  GaussianMixture::sort () {
    std::sort(m_gaussians.begin(), m_gaussians.end(), [](GaussianModel const & lhs, GaussianModel const & rhs)
    {
        return lhs.m_weight > rhs.m_weight;
    });
}


void  GaussianMixture::normalize (float linear_offset)
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

void  GaussianMixture::normalize (float linear_offset,
                                  int start_pos,
                                  int stop_pos,
                                  int step)
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

void GaussianMixture::print()
{
    if (m_gaussians.size () > 0)
    {
        printf("Gaussian mixture : \n");

        int i= 0;
        for ( auto const & gaussian : m_gaussians)
        {
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
        }
        printf("\n");
    }
}

void GaussianMixture::changeReferential( Eigen::Matrix4f const & transform)
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


GaussianModel  GaussianMixture::mergeGaussians (std::vector<int> &i_gaussians_to_merge, bool b_remove_from_mixture)
{
    // TODO: Ben - rewrite this crap, could be half way long

    GaussianModel merged_model( m_gaussians[0].m_dim );

    Eigen::MatrixXf diff(m_dim, 1);

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
        merged_model.m_cov.setZero(m_dim, m_dim);
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
        std::vector<GaussianModel>::iterator it = m_gaussians.begin ();

        for (int i=i_gaussians_to_merge.size () -1; i>-1; ++i) {
            m_gaussians.erase ( it + i);
        }
    }

    return merged_model;
}

void  GaussianMixture::prune(float  trunc_threshold, float  merge_threshold, unsigned int max_gaussians)
{
    // Sort the gaussians mixture, ascending order
    sort ();

    int index, i_best;

    std::vector<int> i_close_to_best;
    std::vector<GaussianModel>::iterator position;

    GaussianMixture pruned_targets(m_dim);
    GaussianModel merged_gaussian(m_dim);

    merged_gaussian.clear();
    pruned_targets.m_gaussians.clear();

    while ( !m_gaussians.empty()
            && pruned_targets.m_gaussians.size () < max_gaussians )
    {
        // - Pick the bigger gaussian (based on weight)
        i_best = selectBestGaussian ();

        if ( i_best == -1 || m_gaussians[i_best].m_weight < trunc_threshold)
        {
            break;
        }
        else
        {
            // - Select all the gaussians close enough, to merge if needed
            i_close_to_best.clear();
            selectCloseGaussians (i_best, merge_threshold, i_close_to_best);

            // - Build a new merged gaussian
            i_close_to_best.push_back (i_best); // Add the initial gaussian

            if (i_close_to_best.size() > 1)
            {
                merged_gaussian = mergeGaussians (i_close_to_best, false);
            }
            else
            {
                merged_gaussian = m_gaussians[i_close_to_best[0]];
            }

            // - Append merged gaussian to the pruned_targets gaussian mixture
            pruned_targets.m_gaussians.push_back (merged_gaussian);

            // - Remove all the merged gaussians from current_targets :
            // -- Sort the indexes
            std::sort(i_close_to_best.begin(), i_close_to_best.end());

            // -- Remove from the last one (to keep previous indexes unchanged)
            while (!i_close_to_best.empty())
            {
                index = i_close_to_best.back();
                i_close_to_best.pop_back();

                position = m_gaussians.erase(m_gaussians.begin() + index);
            }
        }
    }

    m_gaussians = pruned_targets.m_gaussians;
}


int   GaussianMixture::selectBestGaussian () {
    // TODO: Ben - move this to a lambda and std::for_each



    float best_weight = 0.f;
    int   best_index = -1;
    int i= 0;

    std::for_each(m_gaussians.begin(), m_gaussians.end(), [&](GaussianModel const & gaussian)
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

void  GaussianMixture::selectCloseGaussians (int    i_ref, float  threshold,
                                             std::vector<int> &close_gaussians) {

    close_gaussians.clear ();

    float gauss_distance;

    Eigen::Matrix<float, 3,1> diff_vec;
    Eigen::Matrix<float, 3,3> cov_inverse;

    // We only take positions into account there
    int i= 0;
    for (auto const & gaussian : m_gaussians)
    {
        if (i != i_ref)
        {
            // Compute distance
            diff_vec = m_gaussians[i_ref].m_mean.block(0,0,3,1) -
                       gaussian.m_mean.block(0,0,3,1);

            cov_inverse = (m_gaussians[i_ref].m_cov.block(0,0,3,3)).inverse();

            gauss_distance = diff_vec.transpose() *
                             cov_inverse.block(0,0,3,3) *
                             diff_vec;

            // Add to the set of close gaussians, if below threshold
            if ((gauss_distance < threshold) && (gaussian.m_weight != 0.f))
            {
                close_gaussians.push_back (i);
            }
        }
        ++i;
    }
}
