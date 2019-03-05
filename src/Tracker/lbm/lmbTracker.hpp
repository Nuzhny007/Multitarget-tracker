#include <opencv2/opencv.hpp>
#include "global.hpp"
#include <armadillo>

///
/// \brief The lmbTracker class
///
class lmbTracker
{
public:
    typedef std::pair<unsigned, unsigned> label_t;

    lmbTracker();
    std::vector<std::pair<lmbTracker::label_t, arma::fvec>> runFilter(const std::vector<arma::fvec> &_Zk);

private:
    // sub-structures/classes

    ///
    /// \brief The lmb struct
    ///
    struct lmb
    {
        label_t l;
        float r;
        std::vector<arma::fvec> x;
        std::vector<float> w;

        void print()
        {
            std::cout << "r: " << r << std::endl;
            std::cout << "l:<" << l.first << "," << l.second << ">" << std::endl;
            std::cout << "particles:" << std::endl << "------------" << std::endl;
            arma::fvec tmp = arma::zeros<arma::fvec>(X_DIM);
            for (size_t n = 0; n < x.size(); n++) {
                tmp += w[n] * x[n];
            }
            tmp.print("x");
        }
    };
    ///
    /// \brief The dglmb struct
    ///
    struct dglmb
    {
        float w;
        std::vector<unsigned> I;

        void print()
        {
            std::cout << "w: " << w << "; I:(";
            for (auto &i : I) {
                std::cout << i << ", ";
            }
            std::cout << ")" << std::endl;
        }
    };

    ///
    /// \brief The lmbDensity struct
    ///
    struct lmbDensity
    {
        std::vector<lmb> _lmb;
        std::vector<dglmb> _dglmb;
    };

    // member variables
    std::vector<arma::fvec> Zk, Zk_minus_1;
    std::vector<std::pair<label_t, arma::fvec>> X;
    std::vector<float> zk_r;
    lmbDensity P, U;

    unsigned current_time_step;
    float delta;
    float birth_r_max;
    float birth_V_max;
    float track_r_max;
    float track_r_min;
    float track_merge_U;
    unsigned num_particles;
    unsigned birth_Hmax, survive_Hmax, update_Hmax;

    float pd, ps;
    float lambda_c;
    float scan_volume;

    arma::fmat F, H, R;
    float sigma_v;

    // member functions
    void initFilter();

    void lmb2dglmb(std::vector<dglmb> *dglmb, std::vector<lmb> *lmb, unsigned k);
    void Predict();

    void dglmbUpdate();
    void dglmb2lmb(std::vector<lmb> *lmb_tmp, std::vector<dglmb> *dglmb_tmp);
    void pruneUpdate();
    void Update();

    void estimateState();

    float computePs(const arma::fvec &x);
    float computePd(const arma::fvec &x);
    float computeLikelihood(const arma::fvec &z, const arma::fvec &x);
    float clutterDensity();
    arma::fvec esfMahler(const arma::fvec &z);
};
