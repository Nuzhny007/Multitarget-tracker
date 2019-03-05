#include "lmbTracker.hpp"
#include "shortestPaths.hpp"
#include "bestAssignments.hpp"

///
/// \brief lmbTracker::lmbTracker
///
lmbTracker::lmbTracker()
{
    initFilter();
}

///
/// \brief lmbTracker::initFilter
///
void lmbTracker::initFilter()
{
    current_time_step = 0;
    delta = 1;
    birth_r_max = 0.25;
    birth_V_max = 60;
    track_r_max = 0.7;
    track_r_min = 0.1;
    track_merge_U = 30;
    num_particles = 1000;
    birth_Hmax = 50;
    survive_Hmax = 200;
    update_Hmax = 200;
    pd = 0.95;
    ps = 0.98;
    lambda_c = 4;
    scan_volume = (768 * 576);

    arma::fmat A0 = {{1, delta}, {0, 1}};
    F = arma::join_vert(arma::join_horiz(A0, arma::zeros<arma::fmat>(2, 2)),
                        arma::join_horiz(arma::zeros<arma::fmat>(2, 2), A0));
    sigma_v = 10;
    H = {{1, 0, 0, 0}, {0, 0, 1, 0}};
    arma::fmat D = arma::diagmat(arma::fvec({10, 10}));
    R = D * D.t();

    P._lmb.clear();
    P._dglmb.clear();
    U._lmb.clear();
    X.clear();
    Zk.clear();
    zk_r.clear();
}

///
/// \brief lmbTracker::runFilter
/// \param _Zk
/// \return
///
std::vector<std::pair<lmbTracker::label_t, arma::fvec>> lmbTracker::runFilter(const std::vector<arma::fvec> &_Zk)
{
    current_time_step++;

    Zk_minus_1 = Zk;
    Zk = _Zk;
    zk_r.clear();
    zk_r.resize(Zk.size());

    Update();
    Predict();

    pruneUpdate();
    estimateState();

    return X;
}

///
/// \brief lmbTracker::computePs
/// \param x
/// \return
///
float lmbTracker::computePs(const arma::fvec &x)
{
    assert(x.size());
    return ps;
}

///
/// \brief lmbTracker::computePd
/// \param x
/// \return
///
float lmbTracker::computePd(const arma::fvec &x)
{
    assert(x.size());
    return pd;
}

///
/// \brief lmbTracker::computeLikelihood
/// \param z
/// \param x
/// \return
///
float lmbTracker::computeLikelihood(const arma::fvec &z, const arma::fvec &x)
{
    arma::fvec numer = -0.5 * (z - H * x).t() * R.i() * (z - H * x);
    return (exp(numer(0)) / sqrt(pow(2 * M_PI, z.size()) * arma::det(R)));
}

///
/// \brief lmbTracker::clutterDensity
/// \return
///
float lmbTracker::clutterDensity()
{
    return (lambda_c / scan_volume);
}

///
/// \brief lmbTracker::esfMahler
/// \param x
/// \return
///
arma::fvec lmbTracker::esfMahler(const arma::fvec &x)
{
    if (!x.size()) {
        return arma::fvec({1});
    }

    arma::fmat F = arma::zeros<arma::fmat>(2, x.size());
    size_t idx = 0;
    for (int n = 1; n <= (int)x.size(); n++) {
        F(idx, 0) = F(!idx, 0) + x(n - 1);
        for (int k = 2; k <= n; k++) {
            if (k <= n) {
                if (k == n) {
                    F(idx, k - 1) = x(n - 1) * F(!idx, k - 2);
                } else {
                    F(idx, k - 1) = F(!idx, k - 1) + x(n - 1) * F(!idx, k - 2);
                }
            }
        }
        idx = !idx;
    }

    return (arma::join_vert(arma::fvec({1}), arma::trans(F.row(!idx))));
}

///
/// \brief lmbTracker::dglmbUpdate
///
void lmbTracker::dglmbUpdate()
{
    arma::fvec P_avpd = arma::zeros<arma::fvec>(P._lmb.size());
    for (size_t l = 0; l < P._lmb.size(); l++) {
        float avpd_sum = 0;
        for (size_t n = 0; n < num_particles; n++) {
            avpd_sum += computePd(P._lmb[l].x[n]) * P._lmb[l].w[n];
        }
        P_avpd(l) = avpd_sum;
    }

    arma::fmat allcostm = arma::zeros<arma::fmat>(P._lmb.size(), Zk.size());
    std::vector<lmb> update_lmb_tmp = P._lmb;
    for (size_t z = 0; z < Zk.size(); z++) {
        for (size_t l = 0; l < P._lmb.size(); l++) {
            lmb tmp;
            tmp.r = 0;
            tmp.l = P._lmb[l].l;
            tmp.x.resize(num_particles);
            tmp.w.resize(num_particles);
            for (size_t n = 0; n < num_particles; n++) {
                tmp.x[n] = P._lmb[l].x[n];
                tmp.w[n] = computePd(tmp.x[n]) * P._lmb[l].w[n] *
                        computeLikelihood(Zk[z], tmp.x[n]);
                allcostm(l, z) += tmp.w[n];
            }

            // weight normalization
            for (size_t n = 0; n < num_particles; n++) {
                if (allcostm(l, z) == 0) {
                    tmp.w[n] = 0;
                } else {
                    tmp.w[n] /= allcostm(l, z);
                }
            }
            update_lmb_tmp.emplace_back(tmp);
        }
    }

    std::vector<dglmb> update_dglmb_tmp;
    if (!Zk.size()) {
        for (auto &l : P._dglmb) {
            dglmb tmp;
            float log_sum = 0;
            for (auto &i : l.I) {
                log_sum += log(1 - P_avpd(i));
            }
            tmp.w = -lambda_c * log_sum * log(l.w);
            tmp.I = l.I;
            update_dglmb_tmp.emplace_back(tmp);
        }
    } else {
        float P_dglmb_sqrt_w_sum = 0;
        for (auto &l : P._dglmb) {
            P_dglmb_sqrt_w_sum += sqrt(l.w);
        }

        for (auto &l : P._dglmb) {
            if (!l.I.size()) {
                dglmb tmp;
                tmp.w =
                        -lambda_c + Zk.size() * log(lambda_c * clutterDensity()) + log(l.w);
                update_dglmb_tmp.emplace_back(tmp);
            } else {
                arma::fmat CM(l.I.size(), Zk.size());
                for (size_t i = 0; i < l.I.size(); i++) {
                    for (size_t j = 0; j < Zk.size(); j++) {
                        CM(i, j) = -log(allcostm(l.I[i], j) / (lambda_c * clutterDensity() *
                                                               (1 - P_avpd(l.I[i]))));
                    }
                }

                std::vector<std::pair<std::vector<int>, float>> assigns;
                unsigned m = round(update_Hmax * sqrt(l.w) / P_dglmb_sqrt_w_sum);
                murtyKBestAssignmentWrapper(CM, m, assigns);

                for (size_t i = 0; i < assigns.size(); i++) {
                    dglmb tmp;
                    float log_sum = 0;
                    for (size_t ii = 0; ii < l.I.size(); ii++) {
                        tmp.I.emplace_back(l.I[ii] +
                                           P._lmb.size() * (assigns[i].first[ii] + 1));
                        log_sum += log(1 - P_avpd(l.I[ii]));
                    }
                    tmp.w = -lambda_c + Zk.size() * log(lambda_c * clutterDensity()) +
                            log_sum + log(l.w) - assigns[i].second;
                    update_dglmb_tmp.emplace_back(tmp);

                    for (auto &ii : assigns[i].first) {
                        if (ii >= 0) {
                            zk_r[ii] += exp(tmp.w);
                        }
                    }
                }
            }
        }
    }

    // normalization of weights/probabilities
    // using high precision doubles here to avoid missing out on very small data
    double sumexp_w = 0;
    for (auto &d : update_dglmb_tmp) {
        sumexp_w += exp((double)d.w);
    }

    for (auto &d : update_dglmb_tmp) {
        d.w = exp(d.w - (float)log(sumexp_w));
    }
    for (auto &r : zk_r) {
        r = (float)((double)r / sumexp_w);
    }

    dglmb2lmb(&update_lmb_tmp, &update_dglmb_tmp);

    // particle resampling!!!
    for (auto &l : U._lmb) {
        std::vector<float> cum_w(l.w.size());
        cum_w[0] = l.w[0];
        for (size_t n = 1; n < l.w.size(); n++) {
            cum_w[n] = cum_w[n - 1] + l.w[n];
        }
        std::vector<size_t> rsp_idx;
        for (unsigned n = 0; n < num_particles; n++) {
            float r = (float)rand() / (float)RAND_MAX;
            size_t j;
            for (j = 0; j < cum_w.size(); j++) {
                if (r < cum_w[j]) {
                    rsp_idx.emplace_back(j);
                    break;
                }
            }
            // TODO: due to floating point issues, we may end up here!!
            //       HOW TO DO PROPER NORMALIZATION WITH SMALL PRECISION PARTICLES
            //       IN CASE OF FLOATING POINT HW
            if (j == cum_w.size()) {
                rsp_idx.emplace_back(j - 1);
            }
        }
        assert(rsp_idx.size() == num_particles);

        std::vector<arma::fvec> tmp_x = l.x;
        l.x.clear();
        l.x.resize(num_particles);
        l.w.clear();
        l.w.resize(num_particles);
        for (unsigned n = 0; n < num_particles; n++) {
            l.w[n] = 1 / (float)num_particles;
            l.x[n] = tmp_x[rsp_idx[n]];
        }
    }
}

///
/// \brief lmbTracker::dglmb2lmb
/// \param lmb_tmp
/// \param dglmb_tmp
///
void lmbTracker::dglmb2lmb(std::vector<lmb> *lmb_tmp,
                           std::vector<dglmb> *dglmb_tmp)
{
    std::map<label_t, size_t> unique_labels_map;
    for (auto &l : *lmb_tmp) {
        if (unique_labels_map.find(l.l) == unique_labels_map.end()) {
            unique_labels_map[l.l] = U._lmb.size();
            lmb tmp;
            tmp.l = l.l;
            U._lmb.emplace_back(tmp);
        }
    }

    for (auto &d : *dglmb_tmp) {
        for (auto &i : d.I) {
            for (size_t n = 0; n < num_particles; n++) {
                U._lmb[unique_labels_map[(*lmb_tmp)[i].l]].x.emplace_back(
                            (*lmb_tmp)[i].x[n]);
                U._lmb[unique_labels_map[(*lmb_tmp)[i].l]].w.emplace_back(
                            d.w * (*lmb_tmp)[i].w[n]);
            }
        }
    }

    // weight normalization
    for (auto &l : U._lmb) {
        l.r = 0;
        for (auto &v : l.w) {
            l.r += v;
        }
        for (auto &v : l.w) {
            v /= l.r;
        }

        // TODO: figure out why l.r >= 1..shouldnt happen!!
        //       maybe b/c of float precision issues!
        if (l.r >= 1) {
            l.r = (1 - EPS);
        }
    }

    // removing components that have not been assigned by any hypothesis
    U._lmb.erase(std::remove_if(U._lmb.begin(), U._lmb.end(),
                                [](const lmb &c) { return (c.r == 0); }),
                 U._lmb.end());
}

///
/// \brief lmbTracker::Update
///
void lmbTracker::Update()
{
    U._lmb.clear();
    dglmbUpdate();
}

///
/// \brief lmbTracker::Predict
///
void lmbTracker::Predict()
{
    P._lmb.clear();
    P._dglmb.clear();

    std::vector<lmb> P_birth_lmb;
    unsigned target_cnt = 0;
    for (size_t r = 0; r < zk_r.size(); r++) {
        if ((1 - zk_r[r]) > track_r_min) {
            float min_speed = INF;
            size_t min_speed_idx = 0;
            for (size_t z = 0; z < Zk_minus_1.size(); z++) {
                float this_speed =
                        arma::norm(arma::fvec({(Zk[r](0) - Zk_minus_1[z](0)) / delta,
                                               (Zk[r](1) - Zk_minus_1[z](1)) / delta}));
                if (this_speed < min_speed) {
                    min_speed = this_speed;
                    min_speed_idx = z;
                }
            }

            // TODO: could try to filter targets having ~0 min_speed as this usually
            //      mean clutter
            //          if (min_speed < 0.1) {
            //
            //          } else
            if (min_speed < birth_V_max) {
                lmb tmp;
                tmp.r = (1 - zk_r[r]);
                if (tmp.r > birth_r_max) {
                    tmp.r = birth_r_max;
                }
                tmp.l.first = current_time_step;
                tmp.l.second = target_cnt++;
                float x_speed = (Zk[r](0) - Zk_minus_1[min_speed_idx](0)) / delta;
                        float y_speed = (Zk[r](1) - Zk_minus_1[min_speed_idx](1)) / delta;
                        arma::fvec m = arma::fvec({Zk[r](0) + x_speed * delta, x_speed,
                                                   Zk[r](1) + y_speed * delta, y_speed});
                tmp.x.resize(num_particles);
                tmp.w.resize(num_particles);
                for (size_t n = 0; n < num_particles; n++) {
                    tmp.w[n] = 1 / (float)(num_particles);
                    tmp.x[n] = m + sigma_v * arma::randn<arma::fvec>(X_DIM);
                }
                P_birth_lmb.emplace_back(tmp);
            }
        }
    }
    P._lmb = P_birth_lmb;

    std::vector<lmb> P_survive_lmb;
    for (auto &l : U._lmb) {
        if (l.r >= track_r_min) {
            lmb tmp;
            tmp.l = l.l;
            tmp.x.resize(num_particles);
            tmp.w.resize(num_particles);
            float sum_w = 0;
            for (size_t n = 0; n < num_particles; n++) {
                tmp.x[n] = F * l.x[n] + sigma_v * arma::randn<arma::fvec>(X_DIM);
                tmp.w[n] = computePs(l.x[n]) * l.w[n];
                sum_w += tmp.w[n];
            }
            tmp.r = sum_w * l.r;
            for (size_t n = 0; n < num_particles; n++) {
                tmp.w[n] /= sum_w;
            }
            P_survive_lmb.emplace_back(tmp);
        }
    }
    P._lmb.insert(P._lmb.end(), P_survive_lmb.begin(), P_survive_lmb.end());

    std::vector<dglmb> P_birth_dglmb;
    lmb2dglmb(&P_birth_dglmb, &P_birth_lmb, birth_Hmax);
    std::vector<dglmb> P_survive_dglmb;
    lmb2dglmb(&P_survive_dglmb, &P_survive_lmb, survive_Hmax);

    float sum_w = 0;
    for (auto &b : P_birth_dglmb) {
        for (auto &s : P_survive_dglmb) {
            dglmb tmp;
            tmp.w = b.w * s.w;
            sum_w += tmp.w;
            tmp.I = b.I;
            for (auto &i : s.I) {
                tmp.I.emplace_back(i + P_birth_lmb.size());
            }
            P._dglmb.emplace_back(tmp);
        }
    }
    // weight normalization
    for (auto &d : P._dglmb) {
        d.w /= sum_w;
    }
}

///
/// \brief lmbTracker::lmb2dglmb
/// \param _dglmb
/// \param _lmb
/// \param k
///
void lmbTracker::lmb2dglmb(std::vector<dglmb> *_dglmb, std::vector<lmb> *_lmb,
                           unsigned k)
{
    assert(k > 0);
    assert(_dglmb->size() == 0);

    std::vector<float> costv;
    for (auto &l : *_lmb) {
        costv.emplace_back(-log(l.r / (1 - l.r)));
    }
    std::vector<size_t> costv_sorted_idx(costv.size());
    iota(costv_sorted_idx.begin(), costv_sorted_idx.end(), 0);
    std::sort(costv_sorted_idx.begin(), costv_sorted_idx.end(),
              [&costv](size_t i1, size_t i2) { return (costv[i1] > costv[i2]); });
    std::vector<Edge> E;
    for (unsigned i = 0; i < costv.size() + 1; i++) {
        for (unsigned j = i + 1; j < costv.size() + 1; j++) {
            E.emplace_back(i, j, costv[costv_sorted_idx[j - 1]]);
        }
        E.emplace_back(i, costv.size() + 1, EPS);
    }
    std::vector<std::pair<std::vector<uint>, float>> shortest_paths;
    // YenKShortestPaths(E, costv.size() + 2, 0, costv.size() + 1, k,
    //                   shortest_paths);
    EppsteinKShortestPaths(E, costv.size() + 2, 0, costv.size() + 1, k,
                           shortest_paths);
    for (auto &sp : shortest_paths) {
        dglmb tmp;
        tmp.w = sp.second;
        for (auto &i : sp.first) {
            if (i != 0 && i != (costv.size() + 1)) {
                tmp.I.emplace_back(costv_sorted_idx[i - 1]);
            }
        }
        _dglmb->emplace_back(tmp);
    }

    // weight normalization
    float sum_w = 0;
    for (auto &l : *_lmb) {
        sum_w += log(1 - l.r);
    }
    float sum_exp_w = 0;
    for (auto &d : *_dglmb) {
        d.w = sum_w - d.w;
        sum_exp_w += exp(d.w);
    }
    for (auto &d : *_dglmb) {
        d.w = exp(d.w - log(sum_exp_w));
    }
}

///
/// \brief lmbTracker::pruneUpdate
///
void lmbTracker::pruneUpdate()
{
    // merging close-by targets
    std::vector<lmb> lmb_new;
    float track_r_max_local = track_r_max;
    std::sort(U._lmb.begin(), U._lmb.end(),
              [](const lmb &lhs, const lmb &rhs) { return (lhs.r > rhs.r); });
    for (int i = 0; i < (int)(U._lmb.size() - 1); i++) {
        if (U._lmb[i].r < 0) {
            continue;
        }
        arma::fvec this_x = arma::zeros<arma::fvec>(X_DIM);
        for (size_t n = 0; n < num_particles; n++) {
            this_x += U._lmb[i].w[n] * U._lmb[i].x[n];
        }
        for (size_t j = i + 1; j < U._lmb.size(); j++) {
            if (U._lmb[j].r >= 0) {
                arma::fvec that_x = arma::zeros<arma::fvec>(X_DIM);
                for (size_t n = 0; n < num_particles; n++) {
                    that_x += U._lmb[j].w[n] * U._lmb[j].x[n];
                }
                float dist = arma::norm(arma::fvec({this_x(0), this_x(2)}) -
                                        arma::fvec({that_x(0), that_x(2)}));
                if (dist < track_merge_U) {
                    lmb tmp;
                    tmp.r = U._lmb[i].r + U._lmb[j].r;
                    if (U._lmb[i].l.first < U._lmb[j].l.first) {
                        tmp.l = U._lmb[i].l;
                    } else {
                        tmp.l = U._lmb[j].l;
                    }
                    tmp.x.resize(num_particles);
                    tmp.w.resize(num_particles);
                    float sum_w = 0;
                    for (size_t n = 0; n < num_particles; n++) {
                        tmp.w[n] = (U._lmb[i].r / tmp.r) * U._lmb[i].w[n] +
                                (U._lmb[j].r / tmp.r) * U._lmb[j].w[n];
                        sum_w += tmp.w[n];
                        tmp.x[n] = (U._lmb[i].r / tmp.r) * U._lmb[i].x[n] +
                                (U._lmb[j].r / tmp.r) * U._lmb[j].x[n];
                    }
                    for (size_t n = 0; n < num_particles; n++) {
                        tmp.w[n] /= sum_w;
                    }
                    if (tmp.r >= 1) {
                        tmp.r = 1 - EPS;
                    }
                    lmb_new.emplace_back(tmp);

                    U._lmb[i].r = -1;
                    U._lmb[j].r = -1;
                    // TODO: right now merging is done only upto 2 components so as to
                    // save computational costs...see if this is enough or else
                    // merging of all possible components in *_tentative is needed???
                    break;
                }
            }
        }
    }
    U._lmb.erase(std::remove_if(U._lmb.begin(), U._lmb.end(),
                                [track_r_max_local](const lmb &l) {
        return (l.r == -1 || l.r < track_r_max_local);
    }),
                 U._lmb.end());

    // ii) TODO: cap on max number of tracks

    U._lmb.insert(U._lmb.end(), lmb_new.begin(), lmb_new.end());
}

///
/// \brief lmbTracker::estimateState
///
void lmbTracker::estimateState()
{
    X.clear();
    arma::fvec r_over_1minusr(U._lmb.size());
    float prod_r = 1;
    for (size_t i = 0; i < U._lmb.size(); i++) {
        r_over_1minusr(i) = U._lmb[i].r / (1 - U._lmb[i].r);
        prod_r *= (1 - U._lmb[i].r);
    }
    arma::fvec cdn = prod_r * esfMahler(r_over_1minusr);

    unsigned num_targets = 0;
    float max_val = 0;
    for (size_t i = 0; i < cdn.size(); i++) {
        if (max_val < cdn[i]) {
            max_val = cdn[i];
            num_targets = i;
        }
    }
    std::sort(U._lmb.begin(), U._lmb.end(),
              [](const lmb &lhs, const lmb &rhs) { return (lhs.r > rhs.r); });
    for (size_t i = 0; i < num_targets; i++) {
        std::pair<label_t, arma::fvec> x;
        x.first = U._lmb[i].l;
        x.second = arma::zeros<arma::fvec>(X_DIM);
        for (size_t n = 0; n < num_particles; n++) {
            x.second += U._lmb[i].x[n] * U._lmb[i].w[n];
        }
        X.emplace_back(x);
    }
}
