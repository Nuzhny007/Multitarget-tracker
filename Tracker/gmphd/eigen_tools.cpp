#include "eigen_tools.h"


float pseudo_inv(Eigen::MatrixXf const & mat_in,
                 Eigen::MatrixXf &mat_out) {
    int dim = 0;

    // Get matrices dimension :
    if (mat_in.cols() != mat_in.rows())
    {
        throw("Cannot compute matrix pseudo_inverse");
    }
    else
    {
        dim = mat_in.cols();
    }

    mat_out.resize (dim, dim);

    Eigen::MatrixXf U (dim,dim);
    Eigen::MatrixXf eig_val (dim, 1);
    Eigen::MatrixXf eig_val_inv (dim, dim);
    Eigen::MatrixXf V (dim, dim);

    float det;

    eig_val_inv = Eigen::MatrixXf::Identity(dim,dim);

    // Compute the SVD decomposition
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(mat_in, Eigen::ComputeFullU | Eigen::ComputeFullV);

    eig_val = svd.singularValues();
    U = svd.matrixU();
    V = svd.matrixV();

    // Compute pseudo-inverse
    // - quick'n'dirty inversion of eigen matrix
    for (int i = 0; i<dim; ++i) {
        if (eig_val(i,0) != 0.f)
            eig_val_inv(i,i) = 1.f / eig_val(i,0);
        else
            eig_val_inv(i,i) = 0.f;
    }

    mat_out = V.transpose() * eig_val_inv * U.transpose();

    // Compute determinant from eigenvalues..
    det = 1.f;
    for (int i=0; i<dim; ++i) {
        det *= eig_val(i,0);
    }

    return det;
}

