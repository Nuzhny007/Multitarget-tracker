#ifndef EIGEN_TOOLS_H
#define EIGEN_TOOLS_H

// Author : Benjamin Lefaudeux (blefaudeux@github)

#include <math.h>
#include <vector>
#include <stdio.h>
#include <iostream>
#include <Eigen/StdVector>
#include <Eigen/Eigen>

float pseudo_inv(Eigen::MatrixXf const &mat_in,
                 Eigen::MatrixXf & mat_out);

template <size_t size>
float pseudo_inv(Eigen::Matrix <float, size,size> const & mat_in,
                 Eigen::Matrix <float, size,size> & mat_out)
{
    Eigen::Matrix <float, size,size> U;
    Eigen::Matrix <float, size,1> eig_val;
    Eigen::Matrix <float, size,size> eig_val_inv;
    Eigen::Matrix <float, size,size> V;
    float det;

    eig_val_inv = Eigen::MatrixXf::Identity(size,size);

    // Compute the SVD decomposition
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(*mat_in,
                                          Eigen::ComputeThinU | Eigen::ComputeThinV);

    eig_val = svd.singularValues();
    U = svd.matrixU();
    V = svd.matrixV();

    // Compute pseudo-inverse
    // - quick'n'dirty inversion of eigen matrix
    for (int i = 0; i<size; ++i) {
        if (eig_val(i,0) != 0.f)
            eig_val_inv(i,i) = 1.f / eig_val(i,0);
        else
            eig_val_inv(i,i) = 0.f;
    }

    *mat_out = V.transpose() * eig_val_inv * U.transpose();

    // Compute determinant from eigenvalues..
    det = 1.f;
    for (int i=0; i<size; ++i) {
        det *= eig_val(i,0);
    }

    return det;
}

#endif // EIGEN_TOOLS_H
