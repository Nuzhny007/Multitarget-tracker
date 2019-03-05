#ifndef _BEST_ASSIGNMENTS_H__
#define _BEST_ASSIGNMENTS_H__

#include "global.hpp"
#include <armadillo>

#define CHECK_FOR_INF

void assignmentoptimal(float *assignment, float *cost, float *distMatrix,
                       int nOfRows, int nOfColumns);
void buildassignmentvector(float *assignment, bool *starMatrix, int nOfRows,
                           int nOfColumns);
void computeassignmentcost(float *assignment, float *cost, float *distMatrix,
                           int nOfRows);
void step2a(float *assignment, float *distMatrix, bool *starMatrix,
            bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns,
            bool *coveredRows, int nOfRows, int nOfColumns, int minDim);
void step2b(float *assignment, float *distMatrix, bool *starMatrix,
            bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns,
            bool *coveredRows, int nOfRows, int nOfColumns, int minDim);
void step3(float *assignment, float *distMatrix, bool *starMatrix,
           bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns,
           bool *coveredRows, int nOfRows, int nOfColumns, int minDim);
void step4(float *assignment, float *distMatrix, bool *starMatrix,
           bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns,
           bool *coveredRows, int nOfRows, int nOfColumns, int minDim, int row,
           int col);
void step5(float *assignment, float *distMatrix, bool *starMatrix,
           bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns,
           bool *coveredRows, int nOfRows, int nOfColumns, int minDim);

void
murtyKBestAssignment(const arma::fmat &CM, unsigned m,
                     std::vector<std::pair<std::vector<int>, float>> &assigns);
void murtyKBestAssignmentWrapper(
    const arma::fmat &M, unsigned m,
    std::vector<std::pair<std::vector<int>, float>> &assigns);

#endif
