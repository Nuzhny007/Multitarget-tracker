#include "bestAssignments.hpp"

void assignmentoptimal(float *assignment, float *cost, float *distMatrixIn,
                       int nOfRows, int nOfColumns) {
  float *distMatrix, *distMatrixTemp, *distMatrixEnd, *columnEnd, value,
      minValue;
  bool *coveredColumns, *coveredRows, *starMatrix, *newStarMatrix, *primeMatrix;
  int nOfElements, minDim, row, col;
#ifdef CHECK_FOR_INF
  bool infiniteValueFound;
  float maxFiniteValue, infValue;
#endif

  /* initialization */
  *cost = 0;
  for (row = 0; row < nOfRows; row++)
#ifdef ONE_INDEXING
    assignment[row] = 0.0;
#else
    assignment[row] = -1.0;
#endif

  /* generate working copy of distance Matrix */
  /* check if all matrix elements are positive */
  nOfElements = nOfRows * nOfColumns;
  distMatrix = (float *)malloc(nOfElements * sizeof(float));
  distMatrixEnd = distMatrix + nOfElements;
  for (row = 0; row < nOfElements; row++) {
    value = distMatrixIn[row];
    if (value != INF && (value < 0)) {
      std::cout << "All matrix elements have to be non-negative.";
      assert(0);
    }
    distMatrix[row] = value;
  }

#ifdef CHECK_FOR_INF
  /* check for infinite values */
  maxFiniteValue = -1;
  infiniteValueFound = false;

  distMatrixTemp = distMatrix;
  while (distMatrixTemp < distMatrixEnd) {
    value = *distMatrixTemp++;
    if (value != INF) {
      if (value > maxFiniteValue)
        maxFiniteValue = value;
    } else
      infiniteValueFound = true;
  }
  if (infiniteValueFound) {
    if (maxFiniteValue == -1) /* all elements are infinite */
      return;

    /* set all infinite elements to big finite value */
    if (maxFiniteValue > 0)
      infValue = 10 * maxFiniteValue * nOfElements;
    else
      infValue = 10;
    distMatrixTemp = distMatrix;
    while (distMatrixTemp < distMatrixEnd)
      //      if(mxIsInf(*distMatrixTemp++))
      if (*distMatrixTemp++ == INF)
        *(distMatrixTemp - 1) = infValue;
  }
#endif

  /* memory allocation */
  coveredColumns = (bool *)calloc(nOfColumns, sizeof(bool));
  coveredRows = (bool *)calloc(nOfRows, sizeof(bool));
  starMatrix = (bool *)calloc(nOfElements, sizeof(bool));
  primeMatrix = (bool *)calloc(nOfElements, sizeof(bool));
  newStarMatrix = (bool *)calloc(nOfElements, sizeof(bool)); /* used in step4 */

  /* preliminary steps */
  if (nOfRows <= nOfColumns) {
    minDim = nOfRows;

    for (row = 0; row < nOfRows; row++) {
      /* find the smallest element in the row */
      distMatrixTemp = distMatrix + row;
      minValue = *distMatrixTemp;
      distMatrixTemp += nOfRows;
      while (distMatrixTemp < distMatrixEnd) {
        value = *distMatrixTemp;
        if (value < minValue)
          minValue = value;
        distMatrixTemp += nOfRows;
      }

      /* subtract the smallest element from each element of the row */
      distMatrixTemp = distMatrix + row;
      while (distMatrixTemp < distMatrixEnd) {
        *distMatrixTemp -= minValue;
        distMatrixTemp += nOfRows;
      }
    }

    /* Steps 1 and 2a */
    for (row = 0; row < nOfRows; row++)
      for (col = 0; col < nOfColumns; col++)
        if (distMatrix[row + nOfRows * col] == 0)
          if (!coveredColumns[col]) {
            starMatrix[row + nOfRows * col] = true;
            coveredColumns[col] = true;
            break;
          }
  } else /* if(nOfRows > nOfColumns) */
  {
    minDim = nOfColumns;

    for (col = 0; col < nOfColumns; col++) {
      /* find the smallest element in the column */
      distMatrixTemp = distMatrix + nOfRows * col;
      columnEnd = distMatrixTemp + nOfRows;

      minValue = *distMatrixTemp++;
      while (distMatrixTemp < columnEnd) {
        value = *distMatrixTemp++;
        if (value < minValue)
          minValue = value;
      }

      /* subtract the smallest element from each element of the column */
      distMatrixTemp = distMatrix + nOfRows * col;
      while (distMatrixTemp < columnEnd)
        *distMatrixTemp++ -= minValue;
    }

    /* Steps 1 and 2a */
    for (col = 0; col < nOfColumns; col++)
      for (row = 0; row < nOfRows; row++)
        if (distMatrix[row + nOfRows * col] == 0)
          if (!coveredRows[row]) {
            starMatrix[row + nOfRows * col] = true;
            coveredColumns[col] = true;
            coveredRows[row] = true;
            break;
          }
    for (row = 0; row < nOfRows; row++)
      coveredRows[row] = false;
  }

  /* move to step 2b */
  step2b(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix,
         coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);

  /* compute cost and remove invalid assignments */
  computeassignmentcost(assignment, cost, distMatrixIn, nOfRows);

  /* free allocated memory */
  free(distMatrix);
  free(coveredColumns);
  free(coveredRows);
  free(starMatrix);
  free(primeMatrix);
  free(newStarMatrix);

  return;
}

void buildassignmentvector(float *assignment, bool *starMatrix, int nOfRows,
                           int nOfColumns) {
  int row, col;

  for (row = 0; row < nOfRows; row++)
    for (col = 0; col < nOfColumns; col++)
      if (starMatrix[row + nOfRows * col]) {
#ifdef ONE_INDEXING
        assignment[row] = col + 1; /* MATLAB-Indexing */
#else
        assignment[row] = col;
#endif
        break;
      }
}

void computeassignmentcost(float *assignment, float *cost, float *distMatrix,
                           int nOfRows) {
  int row, col;
#ifdef CHECK_FOR_INF
  float value;
#endif

  for (row = 0; row < nOfRows; row++) {
#ifdef ONE_INDEXING
    col = assignment[row] - 1; /* MATLAB-Indexing */
#else
    col = assignment[row];
#endif

    if (col >= 0) {
#ifdef CHECK_FOR_INF
      value = distMatrix[row + nOfRows * col];
      if (value != INF)
        *cost += value;
      else
#ifdef ONE_INDEXING
        assignment[row] = 0.0;
#else
        assignment[row] = -1.0;
#endif

#else
      *cost += distMatrix[row + nOfRows * col];
#endif
    }
  }
}

void step2a(float *assignment, float *distMatrix, bool *starMatrix,
            bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns,
            bool *coveredRows, int nOfRows, int nOfColumns, int minDim) {
  bool *starMatrixTemp, *columnEnd;
  int col;

  /* cover every column containing a starred zero */
  for (col = 0; col < nOfColumns; col++) {
    starMatrixTemp = starMatrix + nOfRows * col;
    columnEnd = starMatrixTemp + nOfRows;
    while (starMatrixTemp < columnEnd) {
      if (*starMatrixTemp++) {
        coveredColumns[col] = true;
        break;
      }
    }
  }

  /* move to step 3 */
  step2b(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix,
         coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

void step2b(float *assignment, float *distMatrix, bool *starMatrix,
            bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns,
            bool *coveredRows, int nOfRows, int nOfColumns, int minDim) {
  int col, nOfCoveredColumns;

  /* count covered columns */
  nOfCoveredColumns = 0;
  for (col = 0; col < nOfColumns; col++)
    if (coveredColumns[col])
      nOfCoveredColumns++;

  if (nOfCoveredColumns == minDim) {
    /* algorithm finished */
    buildassignmentvector(assignment, starMatrix, nOfRows, nOfColumns);
  } else {
    /* move to step 3 */
    step3(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix,
          coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
  }
}

void step3(float *assignment, float *distMatrix, bool *starMatrix,
           bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns,
           bool *coveredRows, int nOfRows, int nOfColumns, int minDim) {
  bool zerosFound;
  int row, col, starCol;

  zerosFound = true;
  while (zerosFound) {
    zerosFound = false;
    for (col = 0; col < nOfColumns; col++)
      if (!coveredColumns[col])
        for (row = 0; row < nOfRows; row++)
          if ((!coveredRows[row]) && (distMatrix[row + nOfRows * col] == 0)) {
            /* prime zero */
            primeMatrix[row + nOfRows * col] = true;

            /* find starred zero in current row */
            for (starCol = 0; starCol < nOfColumns; starCol++)
              if (starMatrix[row + nOfRows * starCol])
                break;

            if (starCol == nOfColumns) /* no starred zero found */
            {
              /* move to step 4 */
              step4(assignment, distMatrix, starMatrix, newStarMatrix,
                    primeMatrix, coveredColumns, coveredRows, nOfRows,
                    nOfColumns, minDim, row, col);
              return;
            } else {
              coveredRows[row] = true;
              coveredColumns[starCol] = false;
              zerosFound = true;
              break;
            }
          }
  }

  /* move to step 5 */
  step5(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix,
        coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

void step4(float *assignment, float *distMatrix, bool *starMatrix,
           bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns,
           bool *coveredRows, int nOfRows, int nOfColumns, int minDim, int row,
           int col) {
  int n, starRow, starCol, primeRow, primeCol;
  int nOfElements = nOfRows * nOfColumns;

  /* generate temporary copy of starMatrix */
  for (n = 0; n < nOfElements; n++)
    newStarMatrix[n] = starMatrix[n];

  /* star current zero */
  newStarMatrix[row + nOfRows * col] = true;

  /* find starred zero in current column */
  starCol = col;
  for (starRow = 0; starRow < nOfRows; starRow++)
    if (starMatrix[starRow + nOfRows * starCol])
      break;

  while (starRow < nOfRows) {
    /* unstar the starred zero */
    newStarMatrix[starRow + nOfRows * starCol] = false;

    /* find primed zero in current row */
    primeRow = starRow;
    for (primeCol = 0; primeCol < nOfColumns; primeCol++)
      if (primeMatrix[primeRow + nOfRows * primeCol])
        break;

    /* star the primed zero */
    newStarMatrix[primeRow + nOfRows * primeCol] = true;

    /* find starred zero in current column */
    starCol = primeCol;
    for (starRow = 0; starRow < nOfRows; starRow++)
      if (starMatrix[starRow + nOfRows * starCol])
        break;
  }

  /* use temporary copy as new starMatrix */
  /* delete all primes, uncover all rows */
  for (n = 0; n < nOfElements; n++) {
    primeMatrix[n] = false;
    starMatrix[n] = newStarMatrix[n];
  }
  for (n = 0; n < nOfRows; n++)
    coveredRows[n] = false;

  /* move to step 2a */
  step2a(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix,
         coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

void step5(float *assignment, float *distMatrix, bool *starMatrix,
           bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns,
           bool *coveredRows, int nOfRows, int nOfColumns, int minDim) {
  float h, value;
  int row, col;

  /* find smallest uncovered element h */
  h = INF;
  for (row = 0; row < nOfRows; row++)
    if (!coveredRows[row])
      for (col = 0; col < nOfColumns; col++)
        if (!coveredColumns[col]) {
          value = distMatrix[row + nOfRows * col];
          if (value < h)
            h = value;
        }

  /* add h to each covered row */
  for (row = 0; row < nOfRows; row++)
    if (coveredRows[row])
      for (col = 0; col < nOfColumns; col++)
        distMatrix[row + nOfRows * col] += h;

  /* subtract h from each uncovered column */
  for (col = 0; col < nOfColumns; col++)
    if (!coveredColumns[col])
      for (row = 0; row < nOfRows; row++)
        distMatrix[row + nOfRows * col] -= h;

  /* move to step 3 */
  step3(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix,
        coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

void
murtyKBestAssignment(const arma::fmat &CM, unsigned m,
                     std::vector<std::pair<std::vector<int>, float>> &assigns) {
  // first get the optimal assignment for this passed CM
  std::pair<std::vector<int>, float> the_optimal_assignment;
  std::vector<float> assignment(CM.n_rows);
  float cost;
  assignmentoptimal(&assignment[0], &cost, (float *)CM.memptr(), CM.n_rows, CM.n_cols);
  for (auto &a : assignment) {
    the_optimal_assignment.first.emplace_back(a);
  }
  the_optimal_assignment.second = cost;

  std::vector<std::pair<arma::fmat, std::pair<std::vector<int>, float>>>
      potential_assignments;
  potential_assignments.emplace_back(
      std::make_pair(CM, the_optimal_assignment));

  while (potential_assignments.size()) {
    // finding the optimal assignment from all the potential assignments so far
    std::pair<arma::fmat, std::pair<std::vector<int>, float>>
        current_optimal_assign;
    float min_cost = INF;
    size_t min_sol_idx = 0;
    for (size_t i = 0; i < potential_assignments.size(); i++) {
      if (min_cost > potential_assignments[i].second.second) {
        min_cost = potential_assignments[i].second.second;
        min_sol_idx = i;
      }
    }
    current_optimal_assign = potential_assignments[min_sol_idx];
    assigns.emplace_back(std::make_pair(current_optimal_assign.second.first,
                                        current_optimal_assign.second.second));
    // if all requested assignments found then quit
    if (assigns.size() == m) {
      break;
    }

    // else keep on looking or new solutions

    // removing current optimal solution from potential_assignments
    potential_assignments.erase(potential_assignments.begin() + min_sol_idx);
    // finding new potential solutions using current optimal assignment
    for (size_t i = 0; i < current_optimal_assign.second.first.size(); i++) {
      the_optimal_assignment.first.clear();
      the_optimal_assignment.second = 0;
      arma::fmat CM_tmp = current_optimal_assign.first;

      int j = current_optimal_assign.second.first[i];
      if (j >= 0) {
        if (j < (int)(CM_tmp.n_cols - CM_tmp.n_rows)) {
          CM_tmp(i, j) = INF;
        } else {
          for (size_t l = (CM_tmp.n_cols - CM.n_rows); l < CM_tmp.n_cols; l++) {
            CM_tmp(i, l) = INF;
          }
        }
        // finding optimal solution for this new CM
        std::vector<float> assignment(CM_tmp.n_rows);
        float cost;
        assignmentoptimal(&assignment[0], &cost, CM_tmp.memptr(), CM_tmp.n_rows, CM_tmp.n_cols);
        for (auto &a : assignment) {
          the_optimal_assignment.first.emplace_back(a);
        }
        the_optimal_assignment.second = cost;

        bool S_ok = true;
        // if there is a feasible assignment (all non-negative entries) in
        // the optimal_assignment hypothesis only then consider it as a
        // potential assignment
        for (auto &v : the_optimal_assignment.first) {
          if (v == -1) {
            S_ok = false;
            break;
          }
        }
        if (S_ok) {
          potential_assignments.emplace_back(
              std::make_pair(CM_tmp, the_optimal_assignment));
        }

        // finally updating the current optimal solution that was used to
        // generate this new potential solution
        for (size_t l = 0; l < current_optimal_assign.first.n_cols; l++) {
          if (l != (size_t)j) {
            current_optimal_assign.first(i, l) = INF;
          }
        }
        for (size_t k = 0; k < current_optimal_assign.first.n_rows; k++) {
          if (k != i) {
            current_optimal_assign.first(k, j) = INF;
          }
        }
      }
    }
  }
}

void murtyKBestAssignmentWrapper(
    const arma::fmat &M, unsigned m,
    std::vector<std::pair<std::vector<int>, float>> &assigns) {
  // this is C++ version of the Murty KBest algorithm implemented by BT Vo in
  // matlab
  assert(!assigns.size());
  if (m == 0) {
    return;
  }

  arma::fmat CM = M;
  // padding
  CM.insert_cols(CM.n_cols, CM.n_rows);
  // making CM a non-negative matrix
  float min_val = INF;
  for (size_t i = 0; i < CM.size(); i++) {
    if (min_val > CM[i]) {
      min_val = CM[i];
    }
  }
  CM -= min_val;

  murtyKBestAssignment(CM, m, assigns);
  // corrections
  for (auto &a : assigns) {
    for (auto &s : a.first) {
      // costs corrections
      if (s >= 0) {
        a.second += min_val;
      }
      // strip dummy assignment
      if (s >= (int)M.n_cols) {
        s = -1;
      }
    }
  }
}
