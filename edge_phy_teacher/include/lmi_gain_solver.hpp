/*
 * Copyright (c) 2025-2026 Yihao Cai - All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef LMI_OPTIMIZER_HPP
#define LMI_OPTIMIZER_HPP

#include <Eigen/Dense>
#include <cmath>
#include <ctime>
#include <cstdio>

#ifdef __cplusplus
extern "C" {
    #include "ecvxcone/base.h"
    #include "ecvxcone/blas.h"
    #include "ecvxcone/misc.h"
    #include "ecvxcone/solver.h"
    #include "ecvxcone/lapack.h"
    #include "ecvxcone/cvxopt.h"
    #include "ecvxcone/cpg_solve.h"
    #include "ecvxcone/cpg_workspace.h"
}
#endif

ECVXConeWorkspace* ecvxcone_setup(int n_var, int n_ineq, int n_eq, int nnz_G, int nnz_A, DIMs *dims, ECVXConeSettings *settings);
// extern ECVXConeWorkspace* ecvxcone_init(matrix *c, void *G, matrix *h, void *A, matrix *b, DIMs *dims, ECVXConeSettings *settings);


// Sampling period
// constexpr double Ts = 1.0 / 20.0;

namespace lmi {

using Mat3x3 = Eigen::Matrix<double, 3, 3>;
using Mat6x6 = Eigen::Matrix<double, 6, 6>;
using Vec6d  = Eigen::Matrix<double, 6, 1>;

// Utility
void print_matrix6x6(const Mat6x6& mat, const char* name);
inline void matrix_multiply(const Mat3x3& A, const Mat3x3& B, Mat3x3& result) {
    result = A * B;
}

class LMIGainSolver {
public:
    LMIGainSolver() = delete;                 // Delete default constructor
    explicit LMIGainSolver(double Ts);        // Constructor with sampling period

    ~LMIGainSolver();                         // Destructor

    LMIGainSolver(const LMIGainSolver&) = delete;            // Delete copy constructor
    LMIGainSolver& operator=(const LMIGainSolver&) = delete; // Delete copy assignment operator

    Mat6x6 F_kp;  // Contains the solved Kp gain matrices
    Mat6x6 F_kd;  // Contains the solved Kd gain matrices

    double tracking_err[10];
    double tracking_err_square[10];

    void update();             // Update matrices and call solver
    void test_and_print();     // Test inputs and print results
    void benchmark(int iterations = 1000, bool verbose = false, double threshold = 0.1);
    
    // Rotation matrices
    void update_Rx(double roll);
    void update_Ry(double pitch);
    void update_Rz(double yaw);
    void update_Rzyx(double roll, double pitch, double yaw);

private:
    double Ts_;    // Sampling period

    // matrix objects
    matrix* P_mat_ = nullptr;
    matrix* Q_mat_ = nullptr;
    matrix* R_mat_ = nullptr;
    matrix* aB_ = nullptr;
    matrix* aF_ = nullptr;
    matrix* ipiv_ = nullptr;

    Mat3x3 Rx_;
    Mat3x3 Ry_;
    Mat3x3 Rz_;
    Mat3x3 Rzyx_;

    // Initialization helpers
    void lmi_init();                 // Initialize LMI matrices
    void test_tracking_err();       // Set up tracking error for testing
    double time_diff(timespec start, timespec end, int count);

    // Matrix operations
    void update_Matrix_A();
    void update_Matrix_B();
    void update_TrackingErrorSquare();
    void post_processing();

};

} // namespace lmi


#endif  // LMI_OPTIMIZER_HPP
