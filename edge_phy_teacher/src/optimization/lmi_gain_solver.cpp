/*
 * Copyright (c) 2025-2026 Yihao Cai - All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include <iomanip>
#include "lmi_gain_solver.hpp"

#ifdef __cplusplus
extern "C" {
    #include "ecvxcone/cpg_solve.h"
    #include "ecvxcone/cvxopt.h"
    #include "ecvxcone/lapack.h"
    void print_matrix_(matrix *m);
    ECVXConeWorkspace* ecvxcone_init(matrix *c, void *G, matrix *h, void *A, matrix *b, DIMs *dims, 
                                    ECVXConeSettings* settings);
}
#endif

/**
 * Setup the ECVXConeWorkspace structure.
 */
ECVXConeWorkspace* ecvxcone_setup(int n_var, int n_ineq, int n_eq, int nnz_G, int nnz_A, DIMs *dims, ECVXConeSettings *settings)
{
    // c
    if (Canon_Params_conditioning.c == NULL) {
        ecvxcone_data.c = Matrix_New(n_var, 1, DOUBLE);
        if (!ecvxcone_data.c) err_no_memory;
    } else {
        ecvxcone_data.c = (matrix*)malloc(sizeof(matrix));
        ecvxcone_data.c->mat_type = MAT_DENSE;
        ecvxcone_data.c->id = DOUBLE;
        ecvxcone_data.c->nrows = n_var;
        ecvxcone_data.c->ncols = 1;
        ecvxcone_data.c->buffer = Canon_Params_conditioning.c;
    }

    // b
    if (Canon_Params_conditioning.b == NULL) {
        ecvxcone_data.b = Matrix_New(n_eq, 1, DOUBLE);
        if (!ecvxcone_data.b) err_no_memory;
    } else {
        ecvxcone_data.b = (matrix*)malloc(sizeof(matrix));
        ecvxcone_data.b->mat_type = MAT_DENSE;
        ecvxcone_data.b->id = DOUBLE;
        ecvxcone_data.b->nrows = n_eq;
        ecvxcone_data.b->ncols = 1;
        ecvxcone_data.b->buffer = Canon_Params_conditioning.b;
    }

    // h
    if (Canon_Params_conditioning.h == NULL) {
        ecvxcone_data.h = Matrix_New(n_ineq, 1, DOUBLE);
        if (!ecvxcone_data.h) err_no_memory;
    } else {
        ecvxcone_data.h = (matrix*)malloc(sizeof(matrix));
        ecvxcone_data.h->mat_type = MAT_DENSE;
        ecvxcone_data.h->id = DOUBLE;
        ecvxcone_data.h->nrows = n_ineq;
        ecvxcone_data.h->ncols = 1;
        ecvxcone_data.h->buffer = Canon_Params_conditioning.h;
    }

    // G
    if (Canon_Params_conditioning.G == NULL) {
        ecvxcone_data.G = SpMatrix_New(n_ineq, n_var, nnz_G, DOUBLE);
        if (!ecvxcone_data.G) err_no_memory;
    } else{
        ecvxcone_data.G = malloc(sizeof(spmatrix));
        if (!ecvxcone_data.G) err_no_memory;
    
        spmatrix *G_sp = (spmatrix *)ecvxcone_data.G;

        G_sp->obj = (ccs *)malloc(sizeof(ccs));
        if (!G_sp->obj) err_no_memory;

        G_sp->mat_type = MAT_SPARSE;
        G_sp->obj->nrows = n_ineq;
        G_sp->obj->ncols = n_var;
        G_sp->obj->id = DOUBLE;
        G_sp->obj->colptr = (int_t *) Canon_Params_conditioning.G->p;
        G_sp->obj->rowind = (int_t *) Canon_Params_conditioning.G->i;
        G_sp->obj->values = (int_t *) Canon_Params_conditioning.G->x;
    }

    // A
    if (Canon_Params_conditioning.A == NULL) {
        ecvxcone_data.A = SpMatrix_New(n_eq, n_var, nnz_A, DOUBLE);
        if (!ecvxcone_data.A) err_no_memory;
    } else{
        ecvxcone_data.A = malloc(sizeof(spmatrix));
        if (!ecvxcone_data.A) err_no_memory;

        spmatrix *A_sp = (spmatrix *)ecvxcone_data.A;

        A_sp->obj = (ccs *)malloc(sizeof(ccs));
        if (!A_sp->obj) err_no_memory;

        A_sp->mat_type = MAT_SPARSE;
        A_sp->obj->nrows = n_eq;
        A_sp->obj->ncols = n_var;
        A_sp->obj->id = DOUBLE;
        A_sp->obj->colptr = Canon_Params_conditioning.A->p;
        A_sp->obj->rowind = Canon_Params_conditioning.A->i;
        A_sp->obj->values = Canon_Params_conditioning.A->x;
    }

    if (!ecvxcone_data.c || !ecvxcone_data.h || !ecvxcone_data.b || !ecvxcone_data.G || !ecvxcone_data.A) {
        err_no_memory;
    }
    ECVXConeWorkspace *ecvxcone_ws = ecvxcone_init(ecvxcone_data.c, ecvxcone_data.G, ecvxcone_data.h, ecvxcone_data.A, ecvxcone_data.b, dims, settings);
    return ecvxcone_ws;
}

namespace lmi {

LMIGainSolver::LMIGainSolver(double Ts) : Ts_(Ts) {
    lmi_init();
}

LMIGainSolver::~LMIGainSolver() {
    if (P_mat_) {
        Matrix_Free(P_mat_);
        P_mat_ = nullptr;
    }
    if (Q_mat_) {
        free(Q_mat_);
        Q_mat_ = nullptr;
    }
    if (R_mat_) {
        free(R_mat_);
        R_mat_ = nullptr;
    }
    if (aB_) {
        Matrix_Free(aB_);
        aB_ = nullptr;
    }
    if (aF_) {
        Matrix_Free(aF_);
        aF_ = nullptr;
    }
    if (ipiv_) {
        Matrix_Free(ipiv_);
        ipiv_ = nullptr;
    }
}

void LMIGainSolver::lmi_init() {
    /********************************  LMI Initialization  *******************************/
    // Initialize Q
    Q_mat_ = (matrix*)(malloc(sizeof(matrix))); if (!Q_mat_) err_no_memory;
    Q_mat_->mat_type = MAT_DENSE;
    Q_mat_->id = DOUBLE;
    Q_mat_->nrows = 10;
    Q_mat_->ncols = 10;
    Q_mat_->buffer = CPG_Prim.Q;

    // Initialize R
    R_mat_ = (matrix*)(malloc(sizeof(matrix))); if (!R_mat_) err_no_memory;
    R_mat_->mat_type = MAT_DENSE;
    R_mat_->id = DOUBLE;
    R_mat_->nrows = 6;
    R_mat_->ncols = 10;
    R_mat_->buffer = CPG_Prim.R;

    // Initialize P
    // P = Matrix_New(10, 10, DOUBLE);

    // Initialize aF
    aF_ = Matrix_New(10, 10, DOUBLE);

    // Initialize aB
    aB_ = Matrix_New(10, 6, DOUBLE);
    for (int i = 0; i < 6; ++i) {
        int idx = 4 + i * 11;
        MAT_BUFD(aB_)[idx] = 1.0;  // Initialize to identity-like structure
    }

    // Initialize ipiv for LU factorization
    ipiv_ = Matrix_New(10, 1, INT);
}

double LMIGainSolver::time_diff(struct timespec start, struct timespec end, int count) {
  double diff_ms = (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_nsec - start.tv_nsec) / 1e6;
  return diff_ms / count;
}

/********************************  Rotation Matrices  *******************************/
void LMIGainSolver::update_Rx(double roll) {
    Rx_(0, 0) = 1; Rx_(0, 1) = 0; Rx_(0, 2) = 0;
    Rx_(1, 0) = 0; Rx_(1, 1) = cos(roll); Rx_(1, 2) = -sin(roll);
    Rx_(2, 0) = 0; Rx_(2, 1) = sin(roll); Rx_(2, 2) = cos(roll);
}

void LMIGainSolver::update_Ry(double pitch) {
    Ry_(0, 0) = cos(pitch); Ry_(0, 1) = 0; Ry_(0, 2) = sin(pitch);
    Ry_(1, 0) = 0; Ry_(1, 1) = 1; Ry_(1, 2) = 0;
    Ry_(2, 0) = -sin(pitch); Ry_(2, 1) = 0; Ry_(2, 2) = cos(pitch);
}

void LMIGainSolver::update_Rz(double yaw) {
    Rz_(0, 0) = cos(yaw); Rz_(0, 1) = -sin(yaw); Rz_(0, 2) = 0;
    Rz_(1, 0) = sin(yaw); Rz_(1, 1) = cos(yaw); Rz_(1, 2) = 0;
    Rz_(2, 0) = 0; Rz_(2, 1) = 0; Rz_(2, 2) = 1;
}

void LMIGainSolver::update_Rzyx(double roll, double pitch, double yaw) {

    // Reset rotation matrices
    LMIGainSolver::update_Rx(roll);
    LMIGainSolver::update_Ry(pitch);
    LMIGainSolver::update_Rz(yaw);

    // Combine rotations
    Mat3x3 temp;
    matrix_multiply(Rz_, Ry_, temp);
    matrix_multiply(temp, Rx_, Rzyx_);
}

/*******************************  Update Functions  *******************************/

// Function to update the rotation matrices based on tracking error
void LMIGainSolver::update_Matrix_A(){
    cpg_update_A(71, Rzyx_(0, 0) * Ts_);  cpg_update_A(81, Rzyx_(0, 1) * Ts_);  cpg_update_A(91, Rzyx_(0, 2) * Ts_);
    cpg_update_A(72, Rzyx_(1, 0) * Ts_);  cpg_update_A(82, Rzyx_(1, 1) * Ts_);  cpg_update_A(92, Rzyx_(1, 2) * Ts_);
    cpg_update_A(73, Rzyx_(2, 0) * Ts_);  cpg_update_A(83, Rzyx_(2, 1) * Ts_);  cpg_update_A(93, Rzyx_(2, 2) * Ts_);
}

void LMIGainSolver::update_Matrix_B(){

}

void LMIGainSolver::update_TrackingErrorSquare() {
    for (int i = 0; i < 10; ++i) {
        tracking_err_square[i] = tracking_err[i] * tracking_err[i];
        cpg_update_tracking_err_square(i, tracking_err_square[i]);
    }
}

/********************************  Post Processing  *******************************/
void LMIGainSolver::post_processing() {
    P_mat_ = Matrix_NewFromMatrix(Q_mat_, Q_mat_->id);  // Ensure P is initialized

    // P = inverse(Q)
    lapack_getrf(P_mat_, ipiv_, P_mat_->nrows, P_mat_->ncols, P_mat_->nrows, 0);   // LU factorization
    lapack_getri(P_mat_, ipiv_, P_mat_->nrows, P_mat_->nrows, 0);   // Inversion

    // aF = round(aB * R * P)
    matrix *temp = Matrix_NewFromMatrix(aF_, aF_->id);  // Temporary matrix for multiplication
    blas_gemm(aB_, R_mat_, temp, 'N', 'N', NULL, NULL, -1, -1, -1, 0, 0, 0, 0, 0, 0);
    blas_gemm(temp, P_mat_, aF_, 'N', 'N', NULL, NULL, -1, -1, -1, 0, 0, 0, 0, 0, 0);
    Matrix_Free(temp);  // Free the temporary matrix
    Matrix_Free(P_mat_);  // Free P if it was previously allocated
    P_mat_ = nullptr;

    // Fb2 = aF[6:10, 0:4]
    // F_kp = -np.block([
    //     [np.zeros((2, 6))], 
    //     [np.zeros((4, 2)), Fb2]])
    for (int i = 6; i < 10; ++i) {
        for (int j = 0; j < 4; ++j) {
            F_kp(i-4, j+2) = -MAT_ELEMD(aF_, i, j);
        }
    }

    // F_kd = -aF[4:10, 4:10]
    for (int i = 4; i < 10; ++i) {
        for (int j = 4; j < 10; ++j) {
            F_kd(i-4, j-4) = -MAT_ELEMD(aF_, i, j);
        }
    }
}

void LMIGainSolver::test_and_print() {
    test_tracking_err();
    update();
    print_matrix6x6(F_kp, "F_kp");
    print_matrix6x6(F_kd, "F_kd");
}

void LMIGainSolver::update() {
    double roll = tracking_err[1];
    double pitch = tracking_err[2];
    double yaw = tracking_err[3];

    // Update rotation matrices based on tracking error
    update_Rzyx(roll, pitch, yaw);

    // Update matrices A(s) B(s) and tracking error square
    update_Matrix_A();
    update_Matrix_B();
    update_TrackingErrorSquare();

    // Call the solver
    cpg_solve();

    // Check solver status and post-process results if successful
    if(CPG_Info.status == OPTIMAL) {
        std::cout << "LMI optimization successful. Processing results..." << std::endl;
        // Post processing to finalize the matrices
        post_processing();
    } else {
        std::cerr << "LMI optimization failed with status: " << CPG_Info.status << std::endl;
    }

}

void LMIGainSolver::test_tracking_err() {
    double data[10] = {-0.0158, -0.0417, -0.1517, 0.0032, 0.2703, -0.1057, 0.0472, 0.3559, -0.2925, -0.6624};
    std::copy(data, data + 10, tracking_err);
}


void LMIGainSolver::benchmark(int iterations, bool verbose, double threshold) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iterations; ++i) {
        for (int j = 0; j < 10; ++j)
        tracking_err[j] = ((double) rand() / RAND_MAX) * threshold;

        update();

        if (verbose) {
        printf("Iteration %d:\n", i + 1);
        print_matrix6x6(F_kp, "F_kp");
        print_matrix6x6(F_kd, "F_kd");
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("Average solving time per iteration: %.3f ms\n", time_diff(start, end, iterations));
}

void print_matrix6x6(const Mat6x6& mat, const char* name) {
    std::cout << "Matrix " << name << ":" << std::endl;

    for (int i = 0; i < 6; ++i) {
        std::cout << "  [";
        for (int j = 0; j < 6; ++j) {
        std::cout << std::setw(8) << std::fixed << std::setprecision(4) << mat(i, j);
        if (j < 5)
            std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << std::endl; 
}

} // namespace lmi

