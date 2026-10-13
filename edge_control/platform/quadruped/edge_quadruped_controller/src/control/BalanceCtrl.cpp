/*
 * Copyright (c) 2025-2026 Yihao Cai - All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../utils/quadProgpp/QuadProg++.hh"
#include "edge_quadruped_controller/control/BalanceCtrl.h"

#include "edge_quadruped_controller/utils/math/mathTools.h"
#include "edge_quadruped_controller/robot_model/QuadrupedModel.hpp"

namespace edge::control::quadruped
{

BalanceCtrl::BalanceCtrl(const std::shared_ptr<QuadrupedModel> &robot) {

    float jetson_com_offset_z = 0.05;  // Estimated vertical offset of Jetson Nano's COM from the robot's COM in meters
    mass_ = robot->getRobotMass();

    com_offset_ = robot->getComOffset();
    com_offset_(2) += jetson_com_offset_z;

    Ib_ = robot->getRobotInertia();
    g_ << 0, 0, -9.81;

    Vec6 s;
    Vec12 w, u;
    w << 10, 10, 4, 10, 10, 4, 10, 10, 4, 10, 10, 4;
    u << 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3;
    alpha_ = 0.001;
    beta_ = 0.1;
    friction_ratio_ = 0.4;

    s << 20, 20, 50, 450, 450, 450;

    S_ = s.asDiagonal();
    W_ = w.asDiagonal();
    U_ = u.asDiagonal();

    F_prev_.setZero();
    friction_mat_ << 1, 0, friction_ratio_, 
                    -1, 0, friction_ratio_, 
                     0, 1, friction_ratio_, 
                     0, -1, friction_ratio_, 
                     0, 0, 1;
}

Vec34 BalanceCtrl::calcGRF(const Vec3 &ddPcd, const Vec3 &dWbd, const RotMat &rot_matrix,
                        const Vec34 &feet_pos_2_body, const VecInt4 &contact) {
    calMatrixA(feet_pos_2_body, rot_matrix);
    calVectorBd(ddPcd, dWbd, rot_matrix);
    calConstraints(contact);

    G_ = A_.transpose() * S_ * A_ + alpha_ * W_ + beta_ * U_;
    g0T_ = -bd_.transpose() * S_ * A_ - beta_ * F_prev_.transpose() * U_;

    solveQP();

    F_prev_ = F_;
    return vec12ToVec34(F_);
}

void BalanceCtrl::calMatrixA(const Vec34& feet_pos_2_body, const RotMat& rotM) {
    
    for (int i = 0; i < 4; ++i) {
        A_.block(0, 3 * i, 3, 3) = I3();
        A_.block(3, 3 * i, 3, 3) = skew(Vec3(feet_pos_2_body.col(i)) - rotM * com_offset_);
    }
}

void BalanceCtrl::calVectorBd(const Vec3 &ddPcd, const Vec3 &dWbd, const RotMat &rotM) {
    bd_.head(3) = mass_ * (ddPcd - g_);
    bd_.tail(3) = rotM * Ib_ * rotM.transpose() * dWbd;
}

void BalanceCtrl::calConstraints(const VecInt4 &contact) {
    int contactLegNum = 0;
    for (int i(0); i < 4; ++i) {
        if (contact[i] == 1) {
            contactLegNum += 1;
        }
    }

    CI_.resize(5 * contactLegNum, 12);
    ci0_.resize(5 * contactLegNum);
    CE_.resize(3 * (4 - contactLegNum), 12);
    ce0_.resize(3 * (4 - contactLegNum));

    CI_.setZero();
    ci0_.setZero();
    CE_.setZero();
    ce0_.setZero();

    int ceID = 0;
    int ciID = 0;
    for (int i(0); i < 4; ++i) {
        if (contact[i] == 1) {
            CI_.block(5 * ciID, 3 * i, 5, 3) = friction_mat_;
            ++ciID;
        } else {
            CE_.block(3 * ceID, 3 * i, 3, 3) = I3();
            ++ceID;
        }
    }
}

void BalanceCtrl::solveQP() {
    const long n = F_.size();
    const long m = ce0_.size();
    const long p = ci0_.size();

    quadprogpp::Matrix<double> G, CE, CI;
    quadprogpp::Vector<double> g0, ce0, ci0, x;

    G.resize(n, n);
    CE.resize(n, m);
    CI.resize(n, p);
    g0.resize(n);
    ce0.resize(m);
    ci0.resize(p);
    x.resize(n);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            G[i][j] = G_(i, j);
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            CE[i][j] = CE_.transpose()(i, j);
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < p; ++j) {
            CI[i][j] = CI_.transpose()(i, j);
        }
    }

    for (int i = 0; i < n; ++i) {
        g0[i] = g0T_[i];
    }

    for (int i = 0; i < m; ++i) {
        ce0[i] = ce0_[i];
    }

    for (int i = 0; i < p; ++i) {
        ci0[i] = ci0_[i];
    }

    solve_quadprog(G, g0, CE, ce0, CI, ci0, x);

    for (int i = 0; i < n; ++i) {
        F_[i] = x[i];
    }
}

} 