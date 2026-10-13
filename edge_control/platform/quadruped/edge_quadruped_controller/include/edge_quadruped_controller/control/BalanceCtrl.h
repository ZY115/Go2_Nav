/*
 * Copyright (c) 2025-2026 Yihao Cai - All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include "edge_quadruped_controller/utils/math/mathTypes.h"
#include "edge_quadruped_controller/robot_model/QuadrupedModel.hpp"


namespace edge::control::quadruped
{
class QuadrupedModel;

class BalanceCtrl
{
public:
    explicit BalanceCtrl(const std::shared_ptr<QuadrupedModel>& robot);

    ~BalanceCtrl() = default;

    /**
     * Calculate the Ground Reaction Forces (GRF) for the feet
     * @param ddPcd desired body acceleration
     * @param dWbd desired body angular acceleration
     * @param rot_matrix current body rotation matrix
     * @param feet_pos_2_body feet positions to body under world frame
     * @param contact feet contact
     * @return
     */
    Vec34 calcGRF(const Vec3& ddPcd, const Vec3& dWbd, const RotMat& rot_matrix,
               const Vec34& feet_pos_2_body, const VecInt4& contact);

private:
    void calMatrixA(const Vec34& feet_pos_2_body, const RotMat& rotM);

    void calVectorBd(const Vec3& ddPcd, const Vec3& dWbd, const RotMat& rotM);

    void calConstraints(const VecInt4& contact);

    void solveQP();

    Mat12 G_, W_, U_;
    Mat6 S_;
    Mat3 Ib_;
    Vec6 bd_;
    Vec3 g_, com_offset_;
    Vec12 F_, F_prev_, g0T_;
    double mass_, alpha_, beta_, friction_ratio_;
    Eigen::MatrixXd CE_, CI_;
    Eigen::VectorXd ce0_, ci0_;
    Eigen::Matrix<double, 6, 12> A_;
    Eigen::Matrix<double, 5, 3> friction_mat_;
};

} // namespace edge::control::quadruped