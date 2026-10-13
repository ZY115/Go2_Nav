/*
 * Copyright (c) 2025-2026 Yihao Cai - All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <Eigen/Dense>
#include "edge_quadruped_controller/control/BalanceCtrl.h"
#include "edge_quadruped_controller/gait/PhaseScheduler.h"
#include "edge_quadruped_controller/estimator/Estimator.hpp"
#include "edge_quadruped_controller/robot_model/QuadrupedModel.hpp"

namespace edge::control::quadruped
{

struct CtrlComponent
{
    std::shared_ptr<QuadrupedModel> robot_model_;
    std::shared_ptr<Estimator> estimator_;
    std::shared_ptr<BalanceCtrl> balance_ctrl_;
    std::shared_ptr<PhaseScheduler> phase_scheduler_;
    
    // RESIDUAL_GAIN matrix (6x12) loaded from physical_model.yaml
    Eigen::MatrixXd residual_gain_ = Eigen::MatrixXd::Zero(6, 12);

    CtrlComponent() = default;
};

} // namespace edge::control::quadruped
