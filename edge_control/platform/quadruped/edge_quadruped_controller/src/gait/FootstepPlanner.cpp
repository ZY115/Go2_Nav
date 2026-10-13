/*
 * Copyright (c) 2025-2026 Yihao Cai - All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 */

#include "edge_quadruped_controller/gait/FootstepPlanner.hpp"

#include <edge_quadruped_controller/control/CtrlComponent.h>
#include <edge_quadruped_controller/estimator/Estimator.hpp>

namespace edge::control::quadruped
{

FootstepPlanner::FootstepPlanner(CtrlComponent &ctrl_component)
    : ctrl_component_(ctrl_component),
      robot_model_(ctrl_component.robot_model_),
      estimator_(ctrl_component.estimator_) {
    k_x_ = 0.005;
    k_y_ = 0.005;
    k_yaw_ = 0.005;
    
    // Calculate foot radius and initial angle
    Vec34 feet_pos_body = robot_model_->feet_pos_normal_stand_;
    for (int i(0); i < 4; ++i) {
        feet_radius_(i) = sqrt(pow(feet_pos_body(0, i), 2) + pow(feet_pos_body(1, i), 2));
        feet_init_angle_(i) = atan2(feet_pos_body(1, i), feet_pos_body(0, i));
    }
    
    // Set stance and swing time from phase scheduler
    t_stance_ = ctrl_component_.phase_scheduler_->get_t_stance();
    t_swing_ = ctrl_component_.phase_scheduler_->get_t_swing();
}


// For swing foot, calculate the desired foot position in the global frame
Vec3 FootstepPlanner::calcFootPos(const int index, Vec2 vxy_goal_global, const double d_yaw_global, const double phase) {
    Vec3 body_vel_global = estimator_->getVelocity();
    Vec3 next_step;

    next_step(0) = body_vel_global(0) * (1 - phase) * t_swing_ +
                   body_vel_global(0) * t_stance_ / 2 +
                   k_x_ * (body_vel_global(0) - vxy_goal_global(0));
    next_step(1) = body_vel_global(1) * (1 - phase) * t_swing_ +
                   body_vel_global(1) * t_stance_ / 2 +
                   k_y_ * (body_vel_global(1) - vxy_goal_global(1));
    next_step(2) = 0;

    const double yaw = estimator_->getYaw();
    const double d_yaw = estimator_->getDYaw();
    const double next_yaw = d_yaw * (1 - phase) * t_swing_ + d_yaw * t_stance_ / 2 +
                            k_yaw_ * (d_yaw_global - d_yaw);

    next_step(0) += feet_radius_(index) * cos(yaw + feet_init_angle_(index) + next_yaw);
    next_step(1) += feet_radius_(index) * sin(yaw + feet_init_angle_(index) + next_yaw);

    Vec3 foot_pos = estimator_->getPosition() + next_step;
    foot_pos(2) = 0.0;

    return foot_pos;
}

} // namespace edge::control::quadruped