/*
 * Copyright (c) 2025-2026 Yihao Cai - All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef FOOTSTEPPLANNER_HPP
#define FOOTSTEPPLANNER_HPP

#include <memory>
#include "edge_quadruped_controller/utils/math/mathTypes.h"


namespace edge::control::quadruped
{

class QuadrupedModel;
class Estimator;
struct CtrlComponent;

class FootstepPlanner {
public:
    explicit FootstepPlanner(CtrlComponent &ctrl_component);

    ~FootstepPlanner() = default;

    Vec3 calcFootPos(int index, Vec2 vxy_goal_global, double d_yaw_global, double phase);

private:
    CtrlComponent &ctrl_component_;
    std::shared_ptr<QuadrupedModel> &robot_model_;
    std::shared_ptr<Estimator> &estimator_;

    Vec4 feet_radius_, feet_init_angle_;

    double k_x_, k_y_, k_yaw_;
    double t_stance_{}, t_swing_{};
};

} // namespace edge::control::quadruped

#endif //FOOTSTEPPLANNER_HPP
