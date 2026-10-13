// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cmath>

namespace edge::rl_runtime
{
class DeadzoneDiscriminator
{
public:
    DeadzoneDiscriminator(double y_left = 0.5, double y_right = 6.0, double x_front = 30.0, double x_back = 10.0);

    /// Returns true if position (pos_x, pos_y) is considered inside the deadzone.
    bool is_in_deadzone(double pos_x, double pos_y) const;

private:
    double threshold_;
    double y_left_dist_;
    double y_right_dist_;
    double x_front_dist_;
    double x_back_dist_;

    double time_horizon_ = 1.0; // Time horizon for prediction (s)
};

} // namespace edge::rl_runtime

