/*
 * Copyright (c) 2025-2026 Yihao Cai - All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 */

// Implementation for DeadzoneDiscriminator
#include <iostream>
#include "edge_rl_runtime/deadzone_discriminator.hpp"

namespace edge::rl_runtime
{

DeadzoneDiscriminator::DeadzoneDiscriminator(double y_left, double y_right, double x_front, double x_back)
    : threshold_(0.5), y_left_dist_(y_left), y_right_dist_(y_right), x_front_dist_(x_front), x_back_dist_(x_back)
{
    y_left_dist_ = 5.0;
    y_right_dist_ = 8.0;
    std::cout << "Deadzone initialized with threshold: " << threshold_ << std::endl
              << ", y_left_distance: " << y_left_dist_ << std::endl
              << ", y_right_distance: " << y_right_dist_ << std::endl
              << ", x_front_distance: " << x_front_dist_ << std::endl
              << ", x_back_distance: " << x_back_dist_ << std::endl;
}

bool DeadzoneDiscriminator::is_in_deadzone(double pos_x, double pos_y) const
{
    double out_of_left_y = pos_y > y_left_dist_;
    double out_of_right_y = pos_y < -y_right_dist_;
    double out_of_front_x = pos_x > x_front_dist_;
    double out_of_back_x = pos_x < -x_back_dist_;

    std::cout << "Checking deadzone for position (x: " << pos_x << ", y: " << pos_y << ")" << std::endl;
    // Placeholder implementation - replace with actual deadzone logic
    if (out_of_left_y) {
        std::cout << "out of left y zone!" << std::endl;
        return true;
    }
    
    if (out_of_right_y) {
        std::cout << "out of right y zone!" << std::endl;
        return true;
    }

    if (out_of_front_x) {
        std::cout << "out of front x zone!" << std::endl;
        return true;
    }

    if (out_of_back_x) {
        std::cout << "out of back x zone!" << std::endl;
        return true;
    }

    return false;
}

} // namespace edge::rl_runtime