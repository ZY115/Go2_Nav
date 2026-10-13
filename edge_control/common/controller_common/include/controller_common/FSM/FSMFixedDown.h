//
// Created by tlab-uav on 24-9-11.
//

#pragma once

#include "FSMState.h"

class FSMFixedDown final : public FSMState
{
public:
    explicit FSMFixedDown(CtrlInterfaces& ctrl_interfaces,
                            const std::vector<double>& target_pos,
                            double kp,
                            double kd
    );

    void enter() override;

    void run(const rclcpp::Time& time,
             const rclcpp::Duration& period) override;

    void exit() override;

    FSMStateName checkChange() override;

private:
    double target_pos_[12] = {};
    double start_pos_[12] = {};
    rclcpp::Time start_time_;

    double kp_, kd_;

    double duration_ = 600; // steps
    double percent_ = 0; //%
    double phase = 0.0;
};
