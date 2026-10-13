#pragma once
#include "FSMState.h"

class FSMPassive final : public FSMState
{
public:
    explicit FSMPassive(CtrlInterfaces& ctrl_interfaces);

    void enter() override;

    void run(const rclcpp::Time& time,
             const rclcpp::Duration& period) override;

    void exit() override;

    FSMStateName checkChange() override;
};

