#pragma once

#include <controller_common/FSM/BaseFixedStand.h>

struct CtrlInterfaces;

namespace edge::control::quadruped
{
    struct CtrlComponent;
}

using namespace edge::control::quadruped;

class FSMFixedStand final : public BaseFixedStand {
public:
    explicit FSMFixedStand(CtrlInterfaces &ctrl_interfaces,
                            CtrlComponent &ctrl_component,
                            const std::vector<double> &target_pos,
                            double kp,
                            double kd);

    void enter() override;

    FSMStateName checkChange() override;

private:
    CtrlComponent& ctrl_component_;
};
