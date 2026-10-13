#include "controller_common/FSM/FSMPassive.h"

#include <iostream>

FSMPassive::FSMPassive(CtrlInterfaces& ctrl_interfaces) : FSMState(
    FSMStateName::PASSIVE, "FSM Passive", ctrl_interfaces)
{
}

void FSMPassive::enter()
{
    std::cerr << "Entering FSM Passive State" << std::endl;
    for (auto i : ctrl_interfaces_.joint_torque_command_interface_)
    {
        i.get().set_value(0);
    }
    for (auto i : ctrl_interfaces_.joint_position_command_interface_)
    {
        i.get().set_value(0);
    }
    for (auto i : ctrl_interfaces_.joint_velocity_command_interface_)
    {
        i.get().set_value(0);
    }
    for (auto i : ctrl_interfaces_.joint_kp_command_interface_)
    {
        i.get().set_value(0);
    }
    for (auto i : ctrl_interfaces_.joint_kd_command_interface_)
    {
        i.get().set_value(1);
    }
    ctrl_interfaces_.control_inputs_.command = 0;
}

void FSMPassive::run(const rclcpp::Time&/*time*/, const rclcpp::Duration&/*period*/)
{
}

void FSMPassive::exit()
{
}

FSMStateName FSMPassive::checkChange()
{
    if (ctrl_interfaces_.control_inputs_.command == 2)
    {
        return FSMStateName::FIXEDDOWN;
    }
    return FSMStateName::PASSIVE;
}
