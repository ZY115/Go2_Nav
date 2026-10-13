//
// Created by tlab-uav on 24-9-6.
//

#ifndef FSMSTATE_H
#define FSMSTATE_H

#include <string>
#include <utility>
#include <controller_common/common/enumClass.h>
#include <controller_common/CtrlInterfaces.h>
#include <rclcpp/time.hpp>
#include <rclcpp/rclcpp.hpp> 


class FSMState : public rclcpp::Node
{
public:
    virtual ~FSMState() = default;

    FSMState(const FSMStateName& state_name, std::string state_name_string, CtrlInterfaces& ctrl_interfaces)
        : state_name(state_name),
          state_name_string(std::move(state_name_string)),
          ctrl_interfaces_(ctrl_interfaces), 
          Node(sanitizeName(state_name_string) + "_node")
    {
    }

    void setReset(bool reset) {
        is_reset_ = reset;
    }

    void setLaunchAfterReset(bool launch) {
        launch_after_reset_ = launch;
    }

    static std::string sanitizeName(std::string name) {
    std::replace(name.begin(), name.end(), ' ', '_');  // Space → Underscore
    return name;
}

    virtual void enter() = 0;

    virtual void run(const rclcpp::Time& time,
                     const rclcpp::Duration& period) = 0;

    virtual void exit() = 0;

    virtual FSMStateName checkChange() { return FSMStateName::INVALID; }

    FSMStateName state_name;
    std::string state_name_string;

protected:
    CtrlInterfaces& ctrl_interfaces_;
    bool is_reset_{false};
    bool launch_after_reset_{false};
};

#endif //FSMSTATE_H
