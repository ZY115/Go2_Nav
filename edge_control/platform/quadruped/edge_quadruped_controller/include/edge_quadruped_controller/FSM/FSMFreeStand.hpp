#pragma once

#include "edge_quadruped_controller/control/CtrlComponent.h"
#include "edge_quadruped_controller/robot_model/QuadrupedModel.hpp"
#include "controller_common/FSM/FSMState.h"

struct CtrlInterfaces;
namespace edge::control::quadruped
{
    struct CtrlComponent;
    class QuadrupedModel;
}

using namespace edge::control::quadruped;

class FSMFreeStand final : public FSMState {
public:
    FSMFreeStand(CtrlInterfaces &ctrl_interfaces,
                 CtrlComponent &ctrl_component,
                 const double kp,
                 const double kd);

    void enter() override;

    void run(const rclcpp::Time &time,
             const rclcpp::Duration &period) override;

    void exit() override;

    FSMStateName checkChange() override;

private:
    std::shared_ptr<QuadrupedModel> &robot_model_;

    double kp_, kd_;

    double row_max_, row_min_;
    double pitch_max_, pitch_min_;
    double yaw_max_, yaw_min_;
    double height_max_, height_min_;

    Vec12 init_joint_pos_;
    Vec12 target_joint_pos_;

    Vec3 fr_init_pos_;           // 初始机身位置（类似_initVecOX）
    Vec34 init_foot_pos_;    // 初始足端位置（类似_initVecXP）
    
    void calc_body_target(double row, double pitch, double yaw, double height);
    Vec34 calcOP(double row, double pitch, double yaw, double height);  // 对应原版_calcOP
    

};
