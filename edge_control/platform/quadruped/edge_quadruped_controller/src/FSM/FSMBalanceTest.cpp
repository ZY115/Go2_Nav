#include "edge_quadruped_controller/gait/PhaseScheduler.h"
#include "edge_quadruped_controller/FSM/FSMBalanceTest.hpp"
#include "edge_quadruped_controller/utils/math/mathTools.h"
#include "edge_quadruped_controller/edge_quadruped_controller.hpp"

FSMBalanceTest::FSMBalanceTest(CtrlInterfaces& ctrl_interfaces,
                                   CtrlComponent& ctrl_component,
                                   const std::vector<double>& stable_gains)
    : FSMState(FSMStateName::BALANCETEST,
               "FSM Balance Test",
               ctrl_interfaces),
      estimator_(ctrl_component.estimator_),
      robot_model_(ctrl_component.robot_model_),
      balance_ctrl_(ctrl_component.balance_ctrl_),
      phase_scheduler_(
          ctrl_component.phase_scheduler_),
      stable_gains_(stable_gains)
{
    _xMax = 0.05;
    _xMin = -_xMax;
    _yMax = 0.05;
    _yMin = -_yMax;
    _zMax = 0.04;
    _zMin = -_zMax;
    _yawMax = 20 * M_PI / 180;
    _yawMin = -_yawMax;

    Kp_p_ = Vec3(150, 150, 150).asDiagonal();
    Kd_p_ = Vec3(25, 25, 25).asDiagonal();

    kp_w_ = 200;
    Kd_w_ = Vec3(30, 30, 30).asDiagonal();
}

void FSMBalanceTest::enter()
{
    pcd_init_ = estimator_->getPosition();
    pcd_ = pcd_init_;
    init_rotation_ = estimator_->getRotation();
    setStableGain();
}

void FSMBalanceTest::run(const rclcpp::Time&/*time*/, const rclcpp::Duration&/*period*/)
{
    pcd_(0) = pcd_init_(0) + invNormalize(ctrl_interfaces_.control_inputs_.ly, _xMin, _xMax);
    pcd_(1) = pcd_init_(1) - invNormalize(ctrl_interfaces_.control_inputs_.lx, _yMin, _yMax);
    pcd_(2) = pcd_init_(2) + invNormalize(ctrl_interfaces_.control_inputs_.ry, _zMin, _zMax);

    const float yaw = -invNormalize(ctrl_interfaces_.control_inputs_.rx, _yawMin, _yawMax);
    Rd_ = rotz(yaw) * init_rotation_;

    calcTorque();
}

void FSMBalanceTest::exit()
{
    // Reset phase scheduler to SWING_ALL
    phase_scheduler_->status_ = PhaseStatus::SWING_ALL;
}

FSMStateName FSMBalanceTest::checkChange()
{
    switch (ctrl_interfaces_.control_inputs_.command)
    {
    case 1:
        return FSMStateName::PASSIVE;
    case 2:
        return FSMStateName::FIXEDSTAND;
    default:
        return FSMStateName::BALANCETEST;
    }
}

void FSMBalanceTest::calcTorque()
{
    const auto B2G_Rotation = estimator_->getRotation();
    const RotMat G2B_Rotation = B2G_Rotation.transpose();

    const Vec3 pose_body = estimator_->getPosition();
    const Vec3 vel_body = estimator_->getVelocity();

    // expected body acceleration
    dd_pcd_ = Kp_p_ * (pcd_ - pose_body) + Kd_p_ * (Vec3(0, 0, 0) - vel_body);

    // expected body angular acceleration
    d_wbd_ = kp_w_ * rotMatToExp(Rd_ * G2B_Rotation) +
        Kd_w_ * (Vec3(0, 0, 0) - estimator_->getGyroGlobal());

    // calculate foot force
    const Vec34 pos_feet_2_body_global = estimator_->getFeetPosInBodyFrame();
    const Vec34 force_feet_global = -balance_ctrl_->calcGRF(dd_pcd_, d_wbd_, B2G_Rotation,
                                                         pos_feet_2_body_global, phase_scheduler_->contact_);
    const Vec34 force_feet_body = G2B_Rotation * force_feet_global;

    // Get current joint positions
    Vec12 current_joints = robot_model_->current_joint_pos_;

    for (int i = 0; i < 4; i++)
    {
        // Calculate torque for each leg using analytical method (more stable)
        Vec3 foot_force = force_feet_body.col(i);
        Vec3 torque = robot_model_->calcTorque(foot_force, i);

        for (int j = 0; j < 3; j++)
        {
            ctrl_interfaces_.joint_torque_command_interface_[i * 3 + j].get().set_value(torque(j));
            ctrl_interfaces_.joint_position_command_interface_[i * 3 + j].get().set_value(
                current_joints(i * 3 + j));
        }
    }
}

void FSMBalanceTest::setStableGain() const
{
    for (int i = 0; i < 12; i++)
    {
        ctrl_interfaces_.joint_kp_command_interface_[i].get().set_value(stable_gains_[0]);
        ctrl_interfaces_.joint_kd_command_interface_[i].get().set_value(stable_gains_[1]);
    }
}
