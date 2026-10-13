#include <cmath>
#include "edge_quadruped_controller/FSM/FSMFreeStand.hpp"
#include "edge_quadruped_controller/edge_quadruped_controller.hpp"

#include "edge_quadruped_controller/utils/math/mathTools.h"
#include "edge_quadruped_controller/utils/math/mathTypes.h"

FSMFreeStand::FSMFreeStand(CtrlInterfaces& ctrl_interfaces,
                           CtrlComponent& ctrl_component,
                           const double kp,
                           const double kd)
    : FSMState(
          FSMStateName::FREESTAND, "FSM Free Stand",
          ctrl_interfaces),
      robot_model_(ctrl_component.robot_model_),
      kp_(kp), kd_(kd)
{
    row_max_ = 20 * M_PI / 180;
    row_min_ = -row_max_;
    pitch_max_ = 15 * M_PI / 180;
    pitch_min_ = -pitch_max_;
    yaw_max_ = 20 * M_PI / 180;
    yaw_min_ = -yaw_max_;
    height_max_ = 0.1;
    height_min_ = -height_max_;
}

void FSMFreeStand::enter()
{
    for (int i = 0; i < 12; i++)
    {
        ctrl_interfaces_.joint_kp_command_interface_[i].get().set_value(kp_);
        ctrl_interfaces_.joint_kd_command_interface_[i].get().set_value(kd_);
    }

    init_joint_pos_ = robot_model_->current_joint_pos_;
    
    // Get initial foot positions (corresponding to original _initVecXP)
    init_foot_pos_ = robot_model_->getFeetPosInBodyFrame();

    // Take FR leg position as reference point, calculate other foot positions relative to FR leg
    fr_init_pos_ = init_foot_pos_.col(0);
    for (int i = 0; i < 4; ++i)
    {
        init_foot_pos_.col(i) -= fr_init_pos_;
    }
    ctrl_interfaces_.control_inputs_.command = 0;
}

void FSMFreeStand::run(const rclcpp::Time&/*time*/, const rclcpp::Duration&/*period*/)
{
    calc_body_target(invNormalize(ctrl_interfaces_.control_inputs_.lx, row_min_, row_max_),
                     invNormalize(ctrl_interfaces_.control_inputs_.ly, pitch_min_, pitch_max_),
                     invNormalize(ctrl_interfaces_.control_inputs_.rx, yaw_min_, yaw_max_),
                     invNormalize(ctrl_interfaces_.control_inputs_.ry, height_min_, height_max_));
}

void FSMFreeStand::exit()
{
}

FSMStateName FSMFreeStand::checkChange()
{
    switch (ctrl_interfaces_.control_inputs_.command)
    {
    case 1:
        return FSMStateName::PASSIVE;
    case 2:
        return FSMStateName::FIXEDSTAND;
    default:
        return FSMStateName::FREESTAND;
    }
}

void FSMFreeStand::calc_body_target(const double row, const double pitch,
                                      const double yaw, const double height)
{
    // 使用原版算法计算目标足端位置
    Vec34 vecOP = calcOP(row, pitch, yaw, height);

    // 通过逆运动学计算关节角度
    target_joint_pos_ = robot_model_->getJointPos(vecOP);

    // 设置关节位置命令
    for (int i = 0; i < 12; i++)
    {
        ctrl_interfaces_.joint_position_command_interface_[i].get().set_value(
            target_joint_pos_(i));
    }
}

Vec34 FSMFreeStand::calcOP(const double row, const double pitch, const double yaw, const double height)
{
    // 对应原版_calcOP函数
    // 计算机身目标位置：从原点开始，加上高度偏移
    Vec3 vecXO = -fr_init_pos_; // 从原点开始（init_body_pos_ = Vec3::Zero()）
    vecXO(2) += height; // 加上高度偏移

    // 使用mathTools中的函数计算旋转矩阵
    Eigen::Matrix3d rotM = rpyToRotMat(row, pitch, yaw);

    // 构建齐次变换矩阵：从世界坐标系到机身坐标系
    Eigen::Matrix4d Tsb = homoMatrix(vecXO, rotM);

    // 计算逆变换：从机身坐标系到世界坐标系
    Eigen::Matrix4d Tbs = homoMatrixInverse(Tsb);

    Vec34 vecOP;
    for (int i = 0; i < 4; ++i)
    {
        // 将每条腿的初始位置从机身坐标系变换到世界坐标系
        Vec4 tempVec4 = Tbs * homoVec(init_foot_pos_.col(i));
        vecOP.col(i) = noHomoVec(tempVec4);
    }

    return vecOP;
}
