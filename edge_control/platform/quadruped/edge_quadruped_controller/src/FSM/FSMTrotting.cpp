/*
 * Copyright (c) 2025-2026 Yihao Cai - All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 */

#include "edge_msgs/msg/edge_action.hpp"
#include "edge_ros_interfaces/msg_conversion.hpp"
#include "edge_quadruped_controller/FSM/FSMTrotting.hpp"
#include <edge_quadruped_controller/utils/math/mathTools.h>
#include <edge_quadruped_controller/control/CtrlComponent.h>
#include <edge_quadruped_controller/estimator/Estimator.hpp>
#include <edge_quadruped_controller/gait/PhaseScheduler.h>

using namespace edge::control::quadruped;

FSMTrotting::FSMTrotting(CtrlInterfaces& ctrl_interfaces,
                             CtrlComponent& ctrl_component,
                             const std::vector<double>& swing_gains,
                             const std::vector<double>& stable_gains,
                             Vec2 vx_limit, Vec2 vy_limit, Vec2 wz_limit,
                             double gait_height) 
                             : FSMState(FSMStateName::TROTTING, "FSM Trotting",
                                ctrl_interfaces),
                                estimator_(ctrl_component.estimator_),
                                robot_model_(ctrl_component.robot_model_),
                                balance_ctrl_(ctrl_component.balance_ctrl_),
                                phase_scheduler_(ctrl_component.phase_scheduler_),
                                ctrl_component_(ctrl_component),
                                gait_generator_(ctrl_component),
                                swing_gains_(swing_gains),
                                stable_gains_(stable_gains),
                                vx_limit_(vx_limit),
                                vy_limit_(vy_limit),
                                wz_limit_(wz_limit),
                                gait_height_(gait_height)
    {
    Kpp = Vec3(70, 70, 70).asDiagonal();
    Kdp = Vec3(20, 20, 20).asDiagonal();
    kp_w_ = 780;
    Kd_w_ = Vec3(70, 70, 70).asDiagonal();
    Kp_swing_ = Vec3(200, 200, 200).asDiagonal();
    Kd_swing_ = Vec3(10, 10, 10).asDiagonal();

    dt_ = 1.0 / ctrl_interfaces_.frequency_;

    // Initialize last tracking error and action
    last_edge_accel_.setZero();
    last_robot_state_.setZero();
    last_tracking_error_tea_.setZero();
    
    curr_edge_accel_.setZero();
    curr_robot_state_.setZero();
    curr_tracking_error_tea_.setZero();

    // Initialize ROS messages for tracking error
    tracking_error_msg_.state_type = edge_msgs::msg::EdgeState::TRACKING_ERROR;
    tracking_error_msg_.state_value.assign(12, 0.0);

    // Initialize ROS messages for robot state
    robot_state_msg_.state_type = edge_msgs::msg::EdgeState::ROBOT_STATE;
    robot_state_msg_.state_value.assign(12, 0.0);
}

void FSMTrotting::enter()
{
    // pcd_ = estimator_->getPosition();
    // pcd_(2) = -estimator_->getFeetPosInBodyFrame()(2, 0);
    pcd_.setZero();  // Reset x,y to zero at the beginning of trotting
    pcd_(2) = 0.3; // Set a default height 0.3m at the beginning
    std::cout << "Enter Trotting State. Pcd initialized to: " << pcd_.transpose() << std::endl;

    v_cmd_body_.setZero();
    yaw_cmd_ = estimator_->getYaw();
    Rd = rotz(yaw_cmd_);
    w_cmd_global_.setZero();

    ctrl_interfaces_.control_inputs_.command = 0;
    ext_cmd_latched_ = false;
    gait_generator_.restart();
    
    // Reset step counter for episode tracking
    step_count_ = 0;
}

void FSMTrotting::run(const rclcpp::Time&/*time*/, const rclcpp::Duration&/*period*/, 
                        edge_msgs::msg::EdgeAction& latest_action,
                        rclcpp::Publisher<edge_msgs::msg::EdgeState>::SharedPtr tracking_error_tea_pub_,
                        rclcpp::Publisher<edge_msgs::msg::EdgeState>::SharedPtr edge_state_pub,
                        double cmd_vx, double cmd_vy, double cmd_wz, double goalX, double goalY)
{
    // Increment step counter
    ++step_count_;
    
    // Update teacher activation status
    pos_body_ = estimator_->getPosition();
    vel_body_ = estimator_->getVelocity();

    // Robot position in global frame (x, y, yaw)
    robot_pos_(0) = pos_body_(0);
    robot_pos_(1) = pos_body_(1);
    robot_pos_(2) = estimator_->getYaw();

    // Goal position in global frame (x, y)
    nav_goal_(0) = goalX;
    nav_goal_(1) = goalY;

    // std::cout << "Body pos: " << pos_body_.transpose() << std::endl;
    // std::cout << "Body vel: " << vel_body_.transpose() << std::endl;

    B2G_RotMat = estimator_->getRotation();
    G2B_RotMat = B2G_RotMat.transpose();

    getUserCmd(cmd_vx, cmd_vy, cmd_wz);
    calcCmd();

    gait_generator_.setGait(vel_cmd_global_.segment(0, 2), w_cmd_global_(2), gait_height_);
    gait_generator_.generate(pos_feet_global_goal_, vel_feet_global_goal_);

    if (checkStepOrNot())
    {
        phase_scheduler_->status_ = PhaseStatus::WAVE_ALL;
    }
    else
    {
        phase_scheduler_->status_ = PhaseStatus::STANCE_ALL;
    }

    calcTau(edge_state_pub, tracking_error_tea_pub_, latest_action);
    calcQQd();
    calcGain();
}

void FSMTrotting::exit()
{
    phase_scheduler_->status_ = PhaseStatus::SWING_ALL;
}

FSMStateName FSMTrotting::checkChange()
{
    if (is_reset_)
    {
        is_reset_ = false; // Reset the flag
        return FSMStateName::FIXEDSTAND;
    }

    // if (launch_after_reset_)
    // {
    //     launch_after_reset_ = false; // Reset the flag
    //     return FSMStateName::TROTTING;
    // }

    switch (ctrl_interfaces_.control_inputs_.command)
    {
    case 1:
        return FSMStateName::PASSIVE;
    case 2:
        return FSMStateName::FIXEDSTAND;
    default:
        return FSMStateName::TROTTING;
    }
}

void FSMTrotting::getUserCmd(double cmd_vx, double cmd_vy, double cmd_wz)
{
    // /* Movement */
    // v_cmd_body_(0) = invNormalize(ctrl_interfaces_.control_inputs_.ly, vx_limit_(0), vx_limit_(1));
    // v_cmd_body_(1) = -invNormalize(ctrl_interfaces_.control_inputs_.lx, vy_limit_(0), vy_limit_(1));
    // v_cmd_body_(2) = 0;

    // /* Turning */
    // d_yaw_cmd_ = -invNormalize(ctrl_interfaces_.control_inputs_.rx, wz_limit_(0), wz_limit_(1));
    // d_yaw_cmd_ = 0.9 * d_yaw_cmd_past_ + (1 - 0.9) * d_yaw_cmd_;
    // d_yaw_cmd_past_ = d_yaw_cmd_;
    // Pick the command source. Testing |cmd| > 1e-4 per axis (as this used to do)
    // conflates "no external command" with "the planner is commanding a stop":
    // as the policy decelerates onto the goal its command crosses that epsilon,
    // control falls back to the keyboard, and with the sticks centred
    // invNormalize(0, 0.05, 0.5) feeds 0.275 m/s straight back in -- the robot
    // accelerated exactly when it was trying to stop. Latch the external source
    // instead, so its zero is honoured as a stop.
    constexpr double kExtCmdEps = 1e-4;
    if (fabs(cmd_vx) > kExtCmdEps || fabs(cmd_vy) > kExtCmdEps || fabs(cmd_wz) > kExtCmdEps)
    {
        ext_cmd_latched_ = true;
    }

    double vx_in, vy_in, wz_in;
    if (ext_cmd_latched_)
    {
        // vx_limit_(0) is the minimum useful *trot* speed (0.05 m/s), not a floor
        // that a stop command has to respect -- clamping to it would make a
        // commanded halt impossible. Allow zero through, still no reverse.
        vx_in = std::clamp(cmd_vx, std::min(0.0, vx_limit_(0)), vx_limit_(1));
        vy_in = std::clamp(cmd_vy, vy_limit_(0), vy_limit_(1));
        wz_in = std::clamp(cmd_wz, wz_limit_(0), wz_limit_(1));
    }
    else
    {
        vx_in =  invNormalize(ctrl_interfaces_.control_inputs_.ly, vx_limit_(0), vx_limit_(1));
        vy_in = -invNormalize(ctrl_interfaces_.control_inputs_.lx, vy_limit_(0), vy_limit_(1));
        wz_in = -invNormalize(ctrl_interfaces_.control_inputs_.rx, wz_limit_(0), wz_limit_(1));
    }
    v_cmd_body_ << vx_in, vy_in, 0.0;

    d_yaw_cmd_ = wz_in;

    // Smoothing -- vx vy
    double alpha_vx = 0.9;
    double alpha_vy = 0.9;
    v_cmd_body_(0) = alpha_vx * v_cmd_body_past_(0) + (1 - alpha_vx) * v_cmd_body_(0);
    v_cmd_body_(1) = alpha_vy * v_cmd_body_past_(1) + (1 - alpha_vy) * v_cmd_body_(1);
    v_cmd_body_past_ = v_cmd_body_;

    // Smoothing -- wz
    double alpha_wz = 0.9;
    d_yaw_cmd_ = alpha_wz * d_yaw_cmd_past_ + (1 - alpha_wz) * d_yaw_cmd_;
    d_yaw_cmd_past_ = d_yaw_cmd_;
}

void FSMTrotting::calcCmd()
{
    /* Movement */
    vel_cmd_global_ = B2G_RotMat * v_cmd_body_;

    vel_cmd_global_(0) =
        saturation(vel_cmd_global_(0), Vec2(vel_body_(0) - 0.2, vel_body_(0) + 0.2));
    vel_cmd_global_(1) =
        saturation(vel_cmd_global_(1), Vec2(vel_body_(1) - 0.2, vel_body_(1) + 0.2));

    pcd_(0) = saturation(pcd_(0) + vel_cmd_global_(0) * dt_,
                         Vec2(pos_body_(0) - 0.05, pos_body_(0) + 0.05));
    pcd_(1) = saturation(pcd_(1) + vel_cmd_global_(1) * dt_,
                         Vec2(pos_body_(1) - 0.05, pos_body_(1) + 0.05));

    vel_cmd_global_(2) = 0;

    /* Turning */
    // [本地修改] 给偏航积分加防饱和，与上面 pcd_ 的处理对称。
    // 原来 yaw_cmd_ 是纯开环积分：enter() 里取一次实际 yaw，之后只累加指令偏航率，
    // 再不参考实际姿态。机器狗只要跟不上指令偏航率（打滑、动力学延迟——必然发生），
    // 误差就无界累积，下游 kp_w_ * rotMatToExp(...) 把它放大成顶死饱和的偏航力矩。
    // 实测教师输出 accel[5] 均值 -39 rad/s^2（饱和限 ±20），即为此症。
    // 注意先把角度差绕回 [-pi, pi]，否则 yaw 越过 ±pi 翻转时钳位会失控。
    const double yaw_now = estimator_->getYaw();
    double yaw_err = yaw_cmd_ + d_yaw_cmd_ * dt_ - yaw_now;
    while (yaw_err >  M_PI) yaw_err -= 2.0 * M_PI;
    while (yaw_err < -M_PI) yaw_err += 2.0 * M_PI;
    yaw_err = std::clamp(yaw_err, -yaw_leash_, yaw_leash_);
    yaw_cmd_ = yaw_now + yaw_err;
    Rd = rotz(yaw_cmd_);
    w_cmd_global_(2) = d_yaw_cmd_;
}

void FSMTrotting::calcTau(rclcpp::Publisher<edge_msgs::msg::EdgeState>::SharedPtr robot_state_pub,
    rclcpp::Publisher<edge_msgs::msg::EdgeState>::SharedPtr tracking_error_tea_pub_,
    const edge_msgs::msg::EdgeAction& latest_action)
{
    pos_error_ = pcd_ - pos_body_;

    // [本地修改] 高度分量换基准。pcd_(2) 在 enter() 里被写成绝对值 0.3 且此后从不更新，
    // 而 pos_body_(2) 是 leg_odom 系的 z，原点在估计器初始化那一刻的机身位置 ——
    // 两者根本不同基准，相减得到的是一个恒定偏置而不是高度误差
    // （实测 pos_body_(2)=0.034 m，即 pos_error_(2) 恒为 0.27 m）。
    // 这个常值误差会经 tracking_error[2] 污染 phy_teacher，也会经 Kpp 污染本地 PD。
    // 改成"机身相对最低足端的高度"，这才是 0.3 这个数想表达的量。
    // 取 minCoeff：足端 z 在机身系里为负，最低的那只即着地腿，摆动腿不会干扰。
    pos_error_(2) = pcd_(2) + estimator_->getFeetPosInBodyFrame().row(2).minCoeff();
    vel_error_ = vel_cmd_global_ - vel_body_;

    // [本地修改] 这四行原本每个控制周期都执行（400Hz -> 1600 行/秒）。std::cout 是
    // 同步 I/O，写的是 launch 捕获 stdout 的管道；管道一满 write() 就阻塞，直接卡在
    // 实时控制环里。实测站立状态下日志已有 59 KB/s，2% 的周期延迟 >5ms，最坏 11.4ms
    // （标称 2.5ms 的 4.5 倍）。需要调试时取消注释，或改用 RCLCPP_*_THROTTLE。
    // std::cout << "pos_body_: " << pos_body_.transpose() << std::endl;
    // std::cout << "pcd_: " << pcd_.transpose() << std::endl;
    // std::cout << "vel_cmd_global: " << vel_cmd_global_.transpose() << std::endl;
    // std::cout << "vel_body: " << vel_body_.transpose() << std::endl;

    // Vec3 dd_pcd = Kpp * pos_error_ + Kdp * vel_error_;
    // Vec3 d_wbd = kp_w_ * rotMatToExp(Rd * G2B_RotMat) +
        // Kd_w_ * (w_cmd_global_ - estimator_->getGyroGlobal());

    // Get current robot state and tracking error
    curr_robot_state_ = getRobotState(true);
    curr_tracking_error_tea_ = getTrackingError();
    auto curr_tracking_error_stu_ = getDRLStudentGoalState() - curr_robot_state_;

    // Publish tracking error
    tracking_error_msg_.header.stamp = now();
    for (int i = 0; i < 12; ++i) {
        tracking_error_msg_.state_value[i] = curr_tracking_error_tea_[i];
    }
    tracking_error_tea_pub_->publish(tracking_error_msg_);

    // Publish robot state
    robot_state_msg_.header.stamp = now();
    for (int i=0; i < 12; ++i){
        robot_state_msg_.state_value[i] = curr_robot_state_[i];
    }
    robot_state_pub->publish(robot_state_msg_);

    // std::cout << "rotMatToRPY(Rd * G2B_RotMat)(0): " << rotMatToRPY(Rd * G2B_RotMat)(0) << std::endl;
    // std::cout << "rotMatToExp(Rd * G2B_RotMat)(0): " << rotMatToExp(Rd * G2B_RotMat)(0) << std::endl;

    // std::cout << "pos error: " << pos_error_.transpose() << std::endl;
    // std::cout << "vel error: " << vel_error_.transpose() << std::endl;
    // std::cout << "angle error: " << rotMatToExp(Rd * G2B_RotMat).transpose() << std::endl;
    // std::cout << "angular vel error: " << (w_cmd_global_ - estimator_->getGyroGlobal()).transpose() << std::endl;

    Vec6 accel = Vec6::Zero();
    const auto action_size = latest_action.action_value.size();
    const auto copy_size = std::min<size_t>(6, action_size);
    for (size_t i = 0; i < copy_size; ++i) {
        accel(static_cast<int>(i)) = latest_action.action_value[i];
    }

    accel(0) = saturation(accel(0), Vec2(-3, 3));
    accel(1) = saturation(accel(1), Vec2(-3, 3));
    accel(2) = saturation(accel(2), Vec2(-5, 5));
    // accel(3) = saturation(accel(3), Vec2(-10, 10));  // Wrong limit, should be (-40, 40)
    // accel(4) = saturation(accel(4), Vec2(-10, 10));  // Wrong limit, should be (-40, 40)
    // accel(5) = saturation(accel(5), Vec2(-20, 20));  // Wrong limit, should be (-10, 10)
    accel(3) = saturation(accel(3), Vec2(-40, 40));
    accel(4) = saturation(accel(4), Vec2(-40, 40));
    accel(5) = saturation(accel(5), Vec2(-20, 20));

    // std::cout << "acc command: " << accel.transpose() << std::endl;

    edge_msgs::msg::EdgeAction latest_action_saturated = EigenToEdgeActionMsg(
        accel,
        edge_msgs::msg::EdgeAction::BASE_ACCEL,
        latest_action.action_mode
    );

    // [本地修改] 机身镇定的来源。
    // 原来只有 accel 这一条路，即跨 DDS 的 phy_teacher -> trigger -> rl_runtime 链路。
    // 该链路在系统判定 SAFE 时根本不发布（trigger 选 STUDENT 分支而 drl_student 从未
    // 启动，直接 return），于是 latest_action.action_value 为空、accel 恒为零向量，
    // dd_pcd 和 d_wbd 都是 0 —— BalanceCtrl 只做重力补偿，机身姿态完全开环。
    // 这就是行走时机身发飘的直接原因（实测一次 178 秒记录里只有 1.2 秒有数据）。
    // 现在默认走环内本地 PD，增益是原作者在构造函数里写好的那一组；
    // 把 use_local_balance_pd 设为 false 即可切回原来的外部链路。
    Vec3 dd_pcd, d_wbd;
    if (use_local_pd_)
    {
        dd_pcd = Kpp * pos_error_ + Kdp * vel_error_;
        d_wbd  = kp_w_ * rotMatToExp(Rd * G2B_RotMat)
               + Kd_w_ * (w_cmd_global_ - estimator_->getGyroGlobal());
    }
    else
    {
        dd_pcd = accel.segment(0, 3);
        d_wbd  = accel.segment(3, 3);
    }
    // 下面已有的 saturation 对两条来源一视同仁，无需另加限幅。

    // Vec6 all_pos_error;
    // Vec6 all_vel_error;
    // all_pos_error << pos_error_, rotMatToExp(Rd * G2B_RotMat);
    // all_vel_error << vel_error_, (w_cmd_global_ - estimator_->getGyroGlobal());
    // // std::cout << "all pos error: " << all_pos_error.transpose() << std::endl;
    // // std::cout << "all vel error: " << all_vel_error.transpose() << std::endl;
    // Eigen::Matrix<double , 6, 6> kp;
    // Eigen::Matrix<double , 6, 6> kd;
    // kp << 1, 0, 0, 0, 0, 0,
    //       0, 1, 0, 0, 0, 0,
    //       0, 0, 200, 0, 0, 0,
    //       0, 0, 0, 200, 0, 0,
    //       0, 0, 0, 0, 200, 0,
    //       0, 0, 0, 0, 0, 200;

    // kd << 20, 0, 0, 0, 0, 0,
    //       0, 20, 0, 0, 0, 0,
    //       0, 0, 20, 0, 0, 0,
    //       0, 0, 0, 70, 0, 0,
    //       0, 0, 0, 0, 70, 0,
    //       0, 0, 0, 0, 0, 70;
    // auto acc = kp * all_pos_error + kd * all_vel_error;
    
    // Vec3 dd_pcd = acc.segment(0, 3);
    // Vec3 d_wbd = acc.segment(3, 3);

    // Vec3 dd_pcd = Kpp * pos_error_ + Kdp * vel_error_;
    // Vec3 d_wbd = kp_w_ * rotMatToExp(Rd * G2B_RotMat) +
    //     Kd_w_ * (w_cmd_global_ - estimator_->getGyroGlobal());

    dd_pcd(0) = saturation(dd_pcd(0), Vec2(-3, 3));
    dd_pcd(1) = saturation(dd_pcd(1), Vec2(-3, 3));
    dd_pcd(2) = saturation(dd_pcd(2), Vec2(-5, 5));

    d_wbd(0) = saturation(d_wbd(0), Vec2(-40, 40));
    d_wbd(1) = saturation(d_wbd(1), Vec2(-40, 40));
    d_wbd(2) = saturation(d_wbd(2), Vec2(-10, 10));

    // std::cout << "actual dd_pcd: " << dd_pcd.transpose() << std::endl;
    // std::cout << "actual d_wbd: " << d_wbd.transpose() << std::endl;

    Vec34 force_feet_global =
        -balance_ctrl_->calcGRF(dd_pcd, d_wbd, B2G_RotMat, estimator_->getFeetPosInBodyFrame(), phase_scheduler_->contact_);

    Vec34 pos_feet_global = estimator_->getFeetPos();
    Vec34 vel_feet_global = estimator_->getFeetVel();

    for (int i(0); i < 4; ++i)
    {
        if (phase_scheduler_->contact_(i) == 0)
        {
            force_feet_global.col(i) =
                Kp_swing_ * (pos_feet_global_goal_.col(i) - pos_feet_global.col(i)) +
                Kd_swing_ * (vel_feet_global_goal_.col(i) - vel_feet_global.col(i));
        }
    }

    Vec34 force_feet_body_ = G2B_RotMat * force_feet_global;

    // Send torque command
    for (int i = 0; i < 4; i++)
    {
        Vec3 torque = robot_model_->calcTorque(force_feet_body_.col(i), i);
        for (int j = 0; j < 3; j++)
        {
            ctrl_interfaces_.joint_torque_command_interface_[i * 3 + j].get().set_value(torque(j));
        }
    }

    // update last action and tracking error and robot state
    last_edge_action_ = latest_action_saturated;
    last_tracking_error_tea_ = curr_tracking_error_tea_;  
    last_robot_state_ = curr_robot_state_;
}

void FSMTrotting::calcQQd()
{
    const Vec34 pos_feet_body = robot_model_->getFeetPosInBodyFrame();

    Vec34 pos_feet_target, vel_feet_target;
    for (int i(0); i < 4; ++i)
    {
        pos_feet_target.col(i) = G2B_RotMat * (pos_feet_global_goal_.col(i) - pos_body_);
        vel_feet_target.col(i) = G2B_RotMat * (vel_feet_global_goal_.col(i) - vel_body_);
    }

    Vec12 q_goal = robot_model_->getJointPos(pos_feet_target, FrameType::BODY);
    Vec12 qd_goal = robot_model_->getJointVel(pos_feet_body, vel_feet_target, FrameType::BODY);

    for (int i = 0; i < 12; i++)
    {
        ctrl_interfaces_.joint_position_command_interface_[i].get().set_value(q_goal(i));
        ctrl_interfaces_.joint_velocity_command_interface_[i].get().set_value(qd_goal(i));
    }
}

void FSMTrotting::calcGain() const
{
    for (int i(0); i < 4; ++i)
    {
        if (phase_scheduler_->contact_(i) == 0)
        {
            // swing gain
            for (int j = 0; j < 3; j++)
            {
                ctrl_interfaces_.joint_kp_command_interface_[i * 3 + j].get().set_value(swing_gains_[0]);
                ctrl_interfaces_.joint_kd_command_interface_[i * 3 + j].get().set_value(swing_gains_[1]);
            }
        }
        else
        {
            // stable gain
            for (int j = 0; j < 3; j++)
            {
                ctrl_interfaces_.joint_kp_command_interface_[i * 3 + j].get().set_value(stable_gains_[0]);
                ctrl_interfaces_.joint_kd_command_interface_[i * 3 + j].get().set_value(stable_gains_[1]);
            }
        }
    }
}

bool FSMTrotting::checkStepOrNot()
{
    // if (fabs(v_cmd_body_(0)) > 0.03 || fabs(v_cmd_body_(1)) > 0.03 ||
    //     fabs(pos_error_(0)) > 0.08 || fabs(pos_error_(1)) > 0.08 ||
    //     fabs(vel_error_(0)) > 0.05 || fabs(vel_error_(1)) > 0.05 ||
    //     fabs(d_yaw_cmd_) > 0.20)
    // {
    //     return true;
    // }
    // return false;
    return true;
}



Vector12d FSMTrotting::getTrackingError() const
{
    Vector12d tracking_error = Vector12d::Zero();
    
    // Distance to goal
    Vec2 dist_to_goal = nav_goal_ - pos_body_.head<2>();

    // X and Y position error to goal
    // tracking_error[0] = dist_to_goal(0);
    // tracking_error[1] = dist_to_goal(1);

    // Height Position error  
    tracking_error[2] = pos_error_(2);

    // Orientation error
    tracking_error.segment<3>(3) = rotMatToExp(Rd * G2B_RotMat);

    // Linear velocity error
    tracking_error.segment<3>(6) = vel_error_;

    // Angular velocity error
    tracking_error.segment<3>(9) = w_cmd_global_ - estimator_->getGyroGlobal();

    return tracking_error;
}


edge_msgs::msg::EdgeAction FSMTrotting::getLearnedDRLAction() const
{
    /**
     * Calculate DRL action by removing residual effect from teacher action
     * learned_drl_action = terminal_action - residual_gain @ tracking_error
     */
    auto last_residual_action = ctrl_component_.residual_gain_ * last_tracking_error_tea_;
    auto last_terminal_action = EdgeActionMsgToEigen(last_edge_action_);
    if (last_terminal_action.size() != 6) {
        last_terminal_action = Vector6d::Zero();
    }
    auto learned_drl_action = last_terminal_action - last_residual_action;

    // std::cout << "last residual action: " << last_residual_action.transpose() << std::endl;
    // std::cout << "learned drl action: " << learned_drl_action.transpose() << std::endl;

    return EigenToEdgeActionMsg(
        learned_drl_action,
        edge_msgs::msg::EdgeAction::BASE_ACCEL,
        last_edge_action_.action_mode);
}

Vector12d FSMTrotting::getRobotState(bool is_global/* = true*/) const
{
    /**
    * Get the robot state in either global frame or body frame
    */
    Vector12d robotState = Vector12d::Zero();

    // Robot Global Position state (x, y, z)
    robotState.segment<3>(0) = pos_body_;

    if (!is_global) {
        // Convert to body frame (px and py are zero in body frame)
        robotState[0] = 0.0;
        robotState[1] = 0.0;
    }

    // Robot Orientation state
    robotState.segment<3>(3) = rotMatToExp(estimator_->getRotation());

    // Linear velocity state
    robotState.segment<3>(6) = vel_body_;

    // Angular velocity state
    robotState.segment<3>(9) = estimator_->getGyroGlobal();

    return robotState;
}

Vector12d FSMTrotting::getDRLStudentGoalState() const
{
    Vector12d robotState = Vector12d::Zero();

    robotState[0] = nav_goal_(0);
    robotState[1] = nav_goal_(1);

    // Robot Position state
    robotState[2] = 0.3; // Set a default height 0.3m

    // Robot Orientation state
    // robotState.segment<3>(3) = Eigen::Vector3d::Zero();

    // Linear velocity state
    // robotState.segment<3>(6) = Eigen::Vector3d(0.8, 0.0, 0.0);
    // robotState[6] = 0.7;

    // Angular velocity state
    // robotState.segment<3>(9) = Eigen::Vector3d::Zero();

    return robotState;
}