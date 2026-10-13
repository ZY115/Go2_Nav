/*
 * Copyright (c) 2025-2026 Yihao Cai - All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 */

 #pragma once

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>

#include "edge_msgs/msg/edge_state.hpp"
#include "edge_msgs/msg/edge_action.hpp"
#include "edge_msgs/msg/vec12_msg.hpp"
#include "edge_msgs/msg/transition.hpp"
#include "edge_ros_interfaces/msg_conversion.hpp"
#include "controller_common/FSM/FSMState.h"

#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include "edge_quadruped_controller/utils/math/mathTypes.h"
#include "edge_quadruped_controller/gait/GaitGenerator.h"

// Forward declarations for ROS2 types
struct CtrlInterfaces;

namespace edge::control::quadruped
{
    struct CtrlComponent;
    class PhaseScheduler;
    class BalanceCtrl;
    class QuadrupedModel;
    class Estimator;
}

namespace edge::rl_runtime
{
    class RewardComposer;
}


using namespace edge::control::quadruped;
using namespace edge::ros_interfaces;
using namespace edge::rl_runtime;

class FSMTrotting final : public FSMState {

public:
    explicit FSMTrotting(CtrlInterfaces &ctrl_interfaces,
                           CtrlComponent &ctrl_component,
                           const std::vector<double>& swing_gains,
                           const std::vector<double>& stable_gains,
                           Vec2 vx_limit, Vec2 vy_limit, Vec2 wz_limit,
                           double gait_height);

    void enter() override;

    void run(const rclcpp::Time& /*time*/, const rclcpp::Duration& /*period*/) override {};
    void run(const rclcpp::Time &time, const rclcpp::Duration &period, 
             edge_msgs::msg::EdgeAction &latest_action_, 
             rclcpp::Publisher<edge_msgs::msg::EdgeState>::SharedPtr tracking_error_tea_pub_,
             rclcpp::Publisher<edge_msgs::msg::EdgeState>::SharedPtr robot_state_pub_,
             double cmd_vx, double cmd_vy, double cmd_wz, double goalX, double goalY);

    void exit() override;

    GaitGenerator& getGaitGenerator() { return gait_generator_; }
    const GaitGenerator& getGaitGenerator() const { return gait_generator_; }

    FSMStateName checkChange() override;

    void setLastResetTime(const rclcpp::Time& time) {
        last_reset_time_ = time;
    }

    /**
     * 机身镇定来源：true = 环内本地 PD（默认），false = 外部 phy_teacher 链路。
     * 外部链路在系统 SAFE 时不发布，会让 accel 恒为零、机身姿态开环。
     */
    void setUseLocalBalancePD(bool v) { use_local_pd_ = v; }

    /** yaw_cmd_ 相对实际 yaw 的最大超前量 [rad]，防止偏航积分无界累积。 */
    void setYawLeash(double v) { yaw_leash_ = v; }

private:

    // void cmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg);

    void getUserCmd(double, double, double);

    void calcCmd();

    /**
    * Calculate the torque command
    */
    void calcTau(rclcpp::Publisher<edge_msgs::msg::EdgeState>::SharedPtr robot_state_pub,
                 rclcpp::Publisher<edge_msgs::msg::EdgeState>::SharedPtr tracking_error_tea_pub,
                 const edge_msgs::msg::EdgeAction& latest_action_);

    /**
    * Calculate the joint space velocity and acceleration
    */
    void calcQQd();

    /**
    * Calculate the PD gain for the joints
    */
    void calcGain() const;

    /**
     * Check whether the robot should take a step or not
     * @return
     */
    bool checkStepOrNot();
    
    std::shared_ptr<Estimator> &estimator_;
    std::shared_ptr<QuadrupedModel> &robot_model_;
    std::shared_ptr<BalanceCtrl> &balance_ctrl_;
    std::shared_ptr<PhaseScheduler> &phase_scheduler_;
    CtrlComponent &ctrl_component_;

    GaitGenerator gait_generator_;

    // Robot State
    Vec3 pos_body_, vel_body_;
    RotMat B2G_RotMat, G2B_RotMat;

    // Robot command
    Vec3 pcd_;
    Vec3 vel_cmd_global_, v_cmd_body_, v_cmd_body_past_;
    double dt_;
    double yaw_cmd_{}, d_yaw_cmd_{}, d_yaw_cmd_past_{};
    Vec3 w_cmd_global_;
    Vec34 pos_feet_global_goal_, vel_feet_global_goal_;
    RotMat Rd;

    // Control Parameters
    double gait_height_;
    Vec3 pos_error_, vel_error_;
    Mat3 Kpp, Kdp, Kd_w_;
    double kp_w_;
    Mat3 Kp_swing_, Kd_swing_;
    // Vec2 v_x_limit_, v_y_limit_, w_yaw_limit_;
    Vec2 vx_limit_, vy_limit_, wz_limit_;

    Vector12d last_tracking_error_tea_;
    Vector12d curr_tracking_error_tea_;

    Vector12d last_robot_state_;
    Vector12d curr_robot_state_;

    edge_msgs::msg::EdgeState tracking_error_msg_;
    edge_msgs::msg::EdgeState robot_state_msg_;

    edge_msgs::msg::EdgeAction last_edge_action_;
    edge_msgs::msg::EdgeAction curr_edge_action_;
    Vec6 last_edge_accel_;
    Vec6 curr_edge_accel_;

    // Reset time
    rclcpp::Time last_reset_time_;

    // Step counter for statistical tracking
    int step_count_{0};

    bool is_teacher_activated_{false};

    // Gain parameters
    std::vector<double> swing_gains_;
    std::vector<double> stable_gains_;

    // True once an external velocity source (RL policy / path follower) has
    // driven this state. From then on a zero command from that source means
    // "stop", instead of falling back to the keyboard sticks. Reset in enter().
    bool ext_cmd_latched_{false};

    // 机身镇定来源开关，见 setUseLocalBalancePD()。
    bool use_local_pd_{true};

    // yaw_cmd_ 的防饱和带宽 [rad]。与 pcd_ 的 ±0.05 m 位置 leash 同一思路。
    double yaw_leash_{0.10};

    // For reward calculation
    Vec2 nav_goal_;
    Vec3 robot_pos_;

    Vector12d getTrackingError() const;
    Vector12d getDRLStudentGoalState() const;
    Vector12d getRobotState(bool is_global=true) const;
    
    /**
     * Get Learned DRL action by removing residual effect from teacher action
     * @return DRL action message (terminal_action - residual_gain * tracking_error)
     */
    edge_msgs::msg::EdgeAction getLearnedDRLAction() const;

    Mat6 latest_kp_;
    Mat6 latest_kd_;
    bool gain_update_{false};
};

