/*
 * Copyright (c) 2025-2026 Yihao Cai - All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "control/CtrlComponent.h"
#include "FSM/FSMFixedStand.hpp"
#include "FSM/FSMFreeStand.hpp"
#include "FSM/FSMBalanceTest.hpp"
#include "FSM/FSMTrotting.hpp"

#include "edge_msgs/msg/edge_state.hpp"
#include "edge_msgs/msg/edge_action.hpp"
#include "edge_msgs/msg/vec12_msg.hpp"
#include "edge_msgs/msg/transition.hpp"

#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <controller_interface/controller_interface.hpp>
#include <controller_common/FSM/FSMState.h>
#include <controller_common/FSM/FSMPassive.h>
#include <controller_common/FSM/FSMFixedDown.h>
#include <controller_common/common/enumClass.h>
#include <controller_common/common/mathTypes.h>

#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>


namespace edge::control::quadruped
{
    struct CtrlComponent;
    class PhaseScheduler;
    class BalanceCtrl;
    class QuadrupedModel;
    class Estimator;
}

namespace edge::control::quadruped
{
    struct FSMStateList
    {
        std::shared_ptr<FSMState> invalid;
        std::shared_ptr<FSMPassive> passive;
        std::shared_ptr<FSMFixedDown> fixedDown;
        std::shared_ptr<FSMFixedStand> fixedStand;
        std::shared_ptr<FSMFreeStand> freeStand;
        std::shared_ptr<FSMTrotting> trotting;
        std::shared_ptr<FSMBalanceTest> balanceTest;
    };

    class EdgeQuadrupedController final : public controller_interface::ControllerInterface
    {
    public: 

        controller_interface::InterfaceConfiguration state_interface_configuration() const override;
        controller_interface::InterfaceConfiguration command_interface_configuration() const override;

        controller_interface::return_type update(
            const rclcpp::Time& time, const rclcpp::Duration& period) override;

        controller_interface::CallbackReturn on_init() override;

        controller_interface::CallbackReturn on_configure(
            const rclcpp_lifecycle::State& previous_state) override;
 
        controller_interface::CallbackReturn on_activate(
            const rclcpp_lifecycle::State& previous_state) override;

        controller_interface::CallbackReturn on_deactivate(
            const rclcpp_lifecycle::State& previous_state) override;

        controller_interface::CallbackReturn on_cleanup(
            const rclcpp_lifecycle::State& previous_state) override;

        controller_interface::CallbackReturn on_error(
            const rclcpp_lifecycle::State& previous_state) override;

        controller_interface::CallbackReturn on_shutdown(
            const rclcpp_lifecycle::State& previous_state) override;

        /**
         * Initialize robot model with auto-generated URDF path
         */
        void initializeRobotModel();

        /**
         * @brief Setup reset service
         */
        void setupResetService();

        /**
         * @brief Setup launch service
         */
        void setupLaunchService();

        CtrlComponent ctrl_component_;
        CtrlInterfaces ctrl_interfaces_;

    protected:
        std::vector<std::string> joint_names_;
        std::vector<std::string> command_interface_types_;
        std::vector<std::string> state_interface_types_;

        // Robot parameters
        std::string robot_name_;

        std::string imu_name_;
        std::string command_prefix_;
        std::string odom_frame_id_;
        std::string base_frame_id_;
        std::vector<std::string> imu_interface_types_;

        // FR FL RR RL
        std::vector<double> stand_pos_ = {
            0.0, 0.67, -1.3,
            0.0, 0.67, -1.3,
            0.0, 0.67, -1.3,
            0.0, 0.67, -1.3
        };

        std::vector<double> down_pos_ = {
            0.0, 1.3, -2.4,
            0.0, 1.3, -2.4,
            0.0, 1.3, -2.4,
            0.0, 1.3, -2.4
        };

        // stand_gains_[0] = kp, stand_gains_[1] = kd
        std::vector<double> stand_gains_ = {80.0, 3.5};

        // swing_gains_[0] = kp, swing_gains_[1] = kd
        std::vector<double> swing_gains_ = {3.0, 2.0};

        // stable_gains_[0] = kp, stable_gains_[1] = kd
        // std::vector<double> stable_gains_ = {0.8, 0.8};
        std::vector<double> stable_gains_ = {100, 0.8};

        Vec2 vx_limit_ = Vec2(-1.0, 1.0);  // max forward/backward speed limit
        Vec2 vy_limit_ = Vec2(-1.0, 1.0);  // max lateral speed limit
        Vec2 wz_limit_ = Vec2(-1.0, 1.0);  // max yaw rate limit

        // Gait parameters
        double gait_height_ = 0.08;

        rclcpp::Subscription<control_input_msgs::msg::Inputs>::SharedPtr control_input_sub_;

        std::unordered_map<
            std::string, std::vector<std::reference_wrapper<hardware_interface::LoanedCommandInterface>>*>
        command_interface_map_ = {
            {"effort", &ctrl_interfaces_.joint_torque_command_interface_},
            {"position", &ctrl_interfaces_.joint_position_command_interface_},
            {"velocity", &ctrl_interfaces_.joint_velocity_command_interface_},
            {"kp", &ctrl_interfaces_.joint_kp_command_interface_},
            {"kd", &ctrl_interfaces_.joint_kd_command_interface_}
        };

        std::string topic_ns_;  // Topic namespace
        std::string state_name_;
        FSMStateList state_list_;
        double update_frequency_;
        FSMMode mode_ = FSMMode::NORMAL;
        FSMStateName next_state_name_ = FSMStateName::INVALID;
        std::shared_ptr<FSMState> current_state_;
        std::shared_ptr<FSMState> next_state_;

        std::chrono::time_point<std::chrono::steady_clock> last_update_time_;

        std::shared_ptr<FSMState> getNextState(FSMStateName stateName) const;

        std::unordered_map<
            std::string, std::vector<std::reference_wrapper<hardware_interface::LoanedStateInterface>>*>
        state_interface_map_ = {
            {"position", &ctrl_interfaces_.joint_position_state_interface_},
            {"effort", &ctrl_interfaces_.joint_effort_state_interface_},
            {"velocity", &ctrl_interfaces_.joint_velocity_state_interface_}
        };

        // For /cmd_vel subscription
        double cmd_vx_ = 0.0;
        double cmd_vy_ = 0.0;
        double cmd_wz_ = 0.0;

        // void cmdVelCallback(const geometry_msgs::msg::TwistStamped::SharedPtr msg)
        // {
        //     cmd_vx_ = msg->twist.linear.x;
        //     cmd_vy_ = msg->twist.linear.y;
        //     cmd_wz_ = msg->twist.angular.z;
        //     std::cout << "*************************************************************" << std::endl;
        //     std::cout << "[StateTrotting] Received /cmd_vel: "
        //             << "vx = " << cmd_vx_ << ", "
        //             << "vy = " << cmd_vy_ << ", "
        //             << "wz = " << cmd_wz_ << std::endl;
        // }

        // For publishers 
        rclcpp::Publisher<edge_msgs::msg::EdgeState>::SharedPtr tracking_error_tea_pub_;
        rclcpp::Publisher<edge_msgs::msg::EdgeState>::SharedPtr tracking_error_stu_pub_;
        rclcpp::Publisher<edge_msgs::msg::Transition>::SharedPtr transition_pub_;
        rclcpp::Publisher<edge_msgs::msg::EdgeState>::SharedPtr robot_state_pub_;
        rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr is_task_done_pub_;
        
        // For subscribers
        rclcpp::Subscription<edge_msgs::msg::EdgeAction>::SharedPtr action_sub_;
        rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr goal_sub_;
        rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr cmd_vel_sub_;
        rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;

        // For reset service
        rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_service_;
        rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr launch_service_;
        int last_reset_timestep_;  // last reset time for debounce protection
        double reset_cooldown_{1.0};  // reset cooldown duration in seconds

        double latest_goal_x_ = 0.0;
        double latest_goal_y_ = 0.0;

        // ---- Arrival stop -------------------------------------------------
        // Position used for the goal-distance test. Deliberately taken from the
        // SLAM odometry topic (point_lio, map frame) rather than from
        // ctrl_component_.estimator_, whose output lives in the leg_odom frame
        // and drifts: /goal_point is published in the map frame, so mixing the
        // two would compare coordinates from different frames.
        double odom_x_ = 0.0;
        double odom_y_ = 0.0;
        bool has_odom_ = false;

        // 机身镇定来源与偏航防饱和，见 FSMTrotting。
        bool use_local_balance_pd_ = true;
        double yaw_leash_ = 0.10;       // [rad]

        double arrive_radius_ = 0.5;    // [m] inside this, start stopping
        double resume_radius_ = 1.0;    // [m] outside this, allow driving again
        double decel_time_ = 1.5;       // [s] time to ramp full speed down to zero

        // 1.0 = command passed through untouched, 0.0 = fully stopped.
        double approach_scale_ = 1.0;
        bool arrived_latched_ = false;

        // Guards the arrival stop against firing before a goal was ever set:
        // latest_goal_* default to (0,0) and the odometry origin is also (0,0),
        // so without this the robot would consider itself "arrived" at startup
        // and refuse to move at all, including under manual keyboard control.
        bool has_goal_ = false;

        /**
         * Scale a velocity command toward zero as the goal is approached.
         * Rate-limited so the robot decelerates over decel_time_ rather than
         * stopping in one control cycle; latched with hysteresis so odometry
         * jitter around the threshold cannot make it lurch forward again.
         */
        void applyArrivalStop(const rclcpp::Duration& period,
                              double& vx, double& vy, double& wz);

        bool is_fall_ = false;

        edge_msgs::msg::EdgeAction latest_edge_action_;

        Mat6 latest_kp_ = (Mat6() << 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0,
                          0, 0, 14, 0, 0, 0,
                          0, 0, 0, 9, 0, 2,
                          0, 0, 0, 0, 9, -0.5,
                          0, 0, 0, -1.3, 0.39, 15).finished() * 5;
                          
        Mat6 latest_kd_ = (Mat6() << 3.4239, 0, 0, 0, 0, 0,
                          0, 2.755, 0, 0, 0, 0,
                          0, 0, 5.8043, 0, 0, 0,
                          0, 0, 0, 5.4217, 0, 0.0482,
                          0, 0, 0, 0, 5.16, -0.02,
                          0, 0, 0, -0.098, -0.0121, 5.3879).finished() * 5;
        bool gain_update_{false};
    };
} // namespace edge::control::quadruped
