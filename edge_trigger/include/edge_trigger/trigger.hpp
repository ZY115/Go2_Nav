/*
 * Copyright (c) 2025-2026 Yihao Cai - All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cmath>
#include <vector>
#include <string>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include "msg_utils.hpp"

#include "edge_msgs/msg/edge_state.hpp"
#include "edge_msgs/msg/edge_action.hpp"
#include "edge_trigger/safety_monitor.hpp"
#include "edge_msgs/srv/get_switching_stats.hpp"
#include "edge_msgs/srv/reset_switching_stats.hpp"
#include "edge_ros_interfaces/msg_conversion.hpp"

namespace edge::trigger {

#define MAX_VAL 100.0

enum class TriggerMode {
    SELF,
    EVENT,
};

inline TriggerMode TriggerModeFromString(const std::string& s)
{
    if (s == "self")  return TriggerMode::SELF;
    if (s == "event") return TriggerMode::EVENT;

    throw std::invalid_argument("Invalid trigger_mode: " + s);
}

// Trigger class
class Trigger : public rclcpp::Node
{
public:
    explicit Trigger(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:

    void updateLoop();     // Main periodic update loop for trigger
    bool checkSafety();    // Check safety for switching
    void resetStats();     // Reset switching statistics
    void updateStats(edge::utils::ActionMode);      // Update switching statistics
    void resetCachedVariables(); // Reset cached variables
    edge::utils::ActionMode selectAction();   // Select action based on current state and trigger mode

    
    // ROS2 Callbacks
    void edgeStateCallback(const edge_msgs::msg::EdgeState::SharedPtr msg);
    void studentActionCallback(const edge_msgs::msg::EdgeAction::SharedPtr msg);
    void teacherActionCallback(const edge_msgs::msg::EdgeAction::SharedPtr msg);
    void goalDistanceCallback(const std_msgs::msg::Float32::SharedPtr msg);
    void obstacleDistanceCallback(const std_msgs::msg::Float32::SharedPtr msg);
    void getStatsCallback(
        const std::shared_ptr<edge_msgs::srv::GetSwitchingStats::Request> request,
        std::shared_ptr<edge_msgs::srv::GetSwitchingStats::Response> response
    );
    void resetStatsCallback(
        const std::shared_ptr<edge_msgs::srv::ResetSwitchingStats::Request> request,
        std::shared_ptr<edge_msgs::srv::ResetSwitchingStats::Response> response
    );

    // Cached variables
    double goal_distance_;                                // last received goal distance
    double obstacle_distance_;                            // last received distance to obstacle
    bool is_last_state_safe_;                             // last alert state (is safe or not)
    bool is_last_state_fall_;                             // last fall state (is fall or not)
    edge_msgs::msg::EdgeState    last_edge_state_;        // last received edge state 
    edge_msgs::msg::EdgeAction   last_stu_action_;        // last received student action
    edge_msgs::msg::EdgeAction   last_tea_action_;        // last received teacher action

    // Statistical tracking
    double student_duration_;                              // Student Mode Accumulated Duration (s)
    double teacher_duration_;                              // Teacher Mode Accumulated Duration (s)
    uint32_t student_action_count_;                        // Number of actions in Student Mode
    uint32_t teacher_action_count_;                        // Number of actions in Teacher Mode
    uint32_t student_switch_count_;                        // Number of switches to Student Mode
    uint32_t teacher_switch_count_;                        // Number of switches to Teacher Mode
    rclcpp::Time mode_start_time_;                         // Mode start time
    edge::utils::ActionMode curr_mode_;                    // Current mode
    edge::utils::ActionMode last_mode_;                    // Last mode
    std::map<std::string, uint32_t> switching_reasons_;    // Switching reasons statistics

    // ROS subscribers
    rclcpp::Subscription<edge_msgs::msg::EdgeAction>::SharedPtr sub_student_action_;
    rclcpp::Subscription<edge_msgs::msg::EdgeAction>::SharedPtr sub_teacher_action_;
    rclcpp::Subscription<edge_msgs::msg::EdgeState>::SharedPtr sub_edge_state_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr sub_obstacle_distance_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr sub_goal_distance_;

    // ROS publishers
    rclcpp::Publisher<edge_msgs::msg::EdgeAction>::SharedPtr pub_action_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_alert_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_fall_;

    // ROS services
    rclcpp::Service<edge_msgs::srv::GetSwitchingStats>::SharedPtr srv_get_switching_stats_;
    rclcpp::Service<edge_msgs::srv::ResetSwitchingStats>::SharedPtr srv_reset_switching_stats_;

    double update_rate_;                            // Update rate (Hz)
    int action_source_;                             // action source: -1 -- none, 0 -- student, 1 -- teacher
    int dwell_steps_;                               // number of steps to hold teacher action in self-trigger mode
    int dwell_count_{0};                            // counter for dwell time in self-trigger mode
    bool dwell_active_{false};                      // whether currently in dwell time in self-trigger mode
    bool print_processing_;                         // whether to print processing info
    bool is_start_with_teacher_;                    // whether to start with teacher mode
    bool is_robot_state_initialized_{false};        // whether robot state has been initialized
    bool is_stu_action_initialized_{false};         // whether student action has been initialized
    bool is_tea_action_initialized_{false};         // whether teacher action has been initialized
    
    std::string student_topic_name_;                // topic name for student action
    std::string teacher_topic_name_;                // topic name for teacher action
    std::string terminal_action_name_;              // topic name for publishing terminal action
    std::string monitor_state_topic_name_;          // topic name for robot state input to safety monitor
    TriggerMode trigger_mode_;                      // trigger mode: self-trigger or event-trigger


    rclcpp::TimerBase::SharedPtr timer_;    // Periodic timer
    std::shared_ptr<edge::trigger::SafetyMonitor> safety_monitor_;       // Safety monitor
};

}  // namespace edge::trigger