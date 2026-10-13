/*
 * Copyright (c) 2025-2026 Yihao Cai - All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PHY_TEACHER_HPP
#define PHY_TEACHER_HPP

#include <string>
#include <memory>
#include "lmi_gain_solver.hpp"
#include <rclcpp/rclcpp.hpp>
#include "std_msgs/msg/float64_multi_array.hpp"
#include "std_msgs/msg/float32.hpp"
#include "edge_msgs/msg/vec12_msg.hpp"
#include "edge_msgs/msg/edge_action.hpp"
#include "edge_msgs/msg/edge_state.hpp"
#include "edge_msgs/msg/control_gain.hpp"
#include "edge_trigger/safety_monitor.hpp"
#include "edge_ros_interfaces/msg_conversion.hpp"

using namespace lmi;

namespace edge::phy_teacher {

class PhyTeacher : public rclcpp::Node {
public:
    explicit PhyTeacher(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
    ~PhyTeacher() override;

    void set_default_kp(Mat6x6 kp) { default_F_kp = kp; print_matrix6x6(default_F_kp, "Default Kp set to:");}
    void set_default_kd(Mat6x6 kd) { default_F_kd = kd; print_matrix6x6(default_F_kd, "Default Kd set to:");}


private:
    void controlCallback();
    void errorCallback(const edge_msgs::msg::EdgeState::SharedPtr msg);
    void obstacleDistanceCallback(const std_msgs::msg::Float32::SharedPtr msg);
    void goalDistanceCallback(const std_msgs::msg::Float32::SharedPtr msg);
    void sythesizedGainCallback(const edge_msgs::msg::ControlGain::SharedPtr msg);
    
    // For real-time patch
    double chi_ = 0.25;                       // Hyperparameter χ
    bool rt_patch_enable_ = true;             // Whether the real-time patch method is enabled
    std::string synthesis_state_topic_;       // topic name for subscribing the state used for gain synthesis
    std::string synthesized_gain_topic_;      // topic name for publishing synthesized gain

    // For phy-teacher logistics
    int update_rate_;                         // Update rate in Hz
    bool got_msg_ = false;
    bool patch_active_ = false;              // Whether the patch is active

    // For control and optimization
    int opt_status_ = -1;                    // Status of the optimization (0: success, non-zero: failure)
    lmi::Mat6x6 F_kp;                        // Contains the solved Kp gain matrices
    lmi::Mat6x6 F_kd;                        // Contains the solved Kd gain matrices
    lmi::Mat6x6 default_F_kp;                // Dfault Kp gain matrices
    lmi::Mat6x6 default_F_kd;                // Default Kd gain matrices

    edge::ros_interfaces::Vector12d  error_state_;          // received error state
    edge::ros_interfaces::Vector12d  patch_center_;         // Center of the safety patch
    edge::ros_interfaces::Vector6d   curr_action_;          // Current action sent
    edge::ros_interfaces::Vector6d   last_action_;          // Last action sent

    // For profiling
    rclcpp::Time start_time_;   
    rclcpp::Time end_time_;

    // Safety-conditions
    double goal_distance_;
    double obstacle_distance_;
    bool is_last_safe_ = true;
    std::shared_ptr<edge::trigger::SafetyMonitor> safety_monitor_;

    // Subscribers
    rclcpp::Subscription<edge_msgs::msg::EdgeState>::SharedPtr state_sub_;
    rclcpp::Subscription<edge_msgs::msg::ControlGain>::SharedPtr gain_synthesizer_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr obstacle_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr goal_sub_;

    // Publishers
    rclcpp::Publisher<edge_msgs::msg::EdgeAction>::SharedPtr action_pub_;
    rclcpp::Publisher<edge_msgs::msg::Vec12Msg>::SharedPtr gain_synthesizer_pub_;

    // Control timer
    rclcpp::TimerBase::SharedPtr control_timer_;
    
};

}  // namespace edge::phy_teacher

#endif  // PHY_TEACHER_HPP