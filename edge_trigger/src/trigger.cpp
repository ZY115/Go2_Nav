/*
 * Copyright (c) 2025-2026 Yihao Cai - All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 */

#include "edge_trigger/trigger.hpp"

using std::placeholders::_1;
using std::placeholders::_2;
using namespace std::chrono_literals;

namespace edge::trigger {

Trigger::Trigger(const rclcpp::NodeOptions & options)
: Node("trigger", options)
{
    // Load parameters
    dwell_steps_ = this->declare_parameter<int>("dwell_steps", 20);
    action_source_ = this->declare_parameter<int>("action_source", -1);
    update_rate_ = this->declare_parameter<double>("update_rate", 150.0);
    print_processing_ = this->declare_parameter<bool>("print_processing", false);
    is_start_with_teacher_ = this->declare_parameter<bool>("start_with_teacher", true);
    teacher_topic_name_ = this->declare_parameter<std::string>("teacher_topic_name", "/edge/teacher_action");
    student_topic_name_ = this->declare_parameter<std::string>("student_topic_name", "/edge/student_action");
    terminal_action_name_ = this->declare_parameter<std::string>("terminal_action_name", "/edge/terminal_action");
    monitor_state_topic_name_ = this->declare_parameter<std::string>("monitor_state_name", "/edge/robot_state");

    std::string trigger_mode_str = this->declare_parameter<std::string>("trigger_mode", "self");
    
    try {
        trigger_mode_ = TriggerModeFromString(trigger_mode_str);
    }
    catch (const std::exception& e)
    {
        RCLCPP_FATAL(this->get_logger(), "%s", e.what());
        throw;
    }
    

    RCLCPP_INFO(this->get_logger(), "🔘 Trigger action source is set to %s",
        (action_source_ == 1) ? "teacher" :
        (action_source_ == 0) ? "student" : "auto");
    RCLCPP_INFO(this->get_logger(), "🔘 Trigger mode is set to %s", 
        (trigger_mode_ == TriggerMode::SELF) ? "self-trigger" : "event-trigger");
    if(trigger_mode_ == TriggerMode::SELF) {
        RCLCPP_INFO(this->get_logger(), "🔘 Trigger dwell steps in self-trigger mode is set to %d", dwell_steps_);
    }
    RCLCPP_INFO(this->get_logger(), "🔘 Trigger update rate is set to %.1f Hz", update_rate_);
    RCLCPP_INFO(this->get_logger(), "🔘 Trigger print processing is set to %s", print_processing_ ? "true" : "false");
    RCLCPP_INFO(this->get_logger(), "🔘 Trigger start with teacher mode is set to %s", is_start_with_teacher_ ? "true" : "false");
    RCLCPP_INFO(this->get_logger(), "🔘 Trigger teacher topic: %s", teacher_topic_name_.c_str());
    RCLCPP_INFO(this->get_logger(), "🔘 Trigger student topic: %s", student_topic_name_.c_str());
    RCLCPP_INFO(this->get_logger(), "🔘 Trigger terminal action topic: %s", terminal_action_name_.c_str());
    RCLCPP_INFO(this->get_logger(), "🔘 Trigger monitor state topic: %s", monitor_state_topic_name_.c_str());
    // Initialize statistical tracking variables
    resetStats();

    // Initialize cached variables
    resetCachedVariables();

    // Subscribers (state, teacher action, student action, obstacle distance)
    sub_edge_state_ = this->create_subscription<edge_msgs::msg::EdgeState>(
        monitor_state_topic_name_, 50, std::bind(&Trigger::edgeStateCallback, this, _1));

    sub_teacher_action_ = this->create_subscription<edge_msgs::msg::EdgeAction>(
        teacher_topic_name_, 50, std::bind(&Trigger::teacherActionCallback, this, _1));

    sub_student_action_ = this->create_subscription<edge_msgs::msg::EdgeAction>(
        student_topic_name_, 50, std::bind(&Trigger::studentActionCallback, this, _1));

    sub_obstacle_distance_ = this->create_subscription<std_msgs::msg::Float32>(
        "/obstacle_min_distance", 50, std::bind(&Trigger::obstacleDistanceCallback, this, _1));

    sub_goal_distance_ = this->create_subscription<std_msgs::msg::Float32>(
        "/goal_distance", 50, std::bind(&Trigger::goalDistanceCallback, this, _1));

    // Publishers (for alert and terminal action)
    pub_alert_  = this->create_publisher<std_msgs::msg::Bool>("is_safe", 50);
    pub_fall_   = this->create_publisher<std_msgs::msg::Bool>("is_fall", 50);
    pub_action_ = this->create_publisher<edge_msgs::msg::EdgeAction>(terminal_action_name_, 50);

    // Service for getting switching statistics
    srv_get_switching_stats_ = this->create_service<edge_msgs::srv::GetSwitchingStats>(
        "get_switching_stats",
        std::bind(&Trigger::getStatsCallback, this, _1, _2));

    // Service for resetting switching statistics
    srv_reset_switching_stats_ = this->create_service<edge_msgs::srv::ResetSwitchingStats>(
        "reset_switching_stats",
        std::bind(&Trigger::resetStatsCallback, this, _1, _2));

    // Timer for periodic update
    auto period = std::chrono::duration<double>(1.0 / update_rate_);
    timer_ = this->create_wall_timer(period, std::bind(&Trigger::updateLoop, this));

    // Initialize Safety Monitor
    safety_monitor_ = std::make_shared<edge::trigger::SafetyMonitor>();

    RCLCPP_INFO(this->get_logger(), "🔘 The Trigger Component is initialized.");

}

void Trigger::studentActionCallback(const edge_msgs::msg::EdgeAction::SharedPtr msg)
{
    last_stu_action_ = *msg;
    if (!is_stu_action_initialized_) is_stu_action_initialized_ = true;
}

void Trigger::teacherActionCallback(const edge_msgs::msg::EdgeAction::SharedPtr msg)
{
    last_tea_action_ = *msg;
    if (!is_tea_action_initialized_) is_tea_action_initialized_ = true;
}

void Trigger::edgeStateCallback(const edge_msgs::msg::EdgeState::SharedPtr msg)
{
    if(msg->state_type != edge_msgs::msg::EdgeState::ROBOT_STATE) {
        RCLCPP_WARN(this->get_logger(), "Received EdgeState message with unsupported state_type: %d", msg->state_type);
        return;
    }
    last_edge_state_ = *msg;  // Store the entire EdgeState message for later use
    if (!is_robot_state_initialized_) is_robot_state_initialized_ = true;
}

void Trigger::obstacleDistanceCallback(const std_msgs::msg::Float32::SharedPtr msg)
{
    // Placeholder for distance callback
    obstacle_distance_ = msg->data;
}

void Trigger::goalDistanceCallback(const std_msgs::msg::Float32::SharedPtr msg)
{
    // Placeholder for distance callback
    goal_distance_ = msg->data;
}

void Trigger::resetStatsCallback(
    const std::shared_ptr<edge_msgs::srv::ResetSwitchingStats::Request> /*request*/,
    std::shared_ptr<edge_msgs::srv::ResetSwitchingStats::Response> response)
{
    try {
        resetStats();   // Reset the switching statistics

        response->success = true;
        response->message = "The Trigger's switching statistics have been reset successfully";
    }
    catch (const std::exception & e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to reset switching statistics: %s", e.what());
        response->success = false;
        response->message = std::string("Failed to reset switching statistics: ") + e.what();
    }
    catch (...) {
        RCLCPP_ERROR(this->get_logger(), "Failed to reset switching statistics: unknown error");
        response->success = false;
        response->message = "Failed to reset switching statistics: unknown error";
    }
}

void Trigger::getStatsCallback(
    const std::shared_ptr<edge_msgs::srv::GetSwitchingStats::Request> /*request*/,
    std::shared_ptr<edge_msgs::srv::GetSwitchingStats::Response> response)
{
    // Fill in switching reasons
    for (const auto& [reason, count] : switching_reasons_) {
        response->switching_reasons.push_back(reason);
        response->reason_counts.push_back(count);
    }

    // Switching frequency and ratios
    uint32_t total_switches = student_switch_count_ + teacher_switch_count_;
    auto student_switch_ratio = total_switches > 0 ? static_cast<float>(student_switch_count_) / total_switches : 0.0f;
    response->total_switches = total_switches;
    response->student_switch_ratio = student_switch_ratio;

    // Action counts and ratios
    uint32_t total_action_count = student_action_count_ + teacher_action_count_;
    auto student_action_ratio = total_action_count > 0 ? static_cast<float>(student_action_count_) / total_action_count : 0.0f;
    response->total_action_count = total_action_count;
    response->student_action_ratio = student_action_ratio;

    // Duration and time ratios
    double total_duration = student_duration_ + teacher_duration_;
    auto student_time_ratio = total_duration > 0.0 ? static_cast<float>(student_duration_) / total_duration : 0.0f;
    response->total_duration = total_duration;
    response->student_time_ratio = student_time_ratio;

    // Print statistics to console
    RCLCPP_INFO(this->get_logger(), "📊 Statistics Query:");
    RCLCPP_INFO(this->get_logger(), "Total Switches: %u", total_switches);
    RCLCPP_INFO(this->get_logger(), "   - Switches to Student Mode: %u", student_switch_count_);
    RCLCPP_INFO(this->get_logger(), "   - Switches to Teacher Mode: %u", teacher_switch_count_);
    RCLCPP_INFO(this->get_logger(), "   - Student Switch Ratio: %.2f", student_switch_ratio);
    RCLCPP_INFO(this->get_logger(), "Total Actions: %u", total_action_count);
    RCLCPP_INFO(this->get_logger(), "   - Actions in Student Mode: %u", student_action_count_);
    RCLCPP_INFO(this->get_logger(), "   - Actions in Teacher Mode: %u", teacher_action_count_);
    RCLCPP_INFO(this->get_logger(), "   - Student Action Ratio: %.2f", student_action_ratio);
    RCLCPP_INFO(this->get_logger(), "Total Duration: %.2f seconds", total_duration);
    RCLCPP_INFO(this->get_logger(), "   - Time in Student Mode: %.2f seconds", student_duration_);
    RCLCPP_INFO(this->get_logger(), "   - Time in Teacher Mode: %.2f seconds", teacher_duration_);
    RCLCPP_INFO(this->get_logger(), "   - Student Time Ratio: %.2f", student_time_ratio);
    
    // Print switching reasons statistics
    if (!switching_reasons_.empty()) {
        RCLCPP_INFO(this->get_logger(), "   Switching Reasons:");
        for (const auto& [reason, count] : switching_reasons_) {
            RCLCPP_INFO(this->get_logger(), "      - %s: %u times", reason.c_str(), count);
        }
    } else {
        RCLCPP_INFO(this->get_logger(), "   No mode switches recorded yet.");
    }
}

void Trigger::resetCachedVariables()
{
    goal_distance_      = is_start_with_teacher_ ? 0.0 : MAX_VAL;
    obstacle_distance_  = is_start_with_teacher_ ? 0.0 : MAX_VAL;
    is_last_state_fall_ = false;     // initial fall state
    is_last_state_safe_ = !is_start_with_teacher_;      // initial alert state
    
    last_edge_state_.state_type = edge_msgs::msg::EdgeState::ROBOT_STATE;
    last_edge_state_.state_value.assign(12, 0.0);

    last_stu_action_  = edge_msgs::msg::EdgeAction{};     // initial student action
    last_tea_action_  = edge_msgs::msg::EdgeAction{};     // initial teacher action            
}

void Trigger::resetStats()
{
    switching_reasons_.clear();

    mode_start_time_ = this->now();
    
    student_switch_count_ = 0;
    teacher_switch_count_ = 0;

    student_action_count_ = 0;
    teacher_action_count_ = 0;

    student_duration_ = 0.0;
    teacher_duration_ = 0.0;

    curr_mode_ = is_start_with_teacher_ ? edge::utils::ActionMode::TEACHER 
                                        : edge::utils::ActionMode::STUDENT;
    last_mode_ = is_start_with_teacher_ ? edge::utils::ActionMode::TEACHER 
                                        : edge::utils::ActionMode::STUDENT;
}

void Trigger::updateStats(edge::utils::ActionMode terminal_mode)
{
    auto current_time = this->now();
    auto duration = (current_time - mode_start_time_).seconds();

    // Accumulate duration and action count for current mode
    if (curr_mode_ == edge::utils::ActionMode::STUDENT) {
        student_duration_ += duration;
        ++student_action_count_;
    } else {
        teacher_duration_ += duration;
        ++teacher_action_count_;
    }

    // Check for mode switch
    if (terminal_mode != curr_mode_) {
        
        // Record switching reasons (from SafetyMonitor)
        if (terminal_mode == edge::utils::ActionMode::TEACHER) {
            // Student -> Teacher: Record violation reasons
            const auto& reasons = safety_monitor_->getViolationReasons();
            for (const auto& reason : reasons) {
                ++switching_reasons_[reason];
            }
            
            if (print_processing_) {
                RCLCPP_WARN(this->get_logger(), "🔄 Mode Switch: STUDENT -> TEACHER (unsafe)");
            }
            ++teacher_switch_count_;
        } else {
            // Teacher -> Student: Return to safe
            if (print_processing_) {
                RCLCPP_INFO(this->get_logger(), "🔄 Mode Switch: TEACHER -> STUDENT (safe)");
            }
            ++student_switch_count_;
        }
        
        // Update mode and reset timer
        last_mode_ = curr_mode_;
        curr_mode_ = terminal_mode;
        mode_start_time_ = current_time;
    }
}


edge::utils::ActionMode Trigger::selectAction()
{
    // In event-trigger mode, switch immediately based on safety status
    if (trigger_mode_ == TriggerMode::EVENT) {
        return is_last_state_safe_ ? edge::utils::ActionMode::STUDENT
                                    : edge::utils::ActionMode::TEACHER;
    }

    // In self-trigger mode, use dwell time to determine switching
    else if (trigger_mode_ == TriggerMode::SELF) {  
        if (is_last_state_safe_ && !dwell_active_) {
            // If currently safe, reset dwell timer and stay in student mode
            dwell_count_ = 0;
            dwell_active_ = false;
            return edge::utils::ActionMode::STUDENT;
        }
        else {
            // If currently unsafe, start/increment dwell timer
            if (!dwell_active_) {
                dwell_active_ = true;  // Start dwell time
            }
            ++dwell_count_;

            // If dwell time has been reached, switch to student mode
            if (dwell_count_ <= dwell_steps_) {
                if (print_processing_) {
                    RCLCPP_WARN(this->get_logger(), "🔄 Unsafe state detected, dwell count: %d/%d", dwell_count_, dwell_steps_);
                }
                return edge::utils::ActionMode::TEACHER;  // Still in dwell time, stay in teacher mode

            }
            else {
                dwell_active_ = false; // Reset dwell time
                dwell_count_ = 0;   // Reset dwell count
                if (print_processing_) {
                    RCLCPP_WARN(this->get_logger(), "🔄 Dwell time reached, switching to student mode");
                }
                return edge::utils::ActionMode::STUDENT;  // Dwell time reached, switch to student mode
            }
        }
    } else {
        // Default to safety-based switching if trigger mode is unrecognized
        RCLCPP_WARN(this->get_logger(), "Unrecognized trigger mode, defaulting to safety-based switching.");
        return edge::utils::ActionMode::TEACHER;
    }
}


// -------------------- Main periodic update for trigger --------------------
void Trigger::updateLoop()
{
    // Convert states
    std::vector<double> robot_state(last_edge_state_.state_value.begin(), last_edge_state_.state_value.end());
    
    // Evaluate safety and fall status
    is_last_state_safe_ = safety_monitor_->evaluateSafety(edge::trigger::RuleType::RELATIVE, robot_state, obstacle_distance_, goal_distance_);
    is_last_state_fall_ = is_robot_state_initialized_ ? safety_monitor_->detectFall(robot_state) : false;

    if (print_processing_) {
        RCLCPP_INFO(this->get_logger(), "Trigger: is_safe = %s", is_last_state_safe_ ? "true" : "false");
    }

    // Publish alert status (still report safety evaluation)
    std_msgs::msg::Bool alert_msg;
    alert_msg.data = is_last_state_safe_;
    pub_alert_->publish(alert_msg);

    // Publish fall status
    std_msgs::msg::Bool fall_msg;
    fall_msg.data = is_last_state_fall_;
    pub_fall_->publish(fall_msg);

    // Determine terminal action and mode
    edge::utils::ActionMode terminal_mode;
    edge_msgs::msg::EdgeAction terminal_action;

    // Determine terminal mode (override or safety-based)
    if (action_source_ == 0) {
        terminal_mode = edge::utils::ActionMode::STUDENT;
    }
    else if (action_source_ == 1) {
        terminal_mode = edge::utils::ActionMode::TEACHER;
    }
    else {
        terminal_mode = selectAction();
    }

    // Check if the required action has been initialized before publishing
    bool can_publish = (terminal_mode == edge::utils::ActionMode::STUDENT)
                        ? is_stu_action_initialized_
                        : is_tea_action_initialized_;
    if (!can_publish) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
            "Waiting for %s action to be initialized...",
            terminal_mode == edge::utils::ActionMode::STUDENT ? "student" : "teacher");
        return;
    }

    // Prepare terminal action message
    if (terminal_mode == edge::utils::ActionMode::STUDENT) {
        terminal_action = last_stu_action_;
        terminal_action.action_mode = edge_msgs::msg::EdgeAction::STUDENT;
    }
    else {
        terminal_action = last_tea_action_;
        terminal_action.action_mode = edge_msgs::msg::EdgeAction::TEACHER;
    }

    // Set timestamp for terminal action
    terminal_action.header.stamp = this->now();

    // Update switching statistics
    updateStats(terminal_mode);

    // Publish terminal action
    pub_action_->publish(terminal_action);

}



}  // namespace edge::trigger


int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto trigger_node = std::make_shared<edge::trigger::Trigger>(rclcpp::NodeOptions());
    rclcpp::spin(trigger_node);
    rclcpp::shutdown();
    return 0;
}
