/*
 * Copyright (c) 2025-2026 Yihao Cai - All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "edge_msgs/msg/control_gain.hpp"
#include "edge_msgs/msg/vec12_msg.hpp"
#include "lmi_gain_solver.hpp"

class GainSynthesizer : public rclcpp::Node {
public:
  GainSynthesizer() : Node("gain_synthesizer") {

    // Declare parameter and get sampling rate and whether to print process time
    sampling_rate_ = this->declare_parameter<int>("sampling_rate", 50);
    print_process_time_ = this->declare_parameter<bool>("print_process_time", false);
    synthesis_state_topic_ = this->declare_parameter<std::string>("synthesis_state_topic", "/edge/synthesis_state");
    synthesized_gain_topic_ = this->declare_parameter<std::string>("synthesized_gain_topic", "/edge/synthesized_gain");

    lmi_gain_solver_ = std::make_unique<lmi::LMIGainSolver>(1.0 / static_cast<double>(sampling_rate_));

    state_sub_ = this->create_subscription<edge_msgs::msg::Vec12Msg>(
      synthesis_state_topic_, 10,
      std::bind(&GainSynthesizer::gain_scheduler_callback, this, std::placeholders::_1)
    );

    gain_pub_ = this->create_publisher<edge_msgs::msg::ControlGain>(
      synthesized_gain_topic_, 10
    );

    RCLCPP_INFO(this->get_logger(), "Sampling rate: %d Hz", sampling_rate_);
    RCLCPP_INFO(this->get_logger(), "Print process time: %s", print_process_time_ ? "true" : "false");
    RCLCPP_INFO(this->get_logger(), "Synthesis state topic: %s", synthesis_state_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "Synthesized gain topic: %s", synthesized_gain_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "🔩 Gain Synthesizer Node initialized.");
  }

private:
  void gain_scheduler_callback(const edge_msgs::msg::Vec12Msg::SharedPtr msg) {
    // Update tracking error from message (from 2 to 11, total 10 values)
    for (int i = 0; i < 10 && i < msg->data.size(); ++i) {
      lmi_gain_solver_->tracking_err[i] = msg->data[i + 2];
    }

    rclcpp::Time start_time;
    if (print_process_time_) {
        start_time = this->now();
    }
      
    // Update controller (this will call the solver)
    lmi_gain_solver_->update();

    // Generate message
    edge_msgs::msg::ControlGain gain_msg;
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        gain_msg.kp[i * 6 + j] = lmi_gain_solver_->F_kp(i, j);
        gain_msg.kd[i * 6 + j] = lmi_gain_solver_->F_kd(i, j);
      }
    }

    // Publish message
    gain_msg.status = CPG_Info.status;
    gain_pub_->publish(gain_msg);

    if (print_process_time_) {
      rclcpp::Time end_time = this->now();
      double duration_ms = (end_time - start_time).seconds() * 1000.0;
      RCLCPP_INFO(this->get_logger(), "Published control gains. Process time: %.3f ms", duration_ms);
    }
  }

  int sampling_rate_ = 50;
  bool print_process_time_ = false;
  std::string synthesis_state_topic_;
  std::string synthesized_gain_topic_;

  std::unique_ptr<lmi::LMIGainSolver> lmi_gain_solver_;

  rclcpp::Subscription<edge_msgs::msg::Vec12Msg>::SharedPtr state_sub_;
  rclcpp::Publisher<edge_msgs::msg::ControlGain>::SharedPtr gain_pub_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GainSynthesizer>());
  rclcpp::shutdown();
  return 0;
}