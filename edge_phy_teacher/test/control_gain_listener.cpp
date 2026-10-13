/*
 * Copyright (c) 2025-2026 Yihao Cai - All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 */

 #include <memory>
#include <vector>
#include <rclcpp/rclcpp.hpp>
#include "lmi_gain_solver.hpp"
#include "edge_msgs/msg/control_gain.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

using namespace lmi;

class ControlGainListener : public rclcpp::Node {
public:
  ControlGainListener()
  : Node("gain_listener")
  {
    subscription_ = this->create_subscription<edge_msgs::msg::ControlGain>(
      "/edge/synthesized_gain", 10,
      std::bind(&ControlGainListener::on_gain_update, this, std::placeholders::_1));

    synthesized_gain_topic_ = this->declare_parameter<std::string>("synthesized_gain_topic", "/edge/synthesized_gain");

    RCLCPP_INFO(this->get_logger(), "Subscribed to topic: %s", synthesized_gain_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "Synthesized Gain Listener initialized.");
  }

private:
  void on_gain_update(const edge_msgs::msg::ControlGain::SharedPtr msg) {

    if (!msg) {
      RCLCPP_ERROR(this->get_logger(), "Received null message in on_gain_update.");
      return;
    }

    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        F_kp(i, j)  = msg->kp[i * 6 + j];
        F_kd(i, j)  = msg->kd[i * 6 + j];
      }
    }

    print_matrix6x6(F_kp, "F_kp");
    print_matrix6x6(F_kd, "F_kd");
  }

  lmi::Mat6x6 F_kp;
  lmi::Mat6x6 F_kd;
  std::string synthesized_gain_topic_;
  rclcpp::Subscription<edge_msgs::msg::ControlGain>::SharedPtr subscription_;
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ControlGainListener>());
  rclcpp::shutdown();
  return 0;
}
