/*
 * Copyright (c) 2025-2026 Yihao Cai - All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rclcpp/rclcpp.hpp"
#include "edge_ros_interfaces/msg_conversion.hpp"
#include "edge_msgs/msg/edge_action.hpp"
#include "edge_msgs/msg/vec12_msg.hpp"

using namespace edge::ros_interfaces;

class MsgConversionTestNode : public rclcpp::Node {
public:
  MsgConversionTestNode() : Node("test_conversion") {
    edge_msgs::msg::EdgeAction msg;
    msg.action_value = {1.0, 2.0, 3.0};
    auto v = EdgeActionMsgToEigen(msg);
    RCLCPP_INFO(this->get_logger(), "Eigen vector: [%f, %f, %f]", v[0], v[1], v[2]);
  }
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<MsgConversionTestNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
}
