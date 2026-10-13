/*
 * Copyright (c) 2025-2026 Yihao Cai - All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cmath>
#include <functional>
#include <iostream>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "edge_msgs/msg/edge_action.hpp"

#include "unitree_api/msg/request.hpp"
#include "unitree_go2_hardware/common/ros2_sport_client.h"

class Go2RealRobotExecutorNode : public rclcpp::Node
{
public:
  Go2RealRobotExecutorNode()
  : Node("go2_real_robot_executor")
  {
    vel_action_topic_ = declare_parameter<std::string>("vel_action_topic", "/edge/command_action");
    request_topic_ = declare_parameter<std::string>("request_topic", "/api/sport/request");
    stop_epsilon_ = declare_parameter<double>("stop_epsilon", 1e-4);

    request_pub_ = create_publisher<unitree_api::msg::Request>(request_topic_, 10);
    vel_action_sub_ = create_subscription<edge_msgs::msg::EdgeAction>(
      vel_action_topic_, 20,
      std::bind(&Go2RealRobotExecutorNode::velActionCallback, this, std::placeholders::_1)
    );

    RCLCPP_INFO(get_logger(), "Listening %s -> publishing %s", vel_action_topic_.c_str(), request_topic_.c_str());
  }

private:
  void velActionCallback(const edge_msgs::msg::EdgeAction::SharedPtr msg)
  {
    if (msg->action_type != edge_msgs::msg::EdgeAction::CMD_VEL) {
      RCLCPP_WARN(get_logger(), "Received unsupported action type: %u", msg->action_type);
      return;
    }

    std::cout << "Received command_action: vx=" << msg->action_value[0]
              << ", vy=" << msg->action_value[1]
              << ", wz=" << msg->action_value[2] << std::endl;

    const double vx = msg->action_value[0];
    const double vy = msg->action_value[1];
    const double wz = msg->action_value[2];

    if (std::fabs(vx) < stop_epsilon_ && std::fabs(vy) < stop_epsilon_ && std::fabs(wz) < stop_epsilon_) {
      sport_client_.StopMove(request_);
    } else {
      sport_client_.Move(request_, vx, vy, wz);
    }
    request_pub_->publish(request_);
  }

  std::string vel_action_topic_;
  std::string request_topic_;
  double stop_epsilon_;

  unitree_api::msg::Request request_;
  SportClient sport_client_;
  rclcpp::Publisher<unitree_api::msg::Request>::SharedPtr request_pub_;
  rclcpp::Subscription<edge_msgs::msg::EdgeAction>::SharedPtr vel_action_sub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Go2RealRobotExecutorNode>());
  rclcpp::shutdown();
  return 0;
}