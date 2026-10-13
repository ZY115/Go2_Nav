/*
 * Copyright (c) 2025-2026 Yihao Cai - All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cmath>
#include <functional>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"

#include "unitree_api/msg/request.hpp"
#include "unitree_go2_hardware/common/ros2_sport_client.h"

class Go2CmdVelExecutorNode : public rclcpp::Node
{
public:
  Go2CmdVelExecutorNode()
  : Node("go2_cmd_vel_executor"),
    last_cmd_time_(this->now())
  {
    cmd_vel_topic_ = declare_parameter<std::string>("cmd_vel_topic", "/cmd_vel");
    request_topic_ = declare_parameter<std::string>("request_topic", "/api/sport/request");
    publish_rate_hz_ = declare_parameter<double>("publish_rate_hz", 50.0);
    cmd_timeout_s_ = declare_parameter<double>("cmd_timeout_s", 0.25);
    stop_epsilon_ = declare_parameter<double>("stop_epsilon", 1e-4);
    max_vx_ = declare_parameter<double>("max_vx", 1.0);
    max_vy_ = declare_parameter<double>("max_vy", 0.5);
    max_wz_ = declare_parameter<double>("max_wz", 1.4);

    request_pub_ = create_publisher<unitree_api::msg::Request>(request_topic_, 10);
    cmd_vel_sub_ = create_subscription<geometry_msgs::msg::TwistStamped>(
      cmd_vel_topic_, 20,
      std::bind(&Go2CmdVelExecutorNode::cmdVelCallback, this, std::placeholders::_1)
    );

    const auto period = std::chrono::duration<double>(1.0 / std::max(1.0, publish_rate_hz_));
    timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&Go2CmdVelExecutorNode::publishCommand, this)
    );

    RCLCPP_INFO(
      get_logger(),
      "Listening %s -> publishing %s @ %.1fHz",
      cmd_vel_topic_.c_str(), request_topic_.c_str(), publish_rate_hz_);
  }

private:
  void cmdVelCallback(const geometry_msgs::msg::TwistStamped::SharedPtr msg)
  {
    last_cmd_.linear.x = clamp(msg->twist.linear.x, -max_vx_, max_vx_);
    last_cmd_.linear.y = clamp(msg->twist.linear.y, -max_vy_, max_vy_);
    last_cmd_.angular.z = clamp(msg->twist.angular.z, -max_wz_, max_wz_);
    last_cmd_time_ = this->now();
  }

  void publishCommand()
  {
    geometry_msgs::msg::Twist cmd = last_cmd_;
    const double dt = (this->now() - last_cmd_time_).seconds();
    if (dt > cmd_timeout_s_) {
      cmd = geometry_msgs::msg::Twist();
    }

    const double vx = cmd.linear.x;
    const double vy = cmd.linear.y;
    const double wz = cmd.angular.z;

    if (std::fabs(vx) < stop_epsilon_ && std::fabs(vy) < stop_epsilon_ && std::fabs(wz) < stop_epsilon_) {
      sport_client_.StopMove(request_);
    } else {
      sport_client_.Move(request_, vx, vy, wz);
    }
    request_pub_->publish(request_);
  }

  static double clamp(double value, double low, double high)
  {
    return std::max(low, std::min(value, high));
  }

  std::string cmd_vel_topic_;
  std::string request_topic_;
  double publish_rate_hz_;
  double cmd_timeout_s_;
  double stop_epsilon_;
  double max_vx_;
  double max_vy_;
  double max_wz_;

  rclcpp::Time last_cmd_time_;
  geometry_msgs::msg::Twist last_cmd_;

  unitree_api::msg::Request request_;
  SportClient sport_client_;

  rclcpp::Publisher<unitree_api::msg::Request>::SharedPtr request_pub_;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr cmd_vel_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Go2CmdVelExecutorNode>());
  rclcpp::shutdown();
  return 0;
}
