/*
 * Copyright (c) 2025-2026 Yihao Cai - All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cmath>
#include <exception>
#include <functional>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "edge_msgs/msg/edge_action.hpp"

#include "unitree_go2_hardware/api/go2_api_proxy.hpp"

class Go2VelActionExecutor : public rclcpp::Node
{
public:
  Go2VelActionExecutor() : Node("go2_vel_action_executor")
  {
    vel_action_topic_ =
        declare_parameter<std::string>("vel_action_topic", "/edge/command_action");

    network_interface_ =
      declare_parameter<std::string>("network_interface", "eth0");

    stop_epsilon_ =
        declare_parameter<double>("stop_epsilon", 1e-4);

    api_proxy_ = std::make_unique<edge::go2_api::Go2ApiProxy>(network_interface_);

    api_proxy_->Init();

    vel_action_sub_ = create_subscription<edge_msgs::msg::EdgeAction>(
        vel_action_topic_,
        20,
        std::bind(&Go2VelActionExecutor::velActionCallback,
                  this,
                  std::placeholders::_1));

    RCLCPP_INFO(get_logger(),
                "Listening velocity action on %s",
                vel_action_topic_.c_str());
  }

private:
  void velActionCallback(const edge_msgs::msg::EdgeAction::SharedPtr msg)
  {
    if (msg->action_type != edge_msgs::msg::EdgeAction::CMD_VEL)
    {
      RCLCPP_WARN(get_logger(),
                  "Received unsupported action type: %u",
                  msg->action_type);
      return;
    }

    const double vx = msg->action_value[0];
    const double vy = msg->action_value[1];
    const double wz = msg->action_value[2];

    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
                         "Received vel_action: vx=%.3f vy=%.3f wz=%.3f",
                         vx, vy, wz);

    if (std::fabs(vx) < stop_epsilon_ &&
        std::fabs(vy) < stop_epsilon_ &&
        std::fabs(wz) < stop_epsilon_)
    {
      api_proxy_->Stop();
    }
    else
    {
      api_proxy_->Move(vx, vy, wz);
    }
  }

  std::string vel_action_topic_;
  std::string network_interface_;
  double stop_epsilon_;

  std::unique_ptr<edge::go2_api::Go2ApiProxy> api_proxy_;

  rclcpp::Subscription<edge_msgs::msg::EdgeAction>::SharedPtr vel_action_sub_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);

  try
  {
    rclcpp::spin(std::make_shared<Go2VelActionExecutor>());
    rclcpp::shutdown();
    return 0;
  }
  catch (const std::exception &e)
  {
    RCLCPP_FATAL(
      rclcpp::get_logger("go2_vel_action_executor"),
      "Initialization failed: %s. Check the network interface and run with, for example, "
      "'--ros-args -p network_interface:=eth0'.",
      e.what());
  }
  catch (...)
  {
    RCLCPP_FATAL(
      rclcpp::get_logger("go2_vel_action_executor"),
      "Initialization failed with an unknown exception.");
  }

  rclcpp::shutdown();
  return 1;
}