/*
 * Copyright (c) 2025-2026 Yihao Cai - All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <exception>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include <unitree/robot/channel/channel_factory.hpp>

#include "edge_msgs/msg/edge_state.hpp"
#include "unitree_go2_hardware/api/providers/go2_state_provider.hpp"

class Go2EdgeStateBridge : public rclcpp::Node
{
public:
  Go2EdgeStateBridge()
  : Node("go2_edge_state_bridge"),
    network_interface_(declare_parameter<std::string>("network_interface", "eth0")),
    domain_id_(declare_parameter<int>("domain_id", 0)),
    publish_topic_(declare_parameter<std::string>("publish_topic", "/edge/state_input")),
    frame_id_(declare_parameter<std::string>("frame_id", "base_link")),
    publish_frequency_hz_(declare_parameter<int>("publish_frequency_hz", 500)),
    state_provider_(network_interface_)
  {
    if (publish_frequency_hz_ <= 0) {
      throw std::runtime_error("publish_frequency_hz must be > 0");
    }
    
    // Calculate period in milliseconds from frequency
    publish_period_ms_ = 1000 / publish_frequency_hz_;

    edge_state_pub_ = create_publisher<edge_msgs::msg::EdgeState>(publish_topic_, 50);

    unitree::robot::ChannelFactory::Instance()->Init(domain_id_, network_interface_);
    state_provider_.Init();

    timer_ = create_wall_timer(
      std::chrono::milliseconds(publish_period_ms_),
      std::bind(&Go2EdgeStateBridge::publishLatestState, this));

    RCLCPP_INFO(
      get_logger(),
      "Bridge started: rt/sportmodestate -> %s (nic=%s, domain=%d, frequency=%dHz)",
      publish_topic_.c_str(),
      network_interface_.c_str(),
      domain_id_,
      publish_frequency_hz_);
  }

private:
  void publishLatestState()
  {
    if (!state_provider_.HasState()) {
      if (!has_seen_state_) {
        RCLCPP_INFO_THROTTLE(
          get_logger(),
          *get_clock(),
          2000,
          "Waiting for rt/sportmodestate ... (nic=%s, domain=%d)",
          network_interface_.c_str(),
          domain_id_);
      }
      return;
    }

    if (!has_seen_state_) {
      has_seen_state_ = true;
      RCLCPP_INFO(
        get_logger(),
        "Received first rt/sportmodestate, start publishing EdgeState to %s",
        publish_topic_.c_str());
    }

    const auto state = state_provider_.GetLatestVec12Eigen();

    edge_msgs::msg::EdgeState edge_state;
    edge_state.header.stamp = now();
    edge_state.header.frame_id = frame_id_;
    edge_state.state_type = edge_msgs::msg::EdgeState::ROBOT_STATE;
    edge_state.state_value.resize(12);

    for (size_t i = 0; i < 12; ++i) {
      edge_state.state_value[i] = state(static_cast<int>(i));
    }

    edge_state_pub_->publish(edge_state);
  }

  std::string network_interface_;
  int domain_id_;
  std::string publish_topic_;
  std::string frame_id_;
  int publish_frequency_hz_;
  int publish_period_ms_;
  bool has_seen_state_{false};

  edge::go2_api::provider::Go2StateProvider state_provider_;

  rclcpp::Publisher<edge_msgs::msg::EdgeState>::SharedPtr edge_state_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  try {
    rclcpp::spin(std::make_shared<Go2EdgeStateBridge>());
    rclcpp::shutdown();
    return 0;
  } catch (const std::exception& e) {
    RCLCPP_FATAL(
      rclcpp::get_logger("go2_edge_state_bridge"),
      "Initialization failed: %s",
      e.what());
  } catch (...) {
    RCLCPP_FATAL(
      rclcpp::get_logger("go2_edge_state_bridge"),
      "Initialization failed with an unknown exception.");
  }

  rclcpp::shutdown();
  return 1;
}
