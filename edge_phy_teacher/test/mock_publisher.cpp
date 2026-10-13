/*
 * Copyright (c) 2025-2026 Yihao Cai - All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <random>
#include <vector>
#include <memory>
#include <std_msgs/msg/float64_multi_array.hpp>
#include "edge_msgs/msg/vec12_msg.hpp"
#include <rclcpp/rclcpp.hpp>

using namespace std::chrono_literals;

class MockPublisher : public rclcpp::Node {
public:
  MockPublisher()
  : Node("mock_publisher"),
    gen_(rd_()), dist_(-0.1, 0.1)
  {
    synthesis_state_topic_ = this->declare_parameter<std::string>("synthesis_state_topic", "/edge/synthesis_state");
    publisher_ = this->create_publisher<edge_msgs::msg::Vec12Msg>(synthesis_state_topic_, 30);
    timer_ = this->create_wall_timer(100ms, std::bind(&MockPublisher::timer_callback, this));

    RCLCPP_INFO(this->get_logger(), "The frequency of publishing is set to 10 Hz (100 ms period).");
    RCLCPP_INFO(this->get_logger(), "Publishing to topic: %s", synthesis_state_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "MockPublisher started.");
  }

private:
void timer_callback() {
  auto msg = edge_msgs::msg::Vec12Msg();
  
  msg.data[0] = 0.0; 
  msg.data[1] = 0.0; 

  for (size_t i = 2; i < msg.data.size(); i++)
  {
    msg.data[i] = dist_(gen_);
  }
  publisher_->publish(msg);
  
  std::ostringstream ss;
  ss << "[";
  for (size_t i = 0; i < msg.data.size(); ++i)
  {
      ss << msg.data[i];
      if (i < msg.data.size() - 1)
          ss << ", ";
  }
  ss << "]";

  RCLCPP_INFO(this->get_logger(), "Published tracking_error: %s", ss.str().c_str());

}
  std::string synthesis_state_topic_;
  rclcpp::Publisher<edge_msgs::msg::Vec12Msg>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::random_device rd_;
  std::mt19937 gen_;
  std::uniform_real_distribution<double> dist_;
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MockPublisher>());
  rclcpp::shutdown();
  return 0;
}
