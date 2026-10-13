/*
 * Copyright (c) 2025-2026 Yihao Cai - All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 */

#include <random>
#include <vector>
#include <chrono>
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"
#include "edge_msgs/msg/vec12_msg.hpp"


using namespace std::chrono_literals;

class TriggerTestNode : public rclcpp::Node
{
public:
    TriggerTestNode() : Node("trigger_test_node", "/edge")
    {
        // ---- Parameters ----
        this->declare_parameter<double>("publish_rate", 100.0);   // Hz
        this->declare_parameter<double>("max_roll", 0.17);         // radians
        this->declare_parameter<double>("max_pitch", 0.17);
        this->declare_parameter<double>("max_height", 0.3);       // meters

        double rate = this->get_parameter("publish_rate").as_double();
        publish_period_ = std::chrono::duration<double>(1.0 / rate);

        // ---- Publishers & Subscribers ----
        pub_error_ = this->create_publisher<edge_msgs::msg::Vec12Msg>("tracking_error", 10);
        sub_safe_ = this->create_subscription<std_msgs::msg::Bool>(
        "is_safe", 10,
        [this](const std_msgs::msg::Bool::SharedPtr msg) {
            last_is_safe_ = msg->data;
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                "🧩 Current /is_safe = %s", msg->data ? "true" : "false");
        });

        // ---- Timer ----
        timer_ = this->create_wall_timer(publish_period_,
                                        std::bind(&TriggerTestNode::publishRandomState, this));

        // ---- Random generator ----
        rng_ = std::mt19937(std::random_device{}());
        normal_dist_ = std::normal_distribution<double>(0.0, 1.0);

        RCLCPP_INFO(this->get_logger(),
                    "✅ TriggerTestNode started (random mode, %.1f Hz).", rate);
    }

private:
    void publishRandomState()
    {
        edge_msgs::msg::Vec12Msg msg;

        // Generate random state (roll, pitch, height + others)
        msg.data[0] = randomRange(-max_height_, max_height_); // height
        msg.data[1] = randomRange(-max_roll_, max_roll_);     // roll
        msg.data[2] = randomRange(-max_pitch_, max_pitch_);   // pitch

        // Remaining 9 states (random noise)
        for (size_t i = 3; i < 12; ++i)
        msg.data[i] = normal_dist_(rng_) * 0.1;
        pub_error_->publish(msg);

        RCLCPP_DEBUG(this->get_logger(),
                    "Published random state: height=%.2f, roll=%.2f, pitch=%.2f",
                    msg.data[0], msg.data[1], msg.data[2]);
    }

    double randomRange(double min_val, double max_val)
    {
        std::uniform_real_distribution<double> dist(min_val, max_val);
        return dist(rng_);
    }

    // ---- Members ----
    rclcpp::Publisher<edge_msgs::msg::Vec12Msg>::SharedPtr pub_error_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr sub_safe_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::mt19937 rng_;
    std::normal_distribution<double> normal_dist_;
    bool last_is_safe_ = true;
    std::chrono::duration<double> publish_period_;
    double max_roll_ = 0.17, max_pitch_ = 0.17, max_height_ = 0.3;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto trigger_test_node = std::make_shared<TriggerTestNode>();
    rclcpp::spin(trigger_test_node);
    rclcpp::shutdown();
    return 0;
}
