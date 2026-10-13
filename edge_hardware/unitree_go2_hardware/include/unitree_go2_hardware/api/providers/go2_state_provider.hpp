/*
 * Copyright (c) 2025-2026 Yihao Cai - All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <mutex>
#include <string>

#include "edge_msgs/msg/vec12_msg.hpp"
#include "edge_msgs/msg/edge_state.hpp"
#include "edge_ros_interfaces/msg_conversion.hpp"

#include <unitree/idl/go2/SportModeState_.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

namespace edge::go2_api::provider {

class Go2StateProvider {
public:
    using Vector12d = edge::ros_interfaces::Vector12d;
    using Vec12Msg = edge_msgs::msg::Vec12Msg;

    explicit Go2StateProvider(std::string network_interface);

    void Init();
    bool HasState() const;
    unitree_go::msg::dds_::SportModeState_ GetLatestState() const;
    Vector12d GetLatestVec12Eigen() const;
    Vec12Msg GetLatestVec12Msg() const;
    void PrintLatestState() const;
    void Run() const;

private:
    static constexpr const char* kTopicHighState = "rt/sportmodestate";

    static Vector12d ExtractRobotBodyState(const unitree_go::msg::dds_::SportModeState_& state);
    static Vec12Msg ExtractRobotBodyStateVec12Msg(const unitree_go::msg::dds_::SportModeState_& state);

    void HandleMessage(const void* message);

    std::string network_interface_;
    unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::SportModeState_> subscriber_;
    mutable std::mutex state_mutex_;
    unitree_go::msg::dds_::SportModeState_ latest_state_{};
    bool has_state_{false};
};

}  // namespace edge::go2_api::provider
