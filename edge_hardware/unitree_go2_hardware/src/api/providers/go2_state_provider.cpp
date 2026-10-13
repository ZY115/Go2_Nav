/*
 * Copyright (c) 2025-2026 Yihao Cai - All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include <utility>
#include <unistd.h>

#include "unitree_go2_hardware/api/providers/go2_state_provider.hpp"

namespace edge::go2_api::provider {

Go2StateProvider::Go2StateProvider(std::string network_interface)
    : network_interface_(std::move(network_interface)),
      subscriber_(kTopicHighState) {}

void Go2StateProvider::Init() {
    std::cout << "Start to subscribe " << kTopicHighState << std::endl;
    subscriber_.InitChannel(
        [this](auto&& msg)
        {
            HandleMessage(std::forward<decltype(msg)>(msg));
        },
        1);
}

bool Go2StateProvider::HasState() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return has_state_;
}

unitree_go::msg::dds_::SportModeState_ Go2StateProvider::GetLatestState() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return latest_state_;
}

Go2StateProvider::Vector12d Go2StateProvider::GetLatestVec12Eigen() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return ExtractRobotBodyState(latest_state_);
}

Go2StateProvider::Vec12Msg Go2StateProvider::GetLatestVec12Msg() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return ExtractRobotBodyStateVec12Msg(latest_state_);
}

void Go2StateProvider::PrintLatestState() const {
    unitree_go::msg::dds_::SportModeState_ latest;
    bool has_state = false;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        latest = latest_state_;
        has_state = has_state_;
    }

    if (!has_state) {
        std::cout << "[Go2StateProvider] No state received yet." << std::endl;
        return;
    }

    auto print_state = [](const char* label, const unitree_go::msg::dds_::SportModeState_& state)
    {
        std::cout << label
                  << " position: "
                  << "x=" << state.position()[0] << ", "
                  << "y=" << state.position()[1] << ", "
                  << "z=" << state.position()[2] << " | ";

        std::cout << "rpy: "
                  << "r=" << state.imu_state().rpy()[0] << ", "
                  << "p=" << state.imu_state().rpy()[1] << ", "
                  << "y=" << state.imu_state().rpy()[2] << " | ";

        std::cout << "vel: "
                  << "vx=" << state.velocity()[0] << ", "
                  << "vy=" << state.velocity()[1] << ", "
                  << "vz=" << state.velocity()[2] << " | ";

        std::cout << "gyro: "
                  << "wx=" << state.imu_state().gyroscope()[0] << ", "
                  << "wy=" << state.imu_state().gyroscope()[1] << ", "
                  << "wz=" << state.imu_state().gyroscope()[2] << std::endl;

        std::cout << std::endl;
    };

    print_state("[latest]", latest);
}

void Go2StateProvider::Run() const {
    while (true) {
        if (HasState()) {
            const auto state = GetLatestState();
            std::cout << "position: "
                      << "x: " << state.position()[0] << ", "
                      << "y: " << state.position()[1] << ", "
                      << "z: " << state.position()[2] << std::endl;

            std::cout << "velocity: "
                      << "vx: " << state.velocity()[0] << ", "
                      << "vy: " << state.velocity()[1] << ", "
                      << "vz: " << state.velocity()[2] << std::endl;

            std::cout << "euler angles: "
                      << "roll: " << state.imu_state().rpy()[0] << ", "
                      << "pitch: " << state.imu_state().rpy()[1] << ", "
                      << "yaw: " << state.imu_state().rpy()[2] << std::endl;

            std::cout << "angular velocity: "
                      << "wx: " << state.imu_state().gyroscope()[0] << ", "
                      << "wy: " << state.imu_state().gyroscope()[1] << ", "
                      << "wz: " << state.imu_state().gyroscope()[2] << std::endl;

            std::cout << std::endl;
        }
        usleep(20000);
    }
}

Go2StateProvider::Vector12d Go2StateProvider::ExtractRobotBodyState(
    const unitree_go::msg::dds_::SportModeState_& state) {
    Vector12d vec12 = Vector12d::Zero();
    vec12(0) = state.position()[0];                     // px
    vec12(1) = state.position()[1];                     // py
    vec12(2) = state.position()[2];                     // pz   
    vec12(3) = state.imu_state().rpy()[0];              // roll
    vec12(4) = state.imu_state().rpy()[1];              // pitch
    vec12(5) = state.imu_state().rpy()[2];              // yaw
    vec12(6) = state.velocity()[0];                     // vx
    vec12(7) = state.velocity()[1];                     // vy
    vec12(8) = state.velocity()[2];                     // vz
    vec12(9) = state.imu_state().gyroscope()[0];        // wx
    vec12(10) = state.imu_state().gyroscope()[1];       // wy
    vec12(11) = state.imu_state().gyroscope()[2];       // wz
    return vec12;
}

Go2StateProvider::Vec12Msg Go2StateProvider::ExtractRobotBodyStateVec12Msg(
    const unitree_go::msg::dds_::SportModeState_& state) {
    return edge::ros_interfaces::EigenToVec12Msg(ExtractRobotBodyState(state));
}

void Go2StateProvider::HandleMessage(const void* message) {
    const auto* state_msg = static_cast<const unitree_go::msg::dds_::SportModeState_*>(message);

    std::lock_guard<std::mutex> lock(state_mutex_);
    latest_state_ = *state_msg;
    has_state_ = true;
}

}  // namespace edge::go2_api::provider
