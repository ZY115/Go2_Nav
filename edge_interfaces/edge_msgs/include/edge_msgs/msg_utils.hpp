/*
 * Copyright (c) 2025-2026 Yihao Cai - All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cstdint>
#include <string>
#include <stdexcept>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <builtin_interfaces/msg/time.hpp>

namespace edge::utils {

// ============================================================
// Action Mode
// ============================================================
enum class ActionMode : uint8_t {
    STUDENT = 0,
    TEACHER = 1,
    UNCERTAIN = 2,
};

inline uint8_t ToMsg(ActionMode mode) {
    return static_cast<uint8_t>(mode);
}

inline ActionMode FromMsg(uint8_t value) {
    switch (value) {
        case 0: return ActionMode::STUDENT;
        case 1: return ActionMode::TEACHER;
        default: return ActionMode::UNCERTAIN;
    }
}

inline std::string ToString(ActionMode mode) {
    switch (mode) {
        case ActionMode::STUDENT: return "STUDENT";
        case ActionMode::TEACHER: return "TEACHER";
        default: return "UNCERTAIN";
    }
}

inline ActionMode FromInt(uint8_t value) {
    return FromMsg(value);
}

// ============================================================
// Action Type (Control Space)
// ============================================================

enum class ActionType : uint8_t {
    BASE_ACCEL   = 0,
    CMD_VEL      = 1,
    JOINT_POS    = 2,
    JOINT_TORQUE = 3,
};

inline uint8_t ToMsg(ActionType type) {
    return static_cast<uint8_t>(type);
}

inline ActionType FromMsgActionType(uint8_t value) {
    switch (value) {
        case 0: return ActionType::BASE_ACCEL;
        case 1: return ActionType::CMD_VEL;
        case 2: return ActionType::JOINT_POS;
        case 3: return ActionType::JOINT_TORQUE;
        default:
            throw std::runtime_error("Unknown ActionType value");
    }
}

inline std::string ToString(ActionType type) {
    switch (type) {
        case ActionType::BASE_ACCEL:   return "BASE_ACCEL";
        case ActionType::CMD_VEL:      return "CMD_VEL";
        case ActionType::JOINT_POS:    return "JOINT_POS";
        case ActionType::JOINT_TORQUE: return "JOINT_TORQUE";
        default: return "UNKNOWN";
    }
}

inline int dimension(ActionType type, int num_joints = 0)
{
    switch (type)
    {
        case ActionType::CMD_VEL:
            return 3;

        case ActionType::BASE_ACCEL:
            return 6;

        case ActionType::JOINT_POS:
        case ActionType::JOINT_TORQUE:
            if (num_joints <= 0) {
                throw std::runtime_error(
                    "Joint action requires valid num_joints > 0");
            }
            return num_joints;

        default:
            throw std::runtime_error("Unknown ActionType");
    }
}

// ============================================================
// Time Utility
// ============================================================

inline builtin_interfaces::msg::Time getCurrentROSTime() {
  static rclcpp::Clock clock(RCL_ROS_TIME);
  return clock.now();
}

}  // namespace edge::utils