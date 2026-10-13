/*
 * Copyright (c) 2025-2026 Yihao Cai - All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <Eigen/Dense>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <algorithm>

#include "msg_utils.hpp"
#include "edge_msgs/msg/edge_action.hpp"

#include "edge_msgs/msg/vec12_msg.hpp"


namespace edge::ros_interfaces {

// ======================================================
// ==================== Type Aliases ====================
// ======================================================

using VectorXd  = Eigen::VectorXd;
using Vector2d  = Eigen::Matrix<double, 2, 1>;
using Vector3d  = Eigen::Matrix<double, 3, 1>;
using Vector6d  = Eigen::Matrix<double, 6, 1>;
using Vector12d = Eigen::Matrix<double, 12, 1>;
using Mat6x6    = Eigen::Matrix<double, 6, 6>;

// ======================================================
// =============== Eigen ↔ std::vector ==================
// ======================================================

/**
 * @brief Convert any Eigen matrix or vector to a std::vector<double>.
 */
template <typename Derived>
inline std::vector<double> EigenToVector(const Eigen::MatrixBase<Derived>& m) {
  return std::vector<double>(m.derived().data(), m.derived().data() + m.size());
}

/**
 * @brief Convert std::vector<double> to Eigen::VectorXd (dynamic size).
 */
inline Eigen::VectorXd VectorToEigenDynamic(const std::vector<double>& vec) {
  return Eigen::Map<const Eigen::VectorXd>(vec.data(), vec.size());
}

/**
 * @brief Convert std::vector<double> to fixed-size Eigen matrix or vector.
 *
 * Example:
 * ```cpp
 * auto v = VectorToEigen<6>(vec);     // 6x1
 * auto M = VectorToEigen<6,6>(vec);   // 6x6
 * ```
 *
 * @tparam Rows  Number of rows
 * @tparam Cols  Number of columns (default 1)
 * @throws std::runtime_error if vec.size() != Rows * Cols
 */
template <int Rows, int Cols = 1>
inline Eigen::Matrix<double, Rows, Cols> VectorToEigen(const std::vector<double>& vec) {
  const size_t expected = static_cast<size_t>(Rows * Cols);
  if (vec.size() != expected) {
    throw std::runtime_error(
      "VectorToEigen: size mismatch, expected " + std::to_string(expected) +
      " elements but got " + std::to_string(vec.size()) + ".");
  }
  return Eigen::Map<const Eigen::Matrix<double, Rows, Cols>>(vec.data());
}

// ======================================================
// =============== std::vector ↔ ROS msg ================
// ======================================================

/**
 * @brief Convert std::vector<double> to edge_msgs::msg::EdgeAction.
 */
inline edge_msgs::msg::EdgeAction VectorToEdgeActionMsg(const std::vector<double>& data, 
  uint8_t action_type, uint8_t action_mode) {

  edge_msgs::msg::EdgeAction msg;
  msg.header.stamp = edge::utils::getCurrentROSTime();
  msg.action_type = action_type;
  msg.action_mode = action_mode;
  msg.action_value = data;  // std::vector can be directly assigned to the message field
  
  return msg;
}

/**
 * @brief Convert edge_msgs::msg::EdgeAction to std::vector<double>.
 */
inline std::vector<double> EdgeActionMsgToVector(const edge_msgs::msg::EdgeAction& msg) {
  return std::vector<double>(msg.action_value.begin(), msg.action_value.end());
}


/**
 * @brief Convert std::vector<double> to edge_msgs::msg::Vec12.
 */
inline edge_msgs::msg::Vec12Msg VectorToVec12Msg(const std::vector<double>& data) {
  edge_msgs::msg::Vec12Msg msg;
  const size_t n = std::min(msg.data.size(), data.size());
  std::copy_n(data.begin(), n, msg.data.begin());
  return msg;
}

/**
 * @brief Convert edge_msgs::msg::Vec12 to std::vector<double>.
 */
inline std::vector<double> Vec12MsgToVector(const edge_msgs::msg::Vec12Msg& msg) {
  return std::vector<double>(msg.data.begin(), msg.data.end());
}

// ======================================================
// =============== Eigen ↔ ROS message ==================
// ======================================================

/**
 * @brief Convert EdgeAction message to Eigen Xd vector.
 */
inline Eigen::VectorXd EdgeActionMsgToEigen(const edge_msgs::msg::EdgeAction& msg) {
  return VectorToEigenDynamic(EdgeActionMsgToVector(msg));
}

/**
 * @brief Convert Eigen Xd vector to EdgeAction message.
 */
inline edge_msgs::msg::EdgeAction EigenToEdgeActionMsg(const Eigen::VectorXd& v, uint8_t action_type, uint8_t action_mode) {
  return VectorToEdgeActionMsg(EigenToVector(v), action_type, action_mode);
}

/**
 * @brief Convert Vec12 message to Eigen 12D vector.
 */
inline Eigen::VectorXd Vec12MsgToEigen(const edge_msgs::msg::Vec12Msg& msg) {
  return VectorToEigen<12>(Vec12MsgToVector(msg));
}

/**
 * @brief Convert Eigen 12D vector to Vec12 message.
 */
inline edge_msgs::msg::Vec12Msg EigenToVec12Msg(const Eigen::VectorXd& v) {
  return VectorToVec12Msg(EigenToVector(v));
}

// ======================================================
// =============== Utility / Debug Helpers ==============
// ======================================================

/**
 * @brief Convert an Eigen matrix to a formatted string (for logging or debugging).
 */
template <typename Derived>
inline std::string EigenToString(const Eigen::MatrixBase<Derived>& mat, int precision = 4) {
  Eigen::IOFormat fmt(precision, 0, ", ", "\n", "[", "]");
  std::stringstream ss;
  ss << mat.format(fmt);
  return ss.str();
}

/**
 * @brief Safe pretty-print wrapper for ROS2 logging (thread-local buffer).
 * Example:
 * ```cpp
 * RCLCPP_INFO(logger, "Kp:\n%s", EigenToCString(Kp).c_str());
 * ```
 */
template <typename Derived>
inline const char* EigenToCString(const Eigen::MatrixBase<Derived>& mat, int precision = 4) {
  static thread_local std::string buffer;
  buffer = EigenToString(mat, precision);
  return buffer.c_str();
}

/**
 * @brief Convert Eigen matrix to YAML-style flat std::vector (for saving to config files).
 */
template <typename Derived>
inline std::vector<double> EigenToYAMLVector(const Eigen::MatrixBase<Derived>& mat) {
  return EigenToVector(mat);
}

}  // namespace edge::ros_interfaces
