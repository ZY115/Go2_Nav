/*
 * Copyright (c) 2025-2026 Yihao Cai - All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 */

 #pragma once

#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include "edge_quadruped_controller/control/LowPassFilter.h"
#include "edge_quadruped_controller/utils/math/mathTypes.h"
#include "edge_quadruped_controller/utils/math/mathTools.h"
#include "edge_quadruped_controller/robot_model/QuadrupedModel.hpp"

struct CtrlInterfaces;

// Forward declarations for ROS2 types
namespace rclcpp_lifecycle
{
    class LifecycleNode;
}

namespace edge::control::quadruped
{

class PhaseScheduler;
class QuadrupedModel;
struct CtrlComponent;

class Estimator
{
public:
    explicit Estimator(CtrlInterfaces& ctrl_interfaces, CtrlComponent& ctrl_component,
                       std::string odom_frame_id, std::string base_frame_id,
                       std::shared_ptr<rclcpp_lifecycle::LifecycleNode> node);

    ~Estimator() = default;

    /**
     * Get the estimated robot central position in world frame
     * @return robot centriodal position (CoM position)
     */
    Vec3 getPosition()
    {
        return x_hat_.segment(0, 3);
    }

    /**
     * Get the estimated robot central velocity in world frame
     * @return robot centriodal velocity (CoM velocity)
     */
    Vec3 getVelocity()
    {
        return x_hat_.segment(3, 3);
    }

    /**
     * Get the estimated foot position in world frame
     * @param index leg index
     * @return foot position in world frame
     */
    Vec3 getFootPos(const int index)
    {
        return getPosition() + rotation_ * robot_model_->getFootPos(index, FrameType::BODY);
    }

    /**
     * Get the estimated feet velocity in world frame
     * @return feet velocity in world frame
     */
    Vec34 getFeetPos()
    {
        Vec34 feet_pos;
        for (int i(0); i < 4; ++i)
        {
            feet_pos.col(i) = getFootPos(i);
        }
        return feet_pos;
    }

    /**
     * Get the estimated feet velocity in world frame
     * @return feet velocity in world frame
     */
    Vec34 getFeetVel()
    {
        Vec34 feet_vel_body = robot_model_->getFeetVelInBodyFrame();
        Vec34 feet_vel_world;
        for (int i(0); i < 4; ++i)
        {
            feet_vel_world.col(i) = feet_vel_body.col(i) + getVelocity();
        }
        // Vec34 feet_vel_world;
        // Vec34 foot_vel_body = robot_model_->getFeetVelInBodyFrame();
        // Vec34 foot_pos_body = robot_model_->getFeetPosInBodyFrame();
        // for (int i = 0; i < 4; ++i) {
        //     feet_vel_world.col(i) = rotation_ * (foot_vel_body.col(i) + skew(gyro_) * 
        //                             foot_pos_body.col(i)) + getVelocity();
        // }

        // std::cout << "feet_vel_world: " << std::endl << feet_vel_world << std::endl;
        return feet_vel_world;
    };

    /**
     * Get the estimated foot position in body frame (similar to original estimator)
     * @return foot positions relative to body in world coordinate frame
     *
     * The importance notice：
     * - Returned foot positions are relative to the robot body frame
     * - This is different from the original implementation which returns foot positions in world frame
     */
    Vec34 getFeetPosInBodyFrame()
    {
        // Transform the vectorized foot positions to a 3x4 matrix
        Vec34 feet_pos_body_matrix;
        for (int i = 0; i < 4; ++i)
        {
            feet_pos_body_matrix.col(i) = feet_pos_body_.segment(3 * i, 3);
        }
        return feet_pos_body_matrix;
    }

    RotMat getRotation()
    {
        return rotation_;
    }

    Vec3 getGyro()
    {
        return gyro_;
    }

    [[nodiscard]] Vec3 getGyroGlobal() const
    {
        return rotation_ * gyro_;
    }

    [[nodiscard]] double getYaw() const;

    [[nodiscard]] double getDYaw() const
    {
        return getGyroGlobal()(2);
    }

    /**
     * Get foot positions in world frame (similar to original estimator)
     * @return foot positions in world frame
     */
    [[nodiscard]] Vec34 getFeetPosInWorldFrame() const;

    /**
     * Get foot velocities in world frame (similar to original estimator)
     * @return foot velocities in world frame
     */
    [[nodiscard]] Vec34 getFeetVelInWorldFrame() const;

    void update();
    void reset();

    /**
     * Publish odometry and tf data
     */
    void publishOdometryAndTf();

private:
    CtrlInterfaces& ctrl_interfaces_;
    std::shared_ptr<QuadrupedModel>& robot_model_;
    std::shared_ptr<PhaseScheduler>& phase_scheduler_;

    Eigen::Matrix<double, 18, 1> x_hat_; // The state of estimator, position(3)+velocity(3)+feet position(3x4)

    Eigen::Matrix<double, 3, 1> u_; // The input of estimator

    Eigen::Matrix<double, 28, 1> y_; // The measurement value of output y
    Eigen::Matrix<double, 28, 1> y_hat_; // The prediction of output y
    Eigen::Matrix<double, 18, 18> A; // The transtion matrix of estimator
    Eigen::Matrix<double, 18, 3> B; // The input matrix
    Eigen::Matrix<double, 28, 18> C; // The output matrix

    // Covariance Matrix
    Eigen::Matrix<double, 18, 18> P; // Prediction covariance
    Eigen::Matrix<double, 18, 18> Ppriori; // Priori prediction covariance
    Eigen::Matrix<double, 18, 18> Q; // Dynamic simulation covariance
    Eigen::Matrix<double, 28, 28> R; // Measurement covariance
    Eigen::Matrix<double, 18, 18> QInit_; // Initial value of Dynamic simulation covariance
    Eigen::Matrix<double, 28, 28> RInit_; // Initial value of Measurement covariance
    Eigen::Matrix<double, 18, 1> Qdig; // adjustable process noise covariance
    Eigen::Matrix<double, 3, 3> Cu; // The covariance of system input u

    // Output Measurement
    Eigen::Matrix<double, 12, 1> feet_pos_body_; // The feet positions to body, in the world frame
    Eigen::Matrix<double, 12, 1> feet_vel_body_; // The feet velocity to body, in the world frame
    Eigen::Matrix<double, 4, 1> feet_h_; // The Height of each foot, in the world frame

    // World frame foot positions (similar to original _feetPosGlobalKine)
    Vec34 feet_pos_global_kine_; // Foot positions in world frame
    Vec34 feet_vel_global_kine_; // Foot velocities in world frame
    Vec4 phase_;
    VecInt4 contact_;

    Eigen::Matrix<double, 28, 28> S; // _S = C*P*C.T + R
    Eigen::PartialPivLU<Eigen::Matrix<double, 28, 28>> Slu; // _S.lu()
    Eigen::Matrix<double, 28, 1> Sy; // _Sy = _S.inv() * (y - yhat)
    Eigen::Matrix<double, 28, 18> Sc; // _Sc = _S.inv() * C
    Eigen::Matrix<double, 28, 28> SR; // _SR = _S.inv() * R
    Eigen::Matrix<double, 28, 18> STC; // _STC = (_S.transpose()).inv() * C
    Eigen::Matrix<double, 18, 18> IKC; // _IKC = I - KC

    Vec3 g_;
    double dt_;

    RotMat rotation_;
    Vec3 acceleration_;
    Vec3 gyro_;

    std::vector<Vec3> foot_poses_;
    std::vector<Eigen::Vector3d> foot_vels_;
    std::vector<std::shared_ptr<LowPassFilter>> low_pass_filters_;

    double large_variance_;

    // ROS2 components for odometry and tf publishing
    std::shared_ptr<rclcpp_lifecycle::LifecycleNode> node_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_publisher_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    // Frame names
    std::string odom_frame_id_;
    std::string base_frame_id_;
};

} // namespace edge::control::quadruped