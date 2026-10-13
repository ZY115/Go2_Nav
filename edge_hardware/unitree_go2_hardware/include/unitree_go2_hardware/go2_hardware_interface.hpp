/*
 * Copyright (c) 2025-2026 Yihao Cai - All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef GO2_HARDWARE_INTERFACE_HPP
#define GO2_HARDWARE_INTERFACE_HPP

#include <rclcpp/rclcpp.hpp>
#include <hardware_interface/system_interface.hpp>
#include <unitree/idl/go2/WirelessController_.hpp>
#include <atomic>
#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/idl/go2/LowCmd_.hpp>
#include <unitree/idl/go2/SportModeState_.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

namespace edge::go2_hardware_interface
{

class Go2HardwareInterface final : public hardware_interface::SystemInterface
{
public:

    // Override Node LifecycleNodeInterface methods
    hardware_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;
    hardware_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;
    // hardware_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;
    // hardware_interface::CallbackReturn on_cleanup(const rclcpp_lifecycle::State& previous_state) override;
    // hardware_interface::CallbackReturn on_shutdown(const rclcpp_lifecycle::State& previous_state) override;
    // hardware_interface::CallbackReturn on_error(const rclcpp_lifecycle::State& previous_state) override;

    // Override necessary functions from HardwareInterface
    hardware_interface::CallbackReturn on_init(const hardware_interface::HardwareInfo& info) override;
    hardware_interface::return_type read(const rclcpp::Time& time, const rclcpp::Duration& period) override;
    hardware_interface::return_type write(const rclcpp::Time& time, const rclcpp::Duration& period) override;
    
    std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
    std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;
    // hardware_interface::return_type prepare_command_mode_switch(
    //     const std::vector<std::string>& start_interfaces,
    //     const std::vector<std::string>& stop_interfaces
    // ) override;  

private:
    std::vector<double> joint_kp_command_;
    std::vector<double> joint_kd_command_;
    std::vector<double> joint_torque_command_;
    std::vector<double> joint_position_command_;
    std::vector<double> joint_velocity_command_;

    std::vector<double> joint_state_position_;
    std::vector<double> joint_state_velocity_;
    std::vector<double> joint_state_effort_;

    std::vector<double> imu_states_;
    std::vector<double> foot_force_;
    std::vector<double> odom_states_;

    std::unordered_map<std::string, std::vector<std::string> > joint_interface_map_ = {
        {"position", {}},
        {"velocity", {}},
        {"effort", {}}
    };


    void InitLowCmd();

    void lowStateMessageHandler(const void *messages);

    void highStateMessageHandler(const void *messages);

    void remoteWirelessHandler(const void *messages);

    unitree_go::msg::dds_::LowCmd_ robot_low_cmd_{}; // default init
    unitree_go::msg::dds_::LowState_ robot_low_state_{}; // default init
    unitree_go::msg::dds_::SportModeState_ robot_high_state_{}; // default init

    int domain_ = 1;
    bool show_foot_force_ = false;

    // ---- 遥控器硬件级急停 ----------------------------------------------------
    // 机器狗固件即使在 App 关闭运动控制后，仍会把遥控器的按键/摇杆原始状态随
    // LowState.wireless_remote 以 500Hz 上报（实测验证）。字节 2-3 是 uint16 按键
    // 位域，L2=bit5(0x0020)、B=bit9(0x0200)，L2+B=0x0220 —— 与遥控器机身标注的
    // 急停组合一致。
    //
    // 这条链路不经过 WiFi / RustDesk / 笔记本 / X / 容器 GUI，只依赖机器狗到
    // Jetson 的网线和本进程存活，比键盘按 1 可靠得多。
    //
    // 锁存后只能重启本节点（go2vi）解除 —— 急停绝不能因为松手或误触而自动恢复。
    // estop_mask_ 设为 0 即禁用该功能。
    std::atomic<bool> estop_latched_{false};
    uint16_t estop_mask_ = 0x0220;   // L2 + B
    bool use_real_robot_ = false;
    std::string network_interface_ = "lo";
    std::string logger_name_ = "Unitree Go2 Hardware";

    /*publisher*/
    unitree::robot::ChannelPublisherPtr<unitree_go::msg::dds_::LowCmd_> low_cmd_publisher_;
    
    /*subscriber*/
    unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> low_state_subscriber_;
    unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::SportModeState_> high_state_subscriber_;
};

} // namespace edge::go2_hardware_interface

#endif // GO2_HARDWARE_INTERFACE_HPP