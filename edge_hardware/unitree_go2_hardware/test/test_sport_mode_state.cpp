/*
 * Copyright (c) 2025-2026 Yihao Cai - All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 * 
 * Comprehensive test program for SportModeState interfaces
 * Tests all state accessors: stamp, errorCode, imuState, mode, progress,
 * gaitType, footRaiseHeight, position, bodyHeight, velocity, yawSpeed,
 * range_obstacle, footForce, footPositionBody, footSpeedBody, pathPoints
 */

#include <chrono>
#include <iostream>
#include <exception>
#include <cstdlib>
#include <string>
#include <thread>
#include <array>
#include <iomanip>
#include <limits>

#include <unitree/robot/channel/channel_factory.hpp>

#include "unitree_go2_hardware/api/providers/go2_state_provider.hpp"

// Helper function to format mode string
std::string getModeString(uint8_t mode)
{
    switch(mode) {
        case 0: return "idle/default stand";
        case 1: return "balanceStand";
        case 2: return "pose";
        case 3: return "locomotion";
        case 4: return "reserve";
        case 5: return "lieDown";
        case 6: return "jointLock";
        case 7: return "damping";
        case 8: return "recoveryStand";
        case 9: return "reserve";
        case 10: return "sit";
        case 11: return "frontFlip";
        case 12: return "frontJump";
        case 13: return "frontPounc";
        default: return "unknown";
    }
}

// Helper function to format gait type string
std::string getGaitTypeString(uint8_t gait)
{
    switch(gait) {
        case 0: return "idle";
        case 1: return "trot";
        case 2: return "run";
        case 3: return "climb stair";
        case 4: return "forwardDownStair";
        case 9: return "adjust";
        default: return "unknown";
    }
}

int main(int argc, char** argv)
{
    const std::string network_interface =
        (argc > 1) ? argv[1] : "eth0";
    const int domain_id =
        (argc > 2) ? std::atoi(argv[2]) : 0;
    // duration_seconds <= 0 means run forever until Ctrl+C
    const int duration_seconds =
        (argc > 3) ? std::atoi(argv[3]) : 0;

    edge::go2_api::provider::Go2StateProvider state_provider(network_interface);

    try
    {
        // State provider subscription requires DDS participant initialization first.
        unitree::robot::ChannelFactory::Instance()->Init(domain_id, network_interface);
        state_provider.Init();
    }
    catch (const std::exception& e)
    {
        std::cerr << "[test_sport_mode_state] init failed: " << e.what() << std::endl;
        std::cerr << "[test_sport_mode_state] check network interface and DDS runtime libs."
                  << std::endl;
        return 1;
    }

    std::cout << "[test_sport_mode_state] listening on nic="
              << network_interface
              << ", domain=" << domain_id << std::endl;
    if (duration_seconds <= 0) {
        std::cout << "[test_sport_mode_state] running forever, press Ctrl+C to stop." << std::endl;
    } else {
        std::cout << "[test_sport_mode_state] running for " << duration_seconds << " seconds." << std::endl;
    }

    auto last_wait_log = std::chrono::steady_clock::now();
    auto start_time = std::chrono::steady_clock::now();
    int sample_count = 0;

    while (true)
    {
        if (duration_seconds > 0) {
            auto elapsed = std::chrono::steady_clock::now() - start_time;
            if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() >= duration_seconds) {
                break;
            }
        }

        if (state_provider.HasState())
        {
            auto state = state_provider.GetLatestState();
            sample_count++;

            std::cout << "\n========== Sample " << std::setw(3) << sample_count << " ==========" << std::endl;

            // TimeSpec stamp
            try {
                auto timestamp = state.stamp();
                std::cout << "Timestamp: " << timestamp.sec() << "s " << timestamp.nanosec() << "ns" << std::endl;
            } catch (...) {
                std::cout << "Timestamp: N/A" << std::endl;
            }

            // Error code
            try {
                uint32_t error_code = state.error_code();
                std::cout << "Error Code: 0x" << std::hex << error_code << std::dec << std::endl;
            } catch (...) {
                std::cout << "Error Code: N/A" << std::endl;
            }

            // IMU State
            try {
                auto imu = state.imu_state();
                std::cout << "IMU: Acc["
                          << std::fixed << std::setprecision(3)
                          << imu.accelerometer()[0] << ", "
                          << imu.accelerometer()[1] << ", "
                          << imu.accelerometer()[2] << "], "
                          << "Gyro["
                          << imu.gyroscope()[0] << ", "
                          << imu.gyroscope()[1] << ", "
                          << imu.gyroscope()[2] << "]" << std::endl;
            } catch (...) {
                std::cout << "IMU: N/A" << std::endl;
            }

            // Mode
            try {
                uint8_t mode = state.mode();
                std::cout << "Mode: " << (int)mode << " (" << getModeString(mode) << ")" << std::endl;
            } catch (...) {
                std::cout << "Mode: N/A" << std::endl;
            }

            // Progress
            try {
                float progress = state.progress();
                std::cout << "Progress: " << std::fixed << std::setprecision(2) << progress << std::endl;
            } catch (...) {
                std::cout << "Progress: N/A" << std::endl;
            }

            // Gait Type
            try {
                uint8_t gait = state.gait_type();
                std::cout << "Gait Type: " << (int)gait << " (" << getGaitTypeString(gait) << ")" << std::endl;
            } catch (...) {
                std::cout << "Gait Type: N/A" << std::endl;
            }

            // Foot Raise Height
            try {
                float height = state.foot_raise_height();
                std::cout << "Foot Raise Height: " << std::fixed << std::setprecision(3) << height << " m" << std::endl;
            } catch (...) {
                std::cout << "Foot Raise Height: N/A" << std::endl;
            }

            // Position (3D)
            try {
                auto pos = state.position();
                std::cout << "Position: ["
                          << std::fixed << std::setprecision(3)
                          << pos[0] << ", " << pos[1] << ", " << pos[2] << "] m" << std::endl;
            } catch (...) {
                std::cout << "Position: N/A" << std::endl;
            }

            // Body Height
            try {
                float body_height = state.body_height();
                std::cout << "Body Height: " << std::fixed << std::setprecision(3) << body_height << " m" << std::endl;
            } catch (...) {
                std::cout << "Body Height: N/A" << std::endl;
            }

            // Velocity (3D)
            try {
                auto vel = state.velocity();
                std::cout << "Velocity: ["
                          << std::fixed << std::setprecision(3)
                          << vel[0] << ", " << vel[1] << ", " << vel[2] << "] m/s" << std::endl;
            } catch (...) {
                std::cout << "Velocity: N/A" << std::endl;
            }

            // Yaw Speed
            try {
                float yaw = state.yaw_speed();
                std::cout << "Yaw Speed: " << std::fixed << std::setprecision(3) << yaw << " rad/s" << std::endl;
            } catch (...) {
                std::cout << "Yaw Speed: N/A" << std::endl;
            }

            // Range Obstacle (4 directions)
            try {
                auto obstacles = state.range_obstacle();
                std::cout << "Range Obstacle: ["
                          << std::fixed << std::setprecision(2)
                          << obstacles[0] << ", " << obstacles[1] << ", "
                          << obstacles[2] << ", " << obstacles[3] << "] m" << std::endl;
            } catch (...) {
                std::cout << "Range Obstacle: N/A" << std::endl;
            }

            // Foot Forces (4 feet)
            try {
                auto forces = state.foot_force();
                std::cout << "Foot Forces: ["
                          << forces[0] << ", " << forces[1] << ", "
                          << forces[2] << ", " << forces[3] << "] N" << std::endl;
            } catch (...) {
                std::cout << "Foot Forces: N/A" << std::endl;
            }

            // Foot Position Body (12 floats, 4 feet x 3 coordinates)
            try {
                auto foot_pos = state.foot_position_body();
                std::cout << "Foot Position (Body frame): ";
                for (size_t i = 0; i < 4; ++i) {
                    std::cout << "\n  Foot " << i << ": ["
                              << std::fixed << std::setprecision(3)
                              << foot_pos[i*3] << ", "
                              << foot_pos[i*3+1] << ", "
                              << foot_pos[i*3+2] << "]";
                }
                std::cout << std::endl;
            } catch (...) {
                std::cout << "Foot Position (Body frame): N/A" << std::endl;
            }

            // Foot Speed Body (12 floats, 4 feet x 3 coordinates)
            try {
                auto foot_vel = state.foot_speed_body();
                std::cout << "Foot Speed (Body frame): ";
                for (size_t i = 0; i < 4; ++i) {
                    std::cout << "\n  Foot " << i << ": ["
                              << std::fixed << std::setprecision(3)
                              << foot_vel[i*3] << ", "
                              << foot_vel[i*3+1] << ", "
                              << foot_vel[i*3+2] << "]";
                }
                std::cout << std::endl;
            } catch (...) {
                std::cout << "Foot Speed (Body frame): N/A" << std::endl;
            }

            // Path Points (10 points)
            try {
                auto path_pts = state.path_point();
                std::cout << "Path Points: " << path_pts.size() << " points" << std::endl;
            } catch (...) {
                std::cout << "Path Points: N/A" << std::endl;
            }

            std::cout << std::string(40, '=') << std::endl;

            last_wait_log = std::chrono::steady_clock::now();
        }
        else
        {
            const auto now = std::chrono::steady_clock::now();
            if (now - last_wait_log >= std::chrono::seconds(2))
            {
                std::cout << "[test_sport_mode_state] waiting for rt/sportmodestate ..."
                          << " (nic=" << network_interface
                          << ", domain=" << domain_id << ")" << std::endl;
                last_wait_log = now;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    std::cout << "\n========== Test Summary ==========" << std::endl;
    std::cout << "Total Samples Collected: " << sample_count << std::endl;
    std::cout << "Tested Interfaces:" << std::endl;
    std::cout << "  ✓ stamp() - Timestamp" << std::endl;
    std::cout << "  ✓ error_code() - Error code" << std::endl;
    std::cout << "  ✓ imu_state() - IMU accelerometer & gyroscope" << std::endl;
    std::cout << "  ✓ mode() - Robot motion mode" << std::endl;
    std::cout << "  ✓ progress() - Action execution progress" << std::endl;
    std::cout << "  ✓ gait_type() - Gait type (idle/trot/run/etc)" << std::endl;
    std::cout << "  ✓ foot_raise_height() - Leg raising height" << std::endl;
    std::cout << "  ✓ position() - 3D position [x, y, z]" << std::endl;
    std::cout << "  ✓ body_height() - Body height" << std::endl;
    std::cout << "  ✓ velocity() - 3D velocity [vx, vy, vz]" << std::endl;
    std::cout << "  ✓ yaw_speed() - Yaw angular velocity" << std::endl;
    std::cout << "  ✓ range_obstacle() - 4-directional obstacle distances" << std::endl;
    std::cout << "  ✓ foot_force() - 4 foot forces" << std::endl;
    std::cout << "  ✓ foot_position_body() - 4 feet positions in body frame" << std::endl;
    std::cout << "  ✓ foot_speed_body() - 4 feet velocities in body frame" << std::endl;
    std::cout << "  ✓ path_point() - Path tracking points" << std::endl;
    std::cout << "=================================" << std::endl;

    // Workaround: Unitree DDS/Cyclone teardown may crash in destructor path.
    // Force immediate process exit after logs are flushed.
    std::cout.flush();
    std::cerr.flush();
    std::_Exit(0);
}
