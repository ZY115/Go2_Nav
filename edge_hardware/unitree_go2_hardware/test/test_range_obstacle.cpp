/*
 * Copyright (c) 2025-2026 Yihao Cai - All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 * 
 * Test program for rangeObstacle() interface
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

int main(int argc, char** argv)
{
    const std::string network_interface =
        (argc > 1) ? argv[1] : "eth0";
    const int domain_id =
        (argc > 2) ? std::atoi(argv[2]) : 0;

    edge::go2_api::provider::Go2StateProvider state_provider(network_interface);

    try
    {
        // State provider subscription requires DDS participant initialization first.
        unitree::robot::ChannelFactory::Instance()->Init(domain_id, network_interface);
        state_provider.Init();
    }
    catch (const std::exception& e)
    {
        std::cerr << "[test_range_obstacle] init failed: " << e.what() << std::endl;
        std::cerr << "[test_range_obstacle] check network interface and DDS runtime libs."
                  << std::endl;
        return 1;
    }

    std::cout << "[test_range_obstacle] listening on nic="
              << network_interface
              << ", domain=" << domain_id << std::endl;

    auto last_wait_log = std::chrono::steady_clock::now();
    auto start_time = std::chrono::steady_clock::now();
    int sample_count = 0;
    
    std::array<float, 4> min_distance = {std::numeric_limits<float>::max(), 
                                         std::numeric_limits<float>::max(),
                                         std::numeric_limits<float>::max(),
                                         std::numeric_limits<float>::max()};
    std::array<float, 4> max_distance = {0.0f, 0.0f, 0.0f, 0.0f};
    std::array<double, 4> sum_distance = {0.0, 0.0, 0.0, 0.0};

    const int duration_seconds = 10;

    while (true)
    {
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() >= duration_seconds) {
            break;
        }

        if (state_provider.HasState())
        {
            auto state = state_provider.GetLatestState();

            // Access rangeObstacle - expecting std::array<float, 4>
            auto obstacle_distances = state.range_obstacle();

            // Format and print obstacle distance data
            std::cout << "Sample " << std::setw(3) << (sample_count + 1) << ": "
                      << "Obstacle["
                      << std::setw(5) << std::fixed << std::setprecision(2) << obstacle_distances[0] << ", "
                      << std::setw(5) << std::fixed << std::setprecision(2) << obstacle_distances[1] << ", "
                      << std::setw(5) << std::fixed << std::setprecision(2) << obstacle_distances[2] << ", "
                      << std::setw(5) << std::fixed << std::setprecision(2) << obstacle_distances[3]
                      << "] m" << std::endl;

            // Update statistics
            for (size_t i = 0; i < 4; ++i) {
                float dist = obstacle_distances[i];
                min_distance[i] = std::min(min_distance[i], dist);
                max_distance[i] = std::max(max_distance[i], dist);
                sum_distance[i] += static_cast<double>(dist);
            }

            sample_count++;
            last_wait_log = std::chrono::steady_clock::now();
        }
        else
        {
            const auto now = std::chrono::steady_clock::now();
            if (now - last_wait_log >= std::chrono::seconds(2))
            {
                std::cout << "[test_range_obstacle] waiting for rt/sportmodestate ..."
                          << " (nic=" << network_interface
                          << ", domain=" << domain_id << ")" << std::endl;
                last_wait_log = now;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Print statistics
    std::cout << std::endl;
    std::cout << "====== rangeObstacle Test Statistics ======" << std::endl;
    std::cout << "Total Samples: " << sample_count << std::endl;
    std::cout << std::endl;

    if (sample_count > 0) {
        for (size_t i = 0; i < 4; ++i) {
            double avg = sum_distance[i] / sample_count;
            std::cout << "Channel " << i << ": "
                      << "Min=" << std::fixed << std::setprecision(2) << min_distance[i] << "m, "
                      << "Max=" << std::fixed << std::setprecision(2) << max_distance[i] << "m, "
                      << "Avg=" << std::fixed << std::setprecision(2) << avg << "m"
                      << std::endl;
        }
    } else {
        std::cout << "No valid samples collected!" << std::endl;
    }

    std::cout << "==========================================" << std::endl;

    return 0;
}
