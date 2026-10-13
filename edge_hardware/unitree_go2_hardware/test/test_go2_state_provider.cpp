#include <chrono>
#include <iostream>
#include <exception>
#include <cstdlib>
#include <string>
#include <thread>

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
        std::cerr << "[go2_state_provider_test] init failed: " << e.what() << std::endl;
        std::cerr << "[go2_state_provider_test] check network interface and DDS runtime libs."
                  << std::endl;
        return 1;
    }

    std::cout << "[go2_state_provider_test] listening on nic="
              << network_interface
              << ", domain=" << domain_id << std::endl;

    auto last_wait_log = std::chrono::steady_clock::now();

    while (true)
    {
        if (state_provider.HasState())
        {
            auto state = state_provider.GetLatestVec12Eigen();

            std::cout << "position: "
                      << state(0) << " "
                      << state(1) << " "
                      << state(2) << std::endl;

            std::cout << "rpy: "
                      << state(3) << " "
                      << state(4) << " "
                      << state(5) << std::endl;

            std::cout << "velocity: "
                      << state(6) << " "
                      << state(7) << " "
                      << state(8) << std::endl;

            std::cout << "gyro: "
                      << state(9) << " "
                      << state(10) << " "
                      << state(11) << std::endl;

            std::cout << "-------------------------" << std::endl;

            last_wait_log = std::chrono::steady_clock::now();
        }
        else
        {
            const auto now = std::chrono::steady_clock::now();
            if (now - last_wait_log >= std::chrono::seconds(2))
            {
                std::cout << "[go2_state_provider_test] waiting for rt/sportmodestate ..."
                          << " (nic=" << network_interface
                          << ", domain=" << domain_id << ")" << std::endl;
                last_wait_log = now;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    return 0;
}