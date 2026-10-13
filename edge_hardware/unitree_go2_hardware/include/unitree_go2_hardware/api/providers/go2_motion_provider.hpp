/*
 * Copyright (c) 2025-2026 Yihao Cai - All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cmath>
#include <memory>
#include <string>

#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/go2/sport/sport_client.hpp>

namespace edge::go2_api::provider
{

class Go2MotionProvider
{
public:

    explicit Go2MotionProvider(std::string network_interface,
                            double stop_epsilon = 1e-4);

    void Init();

    void Move(double vx, double vy, double wz);

    void Stop();

    void Stand();

    void Balance();

private:

    std::string network_interface_;

    double stop_epsilon_;

    std::unique_ptr<unitree::robot::go2::SportClient> sport_client_;
};

} // namespace edge::go2_api::provider