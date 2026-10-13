/*
 * Copyright (c) 2025-2026 Yihao Cai - All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 */

#include "unitree_go2_hardware/api/providers/go2_motion_provider.hpp"

#include <stdexcept>

namespace edge::go2_api::provider
{

Go2MotionProvider::Go2MotionProvider(std::string network_interface,
                               double stop_epsilon)
    : network_interface_(std::move(network_interface)),
      stop_epsilon_(stop_epsilon)
{
}

void Go2MotionProvider::Init()
{
    // SportClient internally touches DDS resources; construct it only after
    // ChannelFactory has initialized the participant.
    sport_client_ = std::make_unique<unitree::robot::go2::SportClient>();
    sport_client_->SetTimeout(10.0f);
    sport_client_->Init();
}

void Go2MotionProvider::Move(double vx, double vy, double wz)
{
    if (!sport_client_) {
        throw std::runtime_error("Go2MotionProvider::Init() must be called before Move().");
    }

    if (std::fabs(vx) < stop_epsilon_ &&
        std::fabs(vy) < stop_epsilon_ &&
        std::fabs(wz) < stop_epsilon_)
    {
        Stop();
        return;
    }

    sport_client_->Move(vx, vy, wz);
}

void Go2MotionProvider::Stop()
{
    if (!sport_client_) {
        throw std::runtime_error("Go2MotionProvider::Init() must be called before Stop().");
    }
    sport_client_->StopMove();
}

void Go2MotionProvider::Stand()
{
    if (!sport_client_) {
        throw std::runtime_error("Go2MotionProvider::Init() must be called before Stand().");
    }
    sport_client_->StandUp();
}

void Go2MotionProvider::Balance()
{
    if (!sport_client_) {
        throw std::runtime_error("Go2MotionProvider::Init() must be called before Balance().");
    }
    sport_client_->BalanceStand();
}

} // namespace edge::go2_api::provider