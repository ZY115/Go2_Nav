/*
 * Copyright (c) 2025-2026 Yihao Cai - All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

#include <unitree/robot/channel/channel_factory.hpp>

#include "unitree_go2_hardware/api/providers/go2_motion_provider.hpp"
#include "unitree_go2_hardware/api/providers/go2_state_provider.hpp"

namespace edge::go2_api
{

class Go2ApiProxy
{
public:

  explicit Go2ApiProxy(std::string network_interface)
  : network_interface_(std::move(network_interface)),
    motion_provider_(network_interface_),
    state_provider_(network_interface_)
  {}

  void Init()
  {
    unitree::robot::ChannelFactory::Instance()->Init(0, network_interface_);
    motion_provider_.Init();
    // state_provider_.Init();
  }

  /* motion API */
  void Move(double vx, double vy, double wz)
  {
    motion_provider_.Move(vx, vy, wz);
  }

  void Stop()
  {
    motion_provider_.Stop();
  }

  void Stand()
  {
    motion_provider_.Stand();
  }

  void Balance()
  {
    motion_provider_.Balance();
  }

  /* state API */

  bool HasState() const
  {
    return state_provider_.HasState();
  }

  auto GetState() const
  {
    return state_provider_.GetLatestState();
  }

  auto GetVec12Eigen() const
  {
    return state_provider_.GetLatestVec12Eigen();
  }

  auto GetVec12Msg() const
  {
    return state_provider_.GetLatestVec12Msg();
  }

private:

  std::string network_interface_;

  provider::Go2MotionProvider motion_provider_;

  provider::Go2StateProvider state_provider_;

};

} // namespace edge::go2_api