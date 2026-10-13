/*
 * Copyright (c) 2025-2026 Yihao Cai - All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vector>
#include <iostream>
#include <unordered_map>
#include <yaml-cpp/yaml.h>

#include "edge_rl_runtime/reward_composer.hpp"
#include "edge_rl_runtime/reward_function.hpp"
#include <ament_index_cpp/get_package_share_directory.hpp>

namespace edge::rl_runtime {

RewardComposer::RewardComposer(const std::string& yaml_path) {

    registerAllRewardFunc();       // Ensure all rewards are registered

    std::string final_path;
    if (yaml_path.empty()) {
        std::string pkg_share = ament_index_cpp::get_package_share_directory("edge_rl_runtime");
        final_path = pkg_share + "/config/env_reward.yaml";

        std::cout << "[RewardComposer] Using default reward file: " 
                << final_path << "\n";
    } else {
        final_path = yaml_path;
        std::cout << "[RewardComposer] Using custom reward file: "
                << final_path << "\n";
    }

    YAML::Node node = YAML::LoadFile(final_path);
    if (node["reward"]) {
        for (auto it : node["reward"]) {
            reward_scales_[it.first.as<std::string>()] = it.second.as<double>();
        }
    }
    std::cout << "[RewardComposer] Loaded reward scales:\n";
    for (auto& kv : reward_scales_) {
        std::cout << "  - " << kv.first << ": " << kv.second << "\n";
    }
}

std::vector<edge_msgs::msg::EdgeReward> RewardComposer::computeTotalReward(const RewardFactor& factors) const {
    std::vector<edge_msgs::msg::EdgeReward> reward_msgs;
    reward_msgs.reserve(reward_scales_.size());

    auto& registry = RewardRegistry::instance();

    // Compute each reward component matching the configured ones
    for (auto& kv : reward_scales_) {
        const std::string& name = kv.first;
        double scale = kv.second;

        std::cout << "[RewardComposer] Computing reward for '" << name << "' with scale " << scale << "\n";

        // Lookup the reward function
        auto rew_fn = registry.lookupRewardFunc(name);
        if (!rew_fn) {
            throw std::logic_error(
                "[RewardComposer] INTERNAL ERROR: Reward '" + name + 
                "' doesn't match any registered reward function!"
            );
        }

        // Compute the reward
        double r = rew_fn->compute(factors);

        std::cout << "[RewardComposer] Reward '" << name << "': " << r 
                << " (scale: " << scale << ", scaled: " << r*scale << ")\n";

        // Validate the reward value
        bool valid = std::isfinite(r);
        if (!valid) {
            std::cerr
                << "[RewardComposer][WARN] Reward '" << name
                << "' is NaN/Inf. Set to 0.0\n";
            r = 0.0;
        }

        edge_msgs::msg::EdgeReward msg;
        msg.name = name;
        msg.value = r * scale;
        reward_msgs.emplace_back(std::move(msg));
    }

    return reward_msgs;
}
} // namespace edge::rl_runtime