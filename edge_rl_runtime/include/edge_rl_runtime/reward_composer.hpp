/*
 * Copyright (c) 2025-2026 Yihao Cai - All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <any>
#include <string>
#include <memory>
#include <unordered_map>
#include <functional>
#include <edge_msgs/msg/edge_reward.hpp>

namespace edge::rl_runtime {

using RewardFactor = std::unordered_map<std::string, std::any>;

// Abstract base class for reward functions
class RewardFunction {
public:
    virtual ~RewardFunction() = default;
    virtual double compute(const RewardFactor& factors) const = 0;
};

// Singleton registry for reward functions
class RewardRegistry {
private:
    std::unordered_map<std::string, std::shared_ptr<RewardFunction>> registry_;
    RewardRegistry() = default;

public:
    RewardRegistry(const RewardRegistry&) = delete;
    RewardRegistry& operator=(const RewardRegistry&) = delete;

    static RewardRegistry& instance() {
        static RewardRegistry inst;
        return inst;
    }

    void registerReward(const std::string& name, std::function<double(const RewardFactor&)> fn) {
        struct LambdaWrapper : RewardFunction {
            std::function<double(const RewardFactor&)> f;
            LambdaWrapper(std::function<double(const RewardFactor&)> f_) : f(f_) {}
            double compute(const RewardFactor& factors) const override { return f(factors); }
        };
        registry_[name] = std::make_shared<LambdaWrapper>(fn);
    }

    std::shared_ptr<RewardFunction> lookupRewardFunc(const std::string& name) const {
        auto it = registry_.find(name);
        if (it != registry_.end()) return it->second;
        return nullptr;
    }

    size_t size() const { 
        return registry_.size(); 
    }
};

// Composer class to compute total rewards
class RewardComposer {
private:
    std::unordered_map<std::string, double> reward_scales_;

public:
    explicit RewardComposer(const std::string& yaml_path = "");

    std::vector<edge_msgs::msg::EdgeReward> computeTotalReward(
        const RewardFactor& factors) const;

    const std::unordered_map<std::string, double>& getRewardScales() const {
        return reward_scales_;
    }

    size_t size() const {
        return reward_scales_.size();
    }
};

inline void registerAllRewardFunc();


} // namespace edge::rl_runtime