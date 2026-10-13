/*
 * Copyright (c) 2025-2026 Yihao Cai - All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <yaml-cpp/yaml.h>

/*******   ANSI colors for printing   *******/ 
#define C_RESET   "\033[0m"
#define C_TITLE   "\033[1;36m"   // Cyan
#define C_HEADER  "\033[1;37m"   // White bold
#define C_VALUE   "\033[0;33m"   // Yellow
#define C_UNIT    "\033[0;32m"   // Green


namespace edge::trigger
{

// Rule type: 
// 1. relative = |x - x_ref| <= limit, 
// 2. absolute = lower < x < upper
enum class RuleType { RELATIVE, ABSOLUTE };

struct SafetyRule {
    RuleType type;      // RELATIVE or ABSOLUTE
    std::string name;   // rule name (yaml key)
    int index;          // state index
    double limit;       // limit for relative mode (after unit conversion)
    double lower;       // lower bound for absolute mode
    double upper;       // upper bound for absolute mode
    std::string unit;   // unit used in logging
};

class SafetyMonitor
{
public:
    explicit SafetyMonitor(const std::string& yaml_path="");
    
    void setDebugMode(bool enable) { debug_mode_ = enable; }
    bool isDebugMode() const { return debug_mode_; }

    // type: RELATIVE (state=errorState) or ABSOLUTE (state=robotState)
    bool evaluateSafety(RuleType ruleType,
                        const std::vector<double>& state,
                        double obstacle_distance = 0.0, 
                        double goal_distance = 0.0);
    bool detectFall(const std::vector<double>& robot_state);

    const std::vector<std::string>& getViolationReasons() const {
        return violation_reasons_;
    }

    const std::vector<std::string>& getFallReasons() const {
        return fall_reasons_;
    }

private:
    bool debug_mode_ = false;       // Debug mode flag

    // ===== Switching Conditions =====
    bool switching_enabled_ = true;
    std::vector<SafetyRule> rules_;
    std::vector<std::string> violation_reasons_;
    bool is_curr_safe_ = true;
    bool is_last_safe_ = true;

    // ===== Fall Detection =====
    bool fall_detection_enabled_ = false;
    double fall_height_threshold_ = 0.12;  // meters
    double fall_roll_threshold_ = 1.0;     // radians
    double fall_pitch_threshold_ = 1.0;    // radians
    std::vector<std::string> fall_reasons_;

    // ===== Private Methods =====
    void loadFromYaml(const std::string& yaml_path);
    void loadDebugConfig(const YAML::Node& root);
    void loadSwitchingConfig(const YAML::Node& root);
    void loadFallDetectionConfig(const YAML::Node& root);
    std::string parseYamlPath(const std::string& user_path);
    void safetyStatusLogging(const std::vector<std::string>& violation_details);

    double convertValue(double raw, const std::string& unit);
    void printLoadedRules() const;
};

}  // namespace edge::trigger
