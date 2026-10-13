/*
 * Copyright (c) 2025-2026 Yihao Cai - All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cmath>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include "edge_trigger/safety_monitor.hpp"
#include <ament_index_cpp/get_package_share_directory.hpp>

#define SM_DEBUG(msg) \
    do { if (debug_mode_) { std::cout << msg << std::endl; } } while (0)

#define SM_DEBUG_ERR(msg) \
    do { if (debug_mode_) { std::cerr << msg << std::endl; } } while (0)


// ANSI colors
static const std::string CYAN  = "\033[36m";
static const std::string YELLOW = "\033[33m";
static const std::string GREEN = "\033[32m";
static const std::string RED = "\033[31m";
static const std::string RESET = "\033[0m";

static constexpr bool USE_COLOR = true;

static std::string color(const std::string& s, const std::string& c) {
    return USE_COLOR ? (c + s + RESET) : s;
}

// Helper: get visible length (excluding ANSI codes)
static size_t visibleLength(const std::string& s) {
    size_t len = 0;
    bool in_escape = false;
    for (char c : s) {
        if (c == '\033') {
            in_escape = true;
        } else if (in_escape && c == 'm') {
            in_escape = false;
        } else if (!in_escape) {
            len++;
        }
    }
    return len;
}

// Helper: pad string to width (considering ANSI codes) - always center
static std::string padTo(const std::string& s, size_t width) {
    size_t vis_len = visibleLength(s);
    if (vis_len >= width) return s;
    
    size_t total_pad = width - vis_len;
    size_t left = total_pad / 2;
    size_t right = total_pad - left;
    return std::string(left, ' ') + s + std::string(right, ' ');
}

namespace edge::trigger
{

SafetyMonitor::SafetyMonitor(const std::string& yaml_path)
{
    std::string final_path = parseYamlPath(yaml_path);
    loadFromYaml(final_path);
    printLoadedRules();
}

/**
 * Determine the YAML path to use
 */
std::string SafetyMonitor::parseYamlPath(const std::string& user_path)
{
    if (!user_path.empty()) {
        return user_path; // user override
    }

    // default YAML path
    std::string pkg_share = ament_index_cpp::get_package_share_directory("edge_trigger");
    return pkg_share + "/config/safety_rules.yaml";
}

/**
 * Load safety rules from YAML file
 */
void SafetyMonitor::loadFromYaml(const std::string& yaml_path)
{
    YAML::Node root = YAML::LoadFile(yaml_path);
    loadDebugConfig(root);        // Load debug config
    loadSwitchingConfig(root);   // Load switching config   
    loadFallDetectionConfig(root);         // Load fall detection settings
}

/**
 * Load debug config from YAML
 */
void SafetyMonitor::loadDebugConfig(const YAML::Node& root)
{
    if (root["debug"] && root["debug"]["enabled"]) {
        debug_mode_ = root["debug"]["enabled"].as<bool>();
        SM_DEBUG("Debug mode enabled via YAML");
    }
}

/**
 * Load switching config from YAML
 */
void SafetyMonitor::loadSwitchingConfig(const YAML::Node& root)
{
    if (!root["switching"]) {
        throw std::runtime_error("YAML missing 'switching' section");
    }

    YAML::Node switching_node = root["switching"];
    
    // Check if switching is enabled
    switching_enabled_ = switching_node["enabled"] ? switching_node["enabled"].as<bool>() : true;
    
    if (!switching_enabled_) {
        std::cout << "⚠️ Switching conditions are disabled in config" << std::endl;
        rules_.clear();
        return;
    }

    if (!switching_node["conditions"]) {
        throw std::runtime_error("YAML missing 'switching.conditions' section");
    }

    YAML::Node conditions_node = switching_node["conditions"];
    rules_.clear();

    for (auto it = conditions_node.begin(); it != conditions_node.end(); ++it) {
        std::string name = it->first.as<std::string>();
        YAML::Node node = it->second;

        int index = node["index"].as<int>();
        std::string unit = node["unit"].as<std::string>();
        
        // Determine rule type (default: relative)
        std::string type_str = node["type"] ? node["type"].as<std::string>() : "relative";
        RuleType type = (type_str == "absolute") ? RuleType::ABSOLUTE : RuleType::RELATIVE;
        
        SafetyRule rule;
        rule.name = name;
        rule.index = index;
        rule.unit = unit;
        rule.type = type;
        rule.limit = 0.0;
        rule.lower = 0.0;
        rule.upper = 0.0;
        
        if (type == RuleType::RELATIVE) {
            // Relative mode: |x| <= limit
            double raw_limit = node["limit"].as<double>();
            rule.limit = convertValue(raw_limit, unit);
        } else if (type == RuleType::ABSOLUTE) {
            // Absolute mode: lower < x < upper
            if (!node["bound"] || !node["bound"].IsSequence() || node["bound"].size() != 2) {
                throw std::runtime_error("Absolute rule '" + name + "' requires 'bound: [lower, upper]'");
            }
            double raw_lower = node["bound"][0].as<double>();
            double raw_upper = node["bound"][1].as<double>();
            rule.lower = convertValue(raw_lower, unit);
            rule.upper = convertValue(raw_upper, unit);
        } else {
            throw std::runtime_error("Unknown rule type for rule '" + name + "'");
        }
        
        rules_.push_back(rule);
    }
    
    std::cout << "✅ Loaded " << rules_.size() << " switching conditions" << std::endl;
}

/**
 * Load fall detection configuration from YAML
 */
void SafetyMonitor::loadFallDetectionConfig(const YAML::Node& root)
{
    if (!root["fall_detection"]) {
        std::cout << "⚠️ YAML missing 'fall_detection' section, fall detection disabled" << std::endl;
        fall_detection_enabled_ = false;
        return;
    }

    YAML::Node fall_node = root["fall_detection"];
    
    // Check if fall detection is enabled
    fall_detection_enabled_ = fall_node["enabled"] ? fall_node["enabled"].as<bool>() : true;

    if (!fall_detection_enabled_) {
        std::cout << "⚠️ Fall detection is disabled in config" << std::endl;
        return;
    }

    if (!fall_node["thresholds"]) {
        throw std::runtime_error("YAML missing 'fall_detection.thresholds' section");
    }

    YAML::Node thresholds = fall_node["thresholds"];
    
    // Load thresholds (all in SI units: meters and radians)
    fall_height_threshold_ = thresholds["height"] ? thresholds["height"].as<double>() : 0.15;
    fall_roll_threshold_ = thresholds["roll"] ? thresholds["roll"].as<double>() : 0.5;
    fall_pitch_threshold_ = thresholds["pitch"] ? thresholds["pitch"].as<double>() : 0.5;
    
    std::cout << "✅ Fall detection loaded: "
              << "height=" << fall_height_threshold_ << "m, "
              << "roll=" << fall_roll_threshold_ << "rad, "
              << "pitch=" << fall_pitch_threshold_ << "rad" << std::endl;
}


/**
 * Convert the user-provided value to internal value (always SI)
 */
double SafetyMonitor::convertValue(double raw, const std::string& unit)
{
    if (unit == "deg") {
        return raw * M_PI / 180.0; // convert to rad
    }
    if (unit == "deg/s") {
        return raw * M_PI / 180.0; // convert to rad/s
    }
    if (unit == "rad" || unit == "m" || unit == "m/s" || unit == "rad/s") {
        return raw;
    }
    std::cerr << "⚠️ Unknown unit: " << unit << " (treat as raw SI)" << std::endl;
    return raw; 
}

bool SafetyMonitor::detectFall(const std::vector<double>& robot_state)
{
    if (!fall_detection_enabled_) {
        return false; // Fall detection disabled
    }

    if (robot_state.size() < 6) {
        throw std::runtime_error("State vector too small in detectFall() - need at least 6 elements");
    }

    fall_reasons_.clear();      // Clear previous reasons
    bool is_falling = false;

    // Extract state values
    double height = robot_state[2];  // pz (index 2)
    double roll = robot_state[3];    // roll (index 3)
    double pitch = robot_state[4];   // pitch (index 4)

    std::vector<std::string> fall_details;

    // Check height drop (critical height threshold)
    if ((height < fall_height_threshold_) && (height > 1e-6)) {
        is_falling = true;
        fall_reasons_.push_back("height");

        std::ostringstream ss;
        ss << std::fixed << std::setprecision(3) 
           << "height=" << height << "m (< " << fall_height_threshold_ << "m)";
        fall_details.push_back(ss.str());
    }

    // Check roll angle (critical tilt)
    if (std::abs(roll) > fall_roll_threshold_) {
        is_falling = true;
        fall_reasons_.push_back("roll");
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(3) 
           << "roll=" << roll << " rad (> " << fall_roll_threshold_ << " rad)";
        fall_details.push_back(ss.str());
    }

    // Check pitch angle (critical tilt)
    if (std::abs(pitch) > fall_pitch_threshold_) {
        is_falling = true;
        fall_reasons_.push_back("pitch");
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(3) 
           << "pitch=" << pitch << " rad (> " << fall_pitch_threshold_ << " rad)";
        fall_details.push_back(ss.str());
    }

    // Log fall detection
    if (is_falling) {
        SM_DEBUG_ERR(color("🚨 FALL: ", RED));
        for (size_t i = 0; i < fall_details.size(); ++i) {
            SM_DEBUG_ERR(fall_details[i] + (i < fall_details.size() - 1 ? ", " : ""));
        }
        SM_DEBUG_ERR(""); // newline
    }

    return is_falling;
}


bool SafetyMonitor::evaluateSafety(RuleType ruleType,
                                   const std::vector<double>& state,
                                   double obstacle_distance,
                                   double goal_distance)
{
    if (state.size() < 12) {
        throw std::runtime_error("State vector too small in evaluateSafety()");
    }

    if (!switching_enabled_) {
        return true; // If switching is disabled, always return safe
    }

    violation_reasons_.clear();     // Clear previous reasons
    
    bool is_safe = true;
    std::vector<std::string> violation_details;

    for (const auto& r : rules_) {
        // Only check rules matching the specified type
        if (r.type != ruleType) {
            continue;
        }

        if (r.index == -1) {
            // Special case: obstacle distance
            if (r.name == "obstacle_distance" && obstacle_distance < r.limit) {
                is_safe = false;
                violation_reasons_.push_back("close_to_obstacle");
                violation_details.push_back(
                    r.name + ": " + std::to_string(obstacle_distance) + " < " + 
                    std::to_string(r.limit) + " (" + r.unit + ")"
                );
            }
            // Special case: goal distance
            if (r.name == "goal_distance" && goal_distance > r.limit) {
                is_safe = false;
                violation_reasons_.push_back("close_to_goal");
                violation_details.push_back(
                    r.name + ": " + std::to_string(goal_distance) + " > " + 
                    std::to_string(r.limit) + " (" + r.unit + ")"
                );
            }
            continue;
        }

        double v = state[r.index];
        
        if (r.type == RuleType::RELATIVE) {
            // Relative mode: |error_x| <= limit
            // state is errorState (already contains error/deviation)
            if (std::abs(v) > r.limit) {
                is_safe = false;
                violation_reasons_.push_back("violation_" + r.name);
                violation_details.push_back(
                    r.name + ": |" + std::to_string(v) + "| = " + 
                    std::to_string(std::abs(v)) + " > " + 
                    std::to_string(r.limit) + " (" + r.unit + ")"
                );
            }
        } else if (r.type == RuleType::ABSOLUTE) {
            // Absolute mode: lower < x < upper
            // state is robotState
            if (v <= r.lower || v >= r.upper) {
                is_safe = false;
                violation_reasons_.push_back("violation_" + r.name);
                violation_details.push_back(
                    r.name + ": " + std::to_string(v) + " not in (" + 
                    std::to_string(r.lower) + ", " + std::to_string(r.upper) + ") (" + r.unit + ")"
                );
            }
        } else {
            throw std::runtime_error("Unknown rule type in evaluateSafety()");
        }
    }

    is_last_safe_ = is_curr_safe_;
    is_curr_safe_ = is_safe;

    safetyStatusLogging(violation_details);
    return is_safe;
}

// --------------------------------------------------------
void SafetyMonitor::safetyStatusLogging(const std::vector<std::string>& violation_details)
{
    // Logging
    if (!is_curr_safe_) {
        std::string msg = "⚠️ Current System is UNSAFE! Reasons: ";
        for (auto &r : violation_details) msg += r + " | ";

        if (is_last_safe_)
            msg += "System leaving the learning space! Safety Triggered.";
        else
            msg += "System still outside the learning space.";

        SM_DEBUG_ERR(msg);
    } else {
        if (!is_last_safe_) {
            SM_DEBUG("✅ System returned to the learning space.");
        } else {
            SM_DEBUG("✅ System remains within the learning space.");
        }
    }
}


/**
 * Print loaded safety rules in a formatted table
 */
void SafetyMonitor::printLoadedRules() const
{
    const int NAME_W  = 20;
    const int IDX_W   = 5;
    const int TYPE_W  = 10;
    const int LIMIT_W = 14;
    const int UNIT_W  = 7;

    int total_w = NAME_W + IDX_W + TYPE_W + LIMIT_W + UNIT_W + 10; // +10 for separators

    auto line = [&]() {
        std::cout << "  " << std::string(total_w, '-') << "\n";
    };

    // ------------------------------------------------------------
    // Print Title
    // ------------------------------------------------------------
    std::string title = "🔄 Switching Conditions Loaded";
    std::cout << "\n  " << CYAN << std::string((total_w - title.size() + 2) / 2, '=')
              << " " << title << " "
              << std::string((total_w - title.size() + 2) / 2, '=') << RESET << "\n";

    // ------------------------------------------------------------
    // Print Header (all centered)
    // ------------------------------------------------------------
    std::cout << "  "
              << padTo(YELLOW + "Name" + RESET, NAME_W) << " | "
              << padTo(YELLOW + "Idx" + RESET, IDX_W) << " | "
              << padTo(YELLOW + "Type" + RESET, TYPE_W) << " | "
              << padTo(YELLOW + "Limit/Bounds" + RESET, LIMIT_W) << " | "
              << padTo(YELLOW + "Unit" + RESET, UNIT_W)
              << "\n";

    line();

    // ------------------------------------------------------------
    // Print Rows (all centered)
    // ------------------------------------------------------------
    for (const auto& r : rules_) {
        std::string bound_str;
        std::string type_str;
        
        if (r.type == RuleType::RELATIVE) {
            type_str = "relative";
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(4) << r.limit;
            bound_str = "±" + ss.str();
        } else {
            type_str = "absolute";
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(4) << "(" << r.lower << ", " << r.upper << ")";
            bound_str = ss.str();
        }
        
        std::cout << "  "
                  << padTo(r.name, NAME_W) << " | "
                  << padTo(std::to_string(r.index), IDX_W) << " | "
                  << padTo(type_str, TYPE_W) << " | "
                  << padTo(GREEN + bound_str + RESET, LIMIT_W) << " | "
                  << padTo(CYAN + r.unit + RESET, UNIT_W)
                  << "\n";
    }

    line();

    // ------------------------------------------------------------
    // Print Fall Detection (if enabled)
    // ------------------------------------------------------------
    if (fall_detection_enabled_) {
        std::cout << "\n";
        std::string fall_title = "🚨 Fall Detection";
        std::cout << "  " << RED << std::string((total_w - fall_title.size() + 2) / 2, '=')
                  << " " << fall_title << " "
                  << std::string((total_w - fall_title.size() + 2) / 2, '=') << RESET << "\n";

        line();

        std::ostringstream ss_h, ss_r, ss_p;
        ss_h << std::fixed << std::setprecision(4) << fall_height_threshold_;
        ss_r << std::fixed << std::setprecision(4) << fall_roll_threshold_;
        ss_p << std::fixed << std::setprecision(4) << fall_pitch_threshold_;

        std::cout << "  "
                  << padTo("height", NAME_W) << " | "
                  << padTo("", IDX_W) << " | "
                  << padTo("threshold", TYPE_W) << " | "
                  << padTo(GREEN + ss_h.str() + RESET, LIMIT_W) << " | "
                  << padTo(CYAN + "m" + RESET, UNIT_W)
                  << "\n";

        std::cout << "  "
                  << padTo("roll", NAME_W) << " | "
                  << padTo("", IDX_W) << " | "
                  << padTo("threshold", TYPE_W) << " | "
                  << padTo(GREEN + ss_r.str() + RESET, LIMIT_W) << " | "
                  << padTo(CYAN + "rad" + RESET, UNIT_W)
                  << "\n";

        std::cout << "  "
                  << padTo("pitch", NAME_W) << " | "
                  << padTo("", IDX_W) << " | "
                  << padTo("threshold", TYPE_W) << " | "
                  << padTo(GREEN + ss_p.str() + RESET, LIMIT_W) << " | "
                  << padTo(CYAN + "rad" + RESET, UNIT_W)
                  << "\n";

        line();
    }
    
    std::cout << std::endl;
}

}  // namespace edge::trigger
