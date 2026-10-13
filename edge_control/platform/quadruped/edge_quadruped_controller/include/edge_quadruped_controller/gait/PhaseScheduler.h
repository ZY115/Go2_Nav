/*
 * Copyright (c) 2025-2026 Yihao Cai - All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 */

 #pragma once

#include <chrono>
#include <rclcpp/rclcpp.hpp>
#include <controller_common/common/enumClass.h>
#include <edge_quadruped_controller/utils/math/mathTypes.h>

// Forward declaration
enum class FSMStateName;

namespace edge::control::quadruped {

class PhaseScheduler {
public:
    PhaseScheduler(double period, double st_ratio, const Vec4 &bias, const rclcpp::Time& start_time);

    ~PhaseScheduler() = default;
    void update(const rclcpp::Time& time);

    /**
     * Set phase status
     * @param status new phase status
     */
    void setStatus(PhaseStatus status) { status_ = status; }

    /**
     * Set phase status based on FSM state
     * @param fsm_state FSM state name
     */
    void setStatusFromFSM(FSMStateName fsm_state);

    /**
     * Get current phase status
     * @return current phase status
     */
    [[nodiscard]] PhaseStatus getStatus() const { return status_; }

    [[nodiscard]] double get_t_stance() const { return period_ * st_ratio_; }
    [[nodiscard]] double get_t_swing() const { return period_ * (1 - st_ratio_); }
    [[nodiscard]] double get_t() const { return period_; }

    Vec4 phase_;
    VecInt4 contact_;
    PhaseStatus status_{};

private:
    /**
     * Update phase, contact and status based on current time.
     * @param phase foot phase
     * @param contact foot contact
     * @param status Phase Status
     */
    void calcPhase(Vec4 &phase, VecInt4 &contact, PhaseStatus status);

    double period_{};
    double st_ratio_{}; // stance phase ratio
    Vec4 bias_;

    Vec4 normal_t_; // normalize time [0,1)
    Vec4 phase_past_; // foot phase
    VecInt4 contact_past_; // foot contact
    VecInt4 switch_status_;
    PhaseStatus status_past_;

    rclcpp::Time start_time_;  // The start time of the phase scheduler
    double accumulated_time_{0.0};  // The accumulated time since start_time_
};

} // namespace edge::control::quadruped