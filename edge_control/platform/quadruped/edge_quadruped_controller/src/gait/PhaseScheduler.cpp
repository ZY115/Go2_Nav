/*
 * Copyright (c) 2025-2026 Yihao Cai - All rights reserved
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include "edge_quadruped_controller/gait/PhaseScheduler.h"
#include "controller_common/FSM/FSMState.h"

namespace edge::control::quadruped
{

PhaseScheduler::PhaseScheduler(const double period, const double st_ratio, const Vec4& bias,
                             const rclcpp::Time& start_time) : start_time_(start_time)
{
    phase_past_ << 0.5, 0.5, 0.5, 0.5;
    contact_past_.setZero();
    status_past_ = PhaseStatus::SWING_ALL;
    status_ = PhaseStatus::SWING_ALL;

    period_ = period;
    st_ratio_ = st_ratio;
    bias_ = bias;

    if (st_ratio_ >= 1 || st_ratio_ <= 0)
    {
        std::cerr << "[ERROR] The stancePhaseRatio of PhaseScheduler should between (0, 1)"
            << std::endl;
        exit(-1);
    }

    for (int i(0); i < bias_.rows(); i++)
    {
        if (bias_(i) > 1 || bias_(i) < 0)
        {
            std::cerr << "[ERROR] The bias of PhaseScheduler should between [0, 1]"
                << std::endl;
            exit(-1);
        }
    }
}

auto PhaseScheduler::update(const rclcpp::Time& time) -> void
{
    // Accumulate time since start
    accumulated_time_ = (time - start_time_).seconds();

    calcPhase(phase_, contact_, status_);

    if (status_ != status_past_)
    {
        if (switch_status_.sum() == 0)
        {
            switch_status_.setOnes();
        }
        calcPhase(phase_past_, contact_past_, status_past_);

        if (status_ == PhaseStatus::STANCE_ALL && status_past_ == PhaseStatus::SWING_ALL)
        {
            contact_past_.setOnes();
        }
        else if (status_ == PhaseStatus::SWING_ALL && status_past_ == PhaseStatus::STANCE_ALL)
        {
            contact_past_.setZero();
        }
    }

    if (switch_status_.sum() != 0)
    {
        for (int i(0); i < 4; ++i)
        {
            if (contact_(i) == contact_past_(i))
            {
                switch_status_(i) = 0;
            }
            else
            {
                contact_(i) = contact_past_(i);
                phase_(i) = phase_past_(i);
            }
        }
        if (switch_status_.sum() == 0)
        {
            status_past_ = status_;
        }
    }
}

void PhaseScheduler::calcPhase(Vec4& phase, VecInt4& contact, const PhaseStatus status)
{
    switch (status)
    {
    case PhaseStatus::WAVE_ALL:
        {
            // Calculate phase and contact for each foot (using accumulated time and bias)
            for (int i(0); i < 4; ++i)
            {
                normal_t_(i) =
                    fmod(accumulated_time_ + period_ - period_ * bias_(i), period_) / period_;
                if (normal_t_(i) < st_ratio_)
                {
                    contact(i) = 1;
                    phase(i) = normal_t_(i) / st_ratio_;
                }
                else
                {
                    contact(i) = 0;
                    phase(i) = (normal_t_(i) - st_ratio_) / (1 - st_ratio_);
                }
            }
            break;
        }
    case PhaseStatus::SWING_ALL:
        {
            contact.setZero();
            phase << 0.5, 0.5, 0.5, 0.5;
            break;
        }
    case PhaseStatus::STANCE_ALL:
        {
            contact.setOnes();
            phase << 0.5, 0.5, 0.5, 0.5;
            break;
        }
    }
}

void PhaseScheduler::setStatusFromFSM(FSMStateName fsm_state)
{
    switch (fsm_state)
    {
    case FSMStateName::FIXEDDOWN:
    case FSMStateName::FIXEDSTAND:
    case FSMStateName::FREESTAND:
    case FSMStateName::BALANCETEST:
    case FSMStateName::TROTTING:
        setStatus(PhaseStatus::STANCE_ALL);
        break;

    default:
        // Default to SWING_ALL for safety (PASSIVE, INVALID, and unknown states)
        setStatus(PhaseStatus::SWING_ALL);
        break;
    }
}

} 
