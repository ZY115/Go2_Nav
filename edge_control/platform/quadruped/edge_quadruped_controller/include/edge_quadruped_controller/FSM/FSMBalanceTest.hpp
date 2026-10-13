#pragma once

#include "controller_common/FSM/FSMState.h"
#include "edge_quadruped_controller/utils/math/mathTypes.h"


// Forward declarations for ROS2 types
namespace edge::control::quadruped
{
    class PhaseScheduler;
    class BalanceCtrl;
    class QuadrupedModel;
    class Estimator;
    struct CtrlComponent;
}

struct CtrlInterfaces;

using namespace edge::control::quadruped;

class FSMBalanceTest final : public FSMState {
public:
    explicit FSMBalanceTest(CtrlInterfaces &ctrl_interfaces,
                              CtrlComponent &ctrl_component,
                              const std::vector<double>& stable_gains);

    void enter() override;

    void run(const rclcpp::Time &time,
             const rclcpp::Duration &period) override;

    void exit() override;

    FSMStateName checkChange() override;

private:
    void calcTorque();
    void setStableGain() const;

    std::shared_ptr<Estimator> &estimator_;
    std::shared_ptr<QuadrupedModel> &robot_model_;
    std::shared_ptr<BalanceCtrl> &balance_ctrl_;
    std::shared_ptr<PhaseScheduler> &phase_scheduler_;

    Vec3 pcd_, pcd_init_;
    RotMat Rd_;
    RotMat init_rotation_;

    double kp_w_;
    Mat3 Kp_p_, Kd_p_, Kd_w_;
    Vec3 dd_pcd_, d_wbd_;

    float _xMax, _xMin;
    float _yMax, _yMin;
    float _zMax, _zMin;
    float _yawMax, _yawMin;

    // Stable gain parameters
    std::vector<double> stable_gains_;
};

