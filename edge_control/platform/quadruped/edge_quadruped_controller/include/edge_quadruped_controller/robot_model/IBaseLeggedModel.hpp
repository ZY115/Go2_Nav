#pragma once
#include <string>
#include <vector>
#include <array>
#include <pinocchio/algorithm/frames.hpp>
#include <controller_common/common/enumClass.h>
#include <edge_quadruped_controller/utils/math/mathTypes.h>

struct CtrlInterfaces;

/** 
 * @brief Quadruped Robot Kinematic Model Interface
 * 
 * Defines the kinematic and dynamic interfaces for a quadruped robot,
 * including forward/inverse kinematics, Jacobian computation, and dynamics.
 * This is an abstract base class; specific implementations should derive from this class.
 * 
 */

namespace edge::control::quadruped
{

class IBaseLeggedModel
{
public:
    explicit IBaseLeggedModel(CtrlInterfaces& ctrl_interfaces)
        : ctrl_interfaces_(ctrl_interfaces) {}

    virtual ~IBaseLeggedModel() = default;

    /// Inverse kinematics: Calculate joint angles from foot positions
    virtual Vec12 getJointPos(const Vec34& feet_positions,
                              FrameType frame = FrameType::BODY) const = 0;

    /// Differential inverse kinematics: Calculate joint velocities from foot positions and velocities
    virtual Vec12 getJointVel(const Vec34& pos, const Vec34& vel,
                              FrameType frame = FrameType::BODY) const = 0;

    /// Forward kinematics: Calculate all foot positions
    virtual Vec34 getFeetPosInBodyFrame() const = 0;

    /// Forward kinematics: Calculate single foot position
    virtual Vec3 getFootPos(int leg_id,
                            FrameType frame = FrameType::BODY) const = 0;

    /// Differential forward kinematics: Calculate single leg foot velocity
    virtual Vec3 getFootVelInBodyFrame(int leg_index) const = 0;

    /// Differential forward kinematics: Calculate all foot velocities
    virtual Vec34 getFeetVelInBodyFrame() const = 0;

    /// Inverse dynamics: Calculate joint torques from foot forces
    virtual Vec3 calcTorque(const Vec3& foot_force, int leg_index) const = 0;

    /// Jacobian matrix calculation
    virtual Mat3 getJacobian(const Vec3& q_leg, int leg_index) const = 0;

    /// Single leg inverse kinematics
    virtual Vec3 calcQ(const Vec3& pEe, FrameType frame, int leg_index) const = 0;

    /// Single leg differential inverse kinematics
    virtual Vec3 calcLegQd(const Vec3& pEe, const Vec3& vEe,
                           FrameType frame, int leg_index) const = 0;

    /// Update current joint state (e.g., read from control interface)
    virtual void update() = 0;

    // ========== General parameter access interface ==========
    [[nodiscard]] virtual double getRobotMass() const = 0;
    [[nodiscard]] virtual Vec3 getComOffset() const = 0;
    [[nodiscard]] virtual Mat3 getRobotInertia() const = 0;

protected:
    CtrlInterfaces& ctrl_interfaces_;  /// Control interface reference
};

} // namespace edge::control::quadruped