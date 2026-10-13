
#pragma once
#include <string>
#include <vector>
#include <pinocchio/algorithm/frames.hpp>
#include <controller_common/common/enumClass.h>
#include <edge_quadruped_controller/robot_model/IBaseLeggedModel.hpp>
#include <edge_quadruped_controller/utils/math/mathTypes.h>

struct CtrlInterfaces;


/**
 * Simplified Quadruped Robot Kinematic Model
 * Uses Pinocchio for geometric parameter calculation from URDF
 * - Automatically calculates geometric parameters (link lengths, hip offsets, etc.)
 */

namespace edge::control::quadruped
{   

class QuadrupedModel : public IBaseLeggedModel
{
public:
    explicit QuadrupedModel(CtrlInterfaces& ctrl_interfaces,
                            const std::string& urdf_path,
                            const std::vector<std::string>& joint_names,
                            const Vec12& default_stand_joint_positions);

    ~QuadrupedModel() = default;

    /**
     * Inverse kinematics: Calculate joint angles from foot positions (original interface)
     * @param feet_positions Foot position matrix
     * @param frame Coordinate frame type (default BODY)
     * @return Joint angles (12x1 vector)
     */
    [[nodiscard]] Vec12 getJointPos(const Vec34& feet_positions, 
                                    FrameType frame = FrameType::BODY) const override;

    /**
     * Differential inverse kinematics: Calculate joint velocities from foot positions and velocities (original interface)
     * @param pos Foot position matrix
     * @param vel Foot velocity matrix
     * @param frame Coordinate frame type (default BODY)
     * @return Joint velocities (12x1 vector)
     */
    [[nodiscard]] Vec12 getJointVel(const Vec34& pos, const Vec34& vel, 
                                    FrameType frame = FrameType::BODY) const override;

    /**
     * Forward kinematics: Calculate all foot positions (original interface)
     * @return Foot position matrix (3x4, controller order: FR FL RR RL)
     */
    [[nodiscard]] Vec34 getFeetPosInBodyFrame() const override;

    /**
     * Forward kinematics: Calculate single foot position (original interface)
     * @param leg_id Leg index (controller order: 0=FR, 1=FL, 2=RR, 3=RL)
     * @param frame Coordinate frame type (default BODY)
     * @return Foot position (3x1 vector)
     */
    [[nodiscard]] Vec3 getFootPos(int leg_id, 
                                  FrameType frame = FrameType::BODY) const override;

    /**
     * Differential forward kinematics: Calculate single leg foot velocity (original interface)
     * @param leg_index Leg index
     * @return Foot velocity (3x1 vector)
     */
    [[nodiscard]] Vec3 getFootVelInBodyFrame(int leg_index) const override;

    /**
     * Differential forward kinematics: Calculate all foot velocities (original interface)
     * @return Foot velocity matrix (3x4, controller order: FR FL RR RL)
     */
    [[nodiscard]] Vec34 getFeetVelInBodyFrame() const override;

    /**
     * Inverse kinematics: Calculate joint torques from foot forces (original interface)
     * @param foot_force Foot Force
     * @param leg_index Leg index
     * @return Joint torques (3x1 vector)
     */
    [[nodiscard]] Vec3 calcTorque(const Vec3& foot_force, int leg_index) const override;

    /**
     * Calculate Jacobian matrix (original interface)
     * @param q_leg Single leg joint angles (3x1 vector)
     * @param leg_index Leg index
     * @return Jacobian matrix (3x3)
     */
    [[nodiscard]] Mat3 getJacobian(const Vec3& q_leg, int leg_index) const override;

    /**
     * Single leg inverse kinematics (original interface)
     * @param pEe Foot position
     * @param frame Frame type
     * @param leg_index Leg index
     * @return Joint angles
     */
    [[nodiscard]] Vec3 calcQ(const Vec3& pEe, FrameType frame, int leg_index) const override;

    /**
     * Single leg differential inverse kinematics (original interface)
     * @param pEe Foot position
     * @param vEe Foot velocity
     * @param frame Frame type
     * @param leg_index Leg index
     * @return Joint velocities
     */
    [[nodiscard]] Vec3 calcLegQd(const Vec3& pEe, const Vec3& vEe, FrameType frame, int leg_index) const override;

    // ========== Robot parameters interface ==========
    /**
     * Get robot mass
     */
    [[nodiscard]] double getRobotMass() const override { return mass_; } 

    /**
     * Get robot center of mass offset
     */
    [[nodiscard]] Vec3 getComOffset() const override { return com_offset_body_; }

    /**
     * Get robot inertia tensor
     */
    [[nodiscard]] Mat3 getRobotInertia() const override { return inertia_tensor_; }

    /**
     * Update joint states (read from control interface)
     */
    void update() override;

public:
    // ========== Public member variables ==========
    Vec12 current_joint_pos_;
    Vec12 current_joint_vel_;
    Vec34 feet_pos_normal_stand_;
    double mass_ = 0;

private:
    // ========== Core Kinematics Algorithms (Based on Unitree Original) ==========

    /**
     * Single leg forward kinematics (Unitree original algorithm)
     * @param q_leg Joint angles (3x1 vector)
     * @param leg_index Leg index
     * @return Foot position (hip frame, consistent with calcPEe2H)
     */
    [[nodiscard]] Vec3 calcLegFK(const Vec3& q_leg, int leg_index) const;

    /**
     * Analytical Jacobian calculation for a single leg (Unitree original algorithm)
     * @param q_leg Leg joint angles (3x1 vector)
     * @param leg_index Leg index
     * @return Jacobian matrix (3x3)
     */
    [[nodiscard]] Mat3 calcLegJacobian(const Vec3& q_leg, int leg_index) const;

    // ========== Inverse Kinematics (Unitree Original Algorithm) ==========
    
    /**
     * Calculate q1 joint angle (Abad joint) - Original algorithm
     * @param py Foot Y Coordinate (Hip frame)
     * @param pz Foot Z Coordinate (Hip frame)
     * @param l1 Abad link length
     * @return q1 joint angle
     */
    [[nodiscard]] double q1_ik(double py, double pz, double l1) const;

    /**
     * Calculate q2 joint angle (Hip joint) - Original algorithm
     * @param q1 q1 joint angle
     * @param q3 q3 joint angle
     * @param px Foot X Coordinate (Hip frame)
     * @param py Foot Y Coordinate (Hip frame)
     * @param pz Foot Z Coordinate (Hip frame)
     * @param b3z Hip joint Z offset
     * @param b4z Knee joint Z offset
     * @return q2 joint angle
     */

    [[nodiscard]] double q2_ik(double q1, double q3, double px, double py, double pz, double b3z, double b4z) const;

    /**
     * Calculate q3 joint angle (Knee joint) - Original algorithm
     * @param b3z Hip joint Z offset
     * @param b4z Knee joint Z offset
     * @param b Foot position to shoulder distance
     * @return q3 joint angle
     */
    [[nodiscard]] double q3_ik(double b3z, double b4z, double b) const;


    // ========== Geometric Parameters (Extracted from Pinocchio Model) ==========

    /**
     * Get hip offset
     * @param leg_index Leg index
     * @return Hip offset relative to the body
     */
    [[nodiscard]] Vec3 getHipOffset(int leg_index) const;

private:

    CtrlInterfaces& ctrl_interfaces_;

    // Pinocchio Model and Data
    pinocchio::Model model_;
    pinocchio::Data data_;

    // Cached geometric parameters
    Vec3 link_lengths_; // [abad, hip, knee] - All legs use the same length parameters
    std::array<Vec3, 4> hip_offsets_; // [leg_index] Hip offsets

    // Joint names (used to directly access joint positions)
    std::vector<std::string> joint_names_;

    // Robot parameters
    Vec3 com_offset_body_; // Center of mass offset in body frame
    Mat3 inertia_tensor_; // Inertia tensor

    /**
     * Initialize model from URDF file by Pinocchio
     * @param urdf_path Path to URDF file
     */
    void initializeFromURDF(const std::string& urdf_path);

    /**
     * Initialize geometry parameters (extract from Pinocchio model)
     */
    void initializeGeometryParameters();

    /**
     * Compute foot positions for normal standing posture
     * @param default_stand_joint_positions Default standing joint angles
     */
    void computeStandFootPositions(const Vec12& default_stand_joint_positions);
};

} // namespace edge::control::quadruped
