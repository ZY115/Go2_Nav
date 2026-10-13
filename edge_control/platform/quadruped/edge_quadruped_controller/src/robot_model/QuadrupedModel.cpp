
#include <iostream>
#include <Eigen/Dense>
#include <algorithm>
#include <cctype>
#include "controller_common/CtrlInterfaces.h"
#include "edge_quadruped_controller/robot_model/QuadrupedModel.hpp"

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/centroidal.hpp>
#include <cmath>

namespace edge::control::quadruped
{

QuadrupedModel::QuadrupedModel(CtrlInterfaces& ctrl_interfaces,
                               const std::string& urdf_path,
                               const std::vector<std::string>& joint_names,
                               const Vec12& default_stand_joint_positions)
    : IBaseLeggedModel(ctrl_interfaces),
      ctrl_interfaces_(ctrl_interfaces), 
      joint_names_(joint_names)
{
    // Initialize joint states
    current_joint_pos_ = Vec12::Zero();
    current_joint_vel_ = Vec12::Zero();

    // Initialize geometric parameters from URDF
    initializeFromURDF(urdf_path);

    // Compute foot positions in standing configuration
    computeStandFootPositions(default_stand_joint_positions);

    std::cout << "=== Simplified Kinematic System Initialization Complete ===" << std::endl;
    std::cout << "Mass: " << mass_ << " kg" << std::endl;
    std::cout << "Center of Mass Offset: [" << com_offset_body_.transpose() << "] m" << std::endl;
    std::cout << "Inertia Tensor Diagonal: [" << inertia_tensor_.diagonal().transpose() << "]" << std::endl;
}

void QuadrupedModel::initializeFromURDF(const std::string& urdf_path)
{
    try
    {
        // Load URDF model
        pinocchio::urdf::buildModel(urdf_path, model_);
        data_ = pinocchio::Data(model_);

        // Compute total mass
        mass_ = 0;
        for (const auto& inertia : model_.inertias)
        {
            mass_ += inertia.mass();
        }

        // Compute center of mass and inertia tensor when q is zero position
        Vec12 q_zero = Vec12::Zero();
        pinocchio::forwardKinematics(model_, data_, q_zero);

        // Compute overall center of mass in body frame
        com_offset_body_ = pinocchio::centerOfMass(model_, data_, q_zero);

        // Compute inertia tensor at center of mass using CCRBA algorithm
        const Vec12 v_zero = Vec12::Zero();
        pinocchio::ccrba(model_, data_, q_zero, v_zero);
        inertia_tensor_ = data_.Ig.inertia().matrix();

        // Compute and cache geometric parameters
        initializeGeometryParameters();
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error initializing from URDF: " << e.what() << std::endl;
        throw;
    }
}

void QuadrupedModel::initializeGeometryParameters()
{
    pinocchio::forwardKinematics(model_, data_, Vec12::Zero());
    pinocchio::updateFramePlacements(model_, data_);

    try
    {
        // Calculate link lengths based on joint placements
        constexpr int start_joint_index = 0; 

        // Abad, Hip, Knee joint IDs for the first leg
        pinocchio::JointIndex thigh_joint_id = model_.getJointId(joint_names_[start_joint_index + 1]);
        pinocchio::JointIndex calf_joint_id = model_.getJointId(joint_names_[start_joint_index + 2]);

        // Obtain link lengths
        pinocchio::SE3 hip_to_thigh = model_.jointPlacements[thigh_joint_id];
        link_lengths_[0] = std::abs(hip_to_thigh.translation().y()); // Offset in Y direction

        // Hip to Knee link length: Thigh length
        pinocchio::SE3 thigh_to_calf = model_.jointPlacements[calf_joint_id];
        link_lengths_[1] = std::sqrt(
            thigh_to_calf.translation().x() * thigh_to_calf.translation().x() +
            thigh_to_calf.translation().z() * thigh_to_calf.translation().z()
        );

        // Knee to Ankle link length: Calf length
        double knee_length = link_lengths_[1]; // Default to thigh length

        try
        {
            // Find the end effector of the first leg

            // Get Joint ID（FL_hip_joint）
            pinocchio::JointIndex first_leg_hip = model_.getJointId(joint_names_[0]);

            // Traverse all Frames to find those belonging to the first leg
            std::vector<pinocchio::FrameIndex> first_leg_frames;
            for (size_t i = 0; i < model_.nframes; ++i)
            {
                // Check if Frame belongs to the first leg's joint chain
                pinocchio::JointIndex frame_parent = model_.frames[i].parentJoint;
                if (frame_parent == first_leg_hip ||
                    frame_parent == thigh_joint_id ||
                    frame_parent == calf_joint_id)
                {
                    first_leg_frames.push_back(i);
                }
            }

            // If Frames are found, prioritize the one containing "foot" in its name, otherwise choose the last one
            if (!first_leg_frames.empty())
            {
                pinocchio::FrameIndex end_frame_id = first_leg_frames.back(); // Default to the last one

                // Prioritize the one containing "foot" in its name (case insensitive)
                for (const auto& frame_idx : first_leg_frames)
                {
                    std::string frame_name = model_.frames[frame_idx].name;
                    std::transform(frame_name.begin(), frame_name.end(), frame_name.begin(), ::tolower);
                    if (frame_name.find("foot") != std::string::npos)
                    {
                        end_frame_id = frame_idx;
                        break;
                    }
                }
                
                pinocchio::SE3 end_pose = data_.oMf[end_frame_id];

                // Calculate the distance from the calf joint to the end effector
                pinocchio::SE3 calf_pose = data_.oMi[calf_joint_id];
                Vec3 calf_to_end = end_pose.translation() - calf_pose.translation();

                knee_length = std::sqrt(
                    calf_to_end.x() * calf_to_end.x() +
                    calf_to_end.z() * calf_to_end.z()
                );

                // Validate knee length
                if (knee_length < 0.01 || knee_length > 0.5)
                {
                    link_lengths_[2] = link_lengths_[1];
                }

                std::cout << "Find the end effector Frame: " << model_.frames[end_frame_id].name
                    << ", Calf length: " << knee_length << std::endl;
            }
        }
        catch (const std::exception& e)
        {
            // If any error occurs, default to thigh length
            std::cerr << "Error finding end effector frame: " << e.what() << std::endl;
            link_lengths_[2] = link_lengths_[1];
        }

        link_lengths_[2] = knee_length;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error calculating link lengths: " << e.what() << std::endl;
        throw;
    }

    // Index legs to compute hip offsets
    for (int leg_index = 0; leg_index < 4; ++leg_index)
    {
        // Get hip joint index for the leg
        const int hip_joint_index = leg_index * 3;
        pinocchio::JointIndex hip_joint_id = model_.getJointId(joint_names_[hip_joint_index]);
        pinocchio::SE3 hip_pose = data_.oMi[hip_joint_id];
        hip_offsets_[leg_index] = hip_pose.translation().cast<double>();

        // Print leg geometric parameters
        std::string leg_names[4] = {"FR", "FL", "RR", "RL"};
        std::cout << "Leg " << leg_names[leg_index] << " (controller_idx=" << leg_index
            << "): Hip offset=[" << hip_offsets_[leg_index].transpose()
            << "], Links=[" << link_lengths_[0] << ", "
            << link_lengths_[1] << ", " << link_lengths_[2] << "]" << std::endl;
    }
}

void QuadrupedModel::computeStandFootPositions(const Vec12& default_stand_joint_positions)
{
    // Calculate the standing foot positions for each leg
    for (int leg_index = 0; leg_index < 4; leg_index++)
    {
        Vec3 q_leg = default_stand_joint_positions.segment(3 * leg_index, 3);
        // Vec3 foot_pos = calcLegFK(q_leg, leg_index);
        feet_pos_normal_stand_.col(leg_index) = calcLegFK(q_leg, leg_index) + hip_offsets_[leg_index];
    }

    std::cout << "=== Standing Foot Positions ===" << std::endl;
    std::cout << "FR: " << feet_pos_normal_stand_.col(0).transpose() << std::endl;
    std::cout << "FL: " << feet_pos_normal_stand_.col(1).transpose() << std::endl;
    std::cout << "RR: " << feet_pos_normal_stand_.col(2).transpose() << std::endl;
    std::cout << "RL: " << feet_pos_normal_stand_.col(3).transpose() << std::endl;
}

// ========== Implementation of the core kinematics ==========

Vec12 QuadrupedModel::getJointPos(const Vec34& feet_positions, FrameType frame) const
{
    Vec12 joint_angles = Vec12::Zero();

    for (int leg_index = 0; leg_index < 4; ++leg_index)
    {
        Vec3 target_pos = feet_positions.col(leg_index);
        Vec3 q_leg = calcQ(target_pos, frame, leg_index);
        joint_angles.segment(3 * leg_index, 3) = q_leg;
    }

    return joint_angles;
}

Vec12 QuadrupedModel::getJointVel(const Vec34& pos, const Vec34& vel, FrameType frame) const
{
    Vec12 qd;
    for (int i(0); i < 4; ++i)
    {
        // Directly call the original calcQd logic
        qd.segment(3 * i, 3) = calcLegQd(pos.col(i), vel.col(i), frame, i);
    }
    return qd;
}

Vec34 QuadrupedModel::getFeetPosInBodyFrame() const
{
    Vec34 feetPos;
    for (int i(0); i < 4; ++i)
    {
        feetPos.col(i) = getFootPos(i, FrameType::BODY);
    }
    return feetPos;
}

Vec3 QuadrupedModel::getFootPos(int leg_id, FrameType frame) const
{
    Vec3 q_leg = current_joint_pos_.segment(3 * leg_id, 3);

    if (frame == FrameType::BODY)
    {
        // Hip frame + offset = Body frame
        return calcLegFK(q_leg, leg_id) + hip_offsets_[leg_id];
    }
    if (frame == FrameType::HIP)
    {
        // Return foot position in hip frame
        return calcLegFK(q_leg, leg_id);
    }
    throw std::runtime_error("The frame of function: getFootPosition can only be BODY or HIP.");
}

Vec3 QuadrupedModel::getFootVelInBodyFrame(int leg_index) const
{
    if (leg_index < 0 || leg_index >= 4)
    {
        throw std::out_of_range("Invalid leg index");
    }

    Vec3 q_leg = current_joint_pos_.segment(3 * leg_index, 3);
    Vec3 qd_leg = current_joint_vel_.segment(3 * leg_index, 3);

    Mat3 jacobian = calcLegJacobian(q_leg, leg_index);
    return jacobian * qd_leg;
}

Vec3 QuadrupedModel::calcTorque(const Vec3& foot_force, int leg_index) const
{
    if (leg_index < 0 || leg_index >= 4)
    {
        throw std::out_of_range("Invalid leg index");
    }

    Vec3 q_leg = current_joint_pos_.segment(3 * leg_index, 3);
    Mat3 jacobian = calcLegJacobian(q_leg, leg_index);

    // Inverse dynamics: tau = J^T * f
    return jacobian.transpose() * foot_force;
}


Vec34 QuadrupedModel::getFeetVelInBodyFrame() const
{
    Vec34 feet_velocities;

    for (int leg_index = 0; leg_index < 4; ++leg_index)
    {
        feet_velocities.col(leg_index) = getFootVelInBodyFrame(leg_index);
    }

    return feet_velocities;
}

Mat3 QuadrupedModel::getJacobian(const Vec3& q_leg, int leg_index) const
{
    return calcLegJacobian(q_leg, leg_index);
}

// ========== Implementation of the core kinematics ==========

Vec3 QuadrupedModel::calcLegFK(const Vec3& q_leg, int leg_index) const
{
    // Confirm side sign
    int side_sign = leg_index == 1 || leg_index == 3 ? 1 : -1;
    double l1 = side_sign * link_lengths_[0];
    double l2 = -link_lengths_[1];
    double l3 = -link_lengths_[2];

    // Trigonometric functions
    double s1 = std::sin(q_leg(0)), c1 = std::cos(q_leg(0));
    double s2 = std::sin(q_leg(1)), c2 = std::cos(q_leg(1));
    double s3 = std::sin(q_leg(2)), c3 = std::cos(q_leg(2));

    double c23 = c2 * c3 - s2 * s3;
    double s23 = s2 * c3 + c2 * s3;

    // Unitree original forward kinematics formula (hip coordinate system)
    Vec3 p_ee_hip;
    p_ee_hip(0) = l3 * s23 + l2 * s2;
    p_ee_hip(1) = -l3 * s1 * c23 + l1 * c1 - l2 * c2 * s1;
    p_ee_hip(2) = l3 * c1 * c23 + l1 * s1 + l2 * c1 * c2;

    return p_ee_hip; // Return foot position in hip frame, keep consistent with original logic
}

Mat3 QuadrupedModel::calcLegJacobian(const Vec3& q_leg, int leg_index) const
{
    // Analytical Jacobian based on Unitree formula
    int side_sign = leg_index == 1 || leg_index == 3 ? 1 : -1;
    double l1 = side_sign * link_lengths_[0];
    double l2 = -link_lengths_[1];
    double l3 = -link_lengths_[2];

    double s1 = std::sin(q_leg(0)), c1 = std::cos(q_leg(0));
    double s2 = std::sin(q_leg(1)), c2 = std::cos(q_leg(1));
    double s3 = std::sin(q_leg(2)), c3 = std::cos(q_leg(2));

    double c23 = c2 * c3 - s2 * s3;
    double s23 = s2 * c3 + c2 * s3;

    // Unitree Jacobian formula
    Mat3 jacobian;
    jacobian(0, 0) = 0;
    jacobian(1, 0) = -l3 * c1 * c23 - l2 * c1 * c2 - l1 * s1;
    jacobian(2, 0) = -l3 * s1 * c23 - l2 * c2 * s1 + l1 * c1;
    jacobian(0, 1) = l3 * c23 + l2 * c2;
    jacobian(1, 1) = l3 * s1 * s23 + l2 * s1 * s2;
    jacobian(2, 1) = -l3 * c1 * s23 - l2 * c1 * s2;
    jacobian(0, 2) = l3 * c23;
    jacobian(1, 2) = l3 * s1 * s23;
    jacobian(2, 2) = -l3 * c1 * s23;

    return jacobian;
}

Vec3 QuadrupedModel::getHipOffset(int leg_index) const
{
    if (leg_index < 0 || leg_index >= 4)
    {
        throw std::out_of_range("Invalid leg index");
    }
    return hip_offsets_[leg_index];
}

Vec3 QuadrupedModel::calcQ(const Vec3& pEe, FrameType frame, int leg_index) const
{
    // The original logic: support two coordinate systems
    Vec3 pEe2H;
    if (frame == FrameType::HIP)
        pEe2H = pEe;
    else if (frame == FrameType::BODY)
        pEe2H = pEe - hip_offsets_[leg_index];
    else
        throw std::runtime_error("Frame type must be HIP or BODY");

    // Directly use the original analytical inverse kinematics algorithm (identical to unitreeLeg)
    double q1, q2, q3;
    Vec3 qResult;
    double px, py, pz;
    double b2y, b3z, b4z, a, b, c;

    px = pEe2H(0);
    py = pEe2H(1);
    pz = pEe2H(2);

    // Certify side sign for left/right legs (Order: FR, FL, RR, RL)
    int side_sign = leg_index == 1 || leg_index == 3 ? 1 : -1; // FL,RL为1, FR,RR为-1

    b2y = link_lengths_[0] * side_sign;
    b3z = -link_lengths_[1];
    b4z = -link_lengths_[2];
    a = link_lengths_[0];
    c = sqrt(pow(px, 2) + pow(py, 2) + pow(pz, 2)); // whole length
    b = sqrt(pow(c, 2) - pow(a, 2)); // distance between shoulder and footpoint

    // The original inverse kinematics formulas from Unitree
    q1 = q1_ik(py, pz, b2y);
    q3 = q3_ik(b3z, b4z, b);
    q2 = q2_ik(q1, q3, px, py, pz, b3z, b4z);

    qResult(0) = q1;
    qResult(1) = q2;
    qResult(2) = q3;

    return qResult;
}

Vec3 QuadrupedModel::calcLegQd(const Vec3& pEe, const Vec3& vEe, FrameType frame, int leg_index) const
{
    // The original logic: first calculate joint angles, then calculate joint velocities
    Vec3 q = calcQ(pEe, frame, leg_index);
    return calcLegJacobian(q, leg_index).inverse() * vEe;
}


void QuadrupedModel::update()
{
    // Read current joint states from control interfaces
    for (int i = 0; i < 12; i++)
    {
        current_joint_pos_(i) = ctrl_interfaces_.joint_position_state_interface_[i].get().get_value();
        current_joint_vel_(i) = ctrl_interfaces_.joint_velocity_state_interface_[i].get().get_value();
    }
}

// ========== The original inverse kinematics auxiliary functions ==========

double QuadrupedModel::q1_ik(double py, double pz, double l1) const
{
    double q1;
    double L = sqrt(pow(py, 2) + pow(pz, 2) - pow(l1, 2));
    q1 = atan2(pz * l1 + py * L, py * l1 - pz * L);
    return q1;
}

double QuadrupedModel::q2_ik(double q1, double q3, double px, double py, double pz, double b3z, double b4z) const
{
    double q2, a1, a2, m1, m2;

    a1 = py * sin(q1) - pz * cos(q1);
    a2 = px;
    m1 = b4z * sin(q3);
    m2 = b3z + b4z * cos(q3);
    q2 = atan2(m1 * a1 + m2 * a2, m1 * a2 - m2 * a1);
    return q2;
}

double QuadrupedModel::q3_ik(double b3z, double b4z, double b) const
{
    double q3, temp;
    temp = (pow(b3z, 2) + pow(b4z, 2) - pow(b, 2)) / (2 * fabs(b3z * b4z));
    if (temp > 1) temp = 1;
    if (temp < -1) temp = -1;
    q3 = acos(temp);
    q3 = -(M_PI - q3); // 0 ~ 180
    return q3;
}

} 