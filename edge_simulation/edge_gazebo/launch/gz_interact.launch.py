#!/usr/bin/env python3
# ============================================================================
# Copyright (c) 2025-2026 Yihao Cai - All rights reserved
# SPDX-License-Identifier: Apache-2.0
# ============================================================================

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # ------------------------
    # Launch Arguments
    # ------------------------
    robot_name_arg = DeclareLaunchArgument("robot_name", default_value="go2", description="Robot name (e.g. go2, go1, a1)")
    world_name_arg = DeclareLaunchArgument("world_name", default_value="openwild_outdoor.world", description="Gazebo world name")
    use_sim_time_arg = DeclareLaunchArgument("use_sim_time", default_value="true", description="Use simulation (Gazebo) clock")
    reset_position_x_arg = DeclareLaunchArgument("reset_position_x", default_value="0.0", description="Reset X")
    reset_position_y_arg = DeclareLaunchArgument("reset_position_y", default_value="0.0", description="Reset Y")
    reset_position_z_arg = DeclareLaunchArgument("reset_position_z", default_value="0.35", description="Reset Z")

    robot_name = LaunchConfiguration("robot_name")
    world_name = LaunchConfiguration("world_name")
    use_sim_time = LaunchConfiguration("use_sim_time")
    reset_position_x = LaunchConfiguration("reset_position_x")
    reset_position_y = LaunchConfiguration("reset_position_y")
    reset_position_z = LaunchConfiguration("reset_position_z")
    
    # ------------------------
    # Launch Components
    # ------------------------
    gz_reset_handler_node = Node(
        package='edge_gazebo', 
        executable='gz_reset_handler', 
        name='gz_reset_handler_node',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'world_name': world_name,
            'robot_name': robot_name,
            'reset_position_x': reset_position_x,
            'reset_position_y': reset_position_y,
            'reset_position_z': reset_position_z,
            'reset_orientation_x': 0.0,
            'reset_orientation_y': 0.0,
            'reset_orientation_z': 0.0,
            'reset_orientation_w': 1.0,
            'service_timeout': 3000,
            'reset_world': False,
            'reset_robot_pose': True,
            'wait_after_reset': 0.5,
        }]
    )
    
    gz_force_handler_node = Node(
        package='edge_gazebo', 
        executable='gz_force_handler', 
        name='gz_force_handler_node',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_name': robot_name,
            'force_topic': '/external_force',
            'apply_rate': 100.0,
        }]
    )

    return LaunchDescription([
        # -- launch arguments --
        robot_name_arg,
        use_sim_time_arg,
        reset_position_x_arg,
        reset_position_y_arg,
        reset_position_z_arg,
        world_name_arg,
        
        # -- Launch Components --
        gz_reset_handler_node,
        # gz_force_handler_node,
    ])
