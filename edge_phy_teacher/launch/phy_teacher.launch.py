#!/usr/bin/env python3
# ============================================================================
# Copyright (c) 2025-2026 Yihao Cai - All rights reserved
# SPDX-License-Identifier: Apache-2.0
# ============================================================================

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from ament_index_python import get_package_share_directory
from launch.substitutions import LaunchConfiguration, TextSubstitution, PythonExpression
from launch_ros.actions import Node

def generate_launch_description():

    # default config path
    default_config_path = os.path.join(
        get_package_share_directory('edge_phy_teacher'),
        'config',
        'physical_model.yaml'
    )
    
    # phy_teacher config path
    phy_teacher_config_path = os.path.join(
        get_package_share_directory('edge_phy_teacher'),
        'config',
        'phy_teacher.yaml'
    )

    if not os.path.exists(phy_teacher_config_path):
        raise FileNotFoundError(f"❌ Config file not found: {phy_teacher_config_path}")

    # Physical Teacher Node
    phy_teacher_node = Node(
        package='edge_phy_teacher',
        executable='phy_teacher',
        name='phy_teacher',
        namespace='edge',
        output='screen',
        parameters=[
            default_config_path,
            phy_teacher_config_path,
        ],
    )
    
    # Gain Synthesizer Node
    gain_synthesizer_node = Node(
        package='edge_phy_teacher',
        executable='gain_synthesizer',
        name='gain_synthesizer',
        namespace='edge',
        output='screen',
        prefix=PythonExpression([
            "'taskset -c ' + str(",
            LaunchConfiguration('set_cpu_affinity'),
            ")"
        ]),
        parameters=[
            phy_teacher_config_path,
            {'print_process_time': LaunchConfiguration('print_process_time')}
        ],
    )

    return LaunchDescription([
        # DeclareLaunchArgument(
        #     'update_rate',
        #     default_value='30',
        #     description='Frequency (Hz) of the main control loop'
        # ),
        DeclareLaunchArgument(
            'set_cpu_affinity',
            default_value='1',
            description='Specify which CPU core(s) the node should run on'
        ),
        DeclareLaunchArgument(
            'print_process_time',
            default_value='true',
            description='Whether to print the process time'
        ),
        
        phy_teacher_node, 
        gain_synthesizer_node
        
    ])
