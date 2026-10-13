#!/usr/bin/env python3
# ============================================================================
# Copyright (c) 2025-2026 Yihao Cai - All rights reserved
# SPDX-License-Identifier: Apache-2.0
# ============================================================================

import os

from datetime import datetime
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # ------------------------------------------------
    # Parameter file
    # ------------------------------------------------
    physical_model_file = os.path.join(
        get_package_share_directory("edge_phy_teacher"),
        "config",
        "physical_model.yaml"
    )

    # ------------------------------------------------
    # Launch arguments
    # ------------------------------------------------
    task_arg = DeclareLaunchArgument(
        "task",
        default_value="end2end",
        description="Training task: navigation or locomotion or end2end"
    )

    run_arg = DeclareLaunchArgument(
        "run",
        default_value="cognitive",
        description="Runtime Learning type: cognitive, dual, rlm, shielding"
    )
    
    is_real_robot_arg = DeclareLaunchArgument(
        "is_real_robot",
        default_value="true",
        description="Whether to run on a real robot"
    )

    run = LaunchConfiguration("run")
    task = LaunchConfiguration("task")
    is_real_robot = LaunchConfiguration("is_real_robot")

    # ------------------------------------------------
    # Experiment naming
    # ------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    project_name = PythonExpression([
        "'drl_student_' + '", run, "' + '_", timestamp, "'"
    ])


    # ------------------------------------------------
    # Node
    # ------------------------------------------------
    drl_student_node = Node(
        package="edge_drl_student",
        executable="drl_student_node",
        name="drl_student",
        namespace="edge",
        output="screen",
        parameters=[
            physical_model_file,
            {
                "task": task,
                "run": run,
                "project_name": project_name,
                "is_real_robot": is_real_robot,
            },
        ],
    )

    return LaunchDescription([
        run_arg,
        task_arg,
        is_real_robot_arg,
        
        # DRL Student Node
        drl_student_node,
    ])