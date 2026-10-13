#!/usr/bin/env python3
# ============================================================================
# Copyright (c) 2025-2026 Yihao Cai - All rights reserved
# SPDX-License-Identifier: Apache-2.0
# ============================================================================

import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    RegisterEventHandler,
    TimerAction,
)
from launch_ros.actions import Node
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression


def generate_launch_description():
    # ------------------------------
    # ---    Launch Arguments    ---
    # ------------------------------
    namespace_arg = DeclareLaunchArgument(
        "namespace",
        default_value="edge",
        description="Global namespace for all nodes"
    )

    robot_arg = DeclareLaunchArgument(
        "robot",
        default_value="go2",
        description="Robot name (e.g. go2, go1, a1)"
    )

    use_global_rviz_arg = DeclareLaunchArgument(
        "use_global_rviz", default_value="true", description="Use global RViz (integrated) or not"
    )
    
    robot = LaunchConfiguration("robot")
    namespace = LaunchConfiguration("namespace")
    use_global_rviz = LaunchConfiguration("use_global_rviz")

    # -----------------------------------
    # --  Include Launch Descriptions  --
    # -----------------------------------
    
    # -------- Edge Quadruped Controller Launch --------
    controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("edge_quadruped_controller"),
                "launch",
                "mujoco.launch.py"
            ])
        ),
        launch_arguments={
            "namespace": namespace,
            "robot": robot,
            "use_rviz": PythonExpression(["'", use_global_rviz, "' == 'false'"])
        }.items()
    )  
    
    # -------- PHY Teacher Launch --------
    phy_teacher_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("edge_phy_teacher"),
                "launch",
                "phy_teacher.launch.py"
            ])
        ),
        launch_arguments={
            "namespace": namespace,
            "robot": robot
        }.items()
    )
    
    # -------- DRL Student Launch --------
    drl_student_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("edge_drl_student"),
                "launch",
                "drl_student.launch.py"
            ])
        ),
        launch_arguments={
            "namespace": namespace,
            "robot": robot
        }.items()
    )

    # -------- Trigger Launch --------
    trigger_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("edge_trigger"),
                "launch",
                "trigger.launch.py"
            ])
        ),
        launch_arguments={
            "namespace": namespace,
            "robot": robot
        }.items()
    )

    # --------- Navigation Module Launch ---------
    navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("edge_nav_bringup"),
                "launch",
                "system_simulation_with_route_planner.launch.py"
            ])
        ),
        launch_arguments={
            "namespace": namespace,
            "robot": robot,
            "use_rviz": PythonExpression(["'", use_global_rviz, "' == 'false'"])
        }.items()
    )
    
    # -----------------------------------
    # --       Node Definitions        --
    # -----------------------------------

    # -------- Virtual Status Indicator Node --------
    virtual_status_indicator_node = Node(
        package='edge_trigger',
        executable='virtual_status_indicator_node',
        name='virtual_status_indicator',
        output='screen',
    )
    
    # -------- Goal Publisher Node --------
    goal = [6.0, -6.0, 0.0]
    publish_goal_point = TimerAction(
        period=1.0,
        actions=[ExecuteProcess(
            cmd=[
                'ros2', 'topic', 'pub', '-r', '1',
                '/goal_point',
                'geometry_msgs/msg/PointStamped',
                f'{{header: {{stamp: {{sec: 0, nanosec: 0}}, frame_id: "map"}}, point: {{x: {goal[0]}, y: {goal[1]}, z: {goal[2]}}}}}'
            ],
            # output='screen'
        )]
    )

    # -------- Global Rviz Node --------
    rviz_config_file = PathJoinSubstitution([
        FindPackageShare('edge_bringup'),
        'rviz',
        'global_view.rviz'
    ])
    global_rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='global_rviz',
        output='screen',
        arguments=['-d', rviz_config_file],
        condition=IfCondition(use_global_rviz)
    )
    
    return LaunchDescription([
        # -- Launch Arguments --
        robot_arg,
        namespace_arg,
        use_global_rviz_arg,
        
        # -- Launch Components --
        controller_launch,
        phy_teacher_launch,
        # drl_student_launch,
        trigger_launch,
        virtual_status_indicator_node,
        global_rviz_node,
        
        # navigation_launch,
        publish_goal_point,
    ])
