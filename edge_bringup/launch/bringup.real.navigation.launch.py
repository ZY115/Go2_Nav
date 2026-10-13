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
    SetEnvironmentVariable,
    TimerAction,
)
from datetime import datetime
from launch_ros.actions import Node
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    
    # ------------------------------
    # ---    Launch Arguments    ---
    # ------------------------------
    
    # Global namespace argument
    namespace_arg = DeclareLaunchArgument(
        "namespace",
        default_value="edge",
        description="Global namespace for all nodes"
    )

    # Robot name argument
    robot_arg = DeclareLaunchArgument(
        "robot",
        default_value="go2",
        description="Robot name (e.g. go2, go1, a1)"
    )
    
    # Use Global RViz argument
    use_global_rviz_arg = DeclareLaunchArgument(
        "use_global_rviz", default_value="true", description="Use global RViz (integrated) or not"
    )
        
    # Real Robot Usage argument
    use_real_robot_arg = DeclareLaunchArgument(
        "use_real_robot",
        default_value="true",
        description="Use real robot hardware interface instead of simulation"
    )
    
    # Network Interface for DDS
    network_interface_arg = DeclareLaunchArgument(
        "network_interface",
        default_value="eth0",
        description="Network interface for DDS communication (e.g., eth0, wlan0, lo)"
    )
    
    # DDS Domain ID
    dds_domain_arg = DeclareLaunchArgument(
        "dds_domain_id",
        default_value="0",
        description="DDS domain ID for communication"
    )

    # ROS middleware implementation
    ros_rmw_implementation_arg = DeclareLaunchArgument(
        "ros_rmw_implementation",
        default_value="rmw_fastrtps_cpp",
        description="ROS 2 RMW implementation used by launched nodes"
    )
    
    # Record rosbag argument
    record_bag_arg = DeclareLaunchArgument(
        "record_bag",
        default_value="false",
        description="Record ros2 bag"
    )
    
    robot = LaunchConfiguration("robot")
    record_bag = LaunchConfiguration("record_bag")
    network_interface = LaunchConfiguration("network_interface")
    dds_domain_id = LaunchConfiguration("dds_domain_id")
    ros_rmw_implementation = LaunchConfiguration("ros_rmw_implementation")

    namespace = LaunchConfiguration("namespace")
    use_real_robot = LaunchConfiguration("use_real_robot")
    use_global_rviz = LaunchConfiguration("use_global_rviz")

    set_rmw_implementation = SetEnvironmentVariable(
        "RMW_IMPLEMENTATION",
        ros_rmw_implementation,
    )

    # -----------------------------------
    # --  Include Launch Descriptions  --
    # -----------------------------------
    
    # -------- LED Hardware Launch --------
    led_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("indicator_led_driver"),
                "launch",
                "indicator_led_driver.launch.py"
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
                "drl_student_vel.launch.py"
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

    # -------- RL Runtime Env Split Launch --------
    rl_runtime_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("edge_rl_runtime"),
                "launch",
                "rl_runtime.launch.py"
            ])
        )
        # launch_arguments={
        #     "go2_network_interface": network_interface,
        # }.items()
    )

    # -------- Real Robot Executor Launch --------
    real_robot_executor_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("edge_rl_runtime"),
                "launch",
                "real_robot_executor.launch.py"
            ])
        )
    )

    # --------- Navigation Module Launch ---------
    navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("vehicle_simulator"),
                "launch",
                "system_simulation_with_route_planner.launch.py"
            ])
        ),
        # launch_arguments={
        #     "namespace": namespace,
        #     "robot": robot
        # }.items()
    )

    # -----------------------------------
    # --       Node Definitions        --
    # -----------------------------------

    # -------- State Bridge Node --------_
    state_bridge_node = Node(
        package='unitree_go2_hardware',
        executable='go2_edge_state_bridge',
        name='go2_edge_state_bridge',
        output='screen',
    )

    # -------- Goal Publisher Node --------
    goal = [24.0, 0.0, 0.0]
    # goal = [6, 0.0, 0.0]
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

    # -------- Virtual Status Indicator Node --------
    virtual_status_indicator_node = Node(
        package='edge_trigger',
        executable='virtual_status_indicator_node',
        name='virtual_status_indicator',
        output='screen',
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
    
    # -------- ROS2 Bag Record --------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bag_name = f"edge_bag_{timestamp}"

    rosbag_record = ExecuteProcess(
        cmd=[
            "ros2", "bag", "record",
            "-o", bag_name,
            "-a"   # Record all topics. You can specify specific topics if needed.
        ],
        output="screen",
        condition=IfCondition(record_bag)
    )
    
    # -------- Camera Image Pub Node --------
    camera_image_pub_node = Node(
        package="go2_h264_repub",
        executable="go2_h264_repub",
        parameters=[
            {"publish_rate": 100.0},
            {"multicast_iface": "eth0"},
        ],
    )
    

    return LaunchDescription([
        # -- Launch Arguments --
        namespace_arg,
        robot_arg,
        
        record_bag_arg,
        use_real_robot_arg,         
        network_interface_arg,   
        dds_domain_arg,       
        ros_rmw_implementation_arg,
        use_global_rviz_arg, 

        # -- Runtime Environment --
        # set_rmw_implementation,
    
        # -- Launch Components --
        led_launch,
        # controller_launch,
        # phy_teacher_launch,
        # drl_student_launch,
        
        trigger_launch,
        rl_runtime_launch,
        # real_robot_executor_launch,
        # virtual_status_indicator_node,
        state_bridge_node,
        publish_goal_point,
        # global_rviz_node,
        camera_image_pub_node,
        # navigation_launch,
        
        # rosbag_record,

    ])
