#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

def generate_launch_description():
    odom_topic_arg = DeclareLaunchArgument(
        "odom_topic",
        default_value="odom",
        description="Odometry topic to use"
    )

    sensor_scan_gen_node = Node(
        package="sensor_scan_generation",
        executable="sensorScanGeneration",
        name="sensorScanGeneration",
        output="screen",
        remappings=[
            ("/state_estimation", LaunchConfiguration("odom_topic")),
        ],
    )

    return LaunchDescription([
        odom_topic_arg,
        
        sensor_scan_gen_node
    ])
