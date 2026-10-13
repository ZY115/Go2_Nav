#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration


def generate_launch_description():

    # -----------------------------
    # Declare arguments
    # -----------------------------
    checkTerrainConn_arg = DeclareLaunchArgument(
        "checkTerrainConn",
        default_value="false"
    )
    
    odom_topic_arg = DeclareLaunchArgument(
        "odom_topic",
        default_value="odom",
        description="Odometry topic to use"
    )

    # -----------------------------
    # Node: terrainAnalysisExt
    # -----------------------------
    terrain_analysis_ext_node = Node(
        package="terrain_analysis_ext",
        executable="terrainAnalysisExt",
        name="terrainAnalysisExt",
        output="screen",
        parameters=[{
            "scanVoxelSize": 0.1,
            "decayTime": 10.0,
            "noDecayDis": 0.0,
            "clearingDis": 30.0,
            "useSorting": True,
            "quantileZ": 0.1,
            "vehicleHeight": 1.5,
            "voxelPointUpdateThre": 100,
            "voxelTimeUpdateThre": 2.0,
            "lowerBoundZ": -2.5,
            "upperBoundZ": 1.0,
            "disRatioZ": 0.1,
            "checkTerrainConn": LaunchConfiguration("checkTerrainConn"),
            "terrainConnThre": 0.5,
            "terrainUnderVehicle": -0.75,
            "ceilingFilteringThre": 2.0,
            "localTerrainMapRadius": 4.0,
            # "use_sim_time": True,
        }],
        remappings=[
            ("/state_estimation", LaunchConfiguration("odom_topic")),
        ],
    )

    # -----------------------------
    # Return final LaunchDescription
    # -----------------------------
    return LaunchDescription([
        odom_topic_arg,
        checkTerrainConn_arg,
        
        terrain_analysis_ext_node
    ])
