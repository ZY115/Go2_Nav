#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration


def generate_launch_description():

    # -----------------------------
    # Declare launch arguments
    # -----------------------------
    args = [
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("sensorOffsetX", default_value="0.0"),
        DeclareLaunchArgument("sensorOffsetY", default_value="0.0"),
        DeclareLaunchArgument("vehicleHeight", default_value="0.366"),

        DeclareLaunchArgument("vehicleX", default_value="0.0"),
        DeclareLaunchArgument("vehicleY", default_value="0.0"),
        DeclareLaunchArgument("vehicleZ", default_value="0.0"),
        DeclareLaunchArgument("terrainZ", default_value="0.0"),
        DeclareLaunchArgument("vehicleYaw", default_value="0.0"),

        DeclareLaunchArgument("terrainVoxelSize", default_value="0.05"),
        DeclareLaunchArgument("groundHeightThre", default_value="0.1"),
        DeclareLaunchArgument("adjustZ", default_value="false"),
        DeclareLaunchArgument("terrainRadiusZ", default_value="1.0"),
        DeclareLaunchArgument("minTerrainPointNumZ", default_value="5"),
        DeclareLaunchArgument("smoothRateZ", default_value="0.5"),

        DeclareLaunchArgument("adjustIncl", default_value="false"),
        DeclareLaunchArgument("terrainRadiusIncl", default_value="2.0"),
        DeclareLaunchArgument("minTerrainPointNumIncl", default_value="200"),
        DeclareLaunchArgument("smoothRateIncl", default_value="0.5"),
        DeclareLaunchArgument("InclFittingThre", default_value="0.2"),
        DeclareLaunchArgument("maxIncl", default_value="30.0"),
    ]

    # -----------------------------
    # vehicleSimulator node
    # -----------------------------
    vehicle_sim_node = Node(
        package="vehicle_simulator",
        executable="vehicleSimulator",
        name="vehicleSimulator",
        output="screen",
        parameters=[{
            "sensorOffsetX": LaunchConfiguration("sensorOffsetX"),
            "sensorOffsetY": LaunchConfiguration("sensorOffsetY"),
            "vehicleHeight": LaunchConfiguration("vehicleHeight"),

            "vehicleX": LaunchConfiguration("vehicleX"),
            "vehicleY": LaunchConfiguration("vehicleY"),
            "vehicleZ": LaunchConfiguration("vehicleZ"),
            "terrainZ": LaunchConfiguration("terrainZ"),
            "vehicleYaw": LaunchConfiguration("vehicleYaw"),

            "terrainVoxelSize": LaunchConfiguration("terrainVoxelSize"),
            "groundHeightThre": LaunchConfiguration("groundHeightThre"),
            "adjustZ": LaunchConfiguration("adjustZ"),
            "terrainRadiusZ": LaunchConfiguration("terrainRadiusZ"),
            "minTerrainPointNumZ": LaunchConfiguration("minTerrainPointNumZ"),
            "smoothRateZ": LaunchConfiguration("smoothRateZ"),

            "adjustIncl": LaunchConfiguration("adjustIncl"),
            "terrainRadiusIncl": LaunchConfiguration("terrainRadiusIncl"),
            "minTerrainPointPointNumIncl": LaunchConfiguration("minTerrainPointNumIncl"),
            "smoothRateIncl": LaunchConfiguration("smoothRateIncl"),
            "InclFittingThre": LaunchConfiguration("InclFittingThre"),
            "maxIncl": LaunchConfiguration("maxIncl"),
            
            "use_sim_time": LaunchConfiguration("use_sim_time"),
        }]
    )

    return LaunchDescription(args + [
        vehicle_sim_node
    ])
