#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.event_handlers import OnProcessExit
from launch.actions import RegisterEventHandler, Shutdown


def generate_launch_description():

    # -----------------------------
    # Declare arguments
    # -----------------------------
    args = [
        DeclareLaunchArgument("sensorOffsetX", default_value="0.0"),
        DeclareLaunchArgument("sensorOffsetY", default_value="0.0"),
        DeclareLaunchArgument("cameraOffsetZ", default_value="0.0"),
        DeclareLaunchArgument("twoWayDrive", default_value="false"),
        DeclareLaunchArgument("maxSpeed", default_value="1.0"),   # Original
        # DeclareLaunchArgument("maxSpeed", default_value="0.5"),   # For testing
        DeclareLaunchArgument("autonomyMode", default_value="true"),
        DeclareLaunchArgument("autonomySpeed", default_value="1.0"),  # Original
        # DeclareLaunchArgument("autonomySpeed", default_value="0.5"),  # For testing
        DeclareLaunchArgument("joyToSpeedDelay", default_value="2.0"),
        DeclareLaunchArgument("goalCloseDis", default_value="2.0"),
        DeclareLaunchArgument("goalX", default_value="0.0"),
        DeclareLaunchArgument("goalY", default_value="0.0"),
        DeclareLaunchArgument("test_unitree_controller", default_value="false"),
        DeclareLaunchArgument("useGazebo", default_value="false"),
        DeclareLaunchArgument("odom_topic", default_value="odom", description="Odometry topic to use"),
    ]

    # -----------------------------
    # Node: localPlanner
    # -----------------------------
    local_planner_node = Node(
        package="local_planner",
        executable="localPlanner",
        name="localPlanner",
        output="screen",
        parameters=[{
            "pathFolder": PathJoinSubstitution([
                FindPackageShare("local_planner"), "paths"
            ]),
            "vehicleLength": 0.3,
            "vehicleWidth": 0.7,
            "sensorOffsetX": LaunchConfiguration("sensorOffsetX"),
            "sensorOffsetY": LaunchConfiguration("sensorOffsetY"),
            "twoWayDrive": LaunchConfiguration("twoWayDrive"),
            "laserVoxelSize": 0.05,
            "terrainVoxelSize": 0.2, # 0.2 (orign) 1.0
            "useTerrainAnalysis": True,
            "checkObstacle": True,
            "checkRotObstacle": False, # False (orign)
            # "adjacentRange": 3.0,     # Original
            "adjacentRange": 2.0,
            # "obstacleHeightThre": 0.3,
            "obstacleHeightThre": 0.5,

            "groundHeightThre": 0.1,
            "costHeightThre": 0.1,
            "costScore": 0.02,
            "useCost": False,
            "pointPerPathThre": 2,
            "minRelZ": -0.5,
            "maxRelZ": 0.25,
            "maxSpeed": LaunchConfiguration("maxSpeed"),
            "dirWeight": 0.02,
            "dirThre": 90.0,
            "dirToVehicle": False,
            "pathScale": 0.75,
            "minPathScale": 0.5,
            "pathScaleStep": 0.25,
            "pathScaleBySpeed": True,
            "minPathRange": 1.0,
            "pathRangeStep": 0.5,
            "pathRangeBySpeed": True,
            "pathCropByGoal": True,
            "autonomyMode": LaunchConfiguration("autonomyMode"),
            "autonomySpeed": LaunchConfiguration("autonomySpeed"),
            "joyToSpeedDelay": LaunchConfiguration("joyToSpeedDelay"),
            "joyToCheckObstacleDelay": 5.0,
            "goalClearRange": 0.5,
            "goalX": LaunchConfiguration("goalX"),
            "goalY": LaunchConfiguration("goalY"),
            'use_sim_time': True,
        }],
        remappings=[
            ("/state_estimation", LaunchConfiguration("odom_topic")),
        ],
    )

    # -----------------------------
    # Node: pathFollower
    # -----------------------------
    path_follower_node = Node(
        package="local_planner",
        executable="pathFollower",
        name="pathFollower",
        output="screen",
        parameters=[{
            "sensorOffsetX": LaunchConfiguration("sensorOffsetX"),
            "sensorOffsetY": LaunchConfiguration("sensorOffsetY"),
            "pubSkipNum": 1,
            "twoWayDrive": LaunchConfiguration("twoWayDrive"),
            "lookAheadDis": 0.5,    # Original 0.5
            "yawRateGain": 2.5,    # Original 1.5
            "stopYawRateGain": 1.5,
            "maxYawRate": 80.0,
            "maxSpeed": LaunchConfiguration("maxSpeed"),
            "maxAccel": 2.0,
            "switchTimeThre": 1.0,
            "dirDiffThre": 0.4,
            "omniDirDiffThre": 1.5,
            "noRotSpeed": 10.0,
            "stopDisThre": 0.3,
            "slowDwnDisThre": 0.75,
            "useInclRateToSlow": False,
            "inclRateThre": 120.0,
            "slowRate1": 0.25,
            "slowRate2": 0.5,
            "slowTime1": 2.0,
            "slowTime2": 2.0,
            "useInclToStop": False,
            "inclThre": 45.0,
            "stopTime": 5.0,
            "noRotAtStop": False,
            "noRotAtGoal": True,
            "autonomyMode": LaunchConfiguration("autonomyMode"),
            "autonomySpeed": LaunchConfiguration("autonomySpeed"),
            "joyToSpeedDelay": LaunchConfiguration("joyToSpeedDelay"),
            "goalCloseDis": LaunchConfiguration("goalCloseDis"),
            "test_unitree_controller": LaunchConfiguration("test_unitree_controller"),
            "useGazebo": LaunchConfiguration("useGazebo"),
            # 'use_sim_time': True,
        }],
        remappings=[
            ("/state_estimation", LaunchConfiguration("odom_topic")),
        ],
    )

    # -----------------------------
    # TF: static transform (sensor -> vehicle)
    # ROS1: args="-x -y 0 0 0 0 /sensor /vehicle"
    # ROS2: static_transform_publisher x y z roll pitch yaw frame child
    # -----------------------------
    static_tf_sensor_vehicle = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="vehicleTransPublisher",
        arguments=[
            LaunchConfiguration("sensorOffsetX"),   # x
            LaunchConfiguration("sensorOffsetY"),   # y
            "0",                                    # z
            "0", "0", "0",                          # rpy
            "sensor", "vehicle"
        ],
        # parameters=[{'use_sim_time': True}],
    )

    # -----------------------------
    # TF: sensor -> camera
    # -----------------------------
    static_tf_sensor_camera = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="sensorTransPublisher",
        arguments=[
            "0", "0", LaunchConfiguration("cameraOffsetZ"),
            "-1.5707963", "0", "-1.5707963",
            "sensor", "camera"
        ],
        # parameters=[{'use_sim_time': True}],
    )

    # -----------------------------
    # Return LaunchDescription
    # -----------------------------
    return LaunchDescription(
        args + [
            local_planner_node,
            # RegisterEventHandler(
            #     event_handler=OnProcessExit(
            #         target_action=local_planner_node,
            #         on_exit=[path_follower_node],
            #     )
            # ),
            path_follower_node,
            static_tf_sensor_vehicle,
            static_tf_sensor_camera,
        ]
    )
