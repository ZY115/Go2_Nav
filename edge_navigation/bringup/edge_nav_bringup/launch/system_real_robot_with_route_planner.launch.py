from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # -------------------------------
    # Launch Arguments
    # -------------------------------
    world_name = LaunchConfiguration("world_name")
    sensorOffsetX = LaunchConfiguration("sensorOffsetX")
    sensorOffsetY = LaunchConfiguration("sensorOffsetY")
    cameraOffsetZ = LaunchConfiguration("cameraOffsetZ")
    vehicleX = LaunchConfiguration("vehicleX")
    vehicleY = LaunchConfiguration("vehicleY")
    checkTerrainConn = LaunchConfiguration("checkTerrainConn")
    odom_topic = LaunchConfiguration("odom_topic")
    test_unitree_controller = LaunchConfiguration("test_unitree_controller")
    use_gazebo = LaunchConfiguration("useGazebo")

    declare_args = [
        DeclareLaunchArgument("world_name", default_value="real_world"),
        DeclareLaunchArgument("sensorOffsetX", default_value="0.3"),
        DeclareLaunchArgument("sensorOffsetY", default_value="0.0"),
        DeclareLaunchArgument("cameraOffsetZ", default_value="0.0"),
        DeclareLaunchArgument("vehicleX", default_value="0.0"),
        DeclareLaunchArgument("vehicleY", default_value="0.0"),
        DeclareLaunchArgument("checkTerrainConn", default_value="true"),
        DeclareLaunchArgument("test_unitree_controller", default_value="false"),
        DeclareLaunchArgument("odom_topic", default_value="odom"),
        DeclareLaunchArgument("useGazebo", default_value="false"),
    ]

    # -------------------------------
    # Nodes
    # -------------------------------
    joy_node = Node(
        package="joy",
        executable="joy_node",
        name="ps3_joy",
        output="screen",
        parameters=[
            {"dev": "/dev/input/js0"},
            {"deadzone": 0.12},
            {"autorepeat_rate": 0.0},
        ],
    )

    # Static TF publishers
    tf_map_to_camera_init = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="loamInterfaceTransPubMap",
        arguments=["0", "0", "0", "0", "0", "0", "/map", "/camera_init"]
    )

    tf_aftmapped_to_sensor = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="loamInterfaceTransPubVehicle",
        arguments=["0", "0", "0", "0", "0", "0", "/aft_mapped", "/sensor"]
    )

    # -------------------------------
    # Includes
    # -------------------------------
    point_lio_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                get_package_share_directory("point_lio_unilidar"),
                "launch",
                "mapping_utlidar.launch.py"
            ])
        ),
        launch_arguments={
            "rviz": "false",
            "odom_topic": odom_topic,
        }.items(),
    )

    local_planner_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                get_package_share_directory("local_planner"),
                "launch",
                "local_planner.launch.py"
            ])
        ),
        launch_arguments={
            "sensorOffsetX": sensorOffsetX,
            "sensorOffsetY": sensorOffsetY,
            "cameraOffsetZ": cameraOffsetZ,
            "goalX": vehicleX,
            "goalY": vehicleY,
            "odom_topic": odom_topic,
            "test_unitree_controller": test_unitree_controller,
            "useGazebo": use_gazebo,
        }.items(),
    )

    terrain_analysis_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                get_package_share_directory("terrain_analysis"),
                "launch",
                "terrain_analysis.launch.py"
            ])
        ),
        launch_arguments={
            "odom_topic": odom_topic
        }.items(),
    )

    terrain_analysis_ext_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                get_package_share_directory("terrain_analysis_ext"),
                "launch",
                "terrain_analysis_ext.launch.py"
            ])
        ),
        launch_arguments={
            "checkTerrainConn": checkTerrainConn,
            "odom_topic": odom_topic
        }.items(),
    )

    far_planner_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                get_package_share_directory("far_planner"),
                "launch",
                "far_planner.launch.py"
            ])
        ),
        launch_arguments={
            "odom_topic": odom_topic
        }.items(),
    )

    # -------------------------------
    # Launch Description Assembly
    # -------------------------------
    return LaunchDescription(
        declare_args
        + [
            joy_node,
            point_lio_launch,
            local_planner_launch,
            terrain_analysis_launch,
            terrain_analysis_ext_launch,
            far_planner_launch,
            tf_map_to_camera_init,
            tf_aftmapped_to_sensor,
        ]
    )
