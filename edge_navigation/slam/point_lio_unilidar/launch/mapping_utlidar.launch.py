from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node


def generate_launch_description():

    # -------------------------
    # Launch Argument
    # -------------------------
    rviz = LaunchConfiguration("rviz")

    declare_rviz = DeclareLaunchArgument(
        "rviz",
        default_value="true",
        description="Launch RViz2?"
    )
    
    odom_topic_arg = DeclareLaunchArgument(
        "odom_topic",
        default_value="odom",
        description="Odometry topic to use"
    )

    # -------------------------
    # Paths
    # -------------------------
    pkg_point_lio = get_package_share_directory("point_lio_unilidar")

    # YAML configuration file
    param_yaml = PathJoinSubstitution([
        pkg_point_lio,
        "config",
        "utlidar.yaml"
    ])

    # RViz config
    rviz_cfg = PathJoinSubstitution([
        pkg_point_lio,
        "rviz_cfg",
        "loam_livox2.rviz"
    ])

    # -------------------------
    # Nodes
    # -------------------------
    transform_everything_node = Node(
        package="transform_sensors",
        executable="transform_everything",
        name="transform_everything",
        output="screen"
    )

    laser_mapping_node = Node(
        package="point_lio_unilidar",
        executable="pointlio_mapping",
        name="laserMapping",
        output="screen",
        # prefix="taskset -c 2,3,4,5",
        parameters=[
            param_yaml,
            {"use_imu_as_input": False},
            {"prop_at_freq_of_imu": True},
            {"check_satu": True},
            {"init_map_size": 10},
            {"point_filter_num": 1},
            {"space_down_sample": True},
            {"filter_size_surf": 0.3},      # Original: 0.1
            {"filter_size_map": 0.3},       # Original: 0.1
            {"cube_side_length": 1000.0},
            {"runtime_pos_log_enable": False},
        ],
        remappings=[
            ("/cloud_registered", "/registered_scan"),
            # ("/aft_mapped_to_init", "/state_estimation"),
            ("/aft_mapped_to_init", LaunchConfiguration("odom_topic")),
        ],
    )

    # RViz group
    rviz_group = GroupAction(
        condition=IfCondition(rviz),
        actions=[
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz",
                output="screen",
                arguments=["-d", rviz_cfg],
                prefix="nice",
            )
        ]
    )

    # -------------------------
    # Assemble launch
    # -------------------------
    return LaunchDescription([
        odom_topic_arg,
        
        declare_rviz,
        transform_everything_node,
        laser_mapping_node,
        rviz_group,
    ])
