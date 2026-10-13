from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition
from launch.launch_description_sources import YAMLLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription


def generate_launch_description():

    # -------------------------
    # Launch Arguments
    # -------------------------
    rviz_arg = DeclareLaunchArgument(
        "rviz",
        default_value="true",
        description="Launch RViz?"
    )

    rviz = LaunchConfiguration("rviz")

    # -------------------------
    # Paths
    # -------------------------
    pkg_point_lio = get_package_share_directory("point_lio_unilidar")

    param_file = PathJoinSubstitution([
        pkg_point_lio,
        "config",
        "unilidar.yaml"
    ])

    rviz_cfg = PathJoinSubstitution([
        pkg_point_lio,
        "rviz_cfg",
        "loam_livox.rviz"
    ])

    # -------------------------
    # Main Node (laserMapping)
    # -------------------------
    laser_mapping_node = Node(
        package="point_lio_unilidar",
        executable="pointlio_mapping",
        name="laserMapping",
        output="screen",
        parameters=[
            param_file,
            {"use_imu_as_input": False},
            {"prop_at_freq_of_imu": True},
            {"check_satu": True},
            {"init_map_size": 10},
            {"point_filter_num": 1},
            {"space_down_sample": True},
            {"filter_size_surf": 0.1},
            {"filter_size_map": 0.1},
            {"cube_side_length": 1000.0},
            {"runtime_pos_log_enable": False},
        ],
    )

    # -------------------------
    # Optional RViz Node
    # -------------------------
    rviz_node = GroupAction(
        condition=IfCondition(rviz),
        actions=[
            Node(
                package="rviz",
                executable="rviz",
                name="rviz",
                output="screen",
                arguments=["-d", rviz_cfg],
                # Uncomment if you want gdb
                # prefix="gdb -ex run --args"
            )
        ]
    )

    # -------------------------
    # Launch Description
    # -------------------------
    return LaunchDescription([
        rviz_arg,
        laser_mapping_node,
        rviz_node
    ])
