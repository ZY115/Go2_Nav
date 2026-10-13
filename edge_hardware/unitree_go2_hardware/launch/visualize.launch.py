import os

import xacro
from launch_ros.actions import Node
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, RegisterEventHandler, SetEnvironmentVariable



def launch_setup(context: LaunchContext, *args, **kwargs):
    # Retrieve launch arguments
    package_description = context.launch_configurations['pkg_description']
    pkg_description_path = os.path.join(get_package_share_directory(package_description))
    
    package_hardware_go2 = context.launch_configurations['pkg_hardware_go2']
    use_real_robot = context.launch_configurations['use_real_robot']
    network_interface = context.launch_configurations['network_interface']
    dds_domain_id = context.launch_configurations['dds_domain_id']

    # Process the xacro file to generate the robot description
    xacro_file = os.path.join(pkg_description_path, 'xacro', 'robot.xacro')
    robot_description = xacro.process_file(
        xacro_file,
        mappings={
            'use_real_robot': use_real_robot,
            'network_interface': network_interface,
            'dds_domain_id': dds_domain_id,
        }
    ).toxml()

    robot_controllers = PathJoinSubstitution(
        [
            FindPackageShare(package_hardware_go2),
            "config",
            "robot_control.yaml",
        ]
    )

    rviz_config_file = os.path.join(pkg_description_path, "rviz", "visualize_urdf.rviz")

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz_ocs2',
        output='screen',
        arguments=["-d", rviz_config_file]
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            {
                'publish_frequency': 20.0,
                'use_tf_static': True,
                'robot_description': robot_description,
                'ignore_timestamp': True
            }
        ],
    )

    controller_manager = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[robot_controllers],
        remappings=[
            ("~/robot_description", "/robot_description"),
        ],
        output="both",
    )

    joint_state_publisher = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster",
                   "--controller-manager", "/controller_manager"],
    )

    imu_sensor_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["imu_sensor_broadcaster",
                   "--controller-manager", "/controller_manager"],
    )

    return [
        rviz,
        robot_state_publisher,
        controller_manager,
        joint_state_publisher,
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=joint_state_publisher,
                on_exit=[imu_sensor_broadcaster],
            )
        )
    ]


def generate_launch_description():
    pkg_description = DeclareLaunchArgument(
        'pkg_description',
        default_value='go2_description',
        description='package for robot description'
    )

    pkg_hardware_go2 = DeclareLaunchArgument(
        'pkg_hardware_go2',
        default_value='unitree_go2_hardware',
        description='package for go2 hardware interface'
    )
    
    use_real_robot = DeclareLaunchArgument(
        'use_real_robot',
        default_value='true',
        description='Whether to use real robot or not'
    )
    
    network_interface = DeclareLaunchArgument(
        'network_interface',
        default_value='eth0',
        description='Network interface for DDS communication (e.g., eth0, wlan0, lo)'
    )

    dds_domain_id = DeclareLaunchArgument(
        'dds_domain_id',
        default_value='0',
        description='DDS domain ID for communication'
    )

    ros_rmw_implementation = DeclareLaunchArgument(
        'ros_rmw_implementation',
        default_value='rmw_fastrtps_cpp',
        description='ROS 2 RMW implementation used by launched ROS nodes'
    )

    set_rmw_implementation = SetEnvironmentVariable(
        'RMW_IMPLEMENTATION',
        LaunchConfiguration('ros_rmw_implementation')
    )

    return LaunchDescription([
        pkg_description,
        pkg_hardware_go2,
        use_real_robot,
        network_interface,
        dds_domain_id,
        ros_rmw_implementation,
        set_rmw_implementation,
        
        OpaqueFunction(function=launch_setup),
    ])
