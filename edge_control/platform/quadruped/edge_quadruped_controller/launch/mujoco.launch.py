import os

import xacro
from launch_ros.actions import Node
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch import LaunchContext, LaunchDescription
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
from launch.actions import DeclareLaunchArgument, OpaqueFunction, IncludeLaunchDescription, RegisterEventHandler
from ament_index_python.packages import get_package_share_directory



def launch_setup(context: LaunchContext, *args, **kwargs):
    namespace = "edge"
    
    robot_name = context.launch_configurations['robot']
    use_real_robot = context.launch_configurations['use_real_robot']
    network_interface = context.launch_configurations['network_interface']
    dds_domain_id = context.launch_configurations['dds_domain_id']
    use_rviz = context.launch_configurations['use_rviz']
    
    package_description = f"{robot_name}_description"
    package_controller = "edge_quadruped_controller"
    
    try:
        pkg_path = os.path.join(get_package_share_directory(package_description))
    except Exception as e:
        print(f"Error: Could not find package '{package_description}'. Please ensure it is installed.")
        print(f"Available packages can be checked with: ros2 pkg list | grep {robot_name}")
        raise e

    xacro_file = os.path.join(pkg_path, 'xacro', 'robot.xacro')
    
    if not os.path.exists(xacro_file):
        print(f"Error: Could not find robot description file: {xacro_file}")
        raise FileNotFoundError(f"Robot description file not found: {xacro_file}")
    
    print(f"Using robot description file: {xacro_file}")
    
    robot_description = xacro.process_file(
        xacro_file, 
        mappings={
            'use_real_robot': use_real_robot,
            'network_interface': network_interface,
            'dds_domain_id': dds_domain_id,
        }
    ).toxml()

    # Select controller config based on whether using real robot or simulation
    config_file = "controller_real.yaml" if use_real_robot == 'true' else "controller_sim.yaml"
    print(f"Using controller configuration: {config_file}")
    robot_controllers = PathJoinSubstitution(
        [
            FindPackageShare(package_controller),
            "config",
            config_file,
        ]
    )

    rviz_config_file = os.path.join(get_package_share_directory(package_controller), "rviz", "quadruped.rviz")

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz_robot_go2',
        namespace=namespace,
        output='screen',
        arguments=["-d", rviz_config_file],
        condition=IfCondition(use_rviz)
    )
    

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
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

    edge_quadruped_controller = Node(
        namespace=namespace,
        package="controller_manager",
        executable="spawner",
        arguments=["edge_quadruped_controller", "--controller-manager", "/controller_manager"],
    )

    tf_to_map_node = Node(
        package='lidar_to_map',
        executable='lidar_to_map',
        name='lidar_to_map_node',
        output='screen',
    )

    tf_static = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=["0", "0", "0", "0", "0", "0", "map", "leg_odom"],
    )


    return [
        tf_static,
        # tf_to_map_node,

        rviz,
        robot_state_publisher,
        controller_manager,
        joint_state_publisher,
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=joint_state_publisher,
                on_exit=[imu_sensor_broadcaster],
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=imu_sensor_broadcaster,
                on_exit=[edge_quadruped_controller],
            )
        ),
    ]


def generate_launch_description():
    robot_name = DeclareLaunchArgument(
        'robot',
        default_value='go2',
        description='name of the robot (e.g., go2, go1)'
    )

    use_real_robot = DeclareLaunchArgument(
        'use_real_robot',
        default_value='false',
        description='Whether to use real robot or not'
    )
    
    network_interface = DeclareLaunchArgument(
        'network_interface',
        default_value='lo',
        description='Network interface for DDS communication (e.g., eth0, wlan0, lo)'
    )

    dds_domain_id = DeclareLaunchArgument(
        'dds_domain_id',
        default_value='0',
        description='DDS domain ID for communication'
    )
    
    use_rviz = DeclareLaunchArgument(
        'use_rviz',
        default_value='false',
        description='Whether to launch RViz'
    )

    return LaunchDescription([
        robot_name,
        use_real_robot,
        network_interface,
        dds_domain_id,
        use_rviz,
        OpaqueFunction(function=launch_setup),
    ])
