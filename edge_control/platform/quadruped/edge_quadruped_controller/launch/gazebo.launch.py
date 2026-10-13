import os
import xacro

from launch_ros.actions import Node
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch_ros.substitutions import FindPackageShare
from launch import LaunchContext, LaunchDescription
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import SetEnvironmentVariable, IncludeLaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, RegisterEventHandler
from ament_index_python.packages import get_package_share_directory

def launch_setup(context: LaunchContext, *args, **kwargs):
    
    robot_name = context.launch_configurations['robot']
    init_height = context.launch_configurations['height']
    world_file_name = context.launch_configurations['world']
    use_rviz = context.launch_configurations['use_rviz']
    
    pkg_gazebo = "edge_gazebo"
    pkg_controller = "edge_quadruped_controller"
    pkg_description = f"{robot_name}_description"
    
    # Set GAZEBO model path
    model_resource_path = os.path.join(get_package_share_directory(pkg_gazebo), "models")

    os.environ["IGN_GAZEBO_RESOURCE_PATH"] = \
        os.environ.get("IGN_GAZEBO_RESOURCE_PATH", "") + ":" + model_resource_path

    os.environ["GZ_SIM_RESOURCE_PATH"] = \
        os.environ.get("GZ_SIM_RESOURCE_PATH", "") + ":" + model_resource_path
    
    # Set GAZEBO log level to error to reduce log spam
    disable_gz_logs = SetEnvironmentVariable(
        name='GZ_LOGGING',
        value='0'
    )

    limit_gz_log_level = SetEnvironmentVariable(
        name='GZ_LOG_LEVEL',
        value='error'
    )

    disable_ros_logs = SetEnvironmentVariable(
        name='RCUTILS_LOGGING_SEVERITY_THRESHOLD',
        value='ERROR'
    )

    try:
        pkg_path = os.path.join(get_package_share_directory(pkg_description))
    except Exception as e:
        print(f"Error: Could not find package '{pkg_description}'. Please ensure it is installed.")
        print(f"Available packages can be checked with: ros2 pkg list | grep {robot_name}")
        raise e

    # Load controller configuration
    controller_config_path = os.path.join(
        get_package_share_directory(pkg_controller), 
        'config', 
        'controller_sim.yaml'
    )

    xacro_file = os.path.join(pkg_path, 'xacro', 'robot.xacro')
    
    if not os.path.exists(xacro_file):
        print(f"Error: Could not find robot description file: {xacro_file}")
        raise FileNotFoundError(f"Robot description file not found: {xacro_file}")
    
    print(f"Using robot description file: {xacro_file}")
    
    # Process xacro file and include controller config
    robot_description = xacro.process_file(
        xacro_file, 
        mappings={'GAZEBO': 'true', 
                  'CONTROLLER_CONFIG': controller_config_path}
    ).toxml()

    rviz_config_file = os.path.join(get_package_share_directory(pkg_controller), "rviz", "quadruped.rviz")

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz_robot_go2',
        output='screen',
        arguments=["-d", rviz_config_file],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(use_rviz)
    )
    
    world_file = PathJoinSubstitution([
        FindPackageShare(pkg_gazebo),
        'worlds',
        world_file_name  
    ])
    # world_file = PathJoinSubstitution([
    #     FindPackageShare(pkg_gazebo),
    #     'worlds/terrains/rough_25cm',
    #     "rough_25cm.world" 
    # ])

    physical_model_config = PathJoinSubstitution([
        FindPackageShare('edge_phy_teacher'),
        'config',
        'physical_model.yaml'
    ])

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [PathJoinSubstitution([FindPackageShare('ros_gz_sim'),
                                    'launch',
                                    'gz_sim.launch.py'])]),
            launch_arguments=[
                ('gz_args', ['-r -v 4 ', world_file]),
            ],
    )

    gz_spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=['-topic', 'robot_description', '-name',
                   robot_name, '-allow_renaming', 'true', '-x', '0.', '-y', '0', '-z', init_height],
        parameters=[{'use_sim_time': True}],
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
                'ignore_timestamp': True,
                'use_sim_time': True,
            }
        ],
    )

    joint_state_publisher = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster",
                   "--controller-manager", "/controller_manager"],
        parameters=[{'use_sim_time': True}],
    )

    imu_sensor_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["imu_sensor_broadcaster",
                   "--controller-manager", "/controller_manager"],
        parameters=[{'use_sim_time': True}],
    )

    edge_quadruped_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "edge_quadruped_controller",
            "--controller-manager", "/controller_manager",
            "--param-file", physical_model_config.perform(context)
        ],
        parameters=[
            {'use_sim_time': True},
            # physical_model_config
        ],
    )

    tf_to_map_node = Node(
        package='lidar_to_map',
        executable='lidar_to_map',
        name='lidar_to_map_node',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    tf_static = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=["0", "0", "0", "0", "0", "0", "map", "leg_odom"],
        parameters=[{'use_sim_time': True}],
    )


    return [
        tf_static,
        tf_to_map_node,
        
        disable_gz_logs,
        limit_gz_log_level,
        disable_ros_logs,
        
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
                        '/camera/image/raw@sensor_msgs/msg/Image[gz.msgs.Image',
                        '/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
                        '/camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image',
                        '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
                        '/scan/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
                       ],
            # remappings=[
            #     ('/scan/points', '/registered_scan')
            # ],
            parameters=[{'use_sim_time': True}],
            output='screen'
        ),

        gazebo,
        robot_state_publisher,
        gz_spawn_entity,
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=gz_spawn_entity,
                on_exit=[imu_sensor_broadcaster, joint_state_publisher],
            )
        ),

        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=joint_state_publisher,
                on_exit=[edge_quadruped_controller],
            )
        ),
        
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=edge_quadruped_controller,
                on_exit=[rviz],
            )
        ),
    ]


def generate_launch_description():
    robot_name = DeclareLaunchArgument(
        'robot',
        default_value='go2',
        description='Robot name (will automatically append _description to find the description package)'
    )

    height = DeclareLaunchArgument(
        'height',
        default_value='0.5',
        description='Init height in simulation'
    )
    
    world_file_name = DeclareLaunchArgument(
        'world',
        default_value='openwild_outdoor.world',
        description='World file name (including .world extension)'
    )
    
    use_rviz = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Whether to launch RViz'
    )

    return LaunchDescription([
        robot_name,
        height,
        world_file_name,
        use_rviz,
        OpaqueFunction(function=launch_setup),
    ])
