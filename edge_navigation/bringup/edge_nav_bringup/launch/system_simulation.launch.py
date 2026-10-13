#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare launch arguments
    world_name_arg = DeclareLaunchArgument(
        'world_name',
        default_value='unity',
        description='World name'
    )
    
    vehicle_height_arg = DeclareLaunchArgument(
        'vehicleHeight',
        default_value='0.366',
        description='Vehicle height'
    )
    
    sensor_offset_x_arg = DeclareLaunchArgument(
        'sensorOffsetX',
        default_value='0.0',
        description='Sensor offset X'
    )
    
    sensor_offset_y_arg = DeclareLaunchArgument(
        'sensorOffsetY',
        default_value='0.0',
        description='Sensor offset Y'
    )
    
    camera_offset_z_arg = DeclareLaunchArgument(
        'cameraOffsetZ',
        default_value='0.0',
        description='Camera offset Z'
    )
    
    vehicle_x_arg = DeclareLaunchArgument(
        'vehicleX',
        default_value='0.0',
        description='Vehicle X position'
    )
    
    vehicle_y_arg = DeclareLaunchArgument(
        'vehicleY',
        default_value='0.0',
        description='Vehicle Y position'
    )
    
    terrain_z_arg = DeclareLaunchArgument(
        'terrainZ',
        default_value='0.0',
        description='Terrain Z position'
    )
    
    vehicle_yaw_arg = DeclareLaunchArgument(
        'vehicleYaw',
        default_value='0.0',
        description='Vehicle yaw angle'
    )
    
    check_terrain_conn_arg = DeclareLaunchArgument(
        'checkTerrainConn',
        default_value='true',
        description='Check terrain connection'
    )
    
    # Get launch configurations
    world_name = LaunchConfiguration('world_name')
    vehicle_height = LaunchConfiguration('vehicleHeight')
    sensor_offset_x = LaunchConfiguration('sensorOffsetX')
    sensor_offset_y = LaunchConfiguration('sensorOffsetY')
    camera_offset_z = LaunchConfiguration('cameraOffsetZ')
    vehicle_x = LaunchConfiguration('vehicleX')
    vehicle_y = LaunchConfiguration('vehicleY')
    terrain_z = LaunchConfiguration('terrainZ')
    vehicle_yaw = LaunchConfiguration('vehicleYaw')
    check_terrain_conn = LaunchConfiguration('checkTerrainConn')
    
    # PS3 Joy node
    ps3_joy_node = Node(
        package='joy',
        executable='joy_node',
        name='ps3_joy',
        output='screen',
        parameters=[{
            'dev': '/dev/input/js0',
            'deadzone': 0.12,
            'autorepeat_rate': 0.0
        }]
    )
    
    # Include local_planner launch
    local_planner_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('local_planner'),
                'launch',
                'local_planner.launch.py'
            ])
        ]),
        launch_arguments={
            'sensorOffsetX': sensor_offset_x,
            'sensorOffsetY': sensor_offset_y,
            'cameraOffsetZ': camera_offset_z,
            'goalX': vehicle_x,
            'goalY': vehicle_y
        }.items()
    )
    
    # Include terrain_analysis launch
    terrain_analysis_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('terrain_analysis'),
                'launch',
                'terrain_analysis.launch.py'
            ])
        ])
    )
    
    # Include terrain_analysis_ext launch
    terrain_analysis_ext_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('terrain_analysis_ext'),
                'launch',
                'terrain_analysis_ext.launch.py'
            ])
        ]),
        launch_arguments={
            'checkTerrainConn': check_terrain_conn
        }.items()
    )
    
    # Include vehicle_simulator launch
    vehicle_simulator_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('vehicle_simulator'),
                'launch',
                'vehicle_simulator.launch.py'
            ])
        ]),
        launch_arguments={
            'vehicleHeight': vehicle_height,
            'sensorOffsetX': sensor_offset_x,
            'sensorOffsetY': sensor_offset_y,
            'vehicleX': vehicle_x,
            'vehicleY': vehicle_y,
            'terrainZ': terrain_z,
            'vehicleYaw': vehicle_yaw
        }.items()
    )
    
    # Include sensor_scan_generation launch
    sensor_scan_generation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('sensor_scan_generation'),
                'launch',
                'sensor_scan_generation.launch.py'
            ])
        ])
    )
    
    # Include visualization_tools launch
    visualization_tools_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('visualization_tools'),
                'launch',
                'visualization_tools.launch.py'
            ])
        ]),
        launch_arguments={
            'world_name': world_name
        }.items()
    )
    
    # Sim image repub node
    sim_image_repub_node = Node(
        package='vehicle_simulator',
        executable='sim_image_repub',
        name='sim_image_repub',
        output='screen',
        parameters=[{
            'camera_in_topic': '/camera/image/compressed',
            'camera_raw_out_topic': '/camera/image/raw',
            'sem_in_topic': '/camera/semantic_image/compressed',
            'sem_raw_out_topic': '/camera/semantic_image/raw',
            'depth_in_topic': '/camera/depth/compressed',
            'depth_raw_out_topic': '/camera/depth/raw'
        }]
    )
    
    # RViz2 node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rvizGA',
        prefix='nice',
        arguments=['-d', PathJoinSubstitution([
            FindPackageShare('vehicle_simulator'),
            'rviz',
            'vehicle_simulator.rviz'
        ])]
    )
    
    return LaunchDescription([
        # Declare arguments
        world_name_arg,
        vehicle_height_arg,
        sensor_offset_x_arg,
        sensor_offset_y_arg,
        camera_offset_z_arg,
        vehicle_x_arg,
        vehicle_y_arg,
        terrain_z_arg,
        vehicle_yaw_arg,
        check_terrain_conn_arg,
        
        # Launch nodes and includes
        ps3_joy_node,
        local_planner_launch,
        terrain_analysis_launch,
        terrain_analysis_ext_launch,
        vehicle_simulator_launch,
        sensor_scan_generation_launch,
        visualization_tools_launch,

        sim_image_repub_node,
        rviz_node
    ])
