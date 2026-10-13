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
        default_value='real_world',
        description='World name'
    )
    
    sensor_offset_x_arg = DeclareLaunchArgument(
        'sensorOffsetX',
        default_value='0.3',
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
    
    check_terrain_conn_arg = DeclareLaunchArgument(
        'checkTerrainConn',
        default_value='true',
        description='Check terrain connection'
    )
    
    # Get launch configurations
    world_name = LaunchConfiguration('world_name')
    sensor_offset_x = LaunchConfiguration('sensorOffsetX')
    sensor_offset_y = LaunchConfiguration('sensorOffsetY')
    camera_offset_z = LaunchConfiguration('cameraOffsetZ')
    vehicle_x = LaunchConfiguration('vehicleX')
    vehicle_y = LaunchConfiguration('vehicleY')
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
    
    # GO2 H264 repub node (commented out)
    # go2_h264_repub_node = Node(
    #     package='go2_h264_repub',
    #     executable='go2_h264_repub',
    #     name='go2_h264_repub',
    #     output='screen',
    #     parameters=[{
    #         'multicast_iface': 'enp3s0'
    #     }]
    # )
    
    # Include point_lio_unilidar launch
    point_lio_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('point_lio_unilidar'),
                'launch',
                'mapping_utlidar.launch.py'
            ])
        ]),
        launch_arguments={
            'rviz': 'false'
        }.items()
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
    
    # Include terrain_analysis_ext launch (commented out)
    # terrain_analysis_ext_launch = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource([
    #         PathJoinSubstitution([
    #             FindPackageShare('terrain_analysis_ext'),
    #             'launch',
    #             'terrain_analysis_ext.launch.py'
    #         ])
    #     ]),
    #     launch_arguments={
    #         'checkTerrainConn': check_terrain_conn
    #     }.items()
    # )
    
    # Include sensor_scan_generation launch (commented out)
    # sensor_scan_generation_launch = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource([
    #         PathJoinSubstitution([
    #             FindPackageShare('sensor_scan_generation'),
    #             'launch',
    #             'sensor_scan_generation.launch.py'
    #         ])
    #     ])
    # )
    
    # Include visualization_tools launch (commented out)
    # visualization_tools_launch = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource([
    #         PathJoinSubstitution([
    #             FindPackageShare('visualization_tools'),
    #             'launch',
    #             'visualization_tools.launch.py'
    #         ])
    #     ]),
    #     launch_arguments={
    #         'world_name': world_name
    #     }.items()
    # )
    
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
    
    # Static transform publisher: map -> camera_init
    static_tf_map_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='loamInterfaceTransPubMap',
        arguments=['0', '0', '0', '0', '0', '0', '/map', '/camera_init']
    )
    
    # Static transform publisher: aft_mapped -> sensor
    static_tf_vehicle_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='loamInterfaceTransPubVehicle',
        arguments=['0', '0', '0', '0', '0', '0', '/aft_mapped', '/sensor']
    )
    
    return LaunchDescription([
        # Declare arguments
        world_name_arg,
        sensor_offset_x_arg,
        sensor_offset_y_arg,
        camera_offset_z_arg,
        vehicle_x_arg,
        vehicle_y_arg,
        check_terrain_conn_arg,
        
        # Launch nodes and includes
        ps3_joy_node,
        # go2_h264_repub_node,  # Uncomment if needed
        point_lio_launch,
        local_planner_launch,
        terrain_analysis_launch,
        # terrain_analysis_ext_launch,  # Uncomment if needed
        # sensor_scan_generation_launch,  # Uncomment if needed
        # visualization_tools_launch,  # Uncomment if needed
        rviz_node,
        static_tf_map_node,
        static_tf_vehicle_node
    ])
