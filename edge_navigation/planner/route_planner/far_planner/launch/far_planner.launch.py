import yaml
from launch import LaunchDescription
from launch.conditions import IfCondition
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node, SetParameter
from ament_index_python.packages import get_package_share_directory
import launch_ros

def generate_launch_description():    
    # Far Planner config file path
    config_file = PathJoinSubstitution([
        get_package_share_directory('far_planner'),
        'config',
        [LaunchConfiguration('config'), '.yaml']
    ])
    
    # RViz 
    rviz_config = PathJoinSubstitution([
        get_package_share_directory('far_planner'),
        'rviz',
        [LaunchConfiguration('config'), '.rviz']
    ])
    
    return LaunchDescription([
        DeclareLaunchArgument('odom_topic', default_value='odom', description='Odometry topic to use'),
        DeclareLaunchArgument('config', default_value='default', description='Configuration file for RViz and Far Planner'),
        DeclareLaunchArgument('use_rviz', default_value='true', description='Whether to launch RViz'),

        # DeclareLaunchArgument('config', default_value='matterport'),
        Node(
            package='far_planner',
            executable='far_planner',
            name='far_planner',
            output='screen',
            parameters=[
                config_file,
            ],
            remappings=[
                # ('/odom_world', '/state_estimation'),
                ('/odom_world', LaunchConfiguration("odom_topic")),
                ('/terrain_cloud', '/terrain_map_ext'),
                ('/scan_cloud', '/terrain_map'),
                ('/terrain_local_cloud', '/registered_scan')
            ]
        ),
        
        Node(
            package='rviz2',
            executable='rviz2',
            name='far_rviz',
            arguments=['-d', rviz_config],
            parameters=[{'use_sim_time': True}],
            output='screen',
            respawn=False,
            condition=IfCondition(LaunchConfiguration('use_rviz'))
        ),

        # Including another launch file
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([get_package_share_directory('graph_decoder'), '/launch/decoder.launch'])
        )
    ])
