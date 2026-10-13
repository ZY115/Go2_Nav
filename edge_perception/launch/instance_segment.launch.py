import os, yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dir = get_package_share_directory('edge_perception')
    config_file = os.path.join(pkg_dir, 'config', 'models.yaml')

    declare_model_arg = DeclareLaunchArgument(
        'model_name',
        default_value='sam_vit_b', 
        # default_value='yolo', 
        description='Name of the model to use from models.yaml'
    )
    
    declare_image_arg = DeclareLaunchArgument(
        'image_topic', 
        default_value='/camera/image',
        description='Topic for input images'
    )
        
    declare_mask_arg = DeclareLaunchArgument(
        'mask_topic', 
        default_value='/instance_mask',
        description='Topic for instance masks'
    )

    declare_overlay_arg = DeclareLaunchArgument(
        'overlay_topic', 
        default_value='/instance_overlay',
        description='Topic for instance overlay images'
    )

    return LaunchDescription([
        declare_model_arg,
        declare_image_arg,
        declare_mask_arg,
        declare_overlay_arg,
        
        Node(
            package='edge_perception',
            executable='instance_segment_node',
            namespace='edge',
            name='instance_segment_node',
            output='screen',
            parameters=[{
                'config_file': config_file,
                'model_name': LaunchConfiguration('model_name')
            }]
        )
    ])
