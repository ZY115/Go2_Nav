import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dir = get_package_share_directory('edge_perception')
    default_model_path = os.path.join(pkg_dir, 'models', 'lraspp_mobilenet_w1280_h720_segmentation.onnx')
    model_path = LaunchConfiguration('model_path')
    image_topic = LaunchConfiguration('image_topic')
    camera_info_topic = LaunchConfiguration('camera_info_topic')
    pointcloud_topic = LaunchConfiguration('pointcloud_topic')
    publish_colored_cloud = LaunchConfiguration('publish_colored_cloud')

    return LaunchDescription([
        DeclareLaunchArgument('model_path', default_value=default_model_path, description='Path to ONNX segmentation model'),
        DeclareLaunchArgument('image_topic', default_value='/camera/image'),
        DeclareLaunchArgument('camera_info_topic', default_value='/camera/camera_info'),
        DeclareLaunchArgument('pointcloud_topic', default_value='/points_raw'),
        DeclareLaunchArgument('publish_colored_cloud', default_value='true'),
        Node(
            package='edge_perception',
            executable='semantic_segment_node',
            namespace='edge',
            name='semantic_segment_node',
            output='screen',
            parameters=[{
                'model_path': model_path,
                'image_topic': image_topic,
                'camera_info_topic': camera_info_topic,
                'pointcloud_topic': pointcloud_topic,
                'publish_colored_cloud': publish_colored_cloud,
            }]
        ),
    ])
