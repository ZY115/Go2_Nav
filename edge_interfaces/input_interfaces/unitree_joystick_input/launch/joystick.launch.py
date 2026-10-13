from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    network_interface = LaunchConfiguration('network_interface')
    domain_id = LaunchConfiguration('domain_id')

    network_interface_arg = DeclareLaunchArgument(
        'network_interface',
        default_value='',
        description='Network interface used by Unitree DDS (empty means auto-select)',
    )

    domain_id_arg = DeclareLaunchArgument(
        'domain_id',
        default_value='0',
        description='DDS domain id used by Unitree DDS runtime',
    )

    return LaunchDescription([
        network_interface_arg,
        domain_id_arg,
        Node(
            package='unitree_joystick_input',
            executable='joystick_input',
            name='joystick_input_node',
            parameters=[
                {
                    'network_interface': network_interface,
                    'domain': domain_id,
                }
            ],
        )
    ])
