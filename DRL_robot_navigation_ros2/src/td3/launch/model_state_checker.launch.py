from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='td3',
            executable='model_state_checker',
            name='model_state_checker',
            output='screen'
        )
    ])
