#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    world_file_name = 'hcr_combined.world'
    world = os.path.join(get_package_share_directory('td3'), 'worlds', world_file_name)
    launch_file_dir = os.path.join(get_package_share_directory('td3'), 'launch')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    rviz_file = os.path.join(get_package_share_directory('td3'), 'launch', 'pioneer3dx_combined.rviz')

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
            ),
            launch_arguments={'world': world}.items(),
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
            ),
        ),

        Node(
            package='td3',
            executable='combined_test_velodyne_node.py',
            name='test_velodyne_node_lidar',
            output='screen',
        ),

        Node(
            package='td3',
            executable='combined_vint_test_node.py',
            name='vint_test_node_camera',
            output='screen',
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([launch_file_dir, '/robot_state_publisher_combined.launch.py']),
            launch_arguments={'use_sim_time': use_sim_time}.items(),
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_file],
            output='screen'
        ),
    ])

if __name__ == '__main__':
    generate_launch_description()
