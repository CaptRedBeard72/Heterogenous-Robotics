import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Define the use_sim_time argument with a default value of 'false'
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    # Define the URDF file name
    urdf_file_name = 'combined_td_robot.urdf'
    print('URDF file name: {}'.format(urdf_file_name))

    # Construct the full path to the URDF file
    urdf = os.path.join(
        get_package_share_directory('td3'),
        'urdf',
        urdf_file_name)
    
    # Read the URDF file content
    with open(urdf, 'r') as infp:
        robot_desc = infp.read()

    return LaunchDescription([
        # Declare the use_sim_time launch argument
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'),

        # Define the robot_state_publisher node
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'robot_description': robot_desc,
            }],
            remappings=[
                ('/joint_states', '/joint_states'),
                ('/tf', 'tf'),
                ('/tf_static', 'tf_static')
            ]
        ),
    ])
