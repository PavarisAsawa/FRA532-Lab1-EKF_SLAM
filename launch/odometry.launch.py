from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    odom_node = Node(
        package='lab1_ekf_slam',
        executable='odometry_node',   # console_scripts name in setup.py
        name='odometry_node',
        output='screen',
    )

    odom_node = Node(
        package='lab1_ekf_slam',
        executable='icp_node',   # console_scripts name in setup.py
        name='icp_node',
        output='screen',
    )


    return LaunchDescription([odom_node])