from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, TimerAction, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def _start_rosbag(context, *args, **kwargs):
    bag = LaunchConfiguration('bag').perform(context)
    delay = float(LaunchConfiguration('delay').perform(context))

    bag0 = LaunchConfiguration('bag0').perform(context)
    bag1 = LaunchConfiguration('bag1').perform(context)
    bag2 = LaunchConfiguration('bag2').perform(context)

    bag_map = {'0': bag0, '1': bag1, '2': bag2}

    if bag not in bag_map:
        raise RuntimeError(f"Invalid bag: {bag}. Use bag:=0, bag:=1, or bag:=2")

    bag_path = bag_map[bag]

    rosbag_play = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', bag_path, '--clock'],
        output='screen'
    )

    return [TimerAction(period=delay, actions=[rosbag_play])]


def generate_launch_description():
    # ---- default bag folders ----
    default_bag0 = '/home/pavaris/fra532_ws/src/lab1_ekf_slam/lab1_ekf_slam/FRA532_LAB1_DATASET/fibo_floor3_seq00'
    default_bag1 = '/home/pavaris/fra532_ws/src/lab1_ekf_slam/lab1_ekf_slam/FRA532_LAB1_DATASET/fibo_floor3_seq01'
    default_bag2 = '/home/pavaris/fra532_ws/src/lab1_ekf_slam/lab1_ekf_slam/FRA532_LAB1_DATASET/fibo_floor3_seq02'

    declare_bag = DeclareLaunchArgument(
        'bag',
        default_value='0',
        description='Which bag to play: 0, 1, or 2'
    )

    declare_delay = DeclareLaunchArgument(
        'delay',
        default_value='3.0',
        description='Delay (seconds) before playing rosbag'
    )

    declare_bag0 = DeclareLaunchArgument('bag0', default_value=default_bag0)
    declare_bag1 = DeclareLaunchArgument('bag1', default_value=default_bag1)
    declare_bag2 = DeclareLaunchArgument('bag2', default_value=default_bag2)

    # ---- RViz config from package share ----
    pkg_share = get_package_share_directory('lab1_ekf_slam')
    rviz_config = os.path.join(pkg_share, 'config', 'slam_cfg.rviz')

    ekf_node = Node(
        package='lab1_ekf_slam',
        executable='odometry_node',
        name='odometry_node',
        output='screen',
    )

    icp_node = Node(
        package='lab1_ekf_slam',
        executable='icp_node',
        name='icp_node',
        output='screen',
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
    )

    delayed_rosbag = OpaqueFunction(function=_start_rosbag)

    return LaunchDescription([
        declare_bag,
        declare_delay,
        declare_bag0,
        declare_bag1,
        declare_bag2,
        ekf_node,
        icp_node,
        rviz_node,
        delayed_rosbag,
    ])
