# Launch file for master launcher, use this to update which nodes to launch
# Owner: Dnyandeep Mandaokar
# email: dnyandeep.mandaokar05@gmail.com
#
# 1. SENSOR INPUT
#    └─ MMwave Radar (IWR2243 via serial)
#
# 2. PERCEPTION
#    └─ Detection Pipeline (radar) → bounding boxes

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node


def generate_launch_description():
    enable_radar = DeclareLaunchArgument(
        'enable_radar',
        default_value='true',
        description='Enable MMwave radar'
    )

    enable_detection = DeclareLaunchArgument(
        'enable_detection',
        default_value='true',
        description='Enable detection pipeline'
    )

    mmwave_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare('am273_awr2243'),
            'default_parameters.launch.py'
        ])),
        condition=IfCondition(LaunchConfiguration('enable_radar'))
    )

    detection_tracker_node = Node(
        package='detection_tracker_pkg',
        executable='tracker_node',
        name='tracker_node',
        output='screen',
        arguments=['--ros-args', '--log-level', 'WARN'],
        condition=IfCondition(LaunchConfiguration('enable_detection')),
    )

    return LaunchDescription([
        enable_radar,
        enable_detection,
        mmwave_launch,
        detection_tracker_node,
    ])
