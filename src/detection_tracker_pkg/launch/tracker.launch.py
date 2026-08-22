# Owner: Dnyandeep Mandaokar
# email: dnyandeep.mandaokar05@gmail.com
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # Node(
        #     package='am273_awr2243',
        #     executable='pcl_pub',
        #     name='radar_node',
        #     output='screen'
        # ),

        Node(
            package='detection_tracker_pkg',
            executable='tracker_node',
            name='tracker_node',
            output='screen',
            parameters=[{
                'classified_point_size': 0.2,
            }]
        ),
    ])
