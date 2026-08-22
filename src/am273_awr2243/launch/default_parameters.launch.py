# Launch file for RADAR node
# Owner: Dnyandeep Mandaokar
# email: dnyandeep.mandaokar05@gmail.com


from struct import pack
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
import os

username = os.path.expanduser('~')

def generate_launch_description():

    mmwave = Node(
        package="am273_awr2243",
        executable="pcl_pub",
        parameters=[
            {'cfg_path': username+'/mmwave_ws/src/am273_awr2243/am273_awr2243/cfg_files/profile_3d_3Azim_1ElevTx_awr2944P.cfg'},
            {'cli_port': '/dev/ttyACM1'},
            {'data_port': '/dev/ttyACM2'}
         ]
    )

    return LaunchDescription([

        mmwave

    ])
