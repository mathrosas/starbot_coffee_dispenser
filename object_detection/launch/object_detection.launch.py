import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('object_detection')
    rviz_config = os.path.join(pkg_share, 'rviz', 'config.rviz')

    return LaunchDescription([

        Node(
            package='object_detection',
            executable='object_detection',
            name='object_detection',
            output='screen'
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config]
        ),
    ])