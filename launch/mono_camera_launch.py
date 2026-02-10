from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([

        # v4l2 camera node
        Node(
            package = 'v4l2_camera',
            executable = 'v4l2_camera_node',
            name = 'v4l2_camera',
            parameters = [{
                'device' : '/dev/video0',
                'pixel_format' : 'YUYV',
                'image_size' : [640,480], 
                'camera_info_url': 'file:///home/yqzzy/ros2_ws/src/mono_camera/camera_info/ost.yaml',
                'output_camera_info_from_camera': True
            }]
        ),

        # image proc node (for rectification)
        Node(
            package = 'image_proc',
            executable = 'image_proc',
            name = 'image_proc',
            parameters=[{
                'approx_sync': True
            }],
            remappings = [
                ('image_raw', '/image_raw'),
                ('camera_info', '/camera_info'),
                ('image_rect', '/image_rect'),
                ('image', '/image_raw') 
            ]
        ),
        
        # optical flow node
        Node(
            package='mono_camera',
            executable='optical_flow_node',
            name='optical_flow_node',
            output='screen'
        )
    ])
