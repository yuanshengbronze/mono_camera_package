Optical Flow based velocity estimation for ROS2. Yaw effects are accounted for, roll and pitch are ignored. 

How to Use:

make sure launch.py has the correct path for the camera info

run depth, image, and rpy publishers (depth, image_proc, rpy)

ros2 launch mono_camera mono_camera_launch.py

ros2 run mono_camera optical_flow_node --ros-args \ -p clahe.clip_limit:=3.0 -p clahe.tile_x:=8 -p clahe.tile_y:=8

