from setuptools import find_packages, setup

package_name = 'mono_camera'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/mono_camera_launch.py']),
        ('share/' + package_name + '/camera_info', ['camera_info/ost.yaml']),
    ],
    install_requires=['setuptools', 'numpy', 'opencv-python'],
    zip_safe=True,
    maintainer='yqzzy',
    maintainer_email='yqzzy@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        'optical_flow_node = mono_camera.optical_flow_node:main',
        ],
    },
)
