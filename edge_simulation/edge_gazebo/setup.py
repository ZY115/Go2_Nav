import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'edge_gazebo'

# Gather all launch files and resource files
launch_files = [f for f in glob('launch/*.launch.py') if os.path.isfile(f)]
model_files = [
    (os.path.join('share', package_name, root), [os.path.join(root, f) for f in files])
    for root, _, files in os.walk('models') if files
]
world_files = [
    (os.path.join('share', package_name, root), [os.path.join(root, f) for f in files])
    for root, _, files in os.walk('worlds') if files
]
xacro_files = [
    (os.path.join('share', package_name, root), [os.path.join(root, f) for f in files])
    for root, _, files in os.walk('xacro') if files
]

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), launch_files),
    ] + model_files + world_files + xacro_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Yihao Cai',
    maintainer_email='yihao.cai@wayne.edu',
    description='The ros2 package for Gazebo Simulation in Edge Brain Go2',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "gz_reset_handler = edge_gazebo.gz_reset_handler_node:main",
            "gz_force_handler = edge_gazebo.gz_force_handler_node:main",
        ],
    },
)
