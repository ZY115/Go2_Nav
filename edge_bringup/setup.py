import glob
from setuptools import find_packages, setup

package_name = 'edge_bringup'
launch_files = glob.glob('launch/*.launch.py')
rviz_files = glob.glob('rviz/*.rviz')

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', launch_files),
        ('share/' + package_name + '/rviz', rviz_files),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Yihao Cai',
    maintainer_email='yihao.cai@wayne.edu',
    description='Bringup package for the Edge system with unified namespace',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        ],
    },
)
