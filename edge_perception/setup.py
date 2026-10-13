from glob import glob
from setuptools import find_packages, setup

package_name = 'edge_perception'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/checkpoints', glob('checkpoints/*')),
        ('share/' + package_name + '/scripts', glob('scripts/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools', 'numpy', 'torch', 'opencv-python', 'cv_bridge', 'segment_anything', 'transformers'],
    zip_safe=True,
    maintainer='Yihao Cai',
    maintainer_email='yihao.cai@wayne.edu',
    description='Instance/Semantic/Panoptic segmentation projection onto point clouds for Jetson.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'semantic_segment_node = edge_perception.segmentation.semantic_segment.semantic_segment_node:main',
            'instance_segment_node = edge_perception.segmentation.instance_segment.instance_segment_node:main',
            'panoptic_segment_node = edge_perception.segmentation.panoptic_segment.panoptic_segment_node:main',
        ],
    },
)
