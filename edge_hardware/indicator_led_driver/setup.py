from glob import glob
from setuptools import find_packages, setup

package_name = 'indicator_led_driver'

setup(
    name=package_name,
    version='2.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Yihao Cai',
    maintainer_email='yihao.cai@wayne.edu',
    description='The driver for Indicator LED light with launch support',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'task_led_driver_node = indicator_led_driver.task_led_driver_node:main',
            'safety_led_driver_node = indicator_led_driver.safety_led_driver_node:main',
        ],
    },
)
