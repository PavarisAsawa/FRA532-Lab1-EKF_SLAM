from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'lab1_ekf_slam'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.rviz')),
        
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pavaris',
    maintainer_email='101215917+PavarisAsawa@users.noreply.github.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'odometry_node = lab1_ekf_slam.localize_node:main',
            'icp_node = lab1_ekf_slam.icp_node_5:main',
            'slam_node = lab1_ekf_slam.SLAM_node:main',
            'eval_node = lab1_ekf_slam.eval_node:main'
        ],
    },
)
