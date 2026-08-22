# Owner: Dnyandeep Mandaokar
# email: dnyandeep.mandaokar05@gmail.com
from setuptools import setup
import os
from glob import glob

package_name = 'detection_tracker_pkg'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'scipy',
        'scikit-learn',
        'pandas',
    ],
    zip_safe=True,
    maintainer='Dnyandeep Mandaokar',
    maintainer_email='dnyandeep.mandaokar@gmail.com',
    description='Detection and tracking pipeline for mmWave radar data',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tracker_node = detection_tracker_pkg.tracker_node:main',
        ],
    },
)
