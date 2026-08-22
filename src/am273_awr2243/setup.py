# Setup file for RADAR node
# Owner: Dnyandeep Mandaokar
# email: dnyandeep.mandaokar05@gmail.com


import os
from glob import glob
from setuptools import setup

package_name = 'am273_awr2243'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*launch.[pxy][yma]*'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Dnyandeep Mandaokar',
    maintainer_email='dnyandeep.mandaokar05@gmail.com',
    description='Publishes pointcloud data from AM273+AWR2243BOOST mmWave device',
    license='',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
			'pcl_pub = am273_awr2243.publisher_member_function:main',
        ],
    },
)
