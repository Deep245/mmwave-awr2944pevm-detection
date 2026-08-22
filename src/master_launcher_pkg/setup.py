# Setup file for master launcher of all node
# Owner: Dnyandeep Mandaokar
# email: dnyandeep.mandaokar05@gmail.com


from setuptools import setup
import os
from glob import glob

package_name = 'master_launcher_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Dnyandeep Mandaokar',
    maintainer_email='dnyandeep.mandaokar05@gmail.com',
    description='Unified launcher for webcam, v4l2, and mmwave',
    license='',
    tests_require=['pytest'],
    include_package_data=True,
    entry_points={
        'console_scripts': [],
    },
)

