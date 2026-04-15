from setuptools import setup
import os
from glob import glob

package_name = 'Angad_Full_Assembly_description'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
        (os.path.join('share', package_name, 'meshes'), glob('meshes/*')),
        (os.path.join('share', package_name, 'config'), glob('config/*'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='author',
    maintainer_email='todo@todo.com',
    description='The ' + package_name + ' package',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'angad_walk = Angad_Full_Assembly_description.angad_walker:main',
            'test_walk = Angad_Full_Assembly_description.test_walk:main',
            'crouch_only = Angad_Full_Assembly_description.crouch_only:main',
            'slow_walk = Angad_Full_Assembly_description.slow_walk:main',
            'axis_test = Angad_Full_Assembly_description.axis_test:main',
            'lift_leg = Angad_Full_Assembly_description.lift_leg:main',
        ],
    },
)
