from setuptools import setup

package_name = 'teleop_keys'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # no launch files here
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@example.com',
    description='Arrow-keys teleoperation for /cmd_vel',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'teleop_node = teleop_keys.run:main',
            'actuator_cmd_node = teleop_keys.kinematic_model:main',
            'stamp = teleop_keys.twist_stamper:main',
            'stamp_high_lvl = teleop_keys.twist_stamper_high_lvl_cmd:main',
            'rosbag_play = teleop_keys.json_reader:main',
            'global_planner_bridge = teleop_keys.global_planner_bridge:main',
            'il_local_planner = teleop_keys.IL_cmd_gen:main',
        ],
    },
)

