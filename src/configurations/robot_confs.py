__author__ = "Antoine Richard"
__copyright__ = "Copyright 2023-24, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
__license__ = "BSD 3-Clause"
__version__ = "2.0.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Pose:
    position: List[float] = field(default_factory=list)
    orientation: List[float] = field(default_factory=list)


@dataclass
class SensorSettings:
    camera_frame_skip_count: Optional[int] = None
    depth_frame_skip_count: Optional[int] = None
    lidar_horizontal_resolution_deg: Optional[float] = None
    lidar_max_range_m: Optional[float] = None
    lidar_min_range_m: Optional[float] = None
    lidar_rotation_rate_hz: Optional[float] = None
    lidar_vertical_fov_deg: Optional[float] = None
    lidar_vertical_resolution_deg: Optional[float] = None


@dataclass
class ControllerSettings:
    max_acceleration: Optional[float] = None
    max_angular_acceleration: Optional[float] = None
    max_angular_speed: Optional[float] = None
    max_deceleration: Optional[float] = None
    max_linear_speed: Optional[float] = None
    max_wheel_speed: Optional[float] = None
    wheel_drive_damping: Optional[float] = None
    wheel_drive_stiffness: Optional[float] = None


@dataclass
class RobotParameters:
    robot_name: str = field(default_factory=str)
    usd_path: str = field(default_factory=str)
    pose: Pose = field(default_factory=dict)
    domain_id: int = field(default_factory=int)
    target_links: List[str] = field(default_factory=list)
    sensor_settings: SensorSettings = field(default_factory=dict)
    controller_settings: ControllerSettings = field(default_factory=dict)

    def __post_init__(self):
        self.usd_path = os.path.join(os.getcwd(), self.usd_path)
        self.pose = Pose(**self.pose)
        self.sensor_settings = SensorSettings(**self.sensor_settings)
        self.controller_settings = ControllerSettings(**self.controller_settings)


@dataclass
class RobotManagerConf:
    uses_nucleus: str = False
    is_ROS2: str = False
    max_robots: int = 5
    robots_root: str = "/Robots"
    parameters: List[RobotParameters] = field(default_factory=list)

    def __post_init__(self):
        if self.parameters:
            assert len(self.parameters) <= self.max_robots, "number of robots to register should not exeed max_robots"
            self.parameters = [RobotParameters(**param) for param in self.parameters]
        else:
            self.parameters = []
