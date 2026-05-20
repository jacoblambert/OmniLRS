__author__ = "Antoine Richard, Junnosuke Kamohara"
__copyright__ = "Copyright 2023-24, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
__license__ = "BSD 3-Clause"
__version__ = "2.0.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"

from typing import Dict, List, Tuple
import numpy as np
import warnings
import os

import omni
from isaacsim.core.api.world import World
import omni.graph.core as og
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.transformations import (
    get_relative_transform,
    pose_from_tf_matrix,
)
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.core.utils.nucleus import get_assets_root_path
from omni.isaac.dynamic_control import _dynamic_control
from isaacsim.core.prims import SingleRigidPrim, RigidPrim
from pxr import Gf, UsdGeom, Usd

from WorldBuilders.pxr_utils import createXform, createObject, setDefaultOps
from src.configurations.robot_confs import RobotManagerConf


class RobotManager:
    """
    RobotManager class.
    It allows to spawn, reset, teleport robots. It also allows to automatically add namespaces to topics,
    and tfs to enable multi-robot operation."""

    def __init__(
        self,
        RM_conf: RobotManagerConf,
    ) -> None:
        """
        Args:
            RM_conf (RobotManagerConf): The configuration of the robot manager.
        """

        self.stage = omni.usd.get_context().get_stage()
        self.RM_conf = RobotManagerConf(**RM_conf)
        self.robot_parameters = self.RM_conf.parameters
        self.uses_nucleus = self.RM_conf.uses_nucleus
        self.is_ROS2 = self.RM_conf.is_ROS2
        self.max_robots = self.RM_conf.max_robots
        self.robots_root = self.RM_conf.robots_root
        createXform(self.stage, self.robots_root)
        self.robots: Dict[str, Robot] = {}
        self.robots_RG: Dict[str, RobotRigidGroup] = {}
        self.num_robots = 0

    def resolve_sensor_settings(self, robot_name: str, usd_path: str, sensor_settings=None):
        """
        Reuses configured sensor settings for matching robots when explicit overrides are absent.
        """

        if sensor_settings is not None:
            return sensor_settings

        normalized_name = robot_name if robot_name.startswith("/") else f"/{robot_name}"
        normalized_usd_path = os.path.normpath(str(usd_path)) if usd_path is not None else None

        for robot_parameter in self.robot_parameters:
            candidate_name = (
                robot_parameter.robot_name
                if robot_parameter.robot_name.startswith("/")
                else f"/{robot_parameter.robot_name}"
            )
            if normalized_name == candidate_name:
                return robot_parameter.sensor_settings

            if normalized_usd_path is not None and normalized_usd_path == os.path.normpath(robot_parameter.usd_path):
                return robot_parameter.sensor_settings

        return None

    def resolve_controller_settings(self, robot_name: str, usd_path: str, controller_settings=None):
        """
        Reuses configured controller settings for matching robots when explicit overrides are absent.
        """

        if controller_settings is not None:
            return controller_settings

        normalized_name = robot_name if robot_name.startswith("/") else f"/{robot_name}"
        normalized_usd_path = os.path.normpath(str(usd_path)) if usd_path is not None else None

        for robot_parameter in self.robot_parameters:
            candidate_name = (
                robot_parameter.robot_name
                if robot_parameter.robot_name.startswith("/")
                else f"/{robot_parameter.robot_name}"
            )
            if normalized_name == candidate_name:
                return robot_parameter.controller_settings

            if normalized_usd_path is not None and normalized_usd_path == os.path.normpath(robot_parameter.usd_path):
                return robot_parameter.controller_settings

        return None

    def preload_robot(
        self,
        world: World,
    ) -> None:
        """
        Preload the robots in the scene.
        Args:
            world (Usd.Stage): The usd stage scene.
        """
        if len(self.robot_parameters) > 0:
            for robot_parameter in self.robot_parameters:
                self.add_robot(
                    robot_parameter.usd_path,
                    robot_parameter.robot_name,
                    robot_parameter.pose.position,
                    robot_parameter.pose.orientation,
                    robot_parameter.domain_id,
                    robot_parameter.sensor_settings,
                    robot_parameter.controller_settings,
                )
                self.add_RRG(
                    robot_parameter.robot_name,
                    robot_parameter.target_links,
                    world,
                )

    def preload_robot_at_pose(
        self,
        world: World,
        position: Tuple[float, float, float],
        orientation: Tuple[float, float, float, float],
    ) -> None:
        """
        Preload the robots in the scene.
        Args:
            world (Usd.Stage): The usd stage scene.
            position (Tuple[float, float, float]): The position of the robot. (x, y, z)
            orientation (Tuple[float, float, float, float]): The orientation of the robot. (w, x, y, z)
        """
        if len(self.robot_parameters) > 0:
            for robot_parameter in self.robot_parameters:
                self.add_robot(
                    robot_parameter.usd_path,
                    robot_parameter.robot_name,
                    position,
                    orientation,
                    robot_parameter.domain_id,
                    robot_parameter.sensor_settings,
                    robot_parameter.controller_settings,
                )
                self.add_RRG(
                    robot_parameter.robot_name,
                    robot_parameter.target_links,
                    world,
                )

    def add_robot(
        self,
        usd_path: str = None,
        robot_name: str = None,
        p: Tuple[float, float, float] = [0, 0, 0],
        q: Tuple[float, float, float, float] = [0, 0, 0, 1],
        domain_id: int = None,
        sensor_settings=None,
        controller_settings=None,
    ) -> None:
        """
        Add a robot to the scene.

        Args:
            usd_path (str): The path of the robot's usd file.
            robot_name (str): The name of the robot.
            p (Tuple[float, float, float]): The position of the robot. (x, y, z)
            q (Tuple[float, float, float, float]): The orientation of the robot. (w, x, y, z)
            domain_id (int): The domain id of the robot. Not required if the robot is not ROS2 enabled.
        """

        if robot_name[0] != "/":
            robot_name = "/" + robot_name
        sensor_settings = self.resolve_sensor_settings(robot_name, usd_path, sensor_settings)
        controller_settings = self.resolve_controller_settings(robot_name, usd_path, controller_settings)
        if self.num_robots >= self.max_robots:
            pass
        else:
            if robot_name in self.robots.keys():
                warnings.warn("Robot already exists. Ignoring request.")
            else:
                self.robots[robot_name] = Robot(
                    usd_path,
                    robot_name,
                    is_on_nucleus=self.uses_nucleus,
                    is_ROS2=self.is_ROS2,
                    domain_id=domain_id,
                    robots_root=self.robots_root,
                    sensor_settings=sensor_settings,
                    controller_settings=controller_settings,
                )
                self.robots[robot_name].load(p, q)
                self.num_robots += 1

    def add_RRG(
        self,
        robot_name: str = None,
        target_links: List[str] = None,
        world=None,
    ) -> None:
        """
        Add a robot rigid group to the scene.

        Args:
            robot_name (str): The name of the robot.
            target_links (List[str]): List of link names.
            world (Usd.Stage): usd stage scene.
        """
        rrg = RobotRigidGroup(
            self.robots_root,
            robot_name,
            target_links,
        )
        rrg.initialize(world)
        self.robots_RG[robot_name] = rrg

    def reset_robots(self) -> None:
        """
        Reset all the robots to their original position.
        """

        for robot in self.robots.keys():
            self.robots[robot].reset()

    def reset_robot(self, robot_name: str = None) -> None:
        """
        Reset a specific robot to its original position.

        Args:
            robot_name (str): The name of the robot.
        """

        if robot_name and robot_name[0] != "/":
            robot_name = "/" + robot_name

        if robot_name in self.robots.keys():
            self.robots[robot_name].reset()
        else:
            warnings.warn("Robot does not exist. Ignoring request.")

    def teleport_robot(
        self, robot_name: str = None, position: np.ndarray = None, orientation: np.ndarray = None
    ) -> None:
        """
        Teleport a specific robot to a specific position and orientation.

        Args:
            robot_name (str): The name of the robot.
        """
        if robot_name and robot_name[0] != "/":
            robot_name = "/" + robot_name

        if robot_name in self.robots.keys():
            self.robots[robot_name].teleport(position, orientation)
        else:
            warnings.warn("Robot does not exist. Ignoring request.")
            print("available robots: ", self.robots.keys())


class Robot:
    """
    Robot class.
    It allows to spawn, reset, teleport a robot. It also allows to automatically add namespaces to topics,
    and tfs to enable multi-robot operation.
    """

    def __init__(
        self,
        usd_path: str,
        robot_name: str,
        robots_root: str = "/Robots",
        is_on_nucleus: bool = False,
        is_ROS2: bool = False,
        domain_id: int = 0,
        sensor_settings=None,
        controller_settings=None,
    ) -> None:
        """
        Args:
            usd_path (str): The path of the robot's usd file.
            robot_name (str): The name of the robot.
            robots_root (str, optional): The root path of the robots. Defaults to "/Robots".
            is_on_nucleus (bool, optional): Whether the robots are loaded from the nucleus or not. Defaults to False.
            is_ROS2 (bool, optional): Whether the robots are ROS2 enabled or not. Defaults to False.
            domain_id (int, optional): The domain id of the robot. Defaults to 0."""

        self.stage: Usd.Stage = omni.usd.get_context().get_stage()
        self.usd_path = str(usd_path)
        self.robots_root = robots_root
        self.robot_name = robot_name
        self.robot_path = os.path.join(self.robots_root, self.robot_name.strip("/"))
        self.is_on_nucleus = is_on_nucleus
        self.is_ROS2 = is_ROS2
        self.domain_id = int(domain_id)
        self.sensor_settings = sensor_settings
        self.controller_settings = controller_settings
        self.dc = _dynamic_control.acquire_dynamic_control_interface()
        self.root_body_id = None

    def _find_descendants_by_name(self, names: List[str]) -> Dict[str, Usd.Prim]:
        """
        Finds named prims under the robot root.
        """

        found = {}
        root_prim = self.stage.GetPrimAtPath(self.robot_path)
        if not root_prim:
            return found

        for prim in Usd.PrimRange(root_prim):
            name = prim.GetName()
            if name in names and name not in found:
                found[name] = prim
        return found

    def _find_first_descendant_of_type(self, type_name: str) -> Usd.Prim:
        """
        Finds the first prim of the requested type under the robot root.
        """

        root_prim = self.stage.GetPrimAtPath(self.robot_path)
        if not root_prim:
            return Usd.Prim()

        for prim in Usd.PrimRange(root_prim):
            if prim.GetTypeName() == type_name:
                return prim
        return Usd.Prim()

    def _set_attr_if_present(self, prim: Usd.Prim, attr_name: str, value) -> bool:
        """
        Sets an attribute when it exists on a prim.
        """

        if not prim:
            return False

        attr = prim.GetAttribute(attr_name)
        if not attr:
            return False

        attr.Set(value)
        return True

    def configure_sensor_streams(self) -> None:
        """
        Applies optional runtime sensor tuning for remote operation.
        """

        if self.sensor_settings is None:
            return

        settings = self.sensor_settings
        camera_nodes = self._find_descendants_by_name(
            [
                "ros2_camera_helper_01",
                "ros2_camera_helper_02",
                "ros2_camera_info_helper",
                "ros2_camera_info_helper_01",
            ]
        )

        if settings.camera_frame_skip_count is not None:
            for node_name in ("ros2_camera_helper_01", "ros2_camera_info_helper"):
                self._set_attr_if_present(
                    camera_nodes.get(node_name),
                    "inputs:frameSkipCount",
                    int(settings.camera_frame_skip_count),
                )

        if settings.depth_frame_skip_count is not None:
            for node_name in ("ros2_camera_helper_02", "ros2_camera_info_helper_01"):
                self._set_attr_if_present(
                    camera_nodes.get(node_name),
                    "inputs:frameSkipCount",
                    int(settings.depth_frame_skip_count),
                )

        lidar_prim = self._find_first_descendant_of_type("Lidar")

        lidar_attr_values = {
            "horizontalResolution": getattr(settings, "lidar_horizontal_resolution_deg", None),
            "maxRange": getattr(settings, "lidar_max_range_m", None),
            "minRange": getattr(settings, "lidar_min_range_m", None),
            "verticalFov": getattr(settings, "lidar_vertical_fov_deg", None),
            "verticalResolution": getattr(settings, "lidar_vertical_resolution_deg", None),
        }
        applied_lidar_attrs = {}
        for attr_name, value in lidar_attr_values.items():
            if value is not None and self._set_attr_if_present(lidar_prim, attr_name, float(value)):
                applied_lidar_attrs[attr_name] = float(value)
        if applied_lidar_attrs:
            print(f"applied lidar geometry settings for {self.robot_path}: {applied_lidar_attrs}")

        if settings.lidar_rotation_rate_hz is not None:
            self._set_attr_if_present(lidar_prim, "rotationRate", float(settings.lidar_rotation_rate_hz))

    def configure_drive_controller(self) -> None:
        """
        Applies optional runtime tuning to the robot differential controller graph.
        """

        array_names_prim = self.stage.GetPrimAtPath(
            os.path.join(self.robot_path, "Graphs", "differential_controller", "ArrayNames")
        )
        array_name_defaults = {
            "inputs:input0": "front_left_wheel",
            "inputs:input1": "front_right_wheel",
            "inputs:input2": "rear_left_wheel",
            "inputs:input3": "rear_right_wheel",
        }
        repaired_array_names = {}
        if array_names_prim:
            for attr_name, expected_value in array_name_defaults.items():
                attr = array_names_prim.GetAttribute(attr_name)
                current_value = attr.Get() if attr else None
                if current_value in (None, "") and self._set_attr_if_present(array_names_prim, attr_name, expected_value):
                    repaired_array_names[attr_name] = expected_value
        if repaired_array_names:
            print(f"repaired drive controller joint names for {self.robot_path}: {repaired_array_names}")

        if self.controller_settings is None:
            return

        controller_prim = self.stage.GetPrimAtPath(
            os.path.join(self.robot_path, "Graphs", "differential_controller", "differential_controller")
        )
        if not controller_prim:
            warnings.warn(f"differential controller prim not found for {self.robot_path}")
            return

        settings = self.controller_settings
        attr_values = {
            "inputs:maxAcceleration": getattr(settings, "max_acceleration", None),
            "inputs:maxAngularAcceleration": getattr(settings, "max_angular_acceleration", None),
            "inputs:maxAngularSpeed": getattr(settings, "max_angular_speed", None),
            "inputs:maxDeceleration": getattr(settings, "max_deceleration", None),
            "inputs:maxLinearSpeed": getattr(settings, "max_linear_speed", None),
            "inputs:maxWheelSpeed": getattr(settings, "max_wheel_speed", None),
        }
        applied = {}
        for attr_name, value in attr_values.items():
            if value is not None:
                if self._set_attr_if_present(controller_prim, attr_name, float(value)):
                    applied[attr_name] = float(value)
        if applied:
            print(f"applied drive controller settings for {self.robot_path}: {applied}")

        wheel_stiffness = getattr(settings, "wheel_drive_stiffness", None)
        wheel_damping = getattr(settings, "wheel_drive_damping", None)
        if wheel_stiffness is None and wheel_damping is None:
            return

        wheel_joint_updates = {}
        root_prim = self.stage.GetPrimAtPath(self.robot_path)
        if not root_prim:
            return
        for prim in Usd.PrimRange(root_prim):
            if prim.GetTypeName() != "PhysicsRevoluteJoint":
                continue
            if "wheel" not in prim.GetName():
                continue
            if wheel_stiffness is not None and self._set_attr_if_present(
                prim, "drive:angular:physics:stiffness", float(wheel_stiffness)
            ):
                wheel_joint_updates.setdefault(prim.GetPath().pathString, {})["stiffness"] = float(wheel_stiffness)
            if wheel_damping is not None and self._set_attr_if_present(
                prim, "drive:angular:physics:damping", float(wheel_damping)
            ):
                wheel_joint_updates.setdefault(prim.GetPath().pathString, {})["damping"] = float(wheel_damping)
        if wheel_joint_updates:
            print(f"applied wheel drive settings for {self.robot_path}: {wheel_joint_updates}")

    def get_root_rigid_body_path(self) -> None:
        """
        Get the root rigid body path of the robot.
        """

        art = self.dc.get_articulation(self.robot_path)
        self.root_body_id = self.dc.get_articulation_root_body(art)

    def edit_graphs(self) -> None:
        """
        Edit the graphs of the robot to add namespaces to topics and tfs.
        """

        selected_paths = []
        for prim in Usd.PrimRange(self.stage.GetPrimAtPath(self.robot_path)):
            l = [attr for attr in prim.GetAttributes() if attr.GetName().split(":")[0] == "graph"]
            if l:
                selected_paths.append(prim.GetPath())

        for path in selected_paths:
            prim = self.stage.GetPrimAtPath(path)
            prim.GetAttribute("graph:variable:Namespace").Set(self.robot_name)
            if self.is_ROS2:
                prim.GetAttribute("graph:variable:Context").Set(self.domain_id)

    def configure_ros_topic_names(self) -> None:
        """
        Normalizes ROS topic names in imported USD graphs.

        OmniLRS robot USDs expose ROS helper topic names as graph attributes.
        Topic expansion via the graph-level Namespace variable is not applied
        consistently by imported Isaac ROS helper graphs, so write fully
        qualified names for robot topics here.
        """

        if not self.is_ROS2:
            return

        topic_updates = {}
        root_prim = self.stage.GetPrimAtPath(self.robot_path)
        if not root_prim:
            return
        robot_namespace = self.robot_name if self.robot_name.startswith("/") else f"/{self.robot_name}"

        for prim in Usd.PrimRange(root_prim):
            prim_path = prim.GetPath().pathString.lower()
            for attr in prim.GetAttributes():
                attr_name = attr.GetName()
                try:
                    current_value = attr.Get()
                except Exception:
                    continue
                is_topic_attr = "topic" in attr_name.lower()
                is_imu_topic_value = "ros2_publish_imu" in prim_path and str(current_value).strip("/") == "imu"
                is_topic_attr = is_topic_attr or is_imu_topic_value
                if not is_topic_attr:
                    continue
                if not isinstance(current_value, str):
                    continue

                context = f"{prim_path} {attr_name.lower()} {current_value.lower()}"
                normalized_value = current_value.strip("/")
                if "tfrobot" in context:
                    normalized_value = "/tf"
                elif "pointcloud" in context or "pcl" in context or "lidar" in context:
                    normalized_value = "lidar_0/pointcloud/raw"
                elif "cmd_vel" in context or "twist" in context:
                    normalized_value = "cmd_vel"
                elif "odom" in context:
                    normalized_value = "odom"
                elif "imu" in context:
                    normalized_value = "imu"
                elif "camera_info" in context or "rgb_info" in context:
                    normalized_value = "camera_0/camera_info"
                elif "depth" in context:
                    normalized_value = "camera_0/depth/image_raw"
                elif "rgb" in context or "image" in context or "camera" in context:
                    normalized_value = "camera_0/image_raw"

                if normalized_value != "/tf":
                    normalized_value = f"{robot_namespace}/{normalized_value.lstrip('/')}"

                if normalized_value != current_value:
                    attr.Set(normalized_value)
                    topic_updates[f"{prim.GetPath().pathString}:{attr_name}"] = normalized_value

        # The IMU helper topic is stored as an OmniGraph input in some USDs and
        # may not appear as a normal USD "topic" attribute during traversal.
        imu_topic = f"{robot_namespace}/imu"
        imu_attr_path = f"{self.robot_path}/Graphs/ROS_Camera/ros2_publish_imu.inputs:topicName"
        try:
            og.Controller.set(og.Controller.attribute(imu_attr_path), imu_topic)
            topic_updates[imu_attr_path] = imu_topic
        except Exception:
            pass

        if topic_updates:
            print(f"normalized ROS topic names for {self.robot_path}: {topic_updates}")

    def load(self, position: np.ndarray, orientation: np.ndarray) -> None:
        """
        Load the robot in the scene, and automatically edit its graphs.

        Args:
            position (np.ndarray): The position of the robot.
            orientation (np.ndarray): The orientation of the robot.
        """

        self.stage = omni.usd.get_context().get_stage()
        self.set_reset_pose(position, orientation)
        if self.is_on_nucleus:
            nucleus = get_assets_root_path()
            self.usd_path = os.path.join(nucleus, self.usd_path)
        createObject(
            self.robot_path,
            self.stage,
            self.usd_path,
            is_instance=False,
            position=Gf.Vec3d(*position),
            rotation=Gf.Quatd(*orientation),
        )
        self.configure_sensor_streams()
        self.configure_drive_controller()
        self.configure_ros_topic_names()
        self.edit_graphs()

    def get_pose(self) -> List[float]:
        """
        Get the pose of the robot.
        Returns:
            List[float]: The pose of the robot. (x, y, z), (qx, qy, qz, qw)
        """
        if self.root_body_id is None:
            self.get_root_rigid_body_path()
        pose = self.dc.get_rigid_body_pose(self.root_body_id)
        return pose.p, pose.r

    def set_reset_pose(self, position: np.ndarray, orientation: np.ndarray) -> None:
        """
        Set the reset pose of the robot.

        Args:
            position (np.ndarray): The position of the robot.
            orientation (np.ndarray): The orientation of the robot.
        """

        self.reset_position = position
        self.reset_orientation = orientation

    def teleport(self, p: List[float], q: List[float]) -> None:
        """
        Teleport the robot to a specific position and orientation.

        Args:
            p (list): The position of the robot.
            q (list): The orientation of the robot. (x, y, z, w)
        """

        self.get_root_rigid_body_path()
        transform = _dynamic_control.Transform(p, q)
        self.dc.set_rigid_body_pose(self.root_body_id, transform)
        self.dc.set_rigid_body_linear_velocity(self.root_body_id, [0, 0, 0])
        self.dc.set_rigid_body_angular_velocity(self.root_body_id, [0, 0, 0])

    def reset(self) -> None:
        """
        Reset the robot to its original position and orientation.
        """

        # w = self.reset_orientation.GetReal()
        # xyz = self.reset_orientation.GetImaginary()
        self.root_body_id = None
        self.teleport(
            [self.reset_position[0], self.reset_position[1], self.reset_position[2]],
            [
                self.reset_orientation[1],
                self.reset_orientation[2],
                self.reset_orientation[3],
                self.reset_orientation[0],
            ],
        )


class RobotRigidGroup:
    """
    Class which deals with rigidprims and rigidprimview of a single robot.
    It is used to retrieve world pose, and contact forces, or apply force/torque.
    """

    def __init__(self, root_path: str = "/Robots", robot_name: str = None, target_links: List[str] = None):
        """
        Args:
            root_path (str): The root path of the robots.
            robot_name (str): The name of the robot.
            target_links (List[str]): List of link names.
        """

        self.root_path = root_path
        self.robot_name = robot_name
        self.target_links = target_links
        self.prims = []
        self.prim_views = []

    def initialize(self, world: World) -> None:
        """
        Initialize the rigidprims and rigidprimviews of the robot.

        Args:
            world (World): A Omni.isaac.core.world.World object.
        """

        world.reset()
        if len(self.target_links) > 0:
            for target_link in self.target_links:
                rigid_prim = SingleRigidPrim(
                    prim_path=os.path.join(self.root_path, self.robot_name, target_link),
                    name=f"{self.robot_name}/{target_link}",
                )
                rigid_prim_view = RigidPrim(
                    prim_paths_expr=os.path.join(self.root_path, self.robot_name, target_link),
                    name=f"{self.robot_name}/{target_link}_view",
                    track_contact_forces=True,
                )
                rigid_prim_view.initialize()
                self.prims.append(rigid_prim)
                self.prim_views.append(rigid_prim_view)
        world.reset()

    def get_world_poses(self) -> np.ndarray:
        """
        Returns the world pose matrix of target links.

        Returns:
            pose (np.ndarray): The world pose matrix of target links.
        """

        n_links = len(self.target_links)
        pose = np.zeros((n_links, 4, 4))
        for i, prim in enumerate(self.prims):
            position, orientation = prim.get_world_pose()
            orientation = quat_to_rot_matrix(orientation)
            pose[i, :3, 3] = 1
            pose[i, :3, :3] = orientation
            pose[i, :3, 3] = position
        return pose

    def get_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the pose of target links.

        Returns:
            positions (np.ndarray): The position of target links. (x, y, z)
            orientations (np.ndarray): The orientation of target links. (w, x, y, z)
        """

        n_links = len(self.target_links)
        positions = np.zeros((n_links, 3))
        orientations = np.zeros((n_links, 4))
        for i, prim in enumerate(self.prims):
            position, orientation = prim.get_world_pose()
            positions[i, :] = position
            orientations[i, :] = orientation
        return positions, orientations

    def get_velocities(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the linear/angular velocity of target links.

        Returns:
            linear_velocities (np.ndarray): The linear velocity of target links.
            angular_velocities (np.ndarray): The angular velocity of target links.
        """

        n_links = len(self.target_links)
        linear_velocities = np.zeros((n_links, 3))
        angular_velocities = np.zeros((n_links, 3))
        for i, prim in enumerate(self.prims):
            linear_velocity, angular_velocity = prim.get_velocities()
            linear_velocities[i, :] = linear_velocity
            angular_velocities[i, :] = angular_velocity
        return linear_velocities, angular_velocities

    def get_net_contact_forces(self) -> np.ndarray:
        """
        Returns net contact forces on each target link.

        Returns:
            contact_forces (np.ndarray): The net contact forces on each target link.
        """

        n_links = len(self.target_links)
        contact_forces = np.zeros((n_links, 3))
        for i, prim_view in enumerate(self.prim_views):
            contact_force = prim_view.get_net_contact_forces().squeeze()
            contact_forces[i, :] = contact_force
        return contact_forces

    def apply_force_torque(self, forces: np.ndarray, torques: np.ndarray) -> None:
        """
        Apply force and torque (defined in local body frame) to body frame of the four wheels.

        Args:
            forces (np.ndarray): The forces to apply to the body origin of the four wheels.
                                 (Fx, Fy, Fz) = (F_DP, F_S, F_N)
            torques (np.ndarray): The torques to apply to the body origin of the four wheels.
                                 (Mx, My, Mz0 = (M_O,-M_R, M_S)
        """

        n_links = len(self.target_links)
        assert forces.shape[0] == n_links, "given force does not have matching shape."
        assert torques.shape[0] == n_links, "given torque does not have matching shape."
        for i, prim_view in enumerate(self.prim_views):
            prim_view.apply_forces_and_torques_at_pos(forces=forces[i], torques=torques[i], is_global=False)
