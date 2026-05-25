__author__ = "OpenAI"

from dataclasses import dataclass
import math
import os
from typing import Optional

import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32


@dataclass(frozen=True)
class ResourceLayer:
    name: str
    field: np.ndarray
    resolution: float
    origin_x: float
    origin_y: float
    threshold: float


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip()


def _normalize_robot_name(robot_name: Optional[str]) -> Optional[str]:
    if robot_name is None or robot_name == "":
        return None
    return robot_name if robot_name.startswith("/") else f"/{robot_name}"


class ROS_ResourceMapPublisher(Node):
    """
    Publishes simulator-owned resource truth and entered-cell samples.

    The full resource map is GT/evaluation only. The autonomy stack should use
    the per-resource sample topics, which expose only the value under the robot.
    """

    def __init__(
        self,
        environment_controller,
        robot_namespace: str = "astrobot_0",
        robot_name: Optional[str] = None,
        map_frame: str = "gt_map",
        publish_rate_hz: float = 1.0,
        sample_rate_hz: float = 2.0,
    ) -> None:
        super().__init__("omnilrs_resource_map")

        if not self.has_parameter("use_sim_time"):
            self.declare_parameter("use_sim_time", True)
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

        self.environment_controller = environment_controller
        self.robot_namespace = robot_namespace.strip("/")
        self.robot_name = _normalize_robot_name(robot_name)
        self.map_frame = map_frame
        self.enabled = _env_bool("OMNILRS_RESOURCE_ENABLED", True)
        self.mode = _env_str("OMNILRS_RESOURCE_MODE", "deterministic").lower()
        self.layers = _env_str("OMNILRS_RESOURCE_LAYERS", "water")
        self.random_seed = _env_int("OMNILRS_RESOURCE_SEED", 4242)
        self.sample_noise_std = max(0.0, _env_float("OMNILRS_RESOURCE_SAMPLE_NOISE_STD", 0.0))

        self.resource_layers: list[ResourceLayer] = []
        self.map_publishers = {}
        self.sample_publishers = {}
        self.sample_pose_publishers = {}

        if self.enabled:
            self._build_layers()

        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        for layer in self.resource_layers:
            self.map_publishers[layer.name] = self.create_publisher(
                OccupancyGrid, f"/gt/resource_maps/{layer.name}", map_qos
            )
            self.sample_publishers[layer.name] = self.create_publisher(
                Float32, f"/{self.robot_namespace}/resource/{layer.name}/sample", 10
            )
            self.sample_pose_publishers[layer.name] = self.create_publisher(
                PoseStamped, f"/{self.robot_namespace}/resource/{layer.name}/sample_pose", 10
            )

        self.create_timer(1.0 / max(float(publish_rate_hz), 0.1), self._publish_maps)
        self.create_timer(1.0 / max(float(sample_rate_hz), 0.1), self._publish_samples)
        self.get_logger().info(
            f"resource maps enabled={self.enabled} mode={self.mode} layers={[layer.name for layer in self.resource_layers]}"
        )

    def _terrain_snapshot(self) -> tuple[np.ndarray, float, float, float]:
        source = self.environment_controller.get_gt_terrain_data()
        dem = source["dem"]
        if dem is None:
            raise RuntimeError("terrain DEM unavailable for resource map generation")
        return (
            np.asarray(dem),
            float(source["resolution"]),
            float(source["origin"][0]),
            float(source["origin"][1]),
        )

    def _build_layers(self) -> None:
        dem, resolution, origin_x, origin_y = self._terrain_snapshot()
        height, width = dem.shape[:2]
        names = [name.strip() for name in self.layers.split(",") if name.strip()]
        rng = np.random.default_rng(self.random_seed)
        for layer_index, name in enumerate(names):
            threshold = _env_float(f"OMNILRS_RESOURCE_{name.upper()}_THRESHOLD", 0.75)
            field = self._make_field(name, layer_index, height, width, resolution, origin_x, origin_y, rng)
            self.resource_layers.append(ResourceLayer(name, field, resolution, origin_x, origin_y, threshold))

    def _make_field(
        self,
        name: str,
        layer_index: int,
        height: int,
        width: int,
        resolution: float,
        origin_x: float,
        origin_y: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        xs = origin_x + (np.arange(width, dtype=np.float32) + 0.5) * resolution
        ys = origin_y + (np.arange(height, dtype=np.float32) + 0.5) * resolution
        xx, yy = np.meshgrid(xs, ys)
        field = np.zeros((height, width), dtype=np.float32)

        if self.mode == "random":
            count = _env_int("OMNILRS_RESOURCE_RANDOM_GAUSSIANS", 3)
            sigma_min = _env_float("OMNILRS_RESOURCE_RANDOM_SIGMA_MIN", 3.5)
            sigma_max = _env_float("OMNILRS_RESOURCE_RANDOM_SIGMA_MAX", 7.5)
            amp_min = _env_float("OMNILRS_RESOURCE_RANDOM_AMPLITUDE_MIN", 0.65)
            amp_max = _env_float("OMNILRS_RESOURCE_RANDOM_AMPLITUDE_MAX", 1.0)
            margin = _env_float("OMNILRS_RESOURCE_RANDOM_MARGIN_M", 5.0)
            min_x = origin_x + margin
            max_x = origin_x + width * resolution - margin
            min_y = origin_y + margin
            max_y = origin_y + height * resolution - margin
            centers = [
                (
                    float(rng.uniform(min_x, max_x)),
                    float(rng.uniform(min_y, max_y)),
                    float(rng.uniform(sigma_min, sigma_max)),
                    float(rng.uniform(amp_min, amp_max)),
                )
                for _ in range(max(count, 1))
            ]
        else:
            offset = float(layer_index) * 1.5
            centers = [
                (-8.0 + offset, 7.0 - offset, 6.0, 1.0),
                (8.0 - offset, -6.0 + offset, 5.0, 0.85),
            ]

        for cx, cy, sigma, amplitude in centers:
            sigma = max(float(sigma), resolution)
            field += float(amplitude) * np.exp(-0.5 * (((xx - cx) / sigma) ** 2 + ((yy - cy) / sigma) ** 2))

        max_value = float(np.max(field))
        if max_value > 0.0:
            field /= max_value
        self.get_logger().info(f"resource layer {name}: mode={self.mode} centers={centers}")
        return np.clip(field, 0.0, 1.0).astype(np.float32)

    def _publish_maps(self) -> None:
        for layer in self.resource_layers:
            msg = OccupancyGrid()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.map_frame
            msg.info.map_load_time = msg.header.stamp
            msg.info.resolution = layer.resolution
            msg.info.width = int(layer.field.shape[1])
            msg.info.height = int(layer.field.shape[0])
            msg.info.origin.position.x = layer.origin_x
            msg.info.origin.position.y = layer.origin_y
            msg.info.origin.orientation.w = 1.0
            msg.data = np.rint(layer.field * 100.0).astype(np.int8).reshape(-1).tolist()
            self.map_publishers[layer.name].publish(msg)

    def _publish_samples(self) -> None:
        if not self.resource_layers:
            return
        world_position, world_orientation = self.environment_controller.get_gt_robot_world_pose(self.robot_name)
        x = float(world_position[0])
        y = float(world_position[1])
        for layer in self.resource_layers:
            col = int(math.floor((x - layer.origin_x) / layer.resolution))
            row = int(math.floor((y - layer.origin_y) / layer.resolution))
            value = 0.0
            if 0 <= row < layer.field.shape[0] and 0 <= col < layer.field.shape[1]:
                value = float(layer.field[row, col])
            if self.sample_noise_std > 0.0:
                value = float(np.clip(value + np.random.normal(0.0, self.sample_noise_std), 0.0, 1.0))
            self.sample_publishers[layer.name].publish(Float32(data=value))

            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = self.map_frame
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = float(world_position[2])
            pose.pose.orientation.x = float(world_orientation[0])
            pose.pose.orientation.y = float(world_orientation[1])
            pose.pose.orientation.z = float(world_orientation[2])
            pose.pose.orientation.w = float(world_orientation[3])
            self.sample_pose_publishers[layer.name].publish(pose)
