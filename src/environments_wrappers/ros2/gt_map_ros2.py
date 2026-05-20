__author__ = "OpenAI"

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence
import math

import numpy as np
import omni
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from pxr import Gf, Usd, UsdGeom
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import TransformBroadcaster


@dataclass(frozen=True)
class GTTerrainSnapshot:
    dem: Optional[np.ndarray]
    mask: Optional[np.ndarray]
    resolution: float
    origin_x: float
    origin_y: float
    stage_roots: tuple[str, ...]
    excluded_paths: tuple[str, ...]
    x_axis: int
    y_axis: int
    flip_x: bool
    flip_y: bool


def _normalize_robot_name(robot_name: Optional[str]) -> Optional[str]:
    if robot_name is None or robot_name == "":
        return None
    return robot_name if robot_name.startswith("/") else f"/{robot_name}"


def _path_matches(path: str, prefixes: Sequence[str]) -> bool:
    return any(path == prefix or path.startswith(f"{prefix}/") for prefix in prefixes)


def _quat_to_yaw(quaternion_xyzw: Sequence[float]) -> float:
    x, y, z, w = quaternion_xyzw
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    half_yaw = yaw * 0.5
    return (0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw))


def _normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _gf_quat_to_yaw(quaternion) -> float:
    if quaternion is None:
        return 0.0
    if hasattr(quaternion, "GetReal") and hasattr(quaternion, "GetImaginary"):
        w = float(quaternion.GetReal())
        imag = quaternion.GetImaginary()
        x = float(imag[0])
        y = float(imag[1])
        z = float(imag[2])
    else:
        return 0.0
    return _quat_to_yaw((x, y, z, w))


class ROS_GTMapPublisher(Node):
    """
    Publishes simulator-owned ground-truth map products under /gt.
    """

    def __init__(
        self,
        environment_controller,
        robot_name: Optional[str] = None,
        map_resolution: float = 0.03,
        map_publish_rate_hz: float = 1.0,
        tf_publish_rate_hz: float = 15.0,
        odom_topic: str = "/astrobot_0/odom",
        map_topic: str = "/gt/map",
        gt_pose_topic: str = "/gt/base_link_pose",
        map_frame: str = "gt_map",
        odom_frame: str = "odom",
        publish_tf: bool = True,
        obstacle_z_min: float = 0.05,
        obstacle_z_max: float = 1.5,
        max_slope_deg: float = 30.0,
        max_mesh_footprint_ratio: float = 0.35,
        mesh_obstacle_padding_m: float = 0.03,
        point_instancer_scale_factor: float = 1.0,
        point_instancer_padding_m: float = 0.03,
    ) -> None:
        super().__init__("omnilrs_gt_map")

        if not self.has_parameter("use_sim_time"):
            self.declare_parameter("use_sim_time", True)
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

        self.stage = omni.usd.get_context().get_stage()
        self.environment_controller = environment_controller
        self.robot_name = _normalize_robot_name(robot_name)
        self.map_resolution = float(map_resolution)
        self.map_frame = map_frame
        self.odom_frame = odom_frame
        self.publish_tf = bool(publish_tf)
        self.obstacle_z_min = float(obstacle_z_min)
        self.obstacle_z_max = float(obstacle_z_max)
        self.max_slope = math.tan(math.radians(float(max_slope_deg)))
        self.max_mesh_footprint_ratio = float(max_mesh_footprint_ratio)
        self.mesh_obstacle_padding_m = max(0.0, float(mesh_obstacle_padding_m))
        self.point_instancer_scale_factor = max(0.1, float(point_instancer_scale_factor))
        self.point_instancer_padding_m = max(0.0, float(point_instancer_padding_m))
        self.latest_odom: Optional[Odometry] = None

        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.map_pub = self.create_publisher(OccupancyGrid, map_topic, map_qos)
        self.gt_pose_pub = self.create_publisher(PoseStamped, gt_pose_topic, 10)
        self.tf_broadcaster = TransformBroadcaster(self) if self.publish_tf else None
        self.create_subscription(Odometry, odom_topic, self._odom_callback, 10)

        self.create_timer(1.0 / max(float(map_publish_rate_hz), 0.1), self._publish_map)
        self.create_timer(1.0 / max(float(tf_publish_rate_hz), 1.0), self._publish_gt_pose)
        if self.publish_tf:
            self.create_timer(1.0 / max(float(tf_publish_rate_hz), 1.0), self._publish_tf)

        self.get_logger().info(
            f"publishing GT map from simulator state on {map_topic}; "
            f"GT pose on {gt_pose_topic}; GT TF {map_frame}->{odom_frame} enabled={self.publish_tf}"
        )

    def _odom_callback(self, msg: Odometry) -> None:
        self.latest_odom = msg

    def _snapshot(self) -> GTTerrainSnapshot:
        source = self.environment_controller.get_gt_terrain_data()
        dem = source["dem"]
        mask = source["mask"]
        return GTTerrainSnapshot(
            dem=np.array(dem, copy=True) if dem is not None else None,
            mask=np.array(mask, copy=True) if mask is not None else None,
            resolution=float(source["resolution"]),
            origin_x=float(source["origin"][0]),
            origin_y=float(source["origin"][1]),
            stage_roots=tuple(source.get("stage_roots", tuple())),
            excluded_paths=tuple(source.get("excluded_paths", tuple())),
            x_axis=int(source.get("x_axis", 1)),
            y_axis=int(source.get("y_axis", 0)),
            flip_x=bool(source.get("flip_x", False)),
            flip_y=bool(source.get("flip_y", False)),
        )

    def _publish_map(self) -> None:
        snapshot = self._snapshot()
        if snapshot.dem is None:
            return

        map_origin_x, map_origin_y, map_max_x, map_max_y = self._compute_map_bounds(snapshot)
        width_cells = max(1, int(math.ceil((map_max_x - map_origin_x) / self.map_resolution)))
        height_cells = max(1, int(math.ceil((map_max_y - map_origin_y) / self.map_resolution)))
        occupancy = np.zeros((height_cells, width_cells), dtype=bool)

        terrain_occupancy = self._build_terrain_occupancy(snapshot)
        self._project_source_occupancy(terrain_occupancy, snapshot, occupancy, map_origin_x, map_origin_y)
        self._mark_boundary(occupancy)
        self._rasterize_stage_obstacles(snapshot, occupancy, map_origin_x, map_origin_y)

        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        msg.info.map_load_time = msg.header.stamp
        msg.info.resolution = self.map_resolution
        msg.info.width = width_cells
        msg.info.height = height_cells
        msg.info.origin.position.x = map_origin_x
        msg.info.origin.position.y = map_origin_y
        msg.info.origin.orientation.w = 1.0
        msg.data = (occupancy.astype(np.int8) * 100).reshape(-1).tolist()
        self.map_pub.publish(msg)

    def _publish_tf(self) -> None:
        if self.latest_odom is None or self.tf_broadcaster is None:
            return

        world_position, world_orientation = self.environment_controller.get_gt_robot_world_pose(self.robot_name)
        odom_pose = self.latest_odom.pose.pose

        world_yaw = _quat_to_yaw(world_orientation)
        odom_yaw = _quat_to_yaw(
            (
                odom_pose.orientation.x,
                odom_pose.orientation.y,
                odom_pose.orientation.z,
                odom_pose.orientation.w,
            )
        )
        map_to_odom_yaw = _normalize_angle(world_yaw - odom_yaw)

        cos_yaw = math.cos(map_to_odom_yaw)
        sin_yaw = math.sin(map_to_odom_yaw)
        odom_xy = np.array([odom_pose.position.x, odom_pose.position.y], dtype=float)
        rotated_odom = np.array(
            [
                cos_yaw * odom_xy[0] - sin_yaw * odom_xy[1],
                sin_yaw * odom_xy[0] + cos_yaw * odom_xy[1],
            ],
            dtype=float,
        )
        translation = np.asarray(world_position[:2], dtype=float) - rotated_odom
        qx, qy, qz, qw = _yaw_to_quaternion(map_to_odom_yaw)

        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = self.map_frame
        transform.child_frame_id = self.odom_frame
        transform.transform.translation.x = float(translation[0])
        transform.transform.translation.y = float(translation[1])
        transform.transform.translation.z = 0.0
        transform.transform.rotation.x = qx
        transform.transform.rotation.y = qy
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(transform)

    def _publish_gt_pose(self) -> None:
        world_position, world_orientation = self.environment_controller.get_gt_robot_world_pose(self.robot_name)

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        msg.pose.position.x = float(world_position[0])
        msg.pose.position.y = float(world_position[1])
        msg.pose.position.z = float(world_position[2]) if len(world_position) > 2 else 0.0
        msg.pose.orientation.x = float(world_orientation[0])
        msg.pose.orientation.y = float(world_orientation[1])
        msg.pose.orientation.z = float(world_orientation[2])
        msg.pose.orientation.w = float(world_orientation[3])
        self.gt_pose_pub.publish(msg)

    def _build_terrain_occupancy(self, snapshot: GTTerrainSnapshot) -> np.ndarray:
        occupancy = np.zeros_like(snapshot.dem, dtype=bool)
        if snapshot.mask is not None:
            occupancy |= snapshot.mask <= 0.5

        if snapshot.dem.size > 1:
            gradients = np.gradient(snapshot.dem, snapshot.resolution, snapshot.resolution)
            slope = np.hypot(gradients[0], gradients[1])
            occupancy |= slope > self.max_slope

        return occupancy

    def _terrain_bounds(self, snapshot: GTTerrainSnapshot) -> tuple[float, float, float, float]:
        width_m = snapshot.dem.shape[snapshot.x_axis] * snapshot.resolution
        height_m = snapshot.dem.shape[snapshot.y_axis] * snapshot.resolution
        return (
            snapshot.origin_x,
            snapshot.origin_x + width_m,
            snapshot.origin_y,
            snapshot.origin_y + height_m,
        )

    def _expand_bounds(
        self,
        bounds: Optional[tuple[float, float, float, float]],
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
    ) -> tuple[float, float, float, float]:
        if bounds is None:
            return (min_x, max_x, min_y, max_y)
        return (
            min(bounds[0], min_x),
            max(bounds[1], max_x),
            min(bounds[2], min_y),
            max(bounds[3], max_y),
        )

    def _compute_map_bounds(self, snapshot: GTTerrainSnapshot) -> tuple[float, float, float, float]:
        bounds = self._terrain_bounds(snapshot)
        stage_bounds = self._compute_stage_bounds(snapshot, terrain_bounds=bounds)
        if stage_bounds is not None:
            bounds = self._expand_bounds(bounds, *stage_bounds)

        min_x = math.floor(bounds[0] / self.map_resolution) * self.map_resolution
        max_x = math.ceil(bounds[1] / self.map_resolution) * self.map_resolution
        min_y = math.floor(bounds[2] / self.map_resolution) * self.map_resolution
        max_y = math.ceil(bounds[3] / self.map_resolution) * self.map_resolution
        return min_x, min_y, max_x, max_y

    def _compute_stage_bounds(
        self,
        snapshot: GTTerrainSnapshot,
        terrain_bounds: tuple[float, float, float, float],
    ) -> Optional[tuple[float, float, float, float]]:
        roots = snapshot.stage_roots or ("/",)
        excluded = snapshot.excluded_paths + ("/Robots",)
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        )

        point_instancers = []
        instancer_paths: list[str] = []
        prototype_paths: list[str] = []
        bounds: Optional[tuple[float, float, float, float]] = None
        terrain_area = max(1e-6, (terrain_bounds[1] - terrain_bounds[0]) * (terrain_bounds[3] - terrain_bounds[2]))
        max_bounds_area = terrain_area * 4.0

        for prim in self._iter_prims(roots):
            path = prim.GetPath().pathString
            if _path_matches(path, excluded):
                continue
            if not self._is_visible(prim):
                continue
            if prim.IsA(UsdGeom.PointInstancer):
                point_instancers.append(UsdGeom.PointInstancer(prim))
                instancer_paths.append(path)
                prototype_paths.extend(str(target) for target in UsdGeom.PointInstancer(prim).GetPrototypesRel().GetTargets())
                continue

            world_bound = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
            min_point = world_bound.GetMin()
            max_point = world_bound.GetMax()
            if max_point[2] < self.obstacle_z_min or min_point[2] > self.obstacle_z_max:
                continue

            footprint_area = max(0.0, float(max_point[0] - min_point[0])) * max(0.0, float(max_point[1] - min_point[1]))
            if footprint_area <= 0.0 or footprint_area > max_bounds_area:
                continue
            bounds = self._expand_bounds(bounds, float(min_point[0]), float(max_point[0]), float(min_point[1]), float(max_point[1]))

        for instancer in point_instancers:
            positions = instancer.GetPositionsAttr().Get()
            if positions is None or len(positions) == 0:
                continue

            proto_indices = instancer.GetProtoIndicesAttr().Get()
            scales = instancer.GetScalesAttr().Get()
            if scales is None or len(scales) == 0:
                scales = [Gf.Vec3f(1.0, 1.0, 1.0)] * len(positions)

            prototype_extents: list[tuple[float, float]] = []
            prototype_z_ranges: list[tuple[float, float]] = []
            for target in instancer.GetPrototypesRel().GetTargets():
                prototype_bound = bbox_cache.ComputeWorldBound(self.stage.GetPrimAtPath(target)).ComputeAlignedBox()
                min_point = prototype_bound.GetMin()
                max_point = prototype_bound.GetMax()
                radius_x = max(0.5 * float(max_point[0] - min_point[0]), self.map_resolution * 0.5)
                radius_y = max(0.5 * float(max_point[1] - min_point[1]), self.map_resolution * 0.5)
                prototype_extents.append((radius_x, radius_y))
                prototype_z_ranges.append((float(min_point[2]), float(max_point[2])))

            world_transform = UsdGeom.Xformable(instancer.GetPrim()).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            for position, scale, proto_index in zip(positions, scales, proto_indices):
                world_position = world_transform.Transform(Gf.Vec3d(position[0], position[1], position[2]))
                radius_x, radius_y = prototype_extents[int(proto_index)]
                radius_x *= max(float(scale[0]), 1e-6)
                radius_y *= max(float(scale[1]), 1e-6)
                prototype_z = prototype_z_ranges[int(proto_index)]
                z_low = float(world_position[2]) + prototype_z[0] * float(scale[2])
                z_high = float(world_position[2]) + prototype_z[1] * float(scale[2])
                if z_high < self.obstacle_z_min or z_low > self.obstacle_z_max:
                    continue
                padding = self.point_instancer_padding_m
                bounds = self._expand_bounds(
                    bounds,
                    float(world_position[0]) - radius_x - padding,
                    float(world_position[0]) + radius_x + padding,
                    float(world_position[1]) - radius_y - padding,
                    float(world_position[1]) + radius_y + padding,
                )

        return bounds

    def _project_source_occupancy(
        self,
        source_occupancy: np.ndarray,
        snapshot: GTTerrainSnapshot,
        occupancy: np.ndarray,
        map_origin_x: float,
        map_origin_y: float,
    ) -> None:
        occupied_indices = np.nonzero(source_occupancy)
        if len(occupied_indices[0]) == 0:
            return

        source_coords = [np.asarray(axis_values, dtype=np.int64) for axis_values in occupied_indices]
        x_idx = source_coords[snapshot.x_axis]
        y_idx = source_coords[snapshot.y_axis]

        if snapshot.flip_x:
            x_idx = source_occupancy.shape[snapshot.x_axis] - 1 - x_idx
        if snapshot.flip_y:
            y_idx = source_occupancy.shape[snapshot.y_axis] - 1 - y_idx

        source_x = snapshot.origin_x + x_idx * snapshot.resolution
        source_y = snapshot.origin_y + y_idx * snapshot.resolution
        cols = np.clip(((source_x - map_origin_x) / self.map_resolution).astype(np.int64), 0, occupancy.shape[1] - 1)
        rows = np.clip(((source_y - map_origin_y) / self.map_resolution).astype(np.int64), 0, occupancy.shape[0] - 1)
        occupancy[rows, cols] = True

    def _mark_boundary(self, occupancy: np.ndarray) -> None:
        occupancy[0, :] = True
        occupancy[-1, :] = True
        occupancy[:, 0] = True
        occupancy[:, -1] = True

    def _rasterize_stage_obstacles(
        self,
        snapshot: GTTerrainSnapshot,
        occupancy: np.ndarray,
        map_origin_x: float,
        map_origin_y: float,
    ) -> None:
        roots = snapshot.stage_roots or ("/",)
        excluded = snapshot.excluded_paths + ("/Robots",)
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        )

        point_instancers = []
        instancer_paths: list[str] = []
        prototype_paths: list[str] = []
        for prim in self._iter_prims(roots):
            path = prim.GetPath().pathString
            if _path_matches(path, excluded):
                continue
            if not self._is_visible(prim):
                continue
            if prim.IsA(UsdGeom.PointInstancer):
                point_instancers.append(UsdGeom.PointInstancer(prim))
                instancer_paths.append(path)
                prototype_paths.extend(str(target) for target in UsdGeom.PointInstancer(prim).GetPrototypesRel().GetTargets())

        map_width_m = occupancy.shape[1] * self.map_resolution
        map_height_m = occupancy.shape[0] * self.map_resolution
        map_area = map_width_m * map_height_m

        geometry_types = (
            UsdGeom.Mesh,
            UsdGeom.Cube,
            UsdGeom.Cylinder,
            UsdGeom.Cone,
            UsdGeom.Capsule,
            UsdGeom.Sphere,
        )
        for prim in self._iter_prims(roots):
            path = prim.GetPath().pathString
            if _path_matches(path, excluded):
                continue
            if not self._is_visible(prim):
                continue
            if _path_matches(path, instancer_paths) or _path_matches(path, prototype_paths):
                continue
            if not any(prim.IsA(geom_type) for geom_type in geometry_types):
                continue

            world_bound = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
            min_point = world_bound.GetMin()
            max_point = world_bound.GetMax()
            footprint_area = max(0.0, max_point[0] - min_point[0]) * max(0.0, max_point[1] - min_point[1])
            if footprint_area > self.max_mesh_footprint_ratio * map_area:
                continue
            self._rasterize_aabb(min_point, max_point, occupancy, map_origin_x, map_origin_y)

        for instancer in point_instancers:
            self._rasterize_point_instancer(instancer, bbox_cache, occupancy, map_origin_x, map_origin_y)

    def _iter_prims(self, roots: Sequence[str]) -> Iterable[Usd.Prim]:
        seen_paths: set[str] = set()
        for root in roots:
            prim = self.stage.GetPrimAtPath(root)
            if not prim:
                continue
            for child in Usd.PrimRange(prim):
                path = child.GetPath().pathString
                if path in seen_paths:
                    continue
                seen_paths.add(path)
                yield child

    @staticmethod
    def _is_visible(prim: Usd.Prim) -> bool:
        imageable = UsdGeom.Imageable(prim)
        if not imageable:
            return True
        return imageable.ComputeVisibility(Usd.TimeCode.Default()) != UsdGeom.Tokens.invisible

    def _rasterize_aabb(
        self,
        min_point: Gf.Vec3d,
        max_point: Gf.Vec3d,
        occupancy: np.ndarray,
        map_origin_x: float,
        map_origin_y: float,
    ) -> None:
        if max_point[2] < self.obstacle_z_min or min_point[2] > self.obstacle_z_max:
            return

        padding = self.mesh_obstacle_padding_m
        col_min = int(math.floor((float(min_point[0]) - padding - map_origin_x) / self.map_resolution))
        col_max = int(math.floor((float(max_point[0]) + padding - map_origin_x) / self.map_resolution))
        row_min = int(math.floor((float(min_point[1]) - padding - map_origin_y) / self.map_resolution))
        row_max = int(math.floor((float(max_point[1]) + padding - map_origin_y) / self.map_resolution))

        col_min = max(col_min, 0)
        row_min = max(row_min, 0)
        col_max = min(col_max, occupancy.shape[1] - 1)
        row_max = min(row_max, occupancy.shape[0] - 1)
        if col_min > col_max or row_min > row_max:
            return
        occupancy[row_min : row_max + 1, col_min : col_max + 1] = True

    def _rasterize_point_instancer(
        self,
        instancer: UsdGeom.PointInstancer,
        bbox_cache: UsdGeom.BBoxCache,
        occupancy: np.ndarray,
        map_origin_x: float,
        map_origin_y: float,
    ) -> None:
        positions = instancer.GetPositionsAttr().Get()
        if positions is None or len(positions) == 0:
            return

        proto_indices = instancer.GetProtoIndicesAttr().Get()
        scales = instancer.GetScalesAttr().Get()
        orientations = instancer.GetOrientationsAttr().Get()
        if scales is None or len(scales) == 0:
            scales = [Gf.Vec3f(1.0, 1.0, 1.0)] * len(positions)
        if orientations is None or len(orientations) == 0:
            orientations = [None] * len(positions)

        prototype_extents: list[tuple[float, float]] = []
        prototype_z_ranges: list[tuple[float, float]] = []
        for target in instancer.GetPrototypesRel().GetTargets():
            prototype_bound = bbox_cache.ComputeWorldBound(self.stage.GetPrimAtPath(target)).ComputeAlignedBox()
            min_point = prototype_bound.GetMin()
            max_point = prototype_bound.GetMax()
            radius_x = max(0.5 * float(max_point[0] - min_point[0]), self.map_resolution * 0.5)
            radius_y = max(0.5 * float(max_point[1] - min_point[1]), self.map_resolution * 0.5)
            prototype_extents.append((radius_x, radius_y))
            prototype_z_ranges.append((float(min_point[2]), float(max_point[2])))

        world_transform = UsdGeom.Xformable(instancer.GetPrim()).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        for position, scale, proto_index, orientation in zip(positions, scales, proto_indices, orientations):
            world_position = world_transform.Transform(Gf.Vec3d(position[0], position[1], position[2]))
            radius_x, radius_y = prototype_extents[int(proto_index)]
            radius_x *= self.point_instancer_scale_factor * max(float(scale[0]), 1e-6)
            radius_y *= self.point_instancer_scale_factor * max(float(scale[1]), 1e-6)
            radius_x += self.point_instancer_padding_m
            radius_y += self.point_instancer_padding_m
            prototype_z = prototype_z_ranges[int(proto_index)]
            z_low = float(world_position[2]) + prototype_z[0] * float(scale[2])
            z_high = float(world_position[2]) + prototype_z[1] * float(scale[2])
            if z_high < self.obstacle_z_min or z_low > self.obstacle_z_max:
                continue
            self._rasterize_ellipse(
                float(world_position[0]),
                float(world_position[1]),
                radius_x,
                radius_y,
                _gf_quat_to_yaw(orientation),
                occupancy,
                map_origin_x,
                map_origin_y,
            )

    def _rasterize_ellipse(
        self,
        center_x: float,
        center_y: float,
        radius_x: float,
        radius_y: float,
        yaw: float,
        occupancy: np.ndarray,
        map_origin_x: float,
        map_origin_y: float,
    ) -> None:
        col_min = int(math.floor((center_x - radius_x - map_origin_x) / self.map_resolution))
        col_max = int(math.floor((center_x + radius_x - map_origin_x) / self.map_resolution))
        row_min = int(math.floor((center_y - radius_y - map_origin_y) / self.map_resolution))
        row_max = int(math.floor((center_y + radius_y - map_origin_y) / self.map_resolution))

        col_min = max(col_min, 0)
        row_min = max(row_min, 0)
        col_max = min(col_max, occupancy.shape[1] - 1)
        row_max = min(row_max, occupancy.shape[0] - 1)
        if col_min > col_max or row_min > row_max:
            return

        rows = np.arange(row_min, row_max + 1, dtype=float)
        cols = np.arange(col_min, col_max + 1, dtype=float)
        yy = map_origin_y + (rows[:, None] + 0.5) * self.map_resolution
        xx = map_origin_x + (cols[None, :] + 0.5) * self.map_resolution
        dx = xx - center_x
        dy = yy - center_y
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        local_x = cos_yaw * dx + sin_yaw * dy
        local_y = -sin_yaw * dx + cos_yaw * dy
        norm_x = local_x / max(radius_x, self.map_resolution * 0.5)
        norm_y = local_y / max(radius_y, self.map_resolution * 0.5)
        mask = norm_x * norm_x + norm_y * norm_y <= 1.0
        occupancy[row_min : row_max + 1, col_min : col_max + 1] |= mask
