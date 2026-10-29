"""Opt-in selected-LiDAR stream for navigation runs (OMNILRS_SELECTED_NAV); never drives.

The same Generic sensor, mount and native windows as the natural holdout capture, publishing
/astrobot_0/lidar_selected/points (GLIM) and /terrain_shadow/native_chunks (v2 terrain) on
every tick. No lease, capture request, PhysX surface audit or raw archive: those qualify a
fixture, and sim nav runs are scored against GT instead. The run directory keeps only the
evaluator geometry (natural_scene.json, a terrain-only clearance_tile.npz) and the window log.
"""
import json
import os
import sys

import numpy as np

TILE_STEP_M = 0.1
TILE_CASTS_PER_TICK = 2000
TERRAIN_PREFIX = "/Lunaryard/Terrain/"


class SelectedNavStream:
    def __init__(self, controller):
        import rclpy
        import omni.usd
        from omni.physx import get_physx_scene_query_interface
        from pxr import Gf, Usd, UsdGeom
        from .terrain_gt_probe import TerrainGTProbe

        sys.path.insert(0, "/workspace/terrain-tools")
        from archive_guard import GIB, allocate_run
        from selected_lidar_profile import declaration, profile_path, selected_setup

        if os.getenv("OMNILRS_NATURAL_HOLDOUT", "false").lower() == "true":
            raise ValueError("selected nav stream and natural holdout are exclusive")
        if os.getenv("OMNILRS_GT_TF_ENABLED", "true").lower() != "false":
            raise ValueError("navigation requires OMNILRS_GT_TF_ENABLED=false")
        if declaration()["name"] != "selected_optical_v2":
            raise ValueError("v2 terrain requires the corrected optical-origin profile")
        self.controller, self.stage = controller, omni.usd.get_context().get_stage()
        self.query = get_physx_scene_query_interface()
        self.driver = None
        # Scene, tile and window log only (MBs); the 50 GiB default reserve is sized for captures.
        self.run = allocate_run("/archive", limit=4 * GIB, reserve=20 * GIB)
        self.write_scene(Usd, UsdGeom)
        robot = next(iter(controller.robotManager.robots.values()))
        legacy = robot._find_first_descendant_of_type("Lidar")
        mount = legacy.GetParent()
        rotation, translation = mount.GetAttribute("xformOp:rotateXYZ"), mount.GetAttribute("xformOp:translate")
        if (mount.GetName() != "vlp16" or mount.GetParent().GetName() != "base_link" or not rotation
                or not translation or not np.allclose(translation.Get(), [.4, 0, .25], atol=1e-7, rtol=0)):
            raise ValueError("unexpected authored LiDAR mount")
        setup, _ = selected_setup(profile_path())
        rotation.Set(Gf.Vec3f(0., 0., 0.))
        translation.Set(type(translation.Get())(*setup["base_from_lidar_translation_m"]))
        self.node = rclpy.create_node("selected_nav_stream")
        from .selected_lidar_capture import SelectedLidar
        from native_chunk_transport import message
        from sensor_msgs.msg import PointCloud2

        self.selected_lidar = SelectedLidar(self, robot, legacy, mount)
        self.pose_method = TerrainGTProbe._pose
        self.message = message
        self.native_publisher = self.node.create_publisher(PointCloud2, "/terrain_shadow/native_chunks", 16)
        self.sequence = 0
        self.failure = None
        self.tile = self.tile_points()
        self.tile_z = []
        print(f"SELECTED_NAV run={self.run}", flush=True)

    def write_scene(self, Usd, UsdGeom):
        from .gt_map_ros2 import _explicit_rock_instance_root

        roots = sorted({root for prim in self.stage.Traverse()
                        if (root := _explicit_rock_instance_root(str(prim.GetPath()))) is not None})
        bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        inventory = []
        for root in roots:
            box = bounds.ComputeWorldBound(self.stage.GetPrimAtPath(root)).ComputeAlignedBox()
            inventory.append(dict(path=root, minimum=list(map(float, box.GetMin())), maximum=list(map(float, box.GetMax()))))
        terrain = bounds.ComputeWorldBound(self.stage.GetPrimAtPath(TERRAIN_PREFIX.rstrip("/"))).ComputeAlignedBox()
        self.terrain_bounds = np.array([list(terrain.GetMin()), list(terrain.GetMax())], dtype=float)
        with (self.run / "natural_scene.json").open("x") as stream:
            json.dump(dict(seed=int(os.getenv("OMNILRS_ENVIRONMENT_SEED", "42")), objects=inventory,
                           source="USD explicit instance world bounds", note="Evaluator-only geometry (nav run)."),
                      stream, indent=2)

    def tile_points(self):
        lo, hi = self.terrain_bounds[0, :2] + TILE_STEP_M / 2, self.terrain_bounds[1, :2] - TILE_STEP_M / 2
        x, y = np.meshgrid(np.arange(lo[0], hi[0], TILE_STEP_M), np.arange(lo[1], hi[1], TILE_STEP_M))
        return np.stack([x.ravel(), y.ravel()], axis=1)

    def survey(self):
        """Terrain-only floor reference for GT rock heights, a bounded batch per tick."""
        from .terrain_gt_probe import raycast_hits

        if self.tile is None:
            return
        top = float(self.terrain_bounds[1, 2]) + 2.
        depth = top - float(self.terrain_bounds[0, 2]) + 2.
        for xy in self.tile[len(self.tile_z):len(self.tile_z) + TILE_CASTS_PER_TICK]:
            hits = [h for h in raycast_hits(self.query, (float(xy[0]), float(xy[1]), top), (0., 0., -1.), depth)
                    if h[3].startswith(TERRAIN_PREFIX)]
            self.tile_z.append(hits[0][1][2] if hits else float("nan"))
        if len(self.tile_z) == len(self.tile):
            z = np.asarray(self.tile_z)
            keep = np.isfinite(z)
            np.savez_compressed(self.run / "clearance_tile.npz", xy=self.tile[keep], z=z[keep])
            print(f"SELECTED_NAV tile {keep.sum()}/{len(z)} terrain samples", flush=True)
            self.tile = None

    def pre_step(self):
        if self.failure is None:
            try:
                self.selected_lidar.pre_step()
            except Exception as exc:
                self.failure = str(exc)

    def tick(self, stamp):
        if self.failure is not None:
            return
        try:
            self.survey()
            pose = self.pose_method(self)[3]
            self.selected_lidar.tick(stamp, pose)
            latest = self.selected_lidar.latest
            if latest is not None and latest[0] == stamp:
                points, depths, intensities, _, _ = latest[2]
                self.native_publisher.publish(self.message(self.run.name, self.sequence,
                    self.selected_lidar.adapter.stamp_ns(stamp), points, depths, intensities))
                self.sequence += 1
        except Exception as exc:
            # Fail silent-closed: the terrain publisher's native-stale fault holds Nav2.
            self.failure = str(exc)
            with (self.run / "failure.json").open("x") as stream:
                json.dump(dict(error=self.failure, stamp=stamp, chunks=self.sequence), stream)
            print(f"SELECTED_NAV FAILED {self.failure}; {self.run}", flush=True)
