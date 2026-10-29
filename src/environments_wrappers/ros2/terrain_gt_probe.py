"""Opt-in, simulation-thread collision reference capture; never navigation input.

Uses active PhysX colliders, not render AABBs or the approximate /gt/map.
One bounded 10x10m, 2cm tile is captured while stationary. Outputs retain
collider identities and normals; semantic drivable labels are NOT asserted.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


def raycast_hits(query, origin, direction, distance, *, both_sides=False):
    """PhysX can swallow callback exceptions: explicitly propagate them."""
    found, errors = [], []

    def report(hit):
        try:
            found.append((float(hit.distance), tuple(hit.position), tuple(hit.normal),
                          str(hit.collision), str(hit.rigid_body), int(hit.protoIndex)))
            return True
        except Exception as exc:
            errors.append(exc)
            return False

    query.raycast_all(origin, direction, distance, report, both_sides)
    if errors:
        raise RuntimeError(f"raycast callback failed: {errors[0]}") from errors[0]
    return sorted(found, key=lambda hit: hit[0])


def collision_identity(stage, hit):
    # protoIndex is meaningful only for a point instancer. Some native builds
    # leave it undefined on ordinary rigid-body shapes.
    ancestor = stage.GetPrimAtPath(hit[3])
    instanced = False
    while ancestor and not ancestor.IsPseudoRoot():
        instanced |= ancestor.GetTypeName() == "PointInstancer"
        ancestor = ancestor.GetParent()
    return hit[3], hit[4], hit[5] if instanced else 0xFFFFFFFF


class TerrainGTProbe:
    def __init__(self, controller):
        from omni.physx import get_physx_scene_query_interface
        from pxr import UsdGeom
        import omni.usd

        sys.path.insert(0, "/workspace/terrain-tools")
        from archive_guard import allocate_run
        self.run = allocate_run("/archive")
        self.controller = controller
        self.query = get_physx_scene_query_interface()
        self.stage = omni.usd.get_context().get_stage()
        if abs(UsdGeom.GetStageMetersPerUnit(self.stage)-1.) > 1e-9:
            raise ValueError("GT probe requires meter stage units")
        self.started = None
        self.done = False
        self.cursor = 0
        self.size = 500
        self.resolution = .02
        self.heights = np.full((self.size, self.size, 2), np.nan, dtype=np.float32)
        self.normals = np.full((self.size, self.size, 3), np.nan, dtype=np.float32)
        self.ids = np.zeros((self.size, self.size), dtype=np.int32)
        self.hits = np.zeros((self.size, self.size), dtype=np.int16)
        self.visible = np.zeros((self.size, self.size), dtype=bool)
        self.visible_sensor = np.zeros((self.size,self.size),dtype=bool)
        self.objects = {}
        self.pose_checks = []
        self.wall_started = time.monotonic()
        self.calibration = None
        self.driver = None
        self.native_capture = None
        self.drive_enabled = os.getenv("OMNILRS_TERRAIN_DRIVE", "false").lower() == "true"
        self.shadow_enabled = os.getenv("OMNILRS_TERRAIN_SHADOW", "false").lower() == "true"
        if self.shadow_enabled and not self.drive_enabled:
            raise ValueError("shadow capture requires the guarded calibration driver")
        if os.getenv("OMNILRS_TERRAIN_CALIBRATION", "false").lower() == "true":
            from .terrain_calibration import create_scene, CENTER
            self.calibration = create_scene(self.stage)
            robot = next(iter(self.controller.robotManager.robots.values()))
            robot.teleport([float(CENTER[0]),float(CENTER[1]),.35], [0.,0.,0.,1.])
            with (self.run/"calibration_spec.json").open("x") as stream:
                json.dump(self.calibration, stream, indent=2)
        if self.drive_enabled and self.calibration is None:
            raise ValueError("GT drive requires the explicit analytic calibration scene")
        self.mount_pitch=float(os.getenv("OMNILRS_TERRAIN_LIDAR_PITCH_DEG","0"))
        if not np.isfinite(self.mount_pitch) or not 0<=self.mount_pitch<=15:
            raise ValueError("calibration LiDAR pitch must be between 0 and 15 degrees")
        if self.mount_pitch:
            if self.calibration is None:raise ValueError("mount experiment requires explicit calibration scene")
            robot=next(iter(self.controller.robotManager.robots.values()))
            lidar=robot._find_first_descendant_of_type("Lidar")
            mount=lidar.GetParent()
            if mount.GetName()!="vlp16" or mount.GetParent().GetName()!="base_link" or mount.GetAppliedSchemas():
                raise ValueError("unexpected LiDAR mount hierarchy/physics schema")
            rotation=mount.GetAttribute("xformOp:rotateXYZ")
            if not rotation or not np.allclose(rotation.Get(),[0,0,0],atol=1e-9):
                raise ValueError("refusing to compound existing LiDAR mount rotation")
            from pxr import Gf
            rotation.Set(Gf.Vec3f(0.,self.mount_pitch,0.))
        from .terrain_sensor_audit import TerrainSensorAudit, graph_contract
        self.sensor_audit = TerrainSensorAudit(self.run,require_tf=True)
        robot = next(iter(self.controller.robotManager.robots.values()))
        with (self.run/"graph_contract.json").open("x") as stream:
            json.dump(graph_contract(self.stage, robot.robot_path), stream, indent=2)
        print(f"TERRAIN_GT_PROBE run={self.run}", flush=True)

    def _pose(self):
        from pxr import UsdGeom, Usd
        robot = next(iter(self.controller.robotManager.robots.values()))
        position, quaternion = robot.get_pose()
        root_path = robot.dc.get_rigid_body_path(robot.root_body_id)
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        selected = getattr(self, "selected_lidar", None)
        base = selected.base_prim if selected is not None else robot._find_descendants_by_name(["base_link"])["base_link"]
        lidar = selected.prim if selected is not None else robot._find_first_descendant_of_type("Lidar")
        if not lidar or str(base.GetPath()) != root_path:
            raise ValueError("GT articulation root is not base_link, or PhysX LiDAR missing")
        world_base = np.asarray(cache.GetLocalToWorldTransform(base), dtype=float).T
        world_lidar = np.asarray(cache.GetLocalToWorldTransform(lidar), dtype=float).T
        # USD matrices use row vectors; exported matrices use column vectors.
        xyz = np.asarray(position, dtype=float)
        q = np.asarray(quaternion, dtype=float)
        if not all(np.isfinite(v).all() for v in (xyz,q,world_base,world_lidar)) or abs(np.linalg.norm(q)-1.)>1e-5:
            raise ValueError("nonfinite/invalid physics or USD pose")
        from pxr import Gf
        rotation = np.asarray(Gf.Matrix3d(Gf.Quatd(float(q[3]), Gf.Vec3d(*q[:3]))), dtype=float).T
        translation_error = float(np.linalg.norm(world_base[:3,3]-xyz))
        rotation_error = float(np.linalg.norm(world_base[:3,:3]-rotation))
        if translation_error > .005 or rotation_error > .005:
            raise ValueError("physics root pose and USD base_link disagree")
        return robot, xyz, q, dict(root_body_path=root_path, lidar_path=str(lidar.GetPath()),
            world_from_base=world_base.tolist(), base_from_lidar=(np.linalg.inv(world_base)@world_lidar).tolist(),
            physics_usd_translation_error_m=translation_error, physics_usd_rotation_matrix_error=rotation_error,
            sensor_type=lidar.GetTypeName(), sensor_attributes=(selected.sensor_attributes if selected is not None else
                {a.GetName(): str(a.Get()) for a in lidar.GetAttributes()}))

    def tick(self, stamp):
        if self.done:
            return
        try:
            if self.driver is not None:
                self._drive_tick(stamp)
            else:
                self._tick(stamp)
        except Exception as exc:
            self.done = True
            # Disk failure must not prevent stop/cleanup or hide the original
            # failure. Never redirect the capture onto the root filesystem.
            print(f"TERRAIN_GT_PROBE FAILED {exc}; archive target {self.run}", flush=True)
            if self.driver is not None:
                self.driver.failure = str(exc)
                try:
                    self.driver.close()
                except Exception as close_error:
                    print(f"TERRAIN_GT_PROBE cleanup/log error: {close_error}",flush=True)
            if self.native_capture is not None:
                try:
                    self.native_capture.close()
                except Exception as native_error:
                    print(f"TERRAIN_GT_PROBE native audit close error: {native_error}",flush=True)
            try:
                with (self.run/"failure.json").open("x") as stream:
                    json.dump(dict(type=type(exc).__name__, error=str(exc), stamp=stamp, rays_completed=self.cursor), stream, indent=2)
            except OSError as archive_error:
                print(f"TERRAIN_GT_PROBE unable to archive failure: {archive_error}",flush=True)

    def _drive_tick(self, stamp):
        from archive_guard import check_run
        from .terrain_sensor_audit import TerrainSensorAudit
        robot,xyz,q,pose = self._pose()
        self.driver.record_step(stamp,xyz,q)
        if not self.driver.shadow_ready(stamp):
            self.driver.hold()
            return
        if self.shadow_enabled and not self.shadow_started:
            from .terrain_native_capture import NativeTerrainCapture
            self.native_capture = NativeTerrainCapture(self.run, pose["lidar_path"])
            self.drive_started = stamp
            self.shadow_started = True
            with (self.run / "shadow_link.json").open("x") as stream:
                json.dump(dict(request=self.shadow_request, first_grant=self.driver.shadow_lease.first_grant), stream, indent=2)
        if self.native_capture is not None:
            self.native_capture.observe(stamp,pose,lambda *args: raycast_hits(self.query,*args,both_sides=True))
        if stamp-self.drive_started>180.:
            raise ValueError("overall calibration drive timeout")
        if self.drive_phase == "moving":
            if self.driver.move(stamp,xyz,q,self.targets[self.target_index]):
                self.drive_phase = "settling"
                self.settle_started = stamp
            return
        self.driver.hold()
        if stamp-self.settle_started<.4:
            return
        if self.view_audit is None:
            self.view_path = self.run/f"view_{self.target_index:02}"
            self.view_path.mkdir(exist_ok=False)
            self.view_start_xyz = xyz.copy()
            self.view_start_q = q.copy()
            self.view_audit = TerrainSensorAudit(self.view_path,frame_limit=2,require_tf=True,
                                                  native_source=self.native_capture,
                                                  ray_budget=100 if self.shadow_enabled else None)
        if np.linalg.norm(xyz-self.view_start_xyz)>.002 or abs(float(q@self.view_start_q))<.9999995:
            raise ValueError("robot did not settle before viewpoint capture")
        self.view_audit.observe(stamp,pose,lambda *args: raycast_hits(self.query,*args,both_sides=True))
        if not self.view_audit.done:
            if stamp-self.settle_started>3.:
                raise ValueError("settled viewpoint acquisition timed out")
            return
        audit = json.loads((self.view_path/"sensor_audit.json").read_text())
        if not audit["passed"]:
            raise ValueError("settled viewpoint failed sensor/surface alignment")
        self.views.append(dict(path=self.view_path.name,target=self.targets[self.target_index],
                               actual_xyz=list(map(float,xyz)),stamp=stamp))
        check_run(self.run)
        self.target_index += 1
        self.view_audit = None
        print(f"TERRAIN_GT_DRIVE view={self.target_index}/{len(self.targets)} x={xyz[0]:.3f}",flush=True)
        if self.target_index == len(self.targets):
            self.driver.hold()
            if self.shadow_enabled:
                # Two-phase completion: withdraw motion before potentially
                # expensive archive finalization or removing the ROS publisher.
                from .terrain_calibration_drive import publish_completion_marker
                publish_completion_marker(self.run / "acquisition_finished.json",
                    dict(schema=1,capture=self.run.name,stamp=stamp,views=len(self.views),commands_stopped=True))
            self.driver.close()
            if self.native_capture is not None:
                self.native_capture.close()
            result = dict(views=self.views,complete=True,mode="physical_stop_scan_translation_and_return",
                          prescribed_pattern=self.drive_pattern, prescribed_targets=self.targets,
                          continuously_moving_scans_scored=False)
            if self.shadow_enabled:
                publish_completion_marker(self.run / "multiview.json", result)
            else:
                with (self.run/"multiview.json").open("x") as stream:
                    json.dump(result,stream,indent=2)
            self.done = True
            print(f"TERRAIN_GT_DRIVE COMPLETE {self.run}",flush=True)
        else:
            self.drive_phase = "moving"

    def _tick(self, stamp):
        from archive_guard import check_run
        if self.started is None:
            self.started = stamp
        if stamp-self.started < (8. if self.calibration else 3.):
            return
        robot, xyz, quaternion, pose = self._pose()
        if not self.pose_checks:
            self.initial_xyz, self.initial_q = xyz.copy(), quaternion.copy()
            self.origin = xyz[:2]-5.
            self.robot_root = robot.robot_path
            self.reference_lidar = np.asarray(pose["world_from_base"])@np.asarray(pose["base_from_lidar"])
            manifest = dict(schema=1, source="active_PhysX_raycast_all", seed=os.getenv("OMNILRS_ENVIRONMENT_SEED"),
                frame="gt_map", resolution=self.resolution, origin=self.origin.tolist(), width=self.size, height=self.size,
                first_stamp=stamp, sensor=pose, root_layer=self.stage.GetRootLayer().identifier,
                experimental_lidar_pitch_deg=self.mount_pitch,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                semantic_labels_validated=False, calibration_geometry_validated=False,
                note="Unsorted all-hits sorted explicitly; top and lowest upward-facing surfaces retained. Robot identity retained for explicit exclusion.")
            paths = [Path(__file__), Path(__file__).with_name("terrain_calibration.py"),
                     Path(__file__).with_name("terrain_sensor_audit.py"),
                     Path(__file__).with_name("terrain_native_capture.py"),
                     Path(__file__).with_name("simulation_manager_ros2.py"),
                     Path(__file__).parents[2]/"physics/physics_scene.py",
                     Path(__file__).parents[2]/"configurations/simulation_timing.py",
                     Path(__file__).with_name("terrain_calibration_drive.py"), Path(robot.usd_path)]
            manifest["source_asset_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
            manifest["visibility"] = dict(range_m=5., min_range_m=.5, ray_tolerance_m=.005,
                vertical_fov_deg=float(pose["sensor_attributes"]["verticalFov"]),
                definition="continuous-FOV line of sight to top surface; independent of measured ray hits and mapper known cells",
                visible="legacy frontface-only conservative target, retained for comparison",
                visible_sensor="both mesh sides, matching Isaac PhysX LiDAR")
            with (self.run/"manifest.json").open("x") as stream:
                json.dump(manifest, stream, indent=2)
            if self.calibration:
                from .terrain_calibration import validate_queries
                checks = validate_queries(self.calibration, lambda *args: raycast_hits(self.query,*args))
                with (self.run/"calibration_checks.json").open("x") as stream:
                    json.dump(checks, stream, indent=2)
                if not checks["passed"]:
                    raise ValueError("analytic collision calibration failed")
        if np.linalg.norm(xyz-self.initial_xyz) > .005 or abs(float(quaternion@self.initial_q)) < .99999:
            raise ValueError("robot moved during static geometry capture")
        self.pose_checks.append(dict(stamp=stamp, **{k:v for k,v in pose.items() if k.startswith("physics_")}))
        self.sensor_audit.observe(stamp, pose, lambda *args: raycast_hits(self.query,*args,both_sides=True))
        # Bounded per-step work. No stage queries or physics access in ROS callbacks.
        end = min(self.cursor+1000, self.size*self.size)
        for i in range(self.cursor, end):
            row, col = divmod(i, self.size)
            x, y = self.origin + (np.array([col,row])+.5)*self.resolution
            found = raycast_hits(self.query, (float(x),float(y),float(xyz[2]+5.)), (0.,0.,-1.), 10.)
            if len(found) > 32767:
                raise ValueError("ray hit count capacity exceeded")
            self.hits[row,col] = len(found)
            if not found:
                continue
            top = found[0]
            key = collision_identity(self.stage, top)
            if key not in self.objects:
                self.objects[key] = len(self.objects)+1
            self.ids[row,col] = self.objects[key]
            self.heights[row,col,0] = top[1][2]
            self.normals[row,col] = top[2]
            delta = np.asarray(top[1])-self.reference_lidar[:3,3]
            distance = np.linalg.norm(delta)
            local = self.reference_lidar[:3,:3].T@delta
            elevation = np.degrees(np.arctan2(local[2],np.hypot(local[0],local[1])))
            if (.5 < distance <= 5. and abs(elevation)<=float(pose["sensor_attributes"]["verticalFov"])/2
                    and not top[3].startswith(self.robot_root+"/")):
                direction = delta/distance
                sight = raycast_hits(self.query, tuple(self.reference_lidar[:3,3]+.5*direction),
                                     tuple(direction), float(distance-.5+.01))
                self.visible[row,col] = bool(sight and abs(sight[0][0]+.5-distance)<=.005)
                sensor_sight = raycast_hits(self.query,tuple(self.reference_lidar[:3,3]+.5*direction),
                                           tuple(direction),float(distance-.5+.01),both_sides=True)
                self.visible_sensor[row,col] = bool(sensor_sight and abs(sensor_sight[0][0]+.5-distance)<=.005)
            upward = [h[1][2] for h in found if h[2][2] > .5 and not h[3].startswith(self.robot_root+"/")]
            if upward:
                self.heights[row,col,1] = min(upward)
        self.cursor = end
        check_run(self.run)
        if self.cursor == self.size*self.size and not self.sensor_audit.done:
            if stamp-self.started > 60.:
                raise ValueError("sensor audit timed out")
            return
        if self.cursor == self.size*self.size:
            np.savez_compressed(self.run/"collision_tile.npz", heights=self.heights, normals=self.normals,
                                object_ids=self.ids, hit_counts=self.hits, visible=self.visible,
                                visible_sensor=self.visible_sensor,
                                origin=self.origin, resolution=self.resolution)
            with (self.run/"summary.json").open("x") as stream:
                json.dump(dict(rays=self.cursor, cells_with_hits=int((self.hits>0).sum()),
                    objects=[dict(id=v, collision=k[0], rigid_body=k[1], proto_index=k[2],
                                  robot=k[0].startswith(self.robot_root+"/")) for k,v in self.objects.items()],
                    pose_checks=self.pose_checks, wall_seconds=time.monotonic()-self.wall_started,
                    note="Raw collision reference only; do not use as a drivable map or movement clearance certificate."), stream, indent=2)
            print(f"TERRAIN_GT_PROBE COMPLETE {self.run}", flush=True)
            if self.drive_enabled:
                if not json.loads((self.run/"sensor_audit.json").read_text())["passed"]:
                    raise ValueError("stationary sensor calibration required before driving")
                from .terrain_calibration_drive import CalibrationDrive, prescribed_targets, validate_acquisition
                self.drive_pattern = os.getenv("OMNILRS_TERRAIN_DRIVE_PATTERN","straight")
                native_enabled=os.getenv("OMNILRS_TERRAIN_NATIVE_CAPTURE","false").lower()=="true"
                validate_acquisition(self.drive_pattern,native_enabled)
                self.shadow_request = None
                self.shadow_started = False
                if self.shadow_enabled:
                    if not native_enabled or self.drive_pattern != "wide_sweep15_return" or self.mount_pitch != 5.:
                        raise ValueError("shadow baseline requires native wide_sweep15_return at 5 degrees")
                    from pxr import UsdGeom, Usd
                    imu = robot._find_descendants_by_name(["Imu_Sensor"])["Imu_Sensor"]
                    if not imu:
                        raise ValueError("shadow acquisition requires USD IMU geometry")
                    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
                    world_imu = np.asarray(cache.GetLocalToWorldTransform(imu), dtype=float).T
                    self.shadow_request = dict(schema=1, capture=self.run.name, pattern=self.drive_pattern,
                        pitch_deg=self.mount_pitch, home_xyz=[-20., -20., 0.],
                        base_from_lidar=pose["base_from_lidar"],
                        base_from_imu=(np.linalg.inv(pose["world_from_base"]) @ world_imu).tolist())
                self.driver = CalibrationDrive(self.calibration["center"],self.query,
                    lambda *args: raycast_hits(self.query,*args),self.run,self.drive_pattern,
                    shadow_request=self.shadow_request)
                if native_enabled and not self.shadow_enabled:
                    from .terrain_native_capture import NativeTerrainCapture
                    self.native_capture=NativeTerrainCapture(self.run,pose["lidar_path"])
                self.targets = prescribed_targets(self.drive_pattern)
                self.target_index = 0
                self.drive_phase = "moving"
                self.drive_started = stamp
                self.view_audit = None
                self.views = []
            else:
                self.done = True
