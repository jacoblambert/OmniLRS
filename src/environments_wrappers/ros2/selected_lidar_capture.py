"""Opt-in single Generic-ray sensor on the physical rover; never drives it.

Estimator payloads contain only sensor-local returns and native acquisition tick
times. PhysX/GT checks and poses go to the separate evaluator archive. The producer
outlives measurement sealing so the recorder can obtain its final pose bracket.
"""

from collections import deque
import hashlib
import json
from pathlib import Path
import os

import numpy as np

from .terrain_native_capture import EncodedFrames, archive_hit_records, surface_audit
from .terrain_sensor_audit import tf_matrix, validate_extrinsic_tf


class SelectedLidar:
    def __init__(self, owner, robot, legacy, mount):
        import omni.kit.commands
        import omni.graph.core as og
        from isaacsim.sensors.physx import _range_sensor
        from isaacsim.core.api import World
        from sensor_msgs.msg import PointCloud2
        from tf2_msgs.msg import TFMessage
        from rclpy.qos import qos_profile_sensor_data
        from generic_lidar_adapter import NativeWindow, message, ASSOCIATION_POLICY
        from selected_lidar_profile import TOPIC, declaration, profile_path
        from pxr import UsdGeom

        self.owner, self.run = owner, owner.run
        # Prim identity is static; pose values are NOT. Re-traversing every robot
        # descendant at 60 Hz creates thousands of temporary USD wrappers per
        # tick and provokes full simulator-heap collections.
        self.base_prim = mount.GetParent()
        self.world = World.instance()
        if any(abs(dt - 1 / 60) > 1e-9 for dt in (self.world.get_physics_dt(), self.world.get_rendering_dt())):
            raise ValueError("selected native profile requires one render per 60 Hz physics tick")
        graph = owner.stage.GetPrimAtPath(robot.robot_path + "/Graphs/ROS_LiDAR")
        if not graph or str(legacy.GetParent().GetPath()) != str(mount.GetPath()):
            raise ValueError("cannot identify legacy sensor/graph for exclusive replacement")
        # A compiled OmniGraph can keep executing after USD SetActive(False).
        # Disconnect its execution edge explicitly before deactivating the old
        # sensor, otherwise it repeatedly queries the removed legacy LiDAR.
        graph_path = str(graph.GetPath())
        source = og.Controller.attribute(graph_path + "/on_playback_tick.outputs:tick")
        target = og.Controller.attribute(graph_path + "/isaac_read_lidar_point_cloud_node.inputs:execIn")
        if not source.is_valid() or not target.is_valid():
            raise ValueError("legacy LiDAR execution attributes missing")
        og.Controller.disconnect(source, target)
        graph.SetActive(False)
        legacy.SetActive(False)
        self.path = str(mount.GetPath()) + "/SelectedGeneric"
        self.adapter = NativeWindow(profile_path(), timestamp_policy="isaac_clock_ns_v2")
        setup = self.adapter.setup
        self.expected_mount = np.eye(4)
        self.expected_mount[:3, 3] = setup["base_from_lidar_translation_m"]
        success, sensor = omni.kit.commands.execute(
            "RangeSensorCreateGeneric",
            path=self.path,
            parent=None,
            min_range=setup.get("raycast_min_range_m", setup["min_range_m"]),
            max_range=setup["max_range_m"],
            sampling_rate=setup["horizontal_columns"] * 32 * setup["rotation_hz"],
            draw_points=False,
            draw_lines=False,
        )
        if not success or not sensor:
            raise ValueError("selected Generic sensor creation failed")
        self.prim = sensor.GetPrim()
        UsdGeom.Xformable(self.prim).ClearXformOpOrder()  # Identity child of the measured vlp16 mount.
        self.sensor_attributes = {a.GetName(): str(a.Get()) for a in self.prim.GetAttributes()}
        self.api = _range_sensor.acquire_generic_sensor_interface()
        aa, zz = np.meshgrid(self.adapter.azimuth, -np.deg2rad(self.adapter.elevations), indexing="ij")
        self.pattern = np.tile(np.stack([aa.ravel(), zz.ravel()]), (1, 4)).astype(np.float32)
        self.publish = owner.node.create_publisher(PointCloud2, TOPIC, qos_profile_sensor_data)
        self.message = message
        self.warmup = 6
        self.latest = None
        self.done = False  # Stationary audit done, NOT producer completion.
        self.audit_frames = []
        self.tf_samples = deque(maxlen=120)
        self.tf_failure = None
        owner.node.create_subscription(TFMessage, "/tf", self.receive_tf, qos_profile_sensor_data)
        self.window_log = (self.run / "selected_windows.jsonl").open("x", buffering=1)
        files = [Path(__file__), Path(__file__).with_name("terrain_native_capture.py")]
        import generic_lidar_adapter, selected_lidar_profile, selected_native_audit, native_rejection, native_archive_io, native_chunk_transport
        from local_terrain import ray_design

        files += [
            Path(m.__file__)
            for m in (
                generic_lidar_adapter,
                selected_lidar_profile,
                selected_native_audit,
                ray_design,
                native_rejection,
                native_archive_io,
                native_chunk_transport,
            )
        ]
        files.append(profile_path())
        with (self.run / "selected_sensor.json").open("x") as stream:
            json.dump(
                dict(
                    profile=declaration(),
                    native_association_policy=ASSOCIATION_POLICY,
                    timestamp_policy=self.adapter.timestamp_policy,
                    legacy_sensor_active=legacy.IsActive(),
                    legacy_graph_active=graph.IsActive(),
                    sensor_path=self.path,
                    source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                    source_note="Real rover/physical IMU; timing/geometry must still pass the whole-run audit.",
                ),
                stream,
                indent=2,
            )

    def receive_tf(self, message):
        for item in message.transforms:
            if item.header.frame_id == "base_link" and item.child_frame_id == "vlp16":
                try:
                    t, q = item.transform.translation, item.transform.rotation
                    self.tf_samples.append(
                        (
                            item.header.stamp.sec + item.header.stamp.nanosec * 1e-9,
                            tf_matrix([t.x, t.y, t.z], [q.x, q.y, q.z, q.w]),
                        )
                    )
                except Exception as exc:
                    self.tf_failure = str(exc)

    def pre_step(self):
        if self.api.send_next_batch(self.path):
            self.api.set_next_batch_rays(self.path, self.pattern)

    def tick(self, stamp, pose):
        if self.warmup:
            self.warmup -= 1
            return
        if not np.allclose(pose["base_from_lidar"], self.expected_mount, atol=1e-5, rtol=0):
            raise ValueError("selected LiDAR mount changed")
        api, path = self.api, self.path
        arrays = [
            np.asarray(getattr(api, method)(path)).copy().reshape(shape)
            for method, shape in (
                ("get_point_cloud_data", (-1, 3)),
                ("get_linear_depth_data", (-1,)),
                ("get_intensity_data", (-1,)),
                ("get_azimuth_data", (-1,)),
                ("get_zenith_data", (-1,)),
            )
        ]
        if int(api.get_num_samples_ticked(path)) != len(arrays[0]):
            raise ValueError("native Generic tick count disagrees with returned arrays")
        sequence = self.adapter.next_sequence
        from native_rejection import guarded_push

        try:
            window = guarded_push(self.adapter, sequence, stamp, arrays, self.run, pose)
        except Exception:
            # Keep the original fault. Diagnostic casts run only after stopping
            # commands, with no intervening world step; never repair a return.
            try:
                if self.owner.driver is not None:
                    self.owner.driver.stop()
                from native_rejection import recast_rejected
                from .terrain_gt_probe import raycast_hits

                cast = lambda *args: raycast_hits(self.owner.query, *args, both_sides=True)
                diagnostic = recast_rejected(arrays, pose, cast)
                diagnostic.update(stamp=stamp, producer_sequence=sequence)
                with (self.run / "rejected_native_recast.json").open("x") as stream:
                    json.dump(diagnostic, stream, indent=2, allow_nan=False)
            except Exception as diagnostic_error:
                with (self.run / "rejected_native_recast_error.json").open("x") as stream:
                    json.dump(dict(error=str(diagnostic_error), acceptance=False), stream)
            raise
        self.latest = (stamp, sequence, arrays)
        if window is not None:
            self.publish.publish(self.message(window))
            metadata = {k: v for k, v in window.items() if k != "data"}
            metadata["sha256"] = hashlib.sha256(window["data"]).hexdigest()
            self.window_log.write(json.dumps(metadata) + "\n")

    def observe(self, stamp, pose, cast):
        if self.done:
            return
        if self.tf_failure:
            raise ValueError(self.tf_failure)
        if "raycast_min_range_m" in self.adapter.setup:
            from omni.physx import get_physx_scene_query_interface

            origin = (np.asarray(pose["world_from_base"]) @ np.asarray(pose["base_from_lidar"]))[:3, 3]
            overlaps = []

            def overlap(hit):
                overlaps.append(str(hit.collision))
                return True

            get_physx_scene_query_interface().overlap_sphere(0.001, tuple(origin), overlap, False)
            if overlaps:
                raise ValueError("optical origin overlaps collision geometry: " + repr(overlaps))
        tf = validate_extrinsic_tf(self.tf_samples, stamp, pose["base_from_lidar"])
        surface = self.surface(stamp, pose, cast)
        if surface["selected"] < 128 or surface["missing"] or max(surface["residuals_m"], default=1.0) > 0.001:
            # Preserve failures before refusing recorder/motion readiness.
            with (self.run / "selected_stationary_failure.json").open("x") as stream:
                json.dump(surface, stream, indent=2)
            raise ValueError("selected native stationary first-hit audit failed")
        self.audit_frames.append(dict(stamp=stamp, surface=surface, tf=tf))
        if len(self.audit_frames) == 12:
            with (self.run / "sensor_audit.json").open("x") as stream:
                json.dump(
                    dict(
                        passed=True,
                        extrinsic_tf_checked=True,
                        frames=self.audit_frames,
                        source="selected native ticks; NOT legacy ROS accumulated scan",
                    ),
                    stream,
                    indent=2,
                )
            self.done = True

    def surface(self, stamp, pose, cast, transform=None):
        if self.latest is None or self.latest[0] != stamp:
            raise ValueError("selected audit lacks exact current acquisition")
        points, depths, intensities, _, _ = self.latest[2]
        eligible = np.flatnonzero((intensities > 0) & (depths > 0.50001) & (depths <= 5.0))
        selected = eligible[np.linspace(0, len(eligible) - 1, min(128, len(eligible)), dtype=int)]
        current = transform is None
        if current:
            transform = np.asarray(pose["world_from_base"]) @ np.asarray(pose["base_from_lidar"])
        result = surface_audit(
            points,
            depths,
            selected,
            transform,
            cast,
            self.adapter.setup.get("raycast_min_range_m", self.adapter.setup["min_range_m"]),
            retain_hits=current,
            record_outliers=current,
        )
        if current and result["outliers"]:
            from selected_native_audit import origin_counterfactuals

            for outlier in result["outliers"]:
                outlier["origin_counterfactuals"] = origin_counterfactuals(outlier, transform[:3, 3], cast)
        return result


class SelectedNativeArchive:
    def __init__(self, sensor):
        import omni.timeline
        from selected_lidar_profile import acquisition_limits

        self.sensor = sensor
        self.timeline = omni.timeline.get_timeline_interface()
        self.output = sensor.run / "native_chunks"
        self.output.mkdir()
        from native_archive_io import PackedNativeArchive
        self.packed = PackedNativeArchive(self.output)
        self.limits = acquisition_limits(sensor.owner.motion_pattern)
        self.frames = EncodedFrames(capacity=self.limits["max_native_frames"])
        self.previous = None
        self.native_publisher = None
        self.completion_announced = False
        self.audit_metadata = None
        if os.getenv("OMNILRS_TERRAIN_NATIVE_PUBLISH", "false").lower() == "true":
            from sensor_msgs.msg import PointCloud2
            from std_msgs.msg import String
            from native_chunk_transport import message

            self.native_message = message
            self.native_publisher = sensor.owner.node.create_publisher(PointCloud2, "/terrain_shadow/native_chunks", 16)
            self.completion_publisher = sensor.owner.node.create_publisher(String, "/terrain_shadow/native_complete", 1)
            self.completion_message = String

    def observe(self, stamp, pose, cast):
        if self.completion_announced:
            raise ValueError("native acquisition already closed")
        if self.previous is not None and abs(stamp - self.previous[0] - 1 / 60) > 1e-7:
            raise ValueError("selected archive lost acquisition ticks")
        points, depths, intensities, azimuth, zenith = self.sensor.latest[2]
        transform = np.asarray(pose["world_from_base"]) @ np.asarray(pose["base_from_lidar"])
        surface = self.sensor.surface(stamp, pose, cast)
        previous = None if self.previous is None else self.sensor.surface(stamp, pose, cast, self.previous[1])
        if previous is not None:
            previous.pop("hit_records", None)
        file = self.output / f"chunk_{len(self.frames):06}.npz"
        archive_hit_records(surface, file.with_suffix(".hits.json"), archive=self.packed)
        from native_archive_io import write_native_npz

        digest = write_native_npz(
            file,
            archive=self.packed,
            points=points,
            depths=depths,
            intensities=intensities,
            azimuth=azimuth,
            zenith=zenith,
            observed_stamp=stamp,
            world_from_lidar=transform,
        )
        world = self.sensor.world
        self.frames.append(
            dict(
                file=file.name,
                sha256=digest,
                observed_stamp=stamp,
                producer_sequence=self.sensor.latest[1],
                samples=len(points),
                ray_layout="per_return_generic",
                world_time=float(world.current_time),
                timeline_time=float(self.timeline.get_current_time()),
                world_step=int(world.current_time_step_index),
                world_physics_dt=float(world.get_physics_dt()),
                world_rendering_dt=float(world.get_rendering_dt()),
                world_from_lidar=transform.tolist(),
                world_from_base=pose["world_from_base"],
                current_pose=surface,
                previous_pose=previous,
            )
        )
        self.previous = (stamp, transform)
        if self.native_publisher is not None:
            self.native_publisher.publish(
                self.native_message(
                    self.sensor.run.name,
                    len(self.frames) - 1,
                    self.sensor.adapter.stamp_ns(stamp),
                    points,
                    depths,
                    intensities,
                )
            )

    def announce_complete(self):
        from selected_lidar_profile import declaration
        from generic_lidar_adapter import ASSOCIATION_POLICY

        if self.audit_metadata is None:
            self.audit_metadata = dict(
                profile=declaration()["name"],
                acquisition_limits=self.limits,
                duplicate_native_reads=0,
                native_association_policy=ASSOCIATION_POLICY,
                timestamp_policy=self.sensor.adapter.timestamp_policy,
                producer_association_diagnostics=dict(self.sensor.adapter.association_diagnostics),
                association_scope="All producer ticks since warmup, including before measurement grant",
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                timing_validated_for_motion=False,
                mapper_input=False,
            )
        if self.native_publisher is not None and self.frames and not self.completion_announced:
            completion = dict(
                capture=self.sensor.run.name,
                sequence=len(self.frames) - 1,
                stamp_ns=self.sensor.adapter.stamp_ns(self.previous[0]),
            )
            self.completion_publisher.publish(self.completion_message(data=json.dumps(completion)))
        self.completion_announced = True

    def close(self):
        self.announce_complete()
        self.packed.close()
        with (self.output.parent / "native_timing_audit.json").open("x") as stream:
            self.frames.write_audit(stream, self.audit_metadata)
