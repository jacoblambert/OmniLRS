"""Opt-in native LiDAR chunk/pose audit, before ROS full-scan accumulation.

This is diagnostic evidence, not an approved moving-cloud input. Native Python
bindings expose tick-local XYZ/angles but not beam timestamps. Record the actual
chunk cadence and current/previous pose counterfactuals instead of guessing a
scan time offset or treating the full ROS scan as instantaneous.
"""
import hashlib
import json
import os
from collections import deque
from pathlib import Path

import numpy as np


def associate_returns(points, stamp, recent, maximum_age):
    """Exact XYZ provenance, not nearest-neighbor fitting or guessed beam times.

    Return EVERY exact candidate inside a bounded acquisition window. Repeated
    XYZ at slightly different poses does not identify a unique beam time; the
    caller must audit all candidates, never select the most favorable pose.
    """
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("invalid ROS return association input")
    if not np.isfinite([stamp, maximum_age]).all() or maximum_age <= 0:
        raise ValueError("invalid native association time window")
    eligible = [(cloud, frame) for cloud, frame in recent
                if stamp - maximum_age <= frame["observed_stamp"] <= stamp + 1e-6]
    if not eligible:
        raise ValueError("no native acquisition history for ROS returns")
    keys = lambda cloud: np.ascontiguousarray(cloud, dtype="<f4").view("V12").ravel()
    sizes = np.array([len(cloud) for cloud, _ in eligible])
    cumulative = np.cumsum(sizes)
    joined = np.concatenate([keys(cloud) for cloud, _ in eligible])
    order = np.argsort(joined, kind="stable")
    sorted_keys, query = joined[order], keys(points)
    left, right = np.searchsorted(sorted_keys, query, side="left"), np.searchsorted(sorted_keys, query, side="right")
    if not (((right - left) >= 1) & ((right - left) <= 32)).all():
        raise ValueError("ROS return has missing or excessive native acquisition candidates")
    offsets = np.r_[0, cumulative[:-1]]
    result=[]
    for lo,hi in zip(left,right):
        source=order[lo:hi]
        frame_indices=np.searchsorted(cumulative,source,side="right")
        result.append([(eligible[k][1],int(index-offsets[k])) for index,k in zip(source,frame_indices)])
    return result


def validate_chunk(points, depths, intensities, azimuth, zenith, columns, rows):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    depths = np.asarray(depths, dtype=np.float32).ravel()
    intensities = np.asarray(intensities, dtype=np.uint8).ravel()
    azimuth = np.asarray(azimuth, dtype=np.float32).ravel()
    zenith = np.asarray(zenith, dtype=np.float32).ravel()
    if not 0 < columns <= 10000 or not 0 < rows <= 256:
        raise ValueError("invalid native LiDAR dimensions")
    if (len(points) != columns * rows or len(depths) != len(points) or len(intensities) != len(points)
            or len(azimuth) != columns or len(zenith) != rows):
        raise ValueError("native XYZ/depth/angle arrays do not share a tick layout")
    if not all(np.isfinite(v).all() for v in (points, depths, azimuth, zenith)):
        raise ValueError("nonfinite native chunk")
    a, z = np.meshgrid(azimuth.astype(float), zenith.astype(float), indexing="ij")
    directions = np.column_stack([(np.cos(a) * np.cos(z)).ravel(),
                                  (np.sin(a) * np.cos(z)).ravel(), -np.sin(z).ravel()])
    valid = (intensities > 0) & (depths > .50001) & (depths <= 5.)
    if valid.any() and np.max(np.linalg.norm(points[valid] - directions[valid] * depths[valid, None], axis=1)) > .001:
        raise ValueError("native angle/XYZ ordering or range convention mismatch")
    return points, depths, intensities, azimuth, zenith


def surface_audit(points, depths, selected, transform, cast, minimum, *, record_outliers=True, retain_hits=False):
    errors, missing, nonhorizontal = [], 0, 0
    outliers = []
    hit_records = []
    for index in selected:
        direction = transform[:3, :3] @ (points[index].astype(float) / np.linalg.norm(points[index]))
        origin = transform[:3, 3] + minimum * direction
        distance = float(depths[index] + .1 - minimum)
        hits = cast(tuple(origin), tuple(direction), distance)
        error = None if not hits else abs(hits[0][0] + minimum - float(depths[index]))
        if retain_hits:
            hit_records.append(dict(point_index=int(index), first_hit=None if not hits else
                dict(distance_m=float(hits[0][0]), position=list(map(float, hits[0][1])),
                     normal=list(map(float, hits[0][2])), collision=str(hits[0][3])), residual_m=error))
        if record_outliers and (error is None or error > .001):
            # Preserve enough geometry to diagnose grazing first-hit failures.
            # Do not choose a later hit matching the recorded depth, normalize
            # a different ray, remove the point, or change audit eligibility.
            outliers.append(dict(point_index=int(index), local_xyz=points[index].astype(float).tolist(),
                native_depth_m=float(depths[index]), ray_origin=origin.tolist(), ray_direction=direction.tolist(),
                direction_norm=float(np.linalg.norm(direction)), cast_distance_m=distance,
                world_endpoint=(transform[:3,:3] @ points[index].astype(float)+transform[:3,3]).tolist(),
                first_hit_residual_m=error, hit_count=len(hits),
                first_hits=[dict(distance_m=float(hit[0]),position=list(map(float,hit[1])),
                                 normal=list(map(float,hit[2])),collision=str(hit[3])) for hit in hits[:8]]))
        if not hits:
            missing += 1
            continue
        errors.append(error)
        nonhorizontal += int(abs(hits[0][2][2]) < .5)
    result = dict(selected=len(selected), missing=missing, residuals_m=errors,
                selected_indices=list(map(int,selected)), outliers=outliers,
                nonhorizontal_surface_samples=nonhorizontal)
    if retain_hits:
        result["hit_records"] = hit_records
    return result


def archive_hit_records(surface, path, *, archive=None):
    """Keep evaluator identity evidence off the long-lived simulator Python heap.

    Retaining 128 nested hit objects per tick provoked a ~0.8 s generation-2
    collection in Isaac. Stream exact records, then retain only their hash/path.
    GC remains enabled; no acquisition cadence or watchdog limit is changed.
    """
    if "hit_records" not in surface:
        return
    payload = json.dumps(surface["hit_records"], allow_nan=False).encode()
    if archive is not None:
        archive.write(path.name, payload)
    else:
        with path.open("xb") as stream:
            stream.write(payload)
    surface["hit_records_file"] = path.name
    surface["hit_records_sha256"] = hashlib.sha256(payload).hexdigest()
    del surface["hit_records"]


class EncodedFrames:
    """Immutable JSON records, not an ever-growing graph of GC-tracked lists.

    The bounded recent association window keeps decoded frames. Long-term audit
    bytes remain exact and are streamed unchanged into the final JSON document.
    No global GC settings or collection thresholds are changed.
    """
    def __init__(self, capacity=20000):
        if type(capacity) is not int or not 0 < capacity <= 100000:  # transit50 declares 55000
            raise ValueError("invalid native frame archive capacity")
        self.capacity = capacity
        self.records = []

    def __len__(self):
        return len(self.records)

    def append(self, frame):
        if len(self.records) >= self.capacity:
            raise ValueError("native frame archive capacity exceeded")
        self.records.append(json.dumps(frame, allow_nan=False, separators=(",", ":")))

    def __getitem__(self, index):
        return json.loads(self.records[index])

    def write_audit(self, stream, metadata):
        stream.write('{"frames":[')
        for index, record in enumerate(self.records):
            if index:
                stream.write(',')
            stream.write(record)
        stream.write('],')
        stream.write(json.dumps(metadata, allow_nan=False)[1:])


class NativeTerrainCapture:
    def __init__(self, output, lidar_path, *, retain_hits=False, node=None):
        from isaacsim.sensors.physx import _range_sensor
        from isaacsim.core.api import World
        import omni.timeline
        self.interface = _range_sensor.acquire_lidar_sensor_interface()
        self.world = World.instance()
        self.timeline = omni.timeline.get_timeline_interface()
        self.path = lidar_path
        self.retain_hits = retain_hits
        self.output = output / "native_chunks"
        self.output.mkdir(exist_ok=False)
        self.frames = EncodedFrames()
        self.previous = None
        self.previous_azimuth = None
        self.duplicate_reads = 0
        self.recent = deque(maxlen=16)
        self.native_publisher = None
        if os.getenv("OMNILRS_TERRAIN_NATIVE_PUBLISH", "false").lower() == "true":
            if node is None:
                raise ValueError("native publication currently requires the guarded natural-capture ROS node")
            from sensor_msgs.msg import PointCloud2
            from std_msgs.msg import String
            from native_chunk_transport import message
            self.native_message = message
            self.capture_identity = output.name
            self.native_publisher = node.create_publisher(PointCloud2, "/terrain_shadow/native_chunks", 16)
            self.completion_publisher = node.create_publisher(String, "/terrain_shadow/native_complete", 1)
            self.completion_message = String

    def associate(self, points, stamp):
        if not self.frames:
            raise ValueError("native source has not captured a tick")
        return associate_returns(points, stamp, self.recent, 2. / self.frames[-1]["configured_rotation_rate"])

    def observe(self, stamp, pose, cast):
        if len(self.frames) >= 20000:
            raise ValueError("native chunk diagnostic exceeded bounded frame count")
        interface, path = self.interface, self.path
        cols, rows = interface.get_num_cols_ticked(path), interface.get_num_rows(path)
        if not cols:
            return
        points, depths, intensities, azimuth, zenith = validate_chunk(
            interface.get_point_cloud_data(path), interface.get_linear_depth_data(path),
            interface.get_intensity_data(path), interface.get_azimuth_data(path),
            interface.get_zenith_data(path), cols, rows)
        transform = np.asarray(pose["world_from_base"]) @ np.asarray(pose["base_from_lidar"])
        if self.previous is not None and stamp <= self.previous[0]:
            raise ValueError("native capture clock regressed")
        if self.previous_azimuth is not None and np.array_equal(azimuth, self.previous_azimuth):
            self.duplicate_reads += 1
            return
        self.previous_azimuth = azimuth.copy()
        minimum = float(pose["sensor_attributes"]["minRange"])
        eligible = np.flatnonzero((intensities > 0) & (depths > minimum + 1e-5) & (depths <= 5.))
        selected = eligible[np.linspace(0, len(eligible) - 1, min(128, len(eligible)), dtype=int)]
        current = surface_audit(points, depths, selected, transform, cast, minimum, retain_hits=self.retain_hits)
        previous = None if self.previous is None else surface_audit(points, depths, selected, self.previous[1], cast, minimum,
                                                                   record_outliers=False)
        file = self.output / f"chunk_{len(self.frames):06}.npz"
        archive_hit_records(current, file.with_suffix(".hits.json"))
        np.savez_compressed(file, points=points, depths=depths, intensities=intensities,
                            azimuth=azimuth, zenith=zenith, observed_stamp=stamp, world_from_lidar=transform)
        self.frames.append(dict(file=file.name, sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                                observed_stamp=stamp, columns=int(cols), rows=int(rows),
                                full_columns=int(interface.get_num_cols(path)),
                                configured_rotation_rate=float(pose["sensor_attributes"]["rotationRate"]),
                                world_time=float(self.world.current_time),
                                world_step=int(self.world.current_time_step_index),
                                world_physics_dt=float(self.world.get_physics_dt()),
                                world_rendering_dt=float(self.world.get_rendering_dt()),
                                timeline_time=float(self.timeline.get_current_time()),
                                world_from_lidar=transform.tolist(), current_pose=current, previous_pose=previous))
        self.recent.append((points.copy(), self.frames[-1]))
        self.previous = (stamp, transform)
        if self.native_publisher is not None:
            # Exact archived raw arrays; no audited GT pose, ray label, filtering
            # or best-hit selection enters the live transport.
            self.native_publisher.publish(self.native_message(self.capture_identity, len(self.frames)-1,
                round(stamp*1e9), points, depths, intensities))

    def close(self):
        if self.native_publisher is not None and self.previous is not None:
            self.completion_publisher.publish(self.completion_message(data=json.dumps(dict(
                capture=self.capture_identity, sequence=len(self.frames)-1, stamp_ns=round(self.previous[0]*1e9)))))
        with (self.output.parent / "native_timing_audit.json").open("x") as stream:
            self.frames.write_audit(stream, dict(duplicate_native_reads=self.duplicate_reads,
                           source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           timing_validated_for_motion=False, mapper_input=False,
                           note=__doc__))
