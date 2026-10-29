"""Stationary ROS raw-cloud versus active-collider audit, on simulation thread."""
from collections import deque
import hashlib
import json
import numpy as np


def tf_matrix(translation,quaternion):
    t=np.asarray(translation,dtype=float);q=np.asarray(quaternion,dtype=float)
    if t.shape!=(3,) or q.shape!=(4,) or not np.isfinite(t).all() or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1)>1e-5:
        raise ValueError("invalid TF pose")
    x,y,z,w=q
    result=np.eye(4)
    result[:3,:3]=[[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                   [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                   [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
    result[:3,3]=t
    return result


def validate_extrinsic_tf(samples,stamp,expected):
    if not samples:raise ValueError("base_link to vlp16 TF missing")
    at,matrix=min(samples,key=lambda item:abs(item[0]-stamp))
    if abs(at-stamp)>.1:raise ValueError("sensor extrinsic TF stale")
    error=float(np.max(np.abs(matrix-np.asarray(expected))))
    if not np.isfinite(error) or error>1e-5:raise ValueError("published sensor TF differs from USD/physics extrinsic")
    return dict(stamp=at,time_error_s=abs(at-stamp),maximum_matrix_error=error,base_from_lidar=matrix.tolist())


def graph_contract(stage, robot_root):
    from pxr import Usd
    nodes = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_root)):
        node_type = prim.GetAttribute("node:type")
        if not node_type:
            continue
        attrs = {}
        for attr in prim.GetAttributes():
            if attr.GetConnections() or attr.GetName().startswith("inputs:") or attr.GetName() == "node:type":
                attrs[attr.GetName()] = dict(value=str(attr.Get()), connections=[str(p) for p in attr.GetConnections()])
        nodes.append(dict(path=str(prim.GetPath()), attributes=attrs,
                          relationships={r.GetName(): [str(p) for p in r.GetTargets()] for r in prim.GetRelationships()}))
    return nodes


def cloud_xyz(msg):
    layout = [(f.name,f.offset,f.datatype,f.count) for f in msg.fields]
    if (layout != [("x",0,7,1),("y",4,7,1),("z",8,7,1)] or msg.point_step != 12
            or msg.row_step != msg.width*12 or len(msg.data) != msg.row_step*msg.height):
        raise ValueError("unexpected raw XYZ layout; do not guess offsets/timing")
    return np.frombuffer(msg.data, dtype=">f4" if msg.is_bigendian else "<f4").reshape(-1,3).copy()


class TerrainSensorAudit:
    def __init__(self, output, frame_limit=10, require_tf=False, native_source=None, ray_budget=None):
        import rclpy
        from sensor_msgs.msg import PointCloud2
        from rclpy.qos import qos_profile_sensor_data
        self.node = rclpy.create_node("terrain_sensor_audit")
        self.queue = deque(maxlen=5)
        self.poses = deque(maxlen=600)
        self.frames = []
        self.boundary = self.dropped = 0
        self.output = output
        self.done = False
        self.frame_limit = frame_limit
        self.require_tf=require_tf
        self.native_source=native_source
        self.tf_samples=deque(maxlen=600)
        self.tf_errors=[]
        if ray_budget is not None and (type(ray_budget) is not int or ray_budget <= 0):
            raise ValueError("invalid per-step ray audit budget")
        self.ray_budget = ray_budget
        self.pending = None
        self.unselected_after_quota = 0

        def receive(msg):
            # A bounded audit reserves exactly frame_limit input scans. Later
            # scans are outside that declared sample, not dropped selected data.
            if (self.ray_budget is not None
                    and len(self.frames)+len(self.queue)+int(self.pending is not None) >= self.frame_limit):
                self.unselected_after_quota += 1
                return
            if len(self.queue) == self.queue.maxlen:
                self.dropped += 1
            self.queue.append(msg)

        self.node.create_subscription(PointCloud2, "/astrobot_0/lidar_0/pointcloud/raw", receive, qos_profile_sensor_data)
        if require_tf:
            from tf2_msgs.msg import TFMessage
            def receive_tf(message):
                for item in message.transforms:
                    if item.header.frame_id!="base_link" or item.child_frame_id!="vlp16":continue
                    try:
                        t,q=item.transform.translation,item.transform.rotation
                        matrix=tf_matrix([t.x,t.y,t.z],[q.x,q.y,q.z,q.w])
                        self.tf_samples.append((item.header.stamp.sec+item.header.stamp.nanosec*1e-9,matrix))
                    except Exception as exc:self.tf_errors.append(str(exc))
            self.node.create_subscription(TFMessage,"/tf",receive_tf,qos_profile_sensor_data)

    def observe(self, stamp, pose, cast):
        import rclpy
        if self.done:
            return
        matrix = np.asarray(pose["world_from_base"])@np.asarray(pose["base_from_lidar"])
        self.poses.append((stamp,matrix))
        for _ in range(8):rclpy.spin_once(self.node, timeout_sec=0.)
        if getattr(self, "ray_budget", None) is None:
            for _ in self._audit_steps(pose, cast):
                pass
            return
        if self.pending is None:
            self.pending = self._audit_steps(pose, cast)
        try:
            next(self.pending)
        except StopIteration:
            self.pending = None

    def _audit_steps(self, pose, cast):
        """All original rays/candidates, optionally spread across held steps.

        Caller must keep the robot settled; the driver checks this every tick.
        Scene queries stay on the simulation thread and retain acquisition poses.
        """
        if getattr(self,"require_tf",False):
            if self.tf_errors:raise ValueError(self.tf_errors[0])
            if not self.tf_samples:return
        if not self.queue:
            return
        msg = self.queue.popleft()
        ts = msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        if ts < self.poses[0][0]:
            self.boundary += 1
            return
        at, transform = min(self.poses, key=lambda p: abs(p[0]-ts))
        if abs(at-ts) > 1e-6:
            raise ValueError("raw cloud has no exact physics-step pose")
        if msg.header.frame_id != "vlp16":
            raise ValueError("unexpected raw sensor frame")
        tf_check=validate_extrinsic_tf(self.tf_samples,ts,pose["base_from_lidar"]) if getattr(self,"require_tf",False) else None
        points = cloud_xyz(msg)
        ranges = np.linalg.norm(points, axis=1)
        eligible = np.flatnonzero(np.isfinite(points).all(axis=1) & (ranges>=.5) & (ranges<=5.))
        if not len(eligible):
            raise ValueError("no finite raw returns within calibration range")
        selected = eligible[np.linspace(0,len(eligible)-1,min(1000,len(eligible)),dtype=int)]
        errors, surface_errors, missing, collisions = [], [], 0, {}
        missing_details, surface_missing = [], 0
        center_errors, center_collisions = [], {}
        minimum = float(pose["sensor_attributes"]["minRange"])
        # Preserve the attempted input even if association or a scene query
        # fails below. Failed attempts must be diagnosable, not silently lost.
        number=len(self.frames)
        raw_path=self.output/f"raw_scan_{number:02}.npz"
        with raw_path.open("xb") as stream:
            np.savez_compressed(stream,points=points,stamp=ts,world_from_lidar=transform)
        native_source=getattr(self,"native_source",None)
        native_matches={}
        if native_source is not None:
            surface_indices=selected[ranges[selected]>minimum+1e-5]
            native_matches=dict(zip(map(int,surface_indices),native_source.associate(points[surface_indices],ts)))
        audited_returns=[]
        for idx in selected:
            candidates=native_matches.get(int(idx))
            if candidates:
                audited_returns.extend((idx,np.asarray(f["world_from_lidar"])) for f,_ in candidates)
            else:
                audited_returns.append((idx,transform))
        for audited_index,(idx,acquisition) in enumerate(audited_returns):
            budget = getattr(self, "ray_budget", None)
            if budget is not None and audited_index and audited_index % budget == 0:
                yield  # Resume before the next query, never discard a candidate.
            direction = acquisition[:3,:3]@(points[idx]/ranges[idx])
            center_hits = cast(tuple(acquisition[:3,3]), tuple(direction), float(ranges[idx]+.1))
            if center_hits:
                center_errors.append(abs(center_hits[0][0]-float(ranges[idx])))
                center_collisions[center_hits[0][3]] = center_collisions.get(center_hits[0][3],0)+1
            # Explicitly audit the minimum-range origin convention, retaining
            # the center-origin counterfactual rather than fitting a transform.
            hits = cast(tuple(acquisition[:3,3]+minimum*direction), tuple(direction), float(ranges[idx]+.1-minimum))
            if not hits:
                missing += 1
                is_surface = bool(ranges[idx]>minimum+1e-5)
                surface_missing += int(is_surface)
                missing_details.append(dict(index=int(idx),range_m=float(ranges[idx]),surface_eligible=is_surface,
                                            center_hit_distance=float(center_hits[0][0]) if center_hits else None))
                continue
            error = abs(hits[0][0]+minimum-float(ranges[idx]))
            errors.append(error)
            if ranges[idx] > minimum+1e-5 and hits[0][0]>1e-5:
                surface_errors.append(error)
            collisions[hits[0][3]] = collisions.get(hits[0][3],0)+1
        frame = dict(stamp=ts, point_count=len(points), selected=len(selected), eligible=len(eligible),
                     extrinsic_tf=tf_check,
                     missing_rays=missing, range_residuals_m=errors, collisions=collisions,
                     world_from_lidar=transform.tolist(), exact_pose_time_error=abs(at-ts))
        frame.update(center_origin_residuals_m=center_errors, center_origin_collisions=center_collisions,
                     minimum_range=minimum, at_minimum_range_points=int((abs(ranges-minimum)<1e-5).sum()),
                     surface_residuals_m=surface_errors, surface_missing_rays=surface_missing,
                     missing_details=missing_details)
        if native_source is not None:
            frame["native_sources"]=[dict(ros_index=i,file=f["file"],sha256=f["sha256"],
                                           observed_stamp=f["observed_stamp"],native_index=k)
                                     for i,candidates in native_matches.items() for f,k in candidates]
            frame["ambiguous_source_returns"]=sum(len(candidates)>1 for candidates in native_matches.values())
        frame["raw_scan_sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        self.frames.append(frame)
        if len(self.frames) == self.frame_limit:
            residuals = [v for f in self.frames for v in f["range_residuals_m"]]
            missing = sum(f["missing_rays"] for f in self.frames)
            surface_missing = sum(f["surface_missing_rays"] for f in self.frames)
            p95 = float(np.percentile(residuals,95)) if residuals else None
            surfaces = [v for f in self.frames for v in f["surface_residuals_m"]]
            surface_p95 = float(np.percentile(surfaces,95)) if surfaces else None
            pose_model="native_tick_all_exact_candidates" if native_source is not None else "scan_end_pose"
            surface_pass=(bool(surfaces) and max(surfaces)<=.001 if native_source is not None
                          else surface_p95 is not None and surface_p95<=.01)
            summary = dict(frames=self.frames, boundary_scans=self.boundary, queue_drops=self.dropped,
                           residual_p95_m=p95, missing_rays=missing,
                           surface_missing_rays=surface_missing,
                           surface_samples=len(surfaces), surface_residual_p95_m=surface_p95,
                           extrinsic_tf_checked=bool(getattr(self,"require_tf",False)),
                           ray_pairs_per_step=getattr(self,"ray_budget",None),
                           unselected_after_frame_quota=getattr(self,"unselected_after_quota",0),
                           passed=bool(surface_pass and surface_missing==0 and self.dropped==0),
                           stationary_only=native_source is None, timing_validated_for_motion=False,
                           surface_pose_model=pose_model,
                           query_convention="origin shifted by sensor minRange; both mesh sides as Isaac LiDAR; center-origin counterfactual retained",
                           note="Range agreement with active collider; no cloud fitting. Native mode audits ALL exact XYZ acquisition candidates, never guesses an ambiguous beam time. Archived ROS XYZ is STILL UNDESKEWED and cannot be mapped at its scan-end pose. Min-clamped misses remain diagnostics.")
            with (self.output/"sensor_audit.json").open("x") as stream:
                json.dump(summary, stream, indent=2)
            self.done = True
            self.node.destroy_node()
