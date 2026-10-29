"""Opt-in GT-controlled stop/scan translation on the analytic test platform.

No navigation stack, no moving-scan accuracy claim. Physics-step GT is used only
by this experiment's driver. Independent watchdog publishes zero on stalled sim.
"""
import threading
import time
import math
import json
import numpy as np


def prescribed_targets(pattern="straight"):
    if pattern == "spin360":
        return [dict(distance=0.,heading=math.radians(degrees)) for degrees in range(0,361,15)]
    if pattern in ("wide_sweep15","wide_sweep15_return"):
        # Separately declared 4.8m acquisition dose. The base pattern ends at
        # the travel heading; only the return experiment adds a final origin turn.
        targets=[dict(distance=0.,heading=0.),
                 *[dict(distance=x,heading=math.radians(15)) for x in (0.,1.2,-1.2,0.)]]
        return targets+([dict(distance=0.,heading=0.)] if pattern=="wide_sweep15_return" else [])
    distances = [x/10 for x in range(21)]+[x/10 for x in range(19,-1,-1)]
    if pattern in ("centered", "centered_diagonal15", "centered_sweep15"):
        distances=[x/10 for x in range(11)]+[x/10 for x in range(9,-11,-1)]+[x/10 for x in range(-9,1)]
    if pattern == "centered_sweep15":
        distances=[0.,1.,-1.,0.]
    if pattern not in ("straight", "diagonal15", "centered", "centered_diagonal15", "centered_sweep15"):
        raise ValueError("unknown prescribed calibration route")
    heading = math.radians(15) if pattern in ("diagonal15", "centered_diagonal15", "centered_sweep15") else 0.
    targets = [dict(distance=x, heading=heading) for x in distances]
    return ([dict(distance=0.,heading=0.), *targets, dict(distance=0.,heading=0.)]
            if heading else targets)


def validate_acquisition(pattern, native_enabled):
    prescribed_targets(pattern)
    if pattern in ("centered_sweep15","wide_sweep15","wide_sweep15_return") and not native_enabled:
        raise ValueError("continuous sweep requires native-tick motion evidence")


def drive_command(position, yaw, target_x, origin, heading=0., pattern="straight"):
    if not np.isfinite([*position,yaw,target_x,*origin,heading]).all():
        raise ValueError("nonfinite drive pose/target")
    local = np.asarray(position)[:2]-origin
    if pattern == "spin360":
        error=math.remainder(heading-yaw,2*math.pi)
        if target_x!=0 or not 0<=heading<=2*math.pi or np.linalg.norm(local)>=.25 or abs(error)>.35:
            raise ValueError("rotation left bounded-origin or incremental-heading envelope")
        if abs(error)<.004:
            return 0.,0.,True
        return 0.,math.copysign(min(.15,max(.04,abs(error)*2)),error),False
    if pattern not in ("straight","diagonal15","centered","centered_diagonal15","centered_sweep15","wide_sweep15","wide_sweep15_return"):
        raise ValueError("unknown calibration drive pattern")
    along = local[0]*math.cos(heading)+local[1]*math.sin(heading)
    across = -local[0]*math.sin(heading)+local[1]*math.cos(heading)
    error = target_x-along
    yaw_error = math.remainder(heading-yaw,2*math.pi)
    lower,upper=(-1.08,1.08) if pattern in ("centered","centered_diagonal15","centered_sweep15") else (-.08,2.08)
    if pattern in ("wide_sweep15","wide_sweep15_return"):
        lower,upper=-1.28,1.28
    if abs(across)>.08 or not lower<=along<=upper or abs(yaw)>.35:
        raise ValueError("calibration drive left tracking envelope")
    # Rotation is allowed only at the origin, before/after a straight leg.
    if abs(target_x)<1e-9 and np.linalg.norm(local)<.04 and abs(yaw_error)>=.004:
        if yaw < -.01 or yaw > math.radians(15)+.01:
            raise ValueError("rotation exceeded the checked swept-yaw envelope")
        return 0.,math.copysign(min(.15,max(.04,abs(yaw_error)*2)),yaw_error),False
    if abs(yaw_error)>.08:
        raise ValueError("calibration drive heading deviated during translation")
    if abs(error)<.008:
        return 0.,0.,True
    speed = math.copysign(min(.15,max(.04,abs(error)*2)),error)
    return speed, float(np.clip(yaw_error*2,-.15,.15)),False


def publish_completion_marker(path, payload):
    """Exclusive atomic visibility; retain the prepared record on failure too."""
    import os
    prepared = path.with_name(path.name + ".pending")
    with prepared.open("x") as stream:
        json.dump(payload, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(prepared, path)  # Same-filesystem atomic create; never replaces an artifact.


class CalibrationDrive:
    def __init__(self, origin, query, cast, output, pattern="straight", shadow_request=None):
        import rclpy
        from geometry_msgs.msg import Twist
        self.node = rclpy.create_node("terrain_calibration_drive")
        self.pub = self.node.create_publisher(Twist,"/astrobot_0/cmd_vel",10)
        self.origin = np.asarray(origin)
        prescribed_targets(pattern)  # Fail before issuing any movement.
        self.pattern = pattern
        self.query,self.cast = query,cast
        self.last_tick = time.monotonic()
        self.armed = False
        self.failure = None
        self.closed = threading.Event()
        self.log = []
        self.pose_log = []
        self.output = output
        self.shadow_lease = None
        self.shadow_request = shadow_request
        self.shadow_created = time.monotonic()
        self.shadow_sim = 0.
        self.shadow_last_request = -float("inf")
        if shadow_request is not None:
            from shadow_capture_gate import ShadowLease
            from std_msgs.msg import String
            self.shadow_lease = ShadowLease(output.name)
            self.shadow_pub = self.node.create_publisher(String, "/terrain_shadow/request", 1)
            self.node.create_subscription(String, "/terrain_shadow/grant", self._shadow_receive, 1)
        self.scan_count = 0
        (output/"continuous_raw").mkdir(exist_ok=False)
        from sensor_msgs.msg import PointCloud2
        from rclpy.qos import qos_profile_sensor_data
        self.node.create_subscription(PointCloud2,"/astrobot_0/lidar_0/pointcloud/raw",self._record,qos_profile_sensor_data)
        self.target_started = None
        self.progress = None
        self.thread = threading.Thread(target=self._watchdog,daemon=True)
        self.thread.start()

    def _record(self, msg):
        from .terrain_sensor_audit import cloud_xyz
        points = cloud_xyz(msg)
        stamp = msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        np.savez_compressed(self.output/"continuous_raw"/f"scan_{self.scan_count:06}.npz",
                            points=points,stamp=stamp,frame_id=msg.header.frame_id)
        self.scan_count += 1

    def record_step(self, stamp, position, quaternion):
        import rclpy
        self.last_tick = time.monotonic()
        self.shadow_sim = stamp
        self.pose_log.append(dict(stamp=stamp,position=list(map(float,position)),quaternion=list(map(float,quaternion))))
        rclpy.spin_once(self.node,timeout_sec=0.)

    def _publish(self, linear=0.,angular=0.):
        from geometry_msgs.msg import Twist
        message = Twist()
        lease = getattr(self, "shadow_lease", None)
        if lease is not None and (linear or angular):
            try:
                if not lease.ready(self.shadow_sim, time.monotonic()):
                    raise ValueError("shadow capture is not armed")
            except ValueError:
                self.pub.publish(message)
                raise
        message.linear.x,message.angular.z = float(linear),float(angular)
        self.pub.publish(message)

    def _watchdog(self):
        while not self.closed.wait(.05):
            if self.armed and time.monotonic()-self.last_tick>.5:
                self.failure = "physics-step heartbeat timed out"
                self._publish()
            if self.armed and self.shadow_lease is not None:
                try:
                    self.shadow_lease.ready(self.shadow_sim, time.monotonic())
                except ValueError as exc:
                    self.failure = str(exc)
                    self._publish()

    def _shadow_receive(self, message):
        try:
            publishers = self.node.get_publishers_info_by_topic("/terrain_shadow/grant")
            if len(publishers) != 1 or publishers[0].node_name != "terrain_shadow_capture":
                raise ValueError("shadow grant requires one identified publisher")
            self.shadow_lease.receive(json.loads(message.data), self.shadow_sim, time.monotonic())
        except (ValueError, TypeError, KeyError) as exc:
            self.failure = f"shadow grant rejected: {exc}"
            self._publish()

    def shadow_ready(self, stamp):
        if self.shadow_lease is None:
            return True
        self.shadow_sim = stamp
        wall = time.monotonic()
        if self.failure:
            raise ValueError(self.failure)
        if wall-self.shadow_last_request >= .25:
            from std_msgs.msg import String
            self.shadow_pub.publish(String(data=json.dumps(dict(self.shadow_request, stamp=stamp))))
            self.shadow_last_request = wall
        if self.shadow_lease.ready(stamp, wall):
            return True
        if wall-self.shadow_created > 120.:
            raise ValueError("shadow recorder/localization readiness timeout")
        return False

    def hold(self):
        self.last_tick = time.monotonic()
        if self.failure:
            self._publish()
            raise ValueError(self.failure)
        self._publish()
        self.armed = False
        self.target_started = None
        self.progress = None

    def move(self, stamp, position, quaternion, target):
        self.last_tick = time.monotonic()
        if self.failure:
            self._publish()
            raise ValueError(self.failure)
        publishers = self.node.get_publishers_info_by_topic("/astrobot_0/cmd_vel")
        if len(publishers)!=1 or publishers[0].node_name != self.node.get_name():
            self._publish()
            raise ValueError("calibration drive requires sole command publisher")
        qx,qy,qz,qw = quaternion
        yaw = math.atan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz))
        linear,angular,arrived = drive_command(position,yaw,target["distance"],self.origin,target["heading"],self.pattern)
        if arrived:
            self.hold()
            return True
        # Inflated rectangular footprint plus short forward/backward lookahead.
        # The lower query face is above flat ground but below 8cm hazards.
        found,errors = [],[]

        def report(hit):
            try:
                if not str(hit.collision).startswith("/Robots/astrobot_0/"):
                    found.append(str(hit.collision))
                return True
            except Exception as exc:
                errors.append(str(exc))
                return False

        # Bound a complete 0..15 degree yaw sweep with a conservative world-axis
        # rectangle. Translation uses the actual yaw-oriented footprint.
        rotating = linear == 0. and angular != 0.
        half = (.999,.900,.48) if rotating else (.85,.7,.48)
        if rotating and self.pattern=="spin360":
            # Entire circle swept by the inflated rectangular footprint. The
            # circumscribed square is deliberately more conservative.
            # Include the whole allowed 25cm drift envelope about the original
            # pose, rather than chasing the robot with an unbounded safe region.
            radius=math.hypot(.85,.7)+.25
            half=(radius,radius,.48)
        box_yaw = 0. if rotating else yaw
        query_center=self.origin if rotating and self.pattern=="spin360" else position
        self.query.overlap_box(half,(float(query_center[0]),float(query_center[1]),.52),
                               (0.,0.,math.sin(box_yaw/2),math.cos(box_yaw/2)),report,False)
        if found or errors:
            self.hold()
            raise ValueError(f"non-robot swept-volume overlap: {found or errors}")
        for dx in (-half[0],0.,half[0]):
            for dy in (-half[1],0.,half[1]):
                rx,ry=dx*math.cos(box_yaw)-dy*math.sin(box_yaw),dx*math.sin(box_yaw)+dy*math.cos(box_yaw)
                hits = self.cast((float(query_center[0]+rx),float(query_center[1]+ry),1.),(0.,0.,-1.),2.)
                hits = [h for h in hits if not h[3].startswith("/Robots/")]
                if not hits or abs(hits[0][1][2])>.005:
                    self.hold()
                    raise ValueError("unsupported/nonflat swept footprint")
        if self.target_started is None:
            self.target_started = stamp
            self.progress = (stamp,np.asarray(position).copy(),yaw)
        # Native-tick sweeps have declared longest legs of 2m / 2.4m (at most
        # 16s at 0.15m/s plus settling). Only the leg deadline changes; per-step clearance, heartbeat,
        # tracking and 1.5s stall checks remain identical to the stop/scan route.
        timeout=8. if rotating else (20. if self.pattern in ("centered_sweep15","wide_sweep15","wide_sweep15_return") else 5.)
        if stamp-self.target_started>timeout:
            self.hold()
            raise ValueError("waypoint timeout")
        if stamp-self.progress[0]>1.5:
            if np.linalg.norm(np.asarray(position)-self.progress[1])<.01 and abs(yaw-self.progress[2])<.02:
                self.hold()
                raise ValueError("robot stalled; refusing to keep pushing")
            self.progress = (stamp,np.asarray(position).copy(),yaw)
        self.armed = True
        self._publish(linear,angular)
        self.log.append(dict(stamp=stamp,position=list(map(float,position)),yaw=yaw,target=target,
                             linear=linear,angular=angular))
        return False

    def close(self):
        self._publish()
        self.armed = False
        self.closed.set()
        self.thread.join(timeout=1.)
        try:
            with (self.output/"drive_log.json").open("x") as stream:
                json.dump(dict(commands=self.log,poses=self.pose_log,raw_scans=self.scan_count,
                               shadow_first_grant=(self.shadow_lease.first_grant
                                                   if getattr(self,"shadow_lease",None) else None),
                               failure=self.failure,note="Continuous raw scans retained, not deskewed; only settled views are scored."),stream)
        finally:
            self.node.destroy_node()
