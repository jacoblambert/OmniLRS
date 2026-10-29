"""Opt-in physics-rate IMU shadow stream; never replaces the legacy publisher.

Isaac 5.1 component checks show gyro corresponds to the current physics state,
whereas orientation can lag a render tick and sensorTime uses a resettable clock.
Publish acceleration/gyro at the callback's monotonic physics time. Do not claim
orientation is available; retain all clock fields on a separate diagnostic topic.
No GT pose, terrain labels, command publisher or estimator configuration here.
"""
import json
import math


class PhysicsSampleClock:
    """Reject stale/changed timing rather than silently fitting a clock offset."""

    def __init__(self):
        self.epoch = 0
        self.last_world = None
        self.last_monotonic = None
        self.last_sensor = None

    def observe(self, *, monotonic, world, sensor, dt, valid):
        if not all(math.isfinite(v) for v in (monotonic, world, sensor, dt)) or min(monotonic, world, sensor) < 0 or dt <= 0:
            raise ValueError("nonfinite/negative physics IMU clock")
        reset = self.last_world is not None and world < self.last_world-1e-8
        if self.last_monotonic is not None and monotonic <= self.last_monotonic:
            raise ValueError("monotonic physics IMU clock regressed or duplicated")
        if reset:
            self.epoch += 1
            self.last_sensor = None
        if (not reset and self.last_monotonic is not None
                and abs(monotonic-self.last_monotonic-dt) > 1e-5):
            raise ValueError("physics IMU callback skipped a tick")
        if valid:
            # Version-scoped component contract, not a fitted timestamp shift.
            # sensorTime labels the preceding integration interval; gyro is
            # current-state. A direct sensorTime -> ROS stamp copy is incorrect.
            if abs(world-sensor-dt) > 1e-3:
                raise ValueError("unqualified Isaac IMU sensor/physics clock phase")
            if self.last_sensor is not None and sensor <= self.last_sensor:
                raise ValueError("stale physics IMU sample")
        self.last_world, self.last_monotonic = world, monotonic
        if valid:
            self.last_sensor = sensor
        return dict(epoch=self.epoch, monotonic_time=monotonic, world_time=world,
                    sensor_time=sensor, physics_dt=dt, valid=bool(valid), orientation_available=False)


def fill_imu(message, reading, stamp, frame):
    """ROS-independent field writer for mock and serialization contracts."""
    values = (reading.lin_acc_x, reading.lin_acc_y, reading.lin_acc_z,
              reading.ang_vel_x, reading.ang_vel_y, reading.ang_vel_z)
    if not all(math.isfinite(v) for v in values) or not math.isfinite(stamp) or stamp < 0:
        raise ValueError("nonfinite physics IMU payload")
    ns = round(stamp*1e9)
    message.header.stamp.sec, message.header.stamp.nanosec = divmod(ns, 1_000_000_000)
    message.header.frame_id = frame
    message.linear_acceleration.x, message.linear_acceleration.y, message.linear_acceleration.z = values[:3]
    message.angular_velocity.x, message.angular_velocity.y, message.angular_velocity.z = values[3:]
    message.orientation.w = 1.
    message.orientation_covariance[0] = -1.  # Never pass stale orientation as current truth.
    return message


class PhysicsIMUShadow:
    """Uses the existing robot-manager node; physics callback owns all state."""

    def __init__(self, node, world, core_clock):
        from isaacsim.sensors.physics import _sensor
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Imu
        from std_msgs.msg import String
        from pxr import Usd

        self.node, self.world, self.core_clock = node, world, core_clock
        self.interface = _sensor.acquire_imu_sensor_interface()
        self.Imu, self.String = Imu, String
        self.entries = []
        self.failed = False
        for robot in node.RM.robots.values():
            prims = [p for p in Usd.PrimRange(robot.stage.GetPrimAtPath(robot.robot_path))
                     if p.GetTypeName() == "IsaacImuSensor"]
            if len(prims) != 1:
                raise ValueError("physics IMU shadow requires exactly one sensor per robot")
            name = robot.robot_name.strip("/")
            topic = f"/{name}/imu_physics"
            if node.count_publishers(topic):
                raise ValueError("physics IMU shadow topic already has a publisher")
            self.entries.append((str(prims[0].GetPath()), PhysicsSampleClock(),
                node.create_publisher(Imu, topic, qos_profile_sensor_data),
                node.create_publisher(String, topic+"/clock", 100)))
        if not self.entries:
            raise ValueError("physics IMU shadow requires loaded robots")
        world.add_physics_callback("physics_imu_shadow", self.tick)
        node.get_logger().info("Opt-in physics IMU shadow enabled; legacy IMU/GLIM inputs unchanged")

    def tick(self, dt):
        if self.failed:
            return
        try:
            for path, clock, publisher, diagnostics in self.entries:
                sample = self.interface.get_sensor_reading(path, use_latest_data=True, read_gravity=True)
                current = float(self.core_clock.get_sim_time_monotonic())
                status = clock.observe(monotonic=current, world=float(self.world.current_time),
                                       sensor=float(sample.time), dt=float(dt), valid=bool(sample.is_valid))
                if sample.is_valid:
                    message = fill_imu(self.Imu(), sample, current, path.rsplit("/", 1)[-1])
                    publisher.publish(message)
                diagnostics.publish(self.String(data=json.dumps(status, allow_nan=False)))
        except Exception as exc:
            # Stop this experimental stream. Never resurrect it or alter the
            # working legacy topic following an exception in a native callback.
            self.failed = True
            for _, _, _, diagnostics in self.entries:
                diagnostics.publish(self.String(data=json.dumps(dict(valid=False, terminal_fault=str(exc)))))
            self.node.get_logger().error(f"Physics IMU shadow stopped: {exc}")
