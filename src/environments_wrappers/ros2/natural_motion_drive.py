"""Opt-in, geometry-guarded acquisition only; never a navigation controller."""
import json
import hashlib
import math
import threading
import time

import numpy as np


def stationary_heap_checkpoint(lease_sequence, motion_sent, capture_started):
    """Collect startup garbage before granting authority, never suppress runtime GC."""
    import gc
    from collections import Counter
    if lease_sequence != -1 or motion_sent or capture_started is not None or not gc.isenabled():
        raise ValueError("heap checkpoint requires pre-lease stationary initialization and enabled GC")
    before = gc.get_stats()
    started = time.monotonic()
    collected = gc.collect(2)
    collection_wall = time.monotonic()-started
    # Diagnostic only, before authority: identify simulator-wide retained types
    # without keeping references to the enumerated object graph.
    types, sizes = Counter(), Counter()
    for item in gc.get_objects():
        kind = type(item)
        types[str(kind.__module__)+"."+str(kind.__name__)] += 1
        if kind in (list, tuple, dict):
            sizes[f"{kind.__name__}:{len(item)}"] += 1
    del item
    return dict(wall_s=collection_wall, checkpoint_wall_s=time.monotonic()-started,
                collected=collected, before=before, after=gc.get_stats(),
                tracked_types=types.most_common(20), tracked_container_sizes=sizes.most_common(20),
                gc_enabled=gc.isenabled(), thresholds=list(gc.get_threshold()), phase="before_request_and_motion_lease")


class NaturalMotionDrive:
    def __init__(self, owner, origin, pattern, cast):
        from natural_motion import route, survey_mask, survey_points, SurveyLookup
        self.owner, self.origin, self.pattern, self.cast = owner, np.asarray(origin), pattern, cast
        self.spec = route(pattern)
        self.points, self.shape = survey_points(self.origin, pattern)
        # Corridor routes survey (and guard against) only the cells near the route.
        self.mask = survey_mask(self.origin, pattern, self.points)
        self.lookup_index = np.arange(len(self.points)) if self.mask is None else np.flatnonzero(self.mask)
        self.survey_lookup = SurveyLookup(self.points[self.lookup_index])
        # Declared demo rocks are exempt from the preflight survey only; the per-step
        # guard below still vetoes any collision with them.
        self.exempt = tuple(self.spec.get("survey_exempt_prefixes", ()))
        self.heights = []
        self.survey_failures = []
        self.normals = []
        self.survey_done = False
        self.publisher = None
        self.closed = threading.Event()
        self.lock = threading.RLock()
        self.failure = None
        self.armed = False
        self.sent_motion = False
        self.last_wall = time.monotonic()
        self.start = self.leg_start = self.settled = self.finished = None
        self.index = 0
        self.segment_start = None
        self.progress = None
        self.reached = []
        self.log = (owner.run / "motion.jsonl").open("x")
        self.thread = threading.Thread(target=self.watchdog, daemon=True)
        self.thread.start()

    def survey(self):
        from natural_motion import MAX_SLOPE, ground_sample, validate_ground_tile
        if self.survey_done:
            return
        # Bounded collision work on the simulator thread, before any motion lease.
        casts = 0
        while casts < 100 and len(self.heights) < len(self.points):
            xy = self.points[len(self.heights)]
            if self.mask is not None and not self.mask[len(self.heights)]:
                self.heights.append(float("nan"))
                self.normals.append([float("nan")]*3)
                continue
            casts += 1
            hits = self.cast((float(xy[0]), float(xy[1]), float(self.origin[2, 3]+2)), (0., 0., -1.), 4.)
            if self.exempt:
                hits = [h for h in hits if not h[3].startswith(self.exempt)]
            if self.spec.get("survey_allows_obstacles", False):
                # Evaluator-only underlying floor reference. Obstacles remain
                # in sensor data and the command-time FIRST-hit/overlap veto.
                hits = [h for h in hits if h[3].startswith("/Lunaryard/Terrain/")]
            nonrobot = [h for h in hits if not h[3].startswith("/Robots/")]
            self.normals.append(nonrobot[0][2] if nonrobot else [float("nan")]*3)
            try:
                # Corridor routes: floor geometry is recorded, not gated (rocks still are).
                self.heights.append(ground_sample(hits, max_slope=None if self.mask is not None else MAX_SLOPE))
            except ValueError as exc:
                self.heights.append(nonrobot[0][1][2] if nonrobot else float("nan"))
                self.survey_failures.append(dict(xy=xy.tolist(), reason=str(exc), hits=nonrobot[:3]))
        if len(self.heights) != len(self.points):
            return
        np.savez_compressed(self.owner.run / "clearance_tile.npz", xy=self.points, z=self.heights, normal=self.normals,
                            **({} if self.mask is None else dict(mask=self.mask, shape=np.asarray(self.shape))))
        if self.survey_failures:
            with (self.owner.run / "clearance.json").open("x") as stream:
                json.dump(dict(passed=False, origin=self.origin.tolist(), pattern=self.pattern,
                    failures=self.survey_failures, samples=len(self.points),
                    tile_sha256=hashlib.sha256((self.owner.run / "clearance_tile.npz").read_bytes()).hexdigest()), stream, indent=2)
            raise ValueError(f"swept floor survey rejected {len(self.survey_failures)}/{len(self.points)} samples")
        result = validate_ground_tile(self.points, self.heights, self.shape, self.normals, self.mask)
        lo, hi = self.points.min(axis=0), self.points.max(axis=0)
        low, high = float(np.nanmin(self.heights)), float(np.nanmax(self.heights))
        if self.mask is not None:
            from natural_motion import corridor_boxes, validate_demo_clearance
            for center, half, angle in corridor_boxes(self.origin, self.pattern):
                self.overlap(center, half, angle, low, high, exempt=self.exempt)
            if self.exempt:
                demo = getattr(self.owner, "demo_rocks", None)
                if not demo or not demo.get("objects"):
                    raise ValueError("survey exemption requires the declared demo rocks")
                result["demo_clearance"] = validate_demo_clearance(self.origin, self.pattern, demo["objects"])
            self.guard(self.origin)
        elif not self.spec.get("survey_allows_obstacles", False):
            self.overlap((lo+hi)/2, (hi-lo)/2, 0., low, high)
        else:
            self.guard(self.origin)  # The initial footprint must still be clear.
        with (self.owner.run / "clearance.json").open("x") as stream:
            json.dump(dict(passed=True, origin=self.origin.tolist(), pattern=self.pattern,
                tile_sha256=hashlib.sha256((self.owner.run / "clearance_tile.npz").read_bytes()).hexdigest(),
                **result), stream, indent=2)
        # Completed survey values are immutable; do not keep thousands of
        # nested per-hit Python lists on the simulator's GC-traversed heap.
        self.heights = np.asarray(self.heights)
        self.normals = np.asarray(self.normals)
        self.heights.flags.writeable = self.normals.flags.writeable = False
        from geometry_msgs.msg import Twist
        self.publisher = self.owner.node.create_publisher(Twist, "/astrobot_0/cmd_vel", 1)
        self.survey_done = True

    def overlap(self, xy, half, angle, low, high, exempt=()):
        collisions, errors = [], []
        def report(hit):
            try:
                path = str(hit.collision)
                if not path.startswith(("/Robots/astrobot_0/", "/Lunaryard/Terrain/", *exempt)):
                    collisions.append(path)
                return True
            except Exception as exc:
                errors.append(str(exc))
                return False
        self.owner.query.overlap_box((float(half[0]), float(half[1]), (high-low+1.)/2),
            (float(xy[0]), float(xy[1]), (high+low+1.)/2),
            (0., 0., math.sin(angle/2), math.cos(angle/2)), report, False)
        if collisions or errors:
            raise ValueError(f"natural swept-volume collision: {collisions or errors}")

    def guard(self, current):
        from natural_motion import GRID, HALF, MAX_SLOPE, ground_sample, yaw
        angle = yaw(current)
        ground = []
        for x in (-HALF[0], 0., HALF[0]):
            for y in (-HALF[1], 0., HALF[1]):
                xy = current[:2, 3] + np.array([x*math.cos(angle)-y*math.sin(angle), x*math.sin(angle)+y*math.cos(angle)])
                nearest = self.lookup_index[self.survey_lookup.nearest(xy)]
                if self.mask is not None and np.linalg.norm(self.points[nearest]-xy) > GRID:
                    raise ValueError("footprint left surveyed corridor")
                height = ground_sample(self.cast((float(xy[0]), float(xy[1]), float(current[2, 3]+2)), (0., 0., -1.), 4.),
                                       max_slope=None if self.mask is not None else MAX_SLOPE)
                normal = np.asarray(self.normals[nearest])
                predicted = self.heights[nearest] - (normal[:2]/normal[2])@(xy-self.points[nearest])
                if abs(height-predicted) > .025:
                    raise ValueError("support changed since preflight")
                ground.append(height)
        self.overlap(current[:2, 3], HALF, angle, min(ground), max(ground))

    def publish(self, linear=0., angular=0.):
        from geometry_msgs.msg import Twist
        with self.lock:
            if self.publisher is None:
                if linear or angular: raise ValueError("motion before clearance")
                return
            message = Twist()
            if linear or angular:
                if self.failure or not self.owner.lease.ready(self.owner.stamp, time.monotonic()):
                    self.publisher.publish(message)
                    raise ValueError(self.failure or "motion before lease")
                self.sent_motion = True
            message.linear.x, message.angular.z = float(linear), float(angular)
            self.publisher.publish(message)

    def watchdog(self):
        from shadow_capture_gate import HEARTBEAT_S  # sim ticks gapped up to 1.4 s on transit50
        while not self.closed.wait(.05):
            if self.armed:
                try:
                    if time.monotonic()-self.last_wall > HEARTBEAT_S:
                        raise ValueError("motion simulation heartbeat expired")
                    if not self.owner.lease.ready(self.owner.stamp, time.monotonic()):
                        raise ValueError("motion lease expired")
                except Exception as exc:
                    with self.lock:
                        self.failure = str(exc)
                        self.publish()

    def step(self, stamp, current):
        from natural_motion import WAYPOINT_LOOPS, motion_command, leg_deadline
        self.last_wall = time.monotonic()
        if self.failure: raise ValueError(self.failure)
        self.guard(current)
        if self.start is None:
            self.start = stamp
        if stamp-self.start < self.spec["pre_roll_s"]:
            self.publish()
            return False
        self.armed = True
        if self.index == len(self.spec["targets"]):
            self.publish()
            if self.finished is None: self.finished = stamp
            return stamp-self.finished >= self.spec["post_roll_s"]
        if self.leg_start is None:
            self.leg_start = stamp
            self.progress = (stamp, current.copy())
            self.segment_start = (current.copy() if self.pattern in WAYPOINT_LOOPS and
                                  self.spec["targets"][self.index][3] != "rotate" else None)
        commanded_start = self.segment_start
        linear, angular, arrived = motion_command(self.origin, current, self.pattern, self.index,
                                                  segment_start=commanded_start)
        commanded_index = self.index
        if stamp-self.leg_start > leg_deadline(self.pattern, self.index):
            raise TimeoutError("bounded natural motion leg deadline")
        if arrived:
            self.publish()
            if self.settled is None: self.settled = stamp
            if stamp-self.settled >= self.spec["settle_s"]:
                self.reached.append(dict(index=self.index, stamp=stamp, world_from_base=current.tolist(),
                    segment_start=None if commanded_start is None else commanded_start.tolist()))
                self.index += 1
                self.leg_start = self.settled = None
        else:
            self.settled = None
            if stamp-self.progress[0] >= 1.5:
                old = self.progress[1]
                movement = np.linalg.norm(current[:3, 3]-old[:3, 3])
                rotation = np.linalg.norm(current[:3, :3]-old[:3, :3])
                if movement < .005 and rotation < .01:
                    raise ValueError("natural motion stalled")
                self.progress = (stamp, current.copy())
            self.publish(linear, angular)
        self.log.write(json.dumps(dict(stamp=stamp, index=self.index, linear=0. if arrived else linear,
            angular=0. if arrived else angular, world_from_base=current.tolist(),
            segment_start=None if commanded_start is None else commanded_start.tolist(),
            arrived_segment=commanded_index if arrived else None))+"\n")
        return False

    def stop(self):
        self.publish()
        self.armed = False
        self.closed.set()
        self.log.close()
