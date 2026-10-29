"""Opt-in seed-43 natural fixture; motion requires a separate declared guarded route."""
import hashlib
import gc
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


def seal_moving_capture(run, request, first_stamp, last_stamp, native, driver, *, announced=False, finalize=None):
    """Stop motion, seal native data, then atomically expose completion."""
    from .terrain_calibration_drive import publish_completion_marker
    if not announced:
        driver.stop()
        publish_completion_marker(run / "acquisition_finished.json",
            dict(schema=1, capture=run.name, views=0, commands_stopped=True, stamp=last_stamp))
    try:
        native.close()
    finally:
        if finalize is not None:
            finalize()
    if (run / "failure.json").exists():
        raise ValueError("cannot complete a failed capture")
    publish_completion_marker(run / "natural_holdout.json",
        dict(complete=True, request=request, chunks=len(native.frames),
            first_stamp=first_stamp, last_stamp=last_stamp, command_publishers=1,
            reached=driver.reached, stationary_only=False, validation="NOT_RUN"))


class NaturalHoldoutCapture:
    def __init__(self, controller):
        import rclpy
        import omni.usd
        from omni.physx import get_physx_scene_query_interface
        from pxr import Gf, UsdGeom
        from std_msgs.msg import String
        from .terrain_gt_probe import TerrainGTProbe
        from .terrain_sensor_audit import TerrainSensorAudit, graph_contract

        sys.path.insert(0, "/workspace/terrain-tools")
        from archive_guard import allocate_run
        from natural_holdout import SEED
        self.motion_pattern = os.getenv("OMNILRS_NATURAL_MOTION", "")
        from demo_rock_profile import validate_profile
        self.demo_profile = validate_profile(os.getenv("OMNILRS_DEMO_ROCK_PROFILE", ""), self.motion_pattern)
        self.demo_rocks = self.demo_declaration = self.demo_created_stamp = None
        from natural_motion import execution_budget
        self.execution_budget = execution_budget(
            self.motion_pattern, os.getenv("OMNILRS_CAPTURE_EXECUTION_PROFILE", "default"))
        self.driver = None
        if self.motion_pattern:
            from natural_motion import route
            route(self.motion_pattern)

        if int(os.getenv("OMNILRS_ENVIRONMENT_SEED", "42")) != SEED:
            raise ValueError("natural holdout requires the predeclared independent seed 43")
        if os.getenv("OMNILRS_ROCKS_ENABLED", "").lower() != "true":
            raise ValueError("natural rock holdout requires explicit rocks enabled")
        if any(os.getenv(k, "false").lower() == "true" for k in
               ("OMNILRS_TERRAIN_GT_PROBE", "OMNILRS_TERRAIN_CALIBRATION", "OMNILRS_TERRAIN_DRIVE", "OMNILRS_GT_TF_ENABLED")):
            raise ValueError("natural holdout forbids calibration, motion and GT TF")
        self.controller, self.stage = controller, omni.usd.get_context().get_stage()
        self.query = get_physx_scene_query_interface()
        if abs(UsdGeom.GetStageMetersPerUnit(self.stage) - 1.) > 1e-9:
            raise ValueError("holdout requires meters")
        from costmap_scenarios import archive_limit
        self.run = allocate_run("/archive", limit=archive_limit(self.motion_pattern, self.execution_budget["profile"]))
        # Require actual explicitly instanced scene geometry, not only a flag.
        from .gt_map_ros2 import _explicit_rock_instance_root
        from pxr import Usd
        roots = sorted({root for prim in self.stage.Traverse()
                        if (root := _explicit_rock_instance_root(str(prim.GetPath()))) is not None})
        if not roots:
            raise ValueError("natural scene has no explicit rock instances")
        bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        inventory = []
        for root in roots:
            box = bounds.ComputeWorldBound(self.stage.GetPrimAtPath(root)).ComputeAlignedBox()
            inventory.append(dict(path=root, minimum=list(map(float, box.GetMin())),
                                  maximum=list(map(float, box.GetMax()))))
        with (self.run / "natural_scene.json").open("x") as stream:
            json.dump(dict(seed=SEED, objects=inventory, source="USD explicit instance world bounds",
                           note="Evaluator-only geometry; first-hit PhysX agreement must pass independently."), stream, indent=2)
        self.pose_method = TerrainGTProbe._pose
        robot = next(iter(controller.robotManager.robots.values()))
        legacy = robot._find_first_descendant_of_type("Lidar")
        mount = legacy.GetParent()
        rotation = mount.GetAttribute("xformOp:rotateXYZ")
        if (mount.GetName() != "vlp16" or mount.GetParent().GetName() != "base_link"
                or mount.GetAppliedSchemas() or not rotation or not np.allclose(rotation.Get(), [0, 0, 0], atol=1e-9)):
            raise ValueError("unexpected natural holdout mount; refusing compounded rotation")
        selected = os.getenv("OMNILRS_SELECTED_LIDAR", "false").lower() == "true"
        if selected and not self.motion_pattern:
            raise ValueError("selected sensor requires guarded natural motion")
        if selected and os.getenv("OMNILRS_TERRAIN_NATIVE_PUBLISH", "false").lower() == "true":
            from selected_lidar_profile import declaration
            if declaration()["name"] != "selected_optical_v2":
                raise ValueError("selected terrain publication requires corrected optical-origin profile")
        rotation.Set(Gf.Vec3f(0., 0. if selected else 5., 0.))
        if selected:
            from selected_lidar_profile import profile_path, selected_setup
            setup, _ = selected_setup(profile_path())
            translation = mount.GetAttribute("xformOp:translate")
            if not translation or not np.allclose(translation.Get(), [.4, 0, .25], atol=1e-7, rtol=0):
                raise ValueError("unexpected authored LiDAR translation")
            translation.Set(type(translation.Get())(*setup["base_from_lidar_translation_m"]))
        self.node = rclpy.create_node("natural_holdout_capture")
        self.selected_lidar = None
        if selected:
            from .selected_lidar_capture import SelectedLidar
            try:
                self.selected_lidar = SelectedLidar(self, robot, legacy, mount)
            except Exception as exc:
                with (self.run / "failure.json").open("x") as stream:
                    json.dump(dict(error=str(exc), phase="selected_sensor_construction", motion_commands_sent=False), stream)
                print(f"NATURAL_HOLDOUT FAILED {exc}; {self.run}", flush=True)
                raise
        self.audit = self.selected_lidar or TerrainSensorAudit(self.run, frame_limit=2, require_tf=True, ray_budget=200)
        self.sensor_audit_sha256 = None
        from rclpy.executors import SingleThreadedExecutor
        # Do not share spin_once's global executor with the sensor-TF auditor.
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.publisher = self.node.create_publisher(String, "/terrain_shadow/request", 1)
        self.String = String
        from lease_process import LeaseProcess
        self.lease = LeaseProcess(self.run.name, command=["/isaac-sim/python.sh"])
        self.failure = None
        self.stamp = 0.
        # Receipt runs in a fresh interpreter: Isaac GC pauses threads too.
        # No USD/PhysX access, commands or independent permission in the child.
        self.lease_closed = False
        # Terrain proposals still need their original independent callbacks;
        # only acquisition grants moved out of process.
        self.lease_node = rclpy.create_node("natural_holdout_proposal_receiver")
        self.lease_executor = SingleThreadedExecutor()
        self.lease_executor.add_node(self.lease_node)
        self.lease_callbacks = None
        self.started = self.initial_pose = self.request = self.native = self.capture_started = None
        self.done = False
        self.sealing = None
        self.wall_started = time.monotonic()
        self.last_request = -float("inf")
        self.last_tick_wall = None
        self.max_tick_wall_gap = 0.
        self.phase = "initialization"
        self.gc_started = {}
        self.gc_pauses = []
        self.max_native_wall = 0.
        from simulation_stage_timing import StageTiming
        self.stage_timing = StageTiming()
        self.forced_gc_ticks = []
        gc.callbacks.append(self.gc_event)
        with (self.run / "graph_contract.json").open("x") as stream:
            json.dump(graph_contract(self.stage, robot.robot_path), stream, indent=2)
        print(f"NATURAL_HOLDOUT run={self.run}", flush=True)
        from lease_callbacks import LeaseCallbacks
        self.lease_callbacks = LeaseCallbacks(self.lease_executor, self.lease_callback_failure)

    def lease_callback_failure(self, error):
        self.failure = self.failure or error

    def gc_event(self, phase, info):
        generation = info["generation"]
        if phase == "start":
            self.gc_started[generation] = time.monotonic()
        elif generation in self.gc_started:
            elapsed = time.monotonic()-self.gc_started.pop(generation)
            if elapsed > .05 and len(self.gc_pauses) < 100:
                import traceback
                self.gc_pauses.append(dict(generation=generation, wall_s=elapsed, stamp=self.stamp, phase=self.phase,
                    collected=info.get("collected"), stack=[f"{s.filename}:{s.lineno}:{s.name}"
                                                          for s in traceback.extract_stack(limit=10)]))

    def close_diagnostics(self):
        if self.lease_callbacks is not None:
            self.lease_callbacks.close()
            self.lease_callbacks = None
            self.lease_executor.remove_node(self.lease_node)
            self.lease_node.destroy_node()
        if not self.lease_closed:
            self.lease.close()
            self.lease_closed = True
        if self.gc_event in gc.callbacks:
            gc.callbacks.remove(self.gc_event)
        with (self.run / "acquisition_timing.json").open("x") as stream:
            json.dump(getattr(self, "seal_timing_snapshot", None) or self.timing_report(), stream, indent=2)

    def timing_report(self):
        return dict(max_tick_wall_gap=self.max_tick_wall_gap, max_native_wall=self.max_native_wall,
                           gc_pauses=self.gc_pauses,
                           forced_gc_ticks=self.forced_gc_ticks,
                           gt_grid_publisher_enabled=os.getenv("OMNILRS_GT_MAP_ENABLED", "true").lower() == "true",
                           resource_map_enabled=os.getenv("OMNILRS_RESOURCE_ENABLED", "true").lower() == "true",
                           stage_timing=self.stage_timing.report(),
                           live_latency_acceptance=False)

    def timed_call(self, name, function, *args, **kwargs):
        return self.stage_timing.call(name, self.stamp, function, *args, **kwargs)

    def check_stationary_audit(self):
        # This is a completed, write-once acquisition artifact, not live sensor
        # state. Re-decoding its nested hit records every tick provokes costly
        # simulator-wide full GC. Final evaluator checks this exact saved hash.
        if self.sensor_audit_sha256 is None:
            payload = (self.run / "sensor_audit.json").read_bytes()
            result = json.loads(payload)
            if result.get("passed") is not True or result.get("extrinsic_tf_checked") is not True:
                raise ValueError("natural stationary sensor/TF surface audit failed")
            self.sensor_audit_sha256 = hashlib.sha256(payload).hexdigest()

    def pre_step(self):
        if self.selected_lidar is not None and self.failure is None:
            try:
                self.selected_lidar.pre_step()
            except Exception as exc:
                self.failure = str(exc)

    def tick(self, stamp):
        if self.done and self.failure:
            return
        try:
            if self.failure:
                raise ValueError(self.failure)
            if self.selected_lidar is not None:
                self.selected_pose_tick = (stamp, self.timed_call("selected_pose", self.pose_method, self))
                self.timed_call("selected_sensor", self.selected_lidar.tick, stamp, self.selected_pose_tick[1][3])
            if self.sealing is not None:
                if self.sealing.poll():
                    self.sealing = None
                    self.done = True
                    print(f"NATURAL_HOLDOUT COMPLETE {self.run}", flush=True)
                return  # Never acquire another native tick or resume commands.
            if self.done:
                return
            self._tick(stamp)
        except Exception as exc:
            self.failure = str(exc)
            self.done = True
            driver = getattr(self, "driver", None)
            if driver is not None:
                try:
                    driver.stop()  # Stop before diagnostics/archive work.
                except Exception as stop_error:
                    print(f"NATURAL_HOLDOUT stop failed: {stop_error}", flush=True)
            print(f"NATURAL_HOLDOUT FAILED {exc}; {self.run}", flush=True)
            # Snapshot at the fault, before any potentially slow archive flush.
            failure = dict(error=str(exc), stamp=stamp, motion_commands_sent=bool(driver and driver.sent_motion),
                phase=self.phase, max_tick_wall_gap=self.max_tick_wall_gap,
                lease_sequence=self.lease.sequence, lease_sim_stamp=self.lease.sim_stamp,
                lease_wall_age=None if self.lease.wall_stamp is None else time.monotonic()-self.lease.wall_stamp)
            try:
                with (self.run / "failure.json").open("x") as stream:
                    json.dump(failure, stream, indent=2)
            except Exception as archive_error:
                print(f"NATURAL_HOLDOUT failure marker write failed: {archive_error}", flush=True)
            # A sealing worker owns these files/receivers until it returns. Do
            # not race a second close against it after a deadline or tick fault.
            if self.sealing is not None:
                return
            for cleanup in (self.close_diagnostics,
                    self.native.close if self.native is not None and not (self.run / "native_timing_audit.json").exists() else lambda: None):
                try:
                    cleanup()
                except Exception as archive_error:
                    print(f"NATURAL_HOLDOUT archive cleanup failed: {archive_error}", flush=True)

    def _tick(self, stamp):
        from pxr import Usd, UsdGeom
        from archive_guard import check_run
        from natural_holdout import PATTERN, SEED, DURATION, HOME, stationary_error, seal_capture
        from .terrain_gt_probe import raycast_hits
        from .terrain_native_capture import NativeTerrainCapture

        self.stamp = stamp
        wall = time.monotonic()
        if self.last_tick_wall is not None:
            self.max_tick_wall_gap = max(self.max_tick_wall_gap, wall-self.last_tick_wall)
        self.last_tick_wall = wall
        self.phase = "grant_callbacks"
        # Bounded draining; stale grants still cannot revive an expired lease.
        for _ in range(8):
            self.executor.spin_once(timeout_sec=0.)
        if self.failure:
            raise ValueError(self.failure)
        publishers = self.node.get_publishers_info_by_topic("/astrobot_0/cmd_vel")
        if publishers and (not self.motion_pattern or len(publishers) != 1 or publishers[0].node_name != self.node.get_name()):
            raise ValueError("natural capture forbids other command publishers")
        from natural_motion import route
        if time.monotonic() - self.wall_started > self.execution_budget["max_wall_s"]:
            raise TimeoutError("natural holdout wall deadline")
        if self.started is None:
            self.started = stamp
        if stamp - self.started < 8.:
            return
        self.phase = "pose_and_sensor_audit"
        robot, xyz, q, pose = (self.selected_pose_tick[1] if self.selected_lidar is not None
                              and self.selected_pose_tick[0] == stamp else self.pose_method(self))
        if self.initial_pose is None:
            self.initial_pose = pose["world_from_base"]
        if not self.motion_pattern or self.capture_started is None:
            stationary_error(self.initial_pose, pose["world_from_base"])
        cast = lambda *args: raycast_hits(self.query, *args, both_sides=True)
        if self.demo_profile:
            import demo_rock_profile
            if self.demo_rocks is None:
                self.demo_rocks = demo_rock_profile.create(self.stage, cast, pose["world_from_base"], self.run,
                                                          profile=self.demo_profile)
                self.demo_created_stamp = stamp
                return
            if stamp - self.demo_created_stamp < 0.5:
                return
            if self.demo_declaration is None:
                self.demo_declaration = demo_rock_profile.verify(self.demo_rocks, cast, self.run)
        self.audit.observe(stamp, pose, cast)
        if not self.audit.done:
            return
        self.check_stationary_audit()
        if self.motion_pattern:
            if self.driver is None:
                from costmap_scenarios import TRIAL_PATTERNS
                if self.motion_pattern in TRIAL_PATTERNS:
                    from .terrain_step_drive import TerrainStepDrive
                    self.driver = TerrainStepDrive(self, self.initial_pose, self.motion_pattern, cast)
                else:
                    from .natural_motion_drive import NaturalMotionDrive
                    self.driver = NaturalMotionDrive(self, self.initial_pose, self.motion_pattern, cast)
            self.driver.survey()
            if not self.driver.survey_done:
                return
        if self.request is None:
            if self.motion_pattern:
                from .natural_motion_drive import stationary_heap_checkpoint
                self.phase = "preflight_heap_checkpoint"
                checkpoint = stationary_heap_checkpoint(self.lease.sequence, self.driver.sent_motion, self.capture_started)
                with (self.run / "heap_checkpoint.json").open("x") as stream:
                    json.dump(checkpoint, stream, indent=2)
                if os.getenv("OMNILRS_HEAP_PROFILE_ONLY", "false").lower() == "true":
                    from simulator_heap_profile import profile_heap
                    profile = profile_heap(self.lease.sequence, self.driver.sent_motion, self.capture_started)
                    with (self.run / "heap_profile.json").open("x") as stream:
                        json.dump(profile, stream, indent=2)
                    # Diagnostic ends before request publication, estimator launch,
                    # native capture or a motion grant. Never resume this run.
                    raise RuntimeError("stationary heap diagnostic complete; no motion requested")
                self.phase = "request_initialization"
            imu = robot._find_descendants_by_name(["Imu_Sensor"])["Imu_Sensor"]
            world_imu = np.asarray(UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(imu)).T
            self.request = dict(schema=1, capture=self.run.name, pattern=PATTERN, seed=SEED,
                pitch_deg=0. if self.selected_lidar else 5., duration_s=DURATION, home_xyz=HOME, motion_authority=False,
                base_from_lidar=pose["base_from_lidar"],
                base_from_imu=(np.linalg.inv(pose["world_from_base"]) @ world_imu).tolist())
            if self.selected_lidar is not None:
                from selected_lidar_profile import declaration
                self.request["lidar_profile"] = declaration()
                from costmap_scenarios import SCENARIOS
                from selected_lidar_profile import EXTENDED_PATTERNS
                if self.motion_pattern in EXTENDED_PATTERNS:
                    from selected_lidar_profile import acquisition_limits
                    self.request["acquisition_limits"] = acquisition_limits(self.motion_pattern)
                self.request["acquisition_resources"] = dict(
                    gt_grid=os.getenv("OMNILRS_GT_MAP_ENABLED", "true").lower() == "true",
                    resource_map=os.getenv("OMNILRS_RESOURCE_ENABLED", "true").lower() == "true",
                    gt_pose_every_physics_tick=True)
            if self.motion_pattern:
                from natural_motion import route_hash
                self.request.update(pattern=self.motion_pattern, motion_authority=True,
                    route_sha256=route_hash(self.motion_pattern), world_from_base=pose["world_from_base"], seed_stamp=stamp)
                if self.execution_budget["profile"] != "default":
                    self.request["execution_budget"] = dict(self.execution_budget)
                self.request.pop("duration_s")
            if os.getenv("OMNILRS_TERRAIN_NATIVE_PUBLISH", "false").lower() == "true":
                self.request["native_shadow_publishing"] = True
            if self.demo_declaration is not None:
                self.request["demo_rocks"] = self.demo_declaration
            sources = [Path(__file__), Path(robot.usd_path),
                       Path(__file__).parents[1] / "__init__.py",
                       Path(__file__).parents[2] / "terrain_management/terrain_manager.py"]
            import archive_guard
            sources.append(Path(archive_guard.__file__))
            import simulation_stage_timing
            sources.append(Path(simulation_stage_timing.__file__))
            if self.demo_profile:
                import demo_rock_profile
                sources.append(Path(demo_rock_profile.__file__))
            import lease_process
            import lease_callbacks
            import shadow_capture_gate
            sources.extend([Path(lease_process.__file__), Path(lease_callbacks.__file__), Path(shadow_capture_gate.__file__)])
            if self.motion_pattern:
                import natural_motion
                sources.extend([Path(natural_motion.__file__), Path(__file__).with_name("natural_motion_drive.py")])
                from costmap_scenarios import TRIAL_PATTERNS, COSTMAP_PATTERNS
                import costmap_scenarios
                sources.append(Path(costmap_scenarios.__file__))
                if self.motion_pattern in TRIAL_PATTERNS:
                    import local_terrain.step_trial
                    sources.extend([Path(__file__).with_name("terrain_step_drive.py"), Path(local_terrain.step_trial.__file__)])
                    if self.motion_pattern in COSTMAP_PATTERNS:
                        import local_terrain.costmap_trial
                        sources.append(Path(local_terrain.costmap_trial.__file__))
            if self.selected_lidar is not None:
                import native_archive_io
                import capture_seal
                sources.append(Path(capture_seal.__file__))
                sources.append(Path(native_archive_io.__file__))
                sources.extend([Path(__file__).with_name(n) for n in
                                ("gt_map_ros2.py", "simulation_manager_ros2.py", "lab_manager_factory.py",
                                 "selected_lidar_capture.py", "terrain_native_capture.py")])
            startup = dict(experience=os.getenv("OMNILRS_SELECTED_EXPERIENCE", "default"),
                           import_stat_cache=os.getenv("OMNILRS_IMPORT_STAT_CACHE", "true"))
            if startup["experience"] == "lean_v1":
                experience = Path(os.environ["OMNILRS_PREPARED_EXPERIENCE"])
                sources.extend([experience, experience.with_name("experience.json")])
                startup["experience_path"] = str(experience)
            with (self.run / "manifest.json").open("x") as stream:
                json.dump(dict(schema=1, experiment=PATTERN, seed=SEED, sensor=pose,
                    simulator_startup=startup,
                    sensor_audit_sha256=self.sensor_audit_sha256,
                    root_layer=self.stage.GetRootLayer().identifier,
                    source_asset_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sources},
                    terrain_labels_in_estimator=False, stationary_only=not bool(self.motion_pattern)), stream, indent=2)
        if self.motion_pattern or time.monotonic() - self.last_request >= .2:
            # Each real simulator tick is evidence of progress. A 5 Hz wall
            # throttle unnecessarily adds up to 200 ms to the observed gap.
            # Never publish from a timer that could outlive stalled physics.
            self.publisher.publish(self.String(data=json.dumps(dict(self.request, stamp=stamp))))
            self.last_request = time.monotonic()
        self.phase = "lease_validation"
        if not self.lease.ready(stamp, time.monotonic()):
            return
        if self.native is None:
            if self.selected_lidar is not None:
                from .selected_lidar_capture import SelectedNativeArchive
                self.native = SelectedNativeArchive(self.selected_lidar)
            else:
                self.native = NativeTerrainCapture(self.run, pose["lidar_path"], retain_hits=True, node=self.node)
            self.capture_started = stamp
            with (self.run / "shadow_link.json").open("x") as stream:
                json.dump(dict(request=self.request, first_grant=self.lease.first_grant), stream, indent=2)
        self.phase = "native_capture"
        native_wall = time.monotonic()
        if len(self.native.frames) in self.execution_budget.get("forced_gc_after_native_ticks", []):
            self.phase = "native_capture_forced_gc_stress"
            self.forced_gc_ticks.append(dict(after_native_ticks=len(self.native.frames), stamp=stamp))
            gc.collect(2)  # Explicit stress protocol; never a nominal capture or GC suppression.
            self.phase = "native_capture"
        self.timed_call("native_audit_archive", self.native.observe, stamp, pose, cast)
        self.max_native_wall = max(self.max_native_wall, time.monotonic()-native_wall)
        if len(self.native.frames) % 30 == 0:
            self.timed_call("archive_check", check_run, self.run)
        finished = (self.timed_call("guarded_driver", self.driver.step, stamp, np.asarray(pose["world_from_base"])) if self.motion_pattern
                    else stamp-self.capture_started >= DURATION)
        if finished:
            if self.motion_pattern and self.selected_lidar is not None:
                from capture_seal import BackgroundSeal
                from .terrain_calibration_drive import publish_completion_marker
                self.driver.stop()
                publish_completion_marker(self.run / "acquisition_finished.json",
                    dict(schema=1, capture=self.run.name, views=0, commands_stopped=True, stamp=stamp))
                # Publish the exact last-tick boundary on this thread before
                # any slow file work. Real clock/scan ticks continue so GLIM and
                # the mapper can drain their original delayed pose brackets.
                self.native.announce_complete()
                import copy
                self.seal_timing_snapshot = copy.deepcopy(self.timing_report())
                self.seal_timing_snapshot["post_motion_sealing_off_physics_thread"] = True
                deadline = time.monotonic() + 20.0  # Existing recorder sealing budget.
                def finalize():
                    self.close_diagnostics()
                    if time.monotonic() > deadline:
                        raise TimeoutError("post-motion archive sealing deadline exceeded")
                self.sealing = BackgroundSeal(lambda: seal_moving_capture(
                    self.run, self.request, self.capture_started, stamp,
                    self.native, self.driver, announced=True, finalize=finalize))
                return
            # Announce measurement completion BEFORE synchronous archive sealing.
            # Otherwise a large JSON flush pauses the simulation clock and can
            # trip the recorder's live-input watchdog on already-finished data.
            if self.motion_pattern:
                seal_moving_capture(self.run, self.request, self.capture_started, stamp,
                                    self.native, self.driver)
            else:
                seal_capture(self.run, self.request, self.capture_started, stamp, self.native)
            self.close_diagnostics()
            self.done = True
            print(f"NATURAL_HOLDOUT COMPLETE {self.run}", flush=True)
