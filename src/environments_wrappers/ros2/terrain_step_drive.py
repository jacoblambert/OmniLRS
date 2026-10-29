"""One explicitly bounded terrain-controlled simulator trial, not production nav.

Reuse the independent clearance survey, footprint/floor checks, lease, sole
publisher and watchdog. GT can STOP the experiment; it never selects a command.
"""

import json
import math
import time
from functools import partial

import numpy as np

from .natural_motion_drive import NaturalMotionDrive


class TerrainStepDrive(NaturalMotionDrive):
    def __init__(self, owner, origin, pattern, cast):
        super().__init__(owner, origin, pattern, cast)
        from std_msgs.msg import String
        from local_terrain.step_trial import TOPIC
        from costmap_scenarios import SCENARIOS, COSTMAP_PATTERNS

        if pattern in COSTMAP_PATTERNS:
            from local_terrain.costmap_trial import validate_proposal
        else:
            from local_terrain.step_trial import validate_proposal
        self.validate = partial(validate_proposal, scenario=pattern) if pattern in SCENARIOS else validate_proposal
        self.displacement_bound = self.spec["goal_distance_m"] + 0.5 if pattern in COSTMAP_PATTERNS else 0.5

        self.proposal = None
        self.proposal_wall = None
        self.proposal_sequence = 0
        self.reference = None
        self.last_revision = -1
        self.outcome = None
        self.mode_start = self.mode = None
        self.command_stamp = None
        self.command_active = False
        self.command_elapsed = 0.0
        self.turn_guard_id = self.turn_guard_pose = self.turn_guard_anchor = None
        self.turn_sample_sequence = self.turn_sample_pose = None
        # Like acquisition grants, proposals must be received while native
        # capture is doing synchronous work on the simulation thread. This
        # callback stores a proposal only; commands still require step().
        self.subscription = owner.lease_node.create_subscription(String, TOPIC, self.receive_proposal, 1)

    def receive_proposal(self, message):
        received = time.monotonic()
        with self.lock:
            self._receive_proposal(message, received)

    def _receive_proposal(self, message, received):
        from local_terrain.step_trial import TOPIC

        try:
            publishers = self.owner.node.get_publishers_info_by_topic(TOPIC)
            if len(publishers) != 1 or publishers[0].node_name != "terrain_shadow_mapper":
                raise ValueError(
                    f"trial requires sole launched terrain_shadow_mapper publisher; got "
                    f"{[p.node_name for p in publishers]}"
                )
            value = json.loads(message.data)
            if value.get("valid") is False and self.proposal is None and not self.armed:
                if value.get("linear") != 0.0 or value.get("angular") != 0.0:
                    raise ValueError("invalid startup proposal requested motion")
                self.publish()
                return  # Expected all-unknown startup; never authorizes motion.
            if value.get("valid") is False:
                raise ValueError(f"terrain proposal invalidated: {value.get('reason')}")
            if self.owner.lease.identity is None:
                raise ValueError("trial proposal before acquisition lease")
            self.validate(
                value,
                self.owner.run.name,
                self.owner.lease.identity,
                # ROS /clock is published inside world.step(), before the
                # capture callback updates owner.stamp. Receipt is not command
                # execution: validate intrinsic timestamps here; _step checks
                # the actual physics tick before granting any authority.
                value.get("query_ns"),
                previous_sequence=self.proposal_sequence,
            )
            reference = np.asarray(value["local_from_map"])
            if self.reference is not None and not np.array_equal(reference, self.reference):
                raise ValueError("trial local frame changed")
            if value["revision"] < self.last_revision:
                raise ValueError("trial revision regressed")
            self.reference, self.last_revision = reference, value["revision"]
            self.proposal_sequence = value["sequence"]
            self.proposal, self.proposal_wall = value, received
        except Exception as exc:
            self.failure = str(exc)
            self.publish()

    def step(self, stamp, current):
        with self.lock:
            return self._step(stamp, current)

    def _step(self, stamp, current):
        from natural_motion import yaw, wrap

        self.last_wall = time.monotonic()
        if self.command_stamp is not None:
            if stamp < self.command_stamp:
                raise ValueError("trial command clock reversed")
            if self.command_active:
                self.command_elapsed += stamp - self.command_stamp
        self.command_stamp, self.command_active = stamp, False
        if self.failure:
            raise ValueError(self.failure)
        self.guard(current)  # Independent GT veto, never path/command selection.
        if "position_bounds_initial_xy" in self.spec:
            from costmap_scenarios import check_position

            check_position(self.origin, current, self.spec["position_bounds_initial_xy"])
        elif np.linalg.norm(current[:2, 3] - self.origin[:2, 3]) > self.displacement_bound:
            raise ValueError("trial GT emergency displacement bound")
        if abs(wrap(yaw(current) - yaw(self.origin))) > self.spec.get("heading_bound_rad", 0.35):
            raise ValueError("trial GT emergency heading bound")
        if self.start is None:
            self.start = stamp
        if stamp - self.start < self.spec["pre_roll_s"]:
            self.publish()
            return False
        self.armed = True
        if self.finished is not None:
            self.publish()
            return stamp - self.finished >= self.spec["post_roll_s"]
        age = None if self.proposal_wall is None else time.monotonic() - self.proposal_wall
        if self.proposal is None or age > 0.5 or not 0 <= round(stamp * 1e9) - self.proposal["query_ns"] <= 200_000_000:
            self.publish()
            # A transport wait is not a new physical manoeuvre. Keep the frozen
            # turn/straight reference; otherwise repeated waits erase skid.
            if "position_bounds_initial_xy" not in self.spec:
                self.mode, self.progress = "hold", None
            self.record_execution(stamp, current, 0.0, 0.0, "waiting_for_fresh_proposal", age)
            return False  # No stale authority. Only a new checked proposal can resume.
        linear, angular = self.validate(
            self.proposal,
            self.owner.run.name,
            self.owner.lease.identity,
            round(stamp * 1e9),
            previous_sequence=self.proposal_sequence - 1,
        )
        if self.proposal["terminal"]:
            self.publish()
            self.finished, self.outcome = stamp, self.proposal["terminal"]
            (self.owner.run / "step_trial_outcome.json").write_text(
                json.dumps(
                    dict(
                        outcome=self.outcome,
                        proposal=self.proposal,
                        world_from_base=current.tolist(),
                        goal_reached_claim_is_estimator_only=True,
                        navigation_qualified=False,
                    ),
                    indent=2,
                )
            )
        else:
            turn_id = self.proposal.get("turn_id")
            if turn_id is not None:
                anchor = self.proposal["turn_origin_xy"]
                if self.turn_guard_id != turn_id:
                    if self.turn_guard_id is not None and turn_id <= self.turn_guard_id:
                        raise ValueError("trial turn identity regressed")
                    self.turn_guard_id, self.turn_guard_pose = turn_id, current.copy()
                    self.turn_guard_anchor = anchor
                elif anchor != self.turn_guard_anchor:
                    raise ValueError("trial turn anchor changed")
                # User-authorized v12 permits bounded cumulative turn drift.
                # Keep the original anchor across holds, never ratchet it.
                if np.linalg.norm(current[:2, 3] - self.turn_guard_pose[:2, 3]) > self.spec.get(
                    "turn_translation_limit_m", 0.25
                ):
                    raise ValueError("trial rotation skid exceeds declared turn limit")
            if angular and "turn_command_drift_allowance_m" in self.spec:
                # Check the previous command before accepting a new sequence;
                # a sequence change must not erase excess inter-query motion.
                if (
                    getattr(self, "turn_sample_pose", None) is not None
                    and self.mode == "rotate"
                    and np.linalg.norm(current[:2, 3] - self.turn_sample_pose[:2, 3])
                    > self.spec["turn_command_drift_allowance_m"]
                ):
                    raise ValueError("trial turn exceeds checked command drift allowance")
                if getattr(self, "turn_sample_sequence", None) != self.proposal_sequence:
                    self.turn_sample_sequence = self.proposal_sequence
                    self.turn_sample_pose = current.copy()
            mode = "rotate" if angular else "translate" if linear else "hold"
            if mode != self.mode and (mode != "hold" or "position_bounds_initial_xy" not in self.spec):
                self.mode, self.mode_start = mode, current.copy()
                self.progress = (stamp, current.copy())
                self.command_elapsed = 0.0
            if mode == "rotate" and np.linalg.norm(current[:2, 3] - self.mode_start[:2, 3]) > self.spec.get(
                "turn_translation_limit_m", 0.25
            ):
                raise ValueError("trial rotation skid exceeds declared turn limit")
            if mode == "translate":
                angle = yaw(self.mode_start)
                lateral = (current[:2, 3] - self.mode_start[:2, 3]) @ [-math.sin(angle), math.cos(angle)]
                if abs(lateral) > 0.08 or abs(wrap(yaw(current) - angle)) > 0.15:
                    raise ValueError("trial translation exceeded original heading/lateral guard")
            progress_time = (
                self.command_elapsed if "position_bounds_initial_xy" in self.spec else stamp - self.progress[0]
            )
            if mode != "hold" and progress_time >= 1.5:
                old = self.progress[1]
                if (
                    np.linalg.norm(current[:3, 3] - old[:3, 3]) < 0.005
                    and np.linalg.norm(current[:3, :3] - old[:3, :3]) < 0.01
                ):
                    raise ValueError("terrain-controlled motion stalled")
                self.progress = (stamp, current.copy())
                self.command_elapsed = 0.0
            self.publish(linear, angular)
            self.command_active = bool(linear or angular)
        self.record_execution(stamp, current, linear, angular, "checked_proposal", age)
        return False

    def record_execution(self, stamp, current, linear, angular, state, age):
        self.log.write(
            json.dumps(
                dict(
                    stamp=stamp,
                    linear=linear,
                    angular=angular,
                    world_from_base=current.tolist(),
                    proposal=self.proposal,
                    execution_state=state,
                    proposal_wall_age_s=age,
                    gt_role="evaluation_and_emergency_veto_only",
                )
            )
            + "\n"
        )
