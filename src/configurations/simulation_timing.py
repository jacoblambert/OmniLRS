"""Resolve timing before creating either World or PhysicsContext.

PhysicsSceneConf already controls the actual PhysX timestep. World must receive
the same value, rather than cache a conflicting environment-level timestep.
"""
import math


def simulation_timing(environment_physics_dt, rendering_dt, scene_physics_dt):
    values = [environment_physics_dt, rendering_dt, scene_physics_dt]
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or not 0 < value <= 1 for value in values):
        raise ValueError("simulation timesteps must be finite and positive")
    if rendering_dt < scene_physics_dt * (1. - 1e-6):
        raise ValueError("rendering timestep cannot be shorter than the authoritative physics timestep")
    # PhysicsContext quantizes dt as int(1 / dt) Hz. Preserve that actual
    # integrator rate. Decimal 0.0333 / 0.016666 is just below two, and Isaac's
    # int(render_dt / physics_dt) would otherwise silently allow only one step.
    physics_hz = int(1. / scene_physics_dt)
    actual_physics_dt = 1. / physics_hz
    ratio = rendering_dt / actual_physics_dt
    substeps = round(ratio)
    if substeps < 1 or not math.isclose(ratio, substeps, rel_tol=.002, abs_tol=1e-9):
        raise ValueError("rendering timestep must be an integral multiple of the physics timestep")
    actual_rendering_dt = substeps * actual_physics_dt
    return dict(physics_dt=actual_physics_dt, rendering_dt=actual_rendering_dt,
                requested_scene_physics_dt=scene_physics_dt, requested_rendering_dt=rendering_dt,
                physics_substeps=substeps,
                environment_physics_dt=environment_physics_dt,
                conflicting_environment_dt=not math.isclose(environment_physics_dt, scene_physics_dt,
                                                            rel_tol=1e-6, abs_tol=1e-9),
                authority="PhysicsSceneConf.dt")
