# MIT License
#
# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES, University of Washington. All rights reserved.
#
# Aubo i3H adaptation of gen_data.py
# Architecture unified with Franka pipeline:
#   - Uses AuboRobot (6-DOF, no gripper) instead of FrankaRobot
#   - Uses AuboCollisionSpheres instead of FrankaSelfCollisionChecker
#   - Uses BulletAubo instead of BulletFranka
#   - Global planner: AIT* primary → RRTConnect fallback (same as Franka)
#   - Hybrid planner (fabric mode): SE3 EEF plan + Geometric Fabrics (same as Franka)
#   - Hybrid planner (curobo mode): cuRobo GPU MotionGen (independent)
#   - HDF5 data format is identical (trajectory dim is 6 instead of 7)

import time
import argparse
import gc
import os
import sys
import traceback
import uuid
import random
import pickle
import numpy as np
import pybullet as p
from ompl.util import noOutputHandler
from multiprocessing import Pool
from tqdm.auto import tqdm
from pathlib import Path
import h5py
from geometrout.primitive import Cuboid, Cylinder, Sphere
from geometrout.transform import SE3
import itertools
import logging
from dataclasses import dataclass, field
from typing import Tuple, List, Union, Sequence, Any

from robofin.bullet import Bullet, BulletAubo
from robofin.robots_aubo import AuboRobot
from robofin.collision_aubo import AuboCollisionSpheres
from robofin.robot_constants_aubo import AuboConstants
from atob.planner import AuboRRTConnectPlanner, AuboAITStarPlanner, AuboAITStarHandPlanner
from atob.trajectory import Trajectory
from pyquaternion import Quaternion

from mpinets.data_pipeline.environments.base_environment import (
    Candidate,
    Environment,
)
from mpinets.data_pipeline.environments.cubby_environment import (
    CubbyEnvironment,
    MergedCubbyEnvironment,
)
from mpinets.data_pipeline.environments.dresser_environment import DresserEnvironment
from mpinets.data_pipeline.environments.tabletop_environment import TabletopEnvironment
from mpinets.mpinets_types import PlanningProblem

# Configuration constants
AUBO_MAX_REACH = 0.72  # Aubo i3H max reach in meters (vs Franka 0.855m)
PLANNED_PATH_LENGTH = 300
END_EFFECTOR_FRAME = "wrist3_Link"
SEQUENCE_LENGTH = 50
NUM_SCENES = 6000
NUM_PLANS_PER_SCENE = 98
MAX_JERK = 10
PIPELINE_TIMEOUT = 36000
CUBOID_CUTOFF = 40
CYLINDER_CUTOFF = 40
import os as _os
NUM_WORKERS = max(1, _os.cpu_count() // 8 or 1)
MAX_TASKS_PER_CHILD = 10
PLANNER_MODE = "fabric"
HDF5_FLOAT_DTYPE = np.float32
TARGET_WRIST_LINK = "wrist3_Link"
TARGET_WRIST_RGBA = [0.15, 0.9, 0.25, 0.9]
TARGET_POINT_RGBA = [1.0, 0.85, 0.1, 0.95]
TARGET_POINT_RADIUS = 0.015
HIDDEN_LINK_RGBA = [0.0, 0.0, 0.0, 0.0]
ENV_GEN_VERBOSE = True
VERIFY_SOLVABLE_RUNTIME = 20.0
TERMINATION_RADIUS = 0.15  # For fabric convergence check
FABRIC_URDF_PATH = ""  # Set from CLI args
LOG_MEMORY = False
MEMORY_LOG_EVERY = 1


@dataclass
class Result:
    start_candidate: Candidate
    target_candidate: Candidate
    error_codes: List[str] = field(default_factory=list)
    cuboids: List[Cuboid] = field(default_factory=list)
    cylinders: List[Cylinder] = field(default_factory=list)
    global_solution: np.ndarray = field(default_factory=lambda: np.array([]))
    hybrid_solution: np.ndarray = field(default_factory=lambda: np.array([]))


def current_rss_mb() -> float:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("nan")


def maybe_log_memory(message: str, *, force: bool = False) -> None:
    if not force and not LOG_MEMORY:
        return
    rss_mb = current_rss_mb()
    if np.isnan(rss_mb):
        print(f"[mem] pid={os.getpid()} {message}", flush=True)
    else:
        print(f"[mem] pid={os.getpid()} rss={rss_mb:.1f}MB {message}", flush=True)


def solve_global_plans(
    start_candidate: Candidate,
    target_candidate: Candidate,
    obstacles: List[Union[Cuboid, Cylinder]],
    selfcc: AuboCollisionSpheres,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs AIT* and smoothing to solve global plan (with RRTConnect fallback),
    matching the Franka pipeline. Returns forward and backward trajectories.
    """
    with Bullet(gui=False) as sim:
        sim.load_primitives(obstacles)
        robot = sim.load_robot(BulletAubo)

        def validate_trajectory(path_like) -> np.ndarray:
            if path_like is None:
                return np.array([])
            trajectory = np.asarray(path_like)
            if trajectory.ndim != 2 or len(trajectory) == 0:
                return np.array([])
            for q in trajectory:
                robot.marionette(q)
                if sim.in_collision(robot, check_self=True) or selfcc.has_self_collision(q):
                    return np.array([])
            return trajectory

        def resample_and_validate(path) -> np.ndarray:
            try:
                trajectory = Trajectory.from_path(
                    np.asarray(path),
                    length=SEQUENCE_LENGTH,
                    robot_constants=AuboConstants,
                )
            except Exception:
                return np.array([])
            if trajectory is None:
                return np.array([])
            return validate_trajectory(trajectory.milestones)

        def smooth_and_validate(planner, path):
            try:
                smoothed = planner.smooth(path, SEQUENCE_LENGTH)
            except Exception:
                smoothed = None
            validated = validate_trajectory(smoothed)
            if len(validated):
                return validated
            return resample_and_validate(path)

        # Primary: AIT*
        planner = AuboAITStarPlanner()
        planner.load_simulation(sim, robot)
        planner.load_self_collision_checker(selfcc)
        try:
            path = planner.plan(
                start=start_candidate.config,
                goal=target_candidate.config,
                max_runtime=20,
                min_solution_time=10,
                exact=True,
                shortcut=True,
                spline=True,
                verbose=False,
            )
        except Exception:
            path = None
        if path is not None:
            forward_smoothed = smooth_and_validate(planner, path)
            backward_smoothed = smooth_and_validate(planner, path[::-1])
            if len(forward_smoothed) and len(backward_smoothed):
                return forward_smoothed, backward_smoothed

        # Fallback: RRTConnect
        fallback_planner = AuboRRTConnectPlanner()
        fallback_planner.load_simulation(sim, robot)
        fallback_planner.load_self_collision_checker(selfcc)
        try:
            fallback_path = fallback_planner.plan(
                start=start_candidate.config,
                goal=target_candidate.config,
                max_runtime=20,
                exact=True,
                shortcut=True,
                spline=True,
                verbose=False,
            )
        except Exception:
            fallback_path = None
        if fallback_path is None:
            return np.array([]), np.array([])
        forward_smoothed = smooth_and_validate(fallback_planner, fallback_path)
        backward_smoothed = smooth_and_validate(fallback_planner, fallback_path[::-1])
        if len(forward_smoothed) and len(backward_smoothed):
            return forward_smoothed, backward_smoothed
        return np.array([]), np.array([])


# ─────────────────────────────────────────────────────────────────────────────
# Fabric hybrid planner (SE3 EEF waypoints → Geometric Fabrics tracking)
# Matches the Franka pipeline: plan_end_effector → get_fabric_chunks
# ─────────────────────────────────────────────────────────────────────────────


def plan_end_effector_aubo(
    start_candidate: Candidate,
    target_candidate: Candidate,
    obstacles: List[Union[Cuboid, Cylinder]],
    selfcc: AuboCollisionSpheres,
) -> "List[SE3] | None":
    """
    Plans an end-effector path in SE(3) using AIT* on the Aubo wrist3_Link.
    Returns a list of SE3 poses or None on failure.
    """
    planner = AuboAITStarHandPlanner(buffer=0.0)
    with Bullet(gui=False) as sim:
        sim.load_primitives(obstacles)
        planner.load_simulation(sim)
        planner.load_self_collision_checker(selfcc)

        start_pose = start_candidate.pose
        goal_pose = target_candidate.pose

        try:
            path = planner.plan(
                start=start_pose,
                goal=goal_pose,
                frame=END_EFFECTOR_FRAME,
                interpolate=PLANNED_PATH_LENGTH,
                max_runtime=5,
            )
        except Exception:
            path = None
        return path


def get_fabric_chunks_aubo(
    end_eff_plan: "List[SE3]",
    q0: np.ndarray,
    cuboids: List[Cuboid],
    cylinders: List[Cylinder],
) -> "Tuple[List[List[np.ndarray]], SE3]":
    """
    Runs RMPflow (lula) to track the SE3 waypoints and produces chunked
    joint-space trajectories. Adapted from Franka fabric version for 6-DOF.

    Uses the newer lula RmpFlow API (Isaac Sim / mpinets-aubo-hybrid image)
    which replaced the deprecated Geometric Fabrics (create_fabric_state /
    create_fabric_config / create_fabric) with:
      lula.create_rmpflow_config() + lula.create_rmpflow()
    eval_accel signature changed from (q, qd, dt, qdd, state) to (q, qd, qdd).
    """
    import lula

    poses = [lula.Pose3(p.matrix) for p in end_eff_plan]

    urdf_path = FABRIC_URDF_PATH
    if not urdf_path:
        # Auto-detect: use the URDF bundled with the robofin third-party package
        _default_urdf = (
            Path(__file__).resolve().parent.parent.parent
            / "mpinets"
            / "third_party"
            / "robofin"
            / "robofin"
            / "urdf"
            / "aubo_i3H"
            / "aubo_i3H.urdf"
        )
        if _default_urdf.exists():
            urdf_path = str(_default_urdf)
    assert Path(urdf_path).exists(), (
        f"Fabric URDF not found at '{urdf_path}'. "
        "Pass --fabric-urdf <path> or ensure the URDF exists at "
        "mpinets/third_party/robofin/robofin/urdf/aubo_i3H/aubo_i3H.urdf"
    )
    fabric_robot_description_path = str(
        Path(__file__).resolve().parent.parent.parent
        / "config"
        / "aubo_lula_robot_description.yaml"
    )
    assert Path(fabric_robot_description_path).exists(), (
        f"{fabric_robot_description_path} not found"
    )
    rmpflow_config_path = str(
        Path(__file__).resolve().parent.parent.parent
        / "config"
        / "aubo_rmpflow_config.yaml"
    )
    assert Path(rmpflow_config_path).exists(), f"{rmpflow_config_path} not found"

    robot_description = lula.load_robot(fabric_robot_description_path, urdf_path)

    world = lula.create_world()
    for o in cuboids:
        if o.is_zero_volume():
            continue
        box_obstacle_pose = lula.Pose3(o.pose.matrix)
        box = lula.create_obstacle(lula.Obstacle.Type.CUBE)
        box.set_attribute(lula.Obstacle.Attribute.SIDE_LENGTHS, np.asarray(o.dims))
        world.add_obstacle(box, box_obstacle_pose)

    for o in cylinders:
        if o.is_zero_volume():
            continue
        cylinder_obstacle_pose = lula.Pose3(o.pose.matrix)
        cylinder = lula.create_obstacle(lula.Obstacle.Type.CYLINDER)
        cylinder.set_attribute(lula.Obstacle.Attribute.RADIUS, o.radius)
        cylinder.set_attribute(lula.Obstacle.Attribute.HEIGHT, o.height)
        world.add_obstacle(cylinder, cylinder_obstacle_pose)

    world_view = world.add_world_view()
    rmpflow_config = lula.create_rmpflow_config(
        rmpflow_config_path,
        robot_description,
        END_EFFECTOR_FRAME,
        world_view,
    )
    rmpflow = lula.create_rmpflow(rmpflow_config)
    joint_position = q0.copy()
    chunked_trajectory = [[joint_position.copy()]]

    kinematics = robot_description.kinematics()
    joint_velocity = np.ones(6) * 0.01
    joint_accel = np.zeros(6)
    dt = 0.005
    for target_pose in poses[1:]:
        rmpflow.set_end_effector_position_attractor(target_pose.translation)
        rmpflow.set_end_effector_orientation_attractor(target_pose.rotation)
        x_pose = kinematics.pose(joint_position, END_EFFECTOR_FRAME)
        time_so_far = 0.0
        chunk = []
        while (
            np.linalg.norm(x_pose.translation - target_pose.translation)
            > TERMINATION_RADIUS
            and time_so_far < 0.5
        ):
            rmpflow.eval_accel(joint_position, joint_velocity, joint_accel)
            joint_position += dt * joint_velocity
            joint_velocity += dt * joint_accel
            chunk.append(joint_position.copy())
            x_pose = kinematics.pose(joint_position, END_EFFECTOR_FRAME)
            time_so_far += dt
        chunked_trajectory.append(chunk)

    # Extra convergence time
    extra_time = 4.0
    time_so_far = 0.0
    while time_so_far < extra_time:
        if np.linalg.norm(x_pose.translation - target_pose.translation) < 0.005:
            break
        rmpflow.eval_accel(joint_position, joint_velocity, joint_accel)
        joint_position += dt * joint_velocity
        joint_velocity += dt * joint_accel
        time_so_far += dt
        chunk.append(joint_position.copy())

    final_pose_matrix = kinematics.pose(joint_position, END_EFFECTOR_FRAME).matrix()
    final_pose_xyz = final_pose_matrix[:3, -1]
    final_pose_q = Quaternion(matrix=final_pose_matrix[:3, :3], atol=1e-5)
    final_pose = SE3(xyz=final_pose_xyz, quaternion=final_pose_q.elements)
    # Explicitly release lula C++ objects so memory is reclaimed promptly
    # rather than waiting for Python GC (important in high-frequency fabric mode).
    del rmpflow, rmpflow_config, world_view, world, robot_description, kinematics
    return chunked_trajectory, final_pose


def solve_fabric_hybrid_plan(
    start_candidate: Candidate,
    target_candidate: Candidate,
    cuboids: List[Cuboid],
    cylinders: List[Cylinder],
    selfcc: AuboCollisionSpheres,
) -> np.ndarray:
    """
    Full Fabric hybrid pipeline: SE3 EEF planning → Fabric tracking → downsample.
    Returns (SEQUENCE_LENGTH, 6) trajectory or empty array on failure.
    """
    end_eff_plan = plan_end_effector_aubo(
        start_candidate, target_candidate, cuboids + cylinders, selfcc
    )
    if end_eff_plan is None or len(end_eff_plan) < 2:
        return np.array([])

    try:
        chunked_trajectory, final_pose = get_fabric_chunks_aubo(
            end_eff_plan, start_candidate.config, cuboids, cylinders
        )
    except Exception:
        return np.array([])

    trajectory = list(itertools.chain.from_iterable(chunked_trajectory))
    if len(trajectory) < 2:
        return np.array([])

    try:
        downsampled = Trajectory.from_path(
            np.asarray(trajectory), length=SEQUENCE_LENGTH, robot_constants=AuboConstants
        ).milestones
    except Exception:
        return np.array([])

    return np.asarray(downsampled)


# ─────────────────────────────────────────────────────────────────────────────
# cuRobo hybrid planner (GPU-accelerated motion generation)
# ─────────────────────────────────────────────────────────────────────────────


def _get_curobo_motion_gen():
    """
    Lazily initialize the cuRobo MotionGen instance (cached per-process).
    cuRobo init is expensive, so we do it once and reuse.
    """
    if not hasattr(_get_curobo_motion_gen, "_instance"):
        import inspect
        from curobo.geom.types import WorldConfig
        from curobo.types.robot import RobotConfig
        from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

        config_path = str(
            Path(__file__).resolve().parent.parent.parent
            / "config"
            / "curobo_aubo_i3H.yaml"
        )
        load_kwargs = dict(
            robot_cfg=config_path,
            world_model=WorldConfig(),
            collision_cache={"obb": 64, "mesh": 64},
            num_ik_seeds=30,
            num_trajopt_seeds=4,
            num_graph_seeds=4,
            interpolation_dt=0.02,
            collision_activation_distance=0.05,
            self_collision_activation_distance=0.02,
            trajopt_dt=0.04,
            evaluate_interpolated_trajectory=True,
        )
        supported = inspect.signature(
            MotionGenConfig.load_from_robot_config
        ).parameters
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            **{k: v for k, v in load_kwargs.items() if k in supported}
        )
        motion_gen = MotionGen(motion_gen_config)
        warmup_supported = inspect.signature(MotionGen.warmup).parameters
        warmup_kwargs = {"warmup_jit_cnt": 1}
        motion_gen.warmup(
            **{k: v for k, v in warmup_kwargs.items() if k in warmup_supported}
        )
        _clear_curobo_stale_link_poses(motion_gen)
        _get_curobo_motion_gen._instance = motion_gen
    return _get_curobo_motion_gen._instance


def _clear_curobo_stale_link_poses(motion_gen) -> None:
    """
    Work around stale link goal buffers left behind by cuRobo warmup().

    Some cuRobo versions keep `links_goal_pose` populated after warmup even
    when subsequent `plan_single()` calls do not pass `link_poses`. Clear the
    cached goal buffers explicitly so ordinary single-EE queries do not inherit
    hidden multi-link constraints.
    """
    for solver_name in ("ik_solver", "trajopt_solver"):
        solver = getattr(motion_gen, solver_name, None)
        goal_buffer = getattr(solver, "_goal_buffer", None)
        if goal_buffer is not None:
            goal_buffer.links_goal_pose = None


def _build_curobo_world(
    cuboids: List[Cuboid], cylinders: List[Cylinder]
):
    """Build a cuRobo WorldConfig from the primitive obstacles."""
    from curobo.types.math import Pose as CuPose
    from curobo.geom.types import WorldConfig, Cuboid as CuCuboid, Cylinder as CuCylinder

    cu_cuboids = []
    for i, c in enumerate(cuboids):
        if c.is_zero_volume():
            continue
        cu_cuboids.append(
            CuCuboid(
                name=f"cuboid_{i}",
                dims=list(c.dims),
                pose=list(c.pose.xyz) + list(c.pose.so3.wxyz),
            )
        )

    cu_cylinders = []
    for i, cyl in enumerate(cylinders):
        if cyl.is_zero_volume():
            continue
        cu_cylinders.append(
            CuCylinder(
                name=f"cylinder_{i}",
                radius=float(cyl.radius),
                height=float(cyl.height),
                pose=list(cyl.pose.xyz) + list(cyl.pose.so3.wxyz),
            )
        )

    return WorldConfig(cuboid=cu_cuboids if cu_cuboids else None,
                       cylinder=cu_cylinders if cu_cylinders else None)


def solve_curobo_hybrid_plan(
    start_candidate: Candidate,
    target_candidate: Candidate,
    cuboids: List[Cuboid],
    cylinders: List[Cylinder],
    selfcc: "AuboCollisionSpheres | None" = None,
) -> np.ndarray:
    """
    Uses cuRobo MotionGen to plan from start config to target joint config.
    Returns (SEQUENCE_LENGTH, 6) trajectory or empty array on failure.

    Uses plan_single_js (joint-space goal) because plan_single (EE-pose goal)
    fails for large-amplitude motions: TrajOpt cannot converge when the IK
    solution is far from the start configuration in C-space.  Since
    target_candidate.config is always available, plan_single_js is both
    more reliable and avoids redundant IK solves.

    If selfcc is provided, the trajectory is validated in PyBullet after
    planning to reject any trajectories that cuRobo's approximate sphere
    model missed.
    """
    import torch
    from curobo.types.state import JointState
    from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig
    import inspect

    try:
        motion_gen = _get_curobo_motion_gen()
    except Exception:
        return np.array([])

    # Update world obstacles
    world_config = _build_curobo_world(cuboids, cylinders)
    try:
        motion_gen.update_world(world_config)
    except Exception:
        return np.array([])

    # Prepare start and goal joint states
    _clear_curobo_stale_link_poses(motion_gen)
    start_state = JointState(
        position=torch.tensor(
            np.array(start_candidate.config, dtype=np.float32)
        ).cuda().unsqueeze(0),
        velocity=torch.zeros((1, 6), dtype=torch.float32).cuda(),
        acceleration=torch.zeros((1, 6), dtype=torch.float32).cuda(),
    )
    goal_state = JointState(
        position=torch.tensor(
            np.array(target_candidate.config, dtype=np.float32)
        ).cuda().unsqueeze(0),
        velocity=torch.zeros((1, 6), dtype=torch.float32).cuda(),
        acceleration=torch.zeros((1, 6), dtype=torch.float32).cuda(),
    )

    # Build MotionGenPlanConfig with graph search for robustness
    pcs = inspect.signature(MotionGenPlanConfig.__init__).parameters
    pc_kwargs = {}
    if "enable_graph_search" in pcs:
        pc_kwargs["enable_graph_search"] = True
    if "max_attempts" in pcs:
        pc_kwargs["max_attempts"] = 4
    if "time_dilation_factor" in pcs:
        pc_kwargs["time_dilation_factor"] = 0.75
    plan_config = MotionGenPlanConfig(**pc_kwargs)

    try:
        result = motion_gen.plan_single_js(
            start_state,
            goal_state,
            plan_config,
        )
    except Exception:
        return np.array([])

    if not result.success.item():
        return np.array([])

    # Extract trajectory and resample to SEQUENCE_LENGTH
    traj = result.get_interpolated_plan()
    if traj is None:
        return np.array([])
    traj_np = traj.position.cpu().numpy().squeeze()  # (T, 6)

    if len(traj_np) < 2:
        return np.array([])

    try:
        resampled = Trajectory.from_path(traj_np, length=SEQUENCE_LENGTH, robot_constants=AuboConstants).milestones
    except Exception:
        return np.array([])

    resampled_np = np.asarray(resampled)

    # PyBullet post-validation: cuRobo's sphere model can miss collisions that
    # the mesh-based PyBullet checker catches.  Reject any trajectory that
    # fails the same checks used by the fabric/global planners.
    if selfcc is not None:
        if has_self_collision(resampled_np, selfcc):
            return np.array([])
        if violates_joint_limits(resampled_np):
            return np.array([])
        with Bullet(gui=False) as sim:
            sim.load_primitives(cuboids + cylinders)
            robot = sim.load_robot(BulletAubo)
            collides = in_collision(resampled_np, sim, robot)
        if collides:
            return np.array([])

    return resampled_np


def has_high_jerk(trajectory: np.ndarray) -> bool:
    velocities = [trajectory[i + 1] - trajectory[i] for i in range(len(trajectory) - 1)]
    accelerations = [velocities[i + 1] - velocities[i] for i in range(len(velocities) - 1)]
    for ai, aj in zip(accelerations[:-1], accelerations[1:]):
        if np.max(np.abs(aj - ai)) > MAX_JERK:
            return True
    return False


def violates_joint_limits(trajectory: np.ndarray) -> bool:
    for q in trajectory:
        if not AuboRobot.within_limits(q):
            return True
    return False


def downsample(trajectory: Sequence[np.ndarray]) -> "np.ndarray | None":
    """
    Retimes the trajectory to have constant-ish velocity.
    Returns None if there is an error.
    """
    with np.errstate(over="raise", divide="raise", under="raise", invalid="raise"):
        try:
            sampled = Trajectory.from_path(
                trajectory, length=SEQUENCE_LENGTH, robot_constants=AuboConstants
            ).milestones
        except Exception:
            return None
    return np.asarray(sampled)


def has_self_collision(
    trajectory: np.ndarray, selfcc: AuboCollisionSpheres
) -> bool:
    """Checks whether there are any self collisions in the trajectory."""
    for q in trajectory:
        if selfcc.has_self_collision(q):
            return True
    return False


def in_collision(trajectory: np.ndarray, sim: Bullet, robot: BulletAubo) -> bool:
    """Checks whether the trajectory collides with the environment."""
    for q in trajectory:
        robot.marionette(q)
        if sim.in_collision(robot, check_self=True):
            return True
    return False


def verify_trajectory(
    sim: Bullet,
    robot: BulletAubo,
    trajectory: np.ndarray,
    final_pose: SE3,
    goal_pose: SE3,
    selfcc: AuboCollisionSpheres,
) -> List[str]:
    """
    Runs a set of checks on the trajectory to determine whether to keep it,
    matching the Franka verification pipeline.
    """
    error_codes = []
    if np.linalg.norm(final_pose._xyz - goal_pose._xyz) > 0.05:
        error_codes.append("miss")
    if has_high_jerk(trajectory):
        error_codes.append("high jerk")
    if has_self_collision(trajectory, selfcc):
        error_codes.append("self collision")
    if in_collision(trajectory, sim, robot):
        error_codes.append("collision")
    if violates_joint_limits(trajectory):
        error_codes.append("joint limit")
    return error_codes


def forward_backward_aubo(
    candidate1: Candidate,
    candidate2: Candidate,
    cuboids: List[Cuboid],
    cylinders: List[Cylinder],
    selfcc: AuboCollisionSpheres,
) -> List[Result]:
    """
    Run the hybrid expert pipeline going forward and backward between two
    candidates.  Matches the Franka pipeline structure:
      - Global: AIT* → RRTConnect fallback (always runs)
      - Hybrid: SE3 EEF plan + Geometric Fabrics tracking (fabric mode, default)
                OR cuRobo GPU MotionGen (curobo mode, independent)

    PLANNER_MODE selects the hybrid expert:
      - "fabric":  SE3 EEF plan + Geometric Fabrics (same as Franka)
      - "curobo":  cuRobo GPU MotionGen (independent planner)
      - "global":  Global only, no hybrid expert
    """
    # cuRobo mode is intentionally independent from the OMPL global planner.
    # We preserve the forward/backward result layout, but leave
    # global_solution empty and record an error instead of falling back.
    if PLANNER_MODE == "curobo":
        forward_result = Result(
            cuboids=cuboids,
            cylinders=cylinders,
            start_candidate=candidate1,
            target_candidate=candidate2,
        )
        backward_result = Result(
            cuboids=cuboids,
            cylinders=cylinders,
            start_candidate=candidate2,
            target_candidate=candidate1,
        )

        cu_fwd = solve_curobo_hybrid_plan(candidate1, candidate2, cuboids, cylinders, selfcc)
        if len(cu_fwd) > 0:
            forward_result.hybrid_solution = cu_fwd
        else:
            forward_result.error_codes.append("curobo")

        cu_bwd = solve_curobo_hybrid_plan(candidate2, candidate1, cuboids, cylinders, selfcc)
        if len(cu_bwd) > 0:
            backward_result.hybrid_solution = cu_bwd
        else:
            backward_result.error_codes.append("curobo")

        return [forward_result, backward_result]

    with Bullet(gui=False) as sim:
        arm = sim.load_robot(BulletAubo)
        sim.load_primitives(cuboids + cylinders)

        # ── Global: AIT* → RRTConnect fallback (same as Franka) ──
        global_forward, global_backward = solve_global_plans(
            candidate1, candidate2, cuboids + cylinders, selfcc
        )
        if len(global_forward) != len(global_backward):
            logging.warning(
                "Length of global forward and backward solutions are different"
                "--something might be buggy"
            )
        if len(global_forward) == 0 or len(global_backward) == 0:
            return []

        forward_result = Result(
            global_solution=global_forward,
            cuboids=cuboids,
            cylinders=cylinders,
            start_candidate=candidate1,
            target_candidate=candidate2,
        )
        backward_result = Result(
            global_solution=global_backward,
            cuboids=cuboids,
            cylinders=cylinders,
            start_candidate=candidate2,
            target_candidate=candidate1,
        )
        results = [forward_result, backward_result]

        # ── Global-only mode: use global as hybrid fallback ──
        if PLANNER_MODE == "global":
            forward_result.hybrid_solution = global_forward
            backward_result.hybrid_solution = global_backward
            return results

        # ── Fabric mode (default): SE3 EEF plan + Geometric Fabrics ──
        #    Same two-stage hybrid expert as the Franka pipeline
        end_eff_plan = plan_end_effector_aubo(
            candidate1, candidate2, cuboids + cylinders, selfcc
        )
        if end_eff_plan is None or len(end_eff_plan) < 2:
            forward_result.error_codes.append("end effector path")
            backward_result.error_codes.append("end effector path")
            return results

        # Forward direction
        chunked_trajectory, final_pose = get_fabric_chunks_aubo(
            end_eff_plan, candidate1.config, cuboids, cylinders
        )
        trajectory = list(itertools.chain.from_iterable(chunked_trajectory))
        downsampled_trajectory = downsample(trajectory)
        if downsampled_trajectory is None:
            forward_result.error_codes.append("lula or downsample")
        else:
            forward_result.error_codes.extend(
                verify_trajectory(
                    sim,
                    arm,
                    downsampled_trajectory,
                    final_pose,
                    end_eff_plan[-1],
                    selfcc,
                ),
            )
            forward_result.hybrid_solution = downsampled_trajectory

        # Backward direction
        end_eff_plan.reverse()
        chunked_trajectory, final_pose = get_fabric_chunks_aubo(
            end_eff_plan, candidate2.config, cuboids, cylinders
        )
        trajectory = list(itertools.chain.from_iterable(chunked_trajectory))
        downsampled_trajectory = downsample(trajectory)
        if downsampled_trajectory is None:
            backward_result.error_codes.append("lula or downsample")
        else:
            backward_result.error_codes.extend(
                verify_trajectory(
                    sim,
                    arm,
                    downsampled_trajectory,
                    final_pose,
                    end_eff_plan[-1],
                    selfcc,
                ),
            )
            backward_result.hybrid_solution = downsampled_trajectory

        return results


def verify_has_solvable_problems_aubo(
    env: Environment,
    selfcc: AuboCollisionSpheres,
    max_runtime: float = 10.0,
) -> bool:
    """Checks that the environment's own demo candidates admit a valid plan."""
    planner = AuboRRTConnectPlanner()
    with Bullet(gui=False) as sim:
        sim.load_primitives(env.obstacles)
        robot = sim.load_robot(BulletAubo)
        planner.load_simulation(sim, robot)
        planner.load_self_collision_checker(selfcc)
        try:
            path = planner.plan(
                start=env.demo_candidates[0].config,
                goal=env.demo_candidates[1].config,
                max_runtime=max_runtime,
                exact=True,
                verbose=False,
            )
        except Exception:
            return False
        return path is not None


def gen_valid_env_aubo(selfcc: AuboCollisionSpheres) -> Environment:
    """Generates a valid environment using the environment-specific candidate logic."""
    env_arguments = {}
    if ENV_TYPE == "tabletop":
        env: Environment = TabletopEnvironment(max_reach=AUBO_MAX_REACH)
        env_arguments["how_many"] = np.random.randint(3, 15)
    elif ENV_TYPE == "cubby":
        env = CubbyEnvironment(max_reach=AUBO_MAX_REACH)
    elif ENV_TYPE == "merged-cubby":
        env = MergedCubbyEnvironment(max_reach=AUBO_MAX_REACH)
    elif ENV_TYPE == "dresser":
        env = DresserEnvironment(max_reach=AUBO_MAX_REACH)
    else:
        raise NotImplementedError(f"{ENV_TYPE} not implemented")

    success = False
    attempt = 0
    while not success:
        attempt += 1
        attempt_start = time.time()
        if ENV_GEN_VERBOSE:
            print(
                f"[{ENV_TYPE}] attempt {attempt}: sampling scene and demo candidates...",
                flush=True,
            )
        generated = env.gen(selfcc=selfcc, **env_arguments)
        obstacle_ok = (
            generated
            and len(env.cuboids) < CUBOID_CUTOFF
            and len(env.cylinders) < CYLINDER_CUTOFF
        )
        success = False
        failure_reason = "unknown"
        if not generated:
            failure_reason = "environment generation returned False"
        elif len(env.cuboids) >= CUBOID_CUTOFF:
            failure_reason = f"too many cuboids ({len(env.cuboids)})"
        elif len(env.cylinders) >= CYLINDER_CUTOFF:
            failure_reason = f"too many cylinders ({len(env.cylinders)})"
        else:
            if ENV_GEN_VERBOSE:
                print(
                    f"[{ENV_TYPE}] attempt {attempt}: verifying demo pair "
                    f"({len(env.cuboids)} cuboids, {len(env.cylinders)} cylinders)...",
                    flush=True,
                )
            success = verify_has_solvable_problems_aubo(
                env,
                selfcc,
                max_runtime=VERIFY_SOLVABLE_RUNTIME,
            )
            if not success:
                failure_reason = (
                    f"demo pair not solvable within {VERIFY_SOLVABLE_RUNTIME:.1f}s"
                )

        if ENV_GEN_VERBOSE:
            elapsed = time.time() - attempt_start
            if success:
                print(
                    f"[{ENV_TYPE}] attempt {attempt}: ok in {elapsed:.1f}s "
                    f"({len(env.cuboids)} cuboids, {len(env.cylinders)} cylinders)",
                    flush=True,
                )
            else:
                print(
                    f"[{ENV_TYPE}] attempt {attempt}: retry after {elapsed:.1f}s "
                    f"({failure_reason})",
                    flush=True,
                )
    return env


def exhaust_environment_aubo(
    env: Environment, num: int, selfcc: AuboCollisionSpheres
) -> List[Result]:
    """Generates problems using the environment's own candidate distributions."""
    n = int(np.round(np.sqrt(num / 2)))
    candidates = env.gen_additional_candidate_sets(n - 1, selfcc)
    candidates[0].append(env.demo_candidates[0])
    candidates[1].append(env.demo_candidates[1])

    if IS_NEUTRAL:
        neutral_candidates = env.gen_neutral_candidates(n, selfcc)
        random.shuffle(candidates[0])
        random.shuffle(candidates[1])
        if n <= 1:
            nonneutral_candidates = candidates[0][:1]
        else:
            nonneutral_candidates = candidates[0][: n // 2] + candidates[1][: n // 2]
        pairs = list(itertools.product(neutral_candidates, nonneutral_candidates))
    else:
        pairs = list(itertools.product(candidates[0], candidates[1]))

    results = []
    for c1, c2 in pairs:
        results.extend(forward_backward_aubo(c1, c2, env.cuboids, env.cylinders, selfcc))
    return results


def gen_single_env_data_aubo():
    selfcc = AuboCollisionSpheres()
    env = gen_valid_env_aubo(selfcc)
    results = exhaust_environment_aubo(env, NUM_PLANS_PER_SCENE, selfcc)
    return env, results


def load_target_wrist_visual(sim: Bullet) -> BulletAubo:
    """Load a ghost Aubo robot and keep only wrist3_Link visible."""
    target_robot = sim.load_robot(BulletAubo)
    for link_name, link_idx in target_robot.links:
        rgba = TARGET_WRIST_RGBA if link_name == TARGET_WRIST_LINK else HIDDEN_LINK_RGBA
        p.changeVisualShape(
            target_robot.id,
            link_idx,
            rgbaColor=rgba,
            physicsClientId=sim.clid,
        )
        p.setCollisionFilterGroupMask(
            target_robot.id,
            link_idx,
            0,
            0,
            physicsClientId=sim.clid,
        )
    return target_robot


def load_target_point_visual(sim: Bullet, point: np.ndarray) -> int:
    """Show the target frame origin as a small visual-only sphere."""
    return sim.load_sphere(
        Sphere(np.asarray(point, dtype=np.float64), TARGET_POINT_RADIUS),
        color=TARGET_POINT_RGBA,
        visual_only=True,
    )


def target_point_for_candidate(env: Environment, candidate: Candidate) -> np.ndarray:
    if isinstance(env, DresserEnvironment):
        return env.target_point_from_pose(candidate.pose)
    return np.asarray(candidate.pose.xyz, dtype=np.float64)


def move_target_point_visual(sim: Bullet, marker_id: int, point: np.ndarray) -> None:
    p.resetBasePositionAndOrientation(
        marker_id,
        np.asarray(point, dtype=np.float64).tolist(),
        [0.0, 0.0, 0.0, 1.0],
        physicsClientId=sim.clid,
    )


def visualize_cases(num_cases: int = 3):
    """在 PyBullet GUI 中逐个可视化数据生成的场景与轨迹。

    每个 case：随机生成环境 -> 规划起终点之间的轨迹 -> GUI 动画播放。
    在终端按 Enter 跳到下一个 case，按 Ctrl-C 退出。
    """
    _worker_init()  # 预编译 Numba 核函数
    selfcc = AuboCollisionSpheres()

    for case_idx in range(num_cases):
        print(f"\n=== Case {case_idx + 1} / {num_cases} ===")

        # 生成随机环境
        env = gen_valid_env_aubo(selfcc)

        # 使用环境自身生成的示例起终点，保持场景语义一致
        c1, c2 = env.demo_candidates

        # 全局规划
        trajectory, _ = solve_global_plans(c1, c2, env.obstacles, selfcc)
        if len(trajectory) == 0:
            print("  规划失败，跳过本 case。")
            continue

        print(f"  场景：{len(env.cuboids)} 个长方体，{len(env.cylinders)} 个圆柱体")
        print(f"  轨迹：{len(trajectory)} 个路径点")

        # 打开 PyBullet GUI
        sim = Bullet(gui=True)
        sim.load_primitives(env.obstacles, color=[0.6, 0.75, 1.0, 0.85])
        robot = sim.load_robot(BulletAubo)
        target_robot = load_target_wrist_visual(sim)
        load_target_point_visual(sim, target_point_for_candidate(env, c2))

        robot.marionette(c1.config)
        target_robot.marionette(c2.config)
        print("  起点配置、目标 wrist3_Link 与目标点已显示（2 秒后开始播放轨迹）...")
        time.sleep(2.0)

        # 播放轨迹动画（~20 fps）
        print("  播放轨迹中...")
        for q in trajectory:
            robot.marionette(q)
            time.sleep(0.05)

        robot.marionette(c2.config)
        print("  已到达终点配置。")

        try:
            input("  → 按 Enter 查看下一个 case（Ctrl-C 退出）... ")
        except KeyboardInterrupt:
            del sim
            print("\n已退出可视化。")
            return
        del sim  # 关闭当前 GUI 再开下一个


def gen_single_env(_: Any):
    if time.time() - START_TIME > PIPELINE_TIMEOUT:
        return
    np.random.seed()
    random.seed()
    maybe_log_memory("worker starting scene generation")
    env, results = gen_single_env_data_aubo()

    n = len(results)
    cuboids = env.cuboids
    cylinders = env.cylinders
    file_name = f"{TMP_DATA_DIR}/{uuid.uuid4()}.hdf5"
    with h5py.File(file_name, "w-") as f:
        f.attrs["planner_mode"] = PLANNER_MODE
        # Use DOF=6 for Aubo
        global_solutions = f.create_dataset(
            "global_solutions",
            (n, SEQUENCE_LENGTH, 6),
            dtype=HDF5_FLOAT_DTYPE,
        )
        # Keep both datasets populated for downstream compatibility even when
        # only one planner mode is generated.
        hybrid_solutions = f.create_dataset(
            "hybrid_solutions",
            (n, SEQUENCE_LENGTH, 6),
            dtype=HDF5_FLOAT_DTYPE,
        )
        cuboid_dims = f.create_dataset(
            "cuboid_dims", (len(cuboids), 3), dtype=HDF5_FLOAT_DTYPE
        )
        cuboid_centers = f.create_dataset(
            "cuboid_centers", (len(cuboids), 3), dtype=HDF5_FLOAT_DTYPE
        )
        cuboid_quats = f.create_dataset(
            "cuboid_quaternions", (len(cuboids), 4), dtype=HDF5_FLOAT_DTYPE
        )
        cylinder_radii = f.create_dataset(
            "cylinder_radii", (len(cylinders), 1), dtype=HDF5_FLOAT_DTYPE
        )
        cylinder_heights = f.create_dataset(
            "cylinder_heights", (len(cylinders), 1), dtype=HDF5_FLOAT_DTYPE
        )
        cylinder_centers = f.create_dataset(
            "cylinder_centers", (len(cylinders), 3), dtype=HDF5_FLOAT_DTYPE
        )
        cylinder_quats = f.create_dataset(
            "cylinder_quaternions", (len(cylinders), 4), dtype=HDF5_FLOAT_DTYPE
        )

        for ii in range(n):
            if results[ii].global_solution.shape == (SEQUENCE_LENGTH, 6):
                global_solutions[ii, :, :] = results[ii].global_solution
            if (
                len(results[ii].error_codes) == 0
                and results[ii].hybrid_solution.shape == (SEQUENCE_LENGTH, 6)
            ):
                hybrid_solutions[ii, :, :] = results[ii].hybrid_solution
        for jj in range(len(cuboids)):
            cuboid_dims[jj, :] = cuboids[jj].dims
            cuboid_centers[jj, :] = cuboids[jj].pose.xyz
            cuboid_quats[jj, :] = cuboids[jj].pose.so3.wxyz
        for kk in range(len(cylinders)):
            cylinder_radii[kk, :] = cylinders[kk].radius
            cylinder_heights[kk, :] = cylinders[kk].height
            cylinder_centers[kk, :] = cylinders[kk].pose.xyz
            cylinder_quats[kk, :] = cylinders[kk].pose.so3.wxyz
    del env
    del results
    gc.collect()
    maybe_log_memory("worker finished scene generation")


def _worker_init():
    """Pre-JIT-compile Numba FK/collision kernels before planning starts.

    Without this, the first call in each worker incurs a ~5-10 s compilation
    delay.  Running a dummy check at worker startup hides that cost.
    """
    dummy_selfcc = AuboCollisionSpheres()
    q = AuboConstants.NEUTRAL.copy()
    dummy_selfcc.has_self_collision(q)  # triggers aubo_arm_link_fk JIT


def gen():
    noOutputHandler()
    non_seeds = np.arange(NUM_SCENES)
    maybe_log_memory(
        (
            f"starting pool scenes={NUM_SCENES} plans_per_scene={NUM_PLANS_PER_SCENE} "
            f"workers={NUM_WORKERS} maxtasksperchild={MAX_TASKS_PER_CHILD}"
        ),
        force=True,
    )
    with Pool(
        processes=NUM_WORKERS,
        initializer=_worker_init,
        maxtasksperchild=MAX_TASKS_PER_CHILD,
    ) as pool:
        results_iter = pool.imap_unordered(gen_single_env, non_seeds)
        pbar = tqdm(total=NUM_SCENES)
        idx = 0
        while True:
            try:
                next(results_iter)
            except StopIteration:
                break
            except Exception as e:
                # Worker OOM or unexpected crash: log and continue so the
                # remaining scenes are still processed.
                logging.warning(f"Worker error (scene skipped): {e}")
            idx += 1
            pbar.update(1)
            if LOG_MEMORY and idx % MEMORY_LOG_EVERY == 0:
                maybe_log_memory(f"parent observed {idx}/{NUM_SCENES} completed scenes")
                gc.collect()
        pbar.close()
    maybe_log_memory("pool generation complete", force=True)

    all_files = list(Path(TMP_DATA_DIR).glob("*.hdf5"))
    maybe_log_memory(f"starting merge over {len(all_files)} temporary files", force=True)
    max_cylinders = 0
    max_cuboids = 0
    total_trajectories = 0
    for fi in all_files:
        with h5py.File(fi) as f:
            total_trajectories += len(f["global_solutions"])
            if len(f["cuboid_dims"]) > max_cuboids:
                max_cuboids = len(f["cuboid_dims"])
            if len(f["cylinder_radii"]) > max_cylinders:
                max_cylinders = len(f["cylinder_radii"])

    with h5py.File(f"{FINAL_DATA_DIR}/all_data.hdf5", "w-") as f:
        f.attrs["planner_mode"] = PLANNER_MODE
        hybrid_solutions = f.create_dataset(
            "hybrid_solutions",
            (total_trajectories, SEQUENCE_LENGTH, 6),
            dtype=HDF5_FLOAT_DTYPE,
        )
        global_solutions = f.create_dataset(
            "global_solutions",
            (total_trajectories, SEQUENCE_LENGTH, 6),
            dtype=HDF5_FLOAT_DTYPE,
        )
        cuboid_dims = f.create_dataset(
            "cuboid_dims",
            (total_trajectories, max_cuboids, 3),
            dtype=HDF5_FLOAT_DTYPE,
        )
        cuboid_centers = f.create_dataset(
            "cuboid_centers",
            (total_trajectories, max_cuboids, 3),
            dtype=HDF5_FLOAT_DTYPE,
        )
        cuboid_quats = f.create_dataset(
            "cuboid_quaternions",
            (total_trajectories, max_cuboids, 4),
            dtype=HDF5_FLOAT_DTYPE,
        )
        cylinder_radii = f.create_dataset(
            "cylinder_radii",
            (total_trajectories, max_cylinders, 1),
            dtype=HDF5_FLOAT_DTYPE,
        )
        cylinder_heights = f.create_dataset(
            "cylinder_heights",
            (total_trajectories, max_cylinders, 1),
            dtype=HDF5_FLOAT_DTYPE,
        )
        cylinder_centers = f.create_dataset(
            "cylinder_centers",
            (total_trajectories, max_cylinders, 3),
            dtype=HDF5_FLOAT_DTYPE,
        )
        cylinder_quats = f.create_dataset(
            "cylinder_quaternions",
            (total_trajectories, max_cylinders, 4),
            dtype=HDF5_FLOAT_DTYPE,
        )

        chunk_start = 0
        for fi in all_files:
            with h5py.File(fi, "r") as g:
                chunk_end = chunk_start + len(g["global_solutions"])
                global_solutions[chunk_start:chunk_end, ...] = g["global_solutions"][...]
                hybrid_solutions[chunk_start:chunk_end, ...] = g["hybrid_solutions"][...]
                nc = len(g["cuboid_dims"])
                nl = len(g["cylinder_radii"])
                batch_len = chunk_end - chunk_start
                if nc > 0:
                    cuboid_dims[chunk_start:chunk_end, :nc, ...] = np.broadcast_to(
                        g["cuboid_dims"][...], (batch_len, nc, 3)
                    )
                    cuboid_centers[
                        chunk_start:chunk_end, :nc, ...
                    ] = np.broadcast_to(g["cuboid_centers"][...], (batch_len, nc, 3))
                    cuboid_quats[
                        chunk_start:chunk_end, :nc, ...
                    ] = np.broadcast_to(
                        g["cuboid_quaternions"][...], (batch_len, nc, 4)
                    )
                if nl > 0:
                    cylinder_radii[
                        chunk_start:chunk_end, :nl, ...
                    ] = np.broadcast_to(g["cylinder_radii"][...], (batch_len, nl, 1))
                    cylinder_heights[
                        chunk_start:chunk_end, :nl, ...
                    ] = np.broadcast_to(
                        g["cylinder_heights"][...], (batch_len, nl, 1)
                    )
                    cylinder_centers[
                        chunk_start:chunk_end, :nl, ...
                    ] = np.broadcast_to(
                        g["cylinder_centers"][...], (batch_len, nl, 3)
                    )
                    cylinder_quats[
                        chunk_start:chunk_end, :nl, ...
                    ] = np.broadcast_to(
                        g["cylinder_quaternions"][...], (batch_len, nl, 4)
                    )
                chunk_start = chunk_end
                if LOG_MEMORY and chunk_start % max(1, MEMORY_LOG_EVERY * NUM_PLANS_PER_SCENE) == 0:
                    maybe_log_memory(
                        f"merged {chunk_start}/{total_trajectories} trajectories"
                    )
    for fi in all_files:
        fi.unlink()
    maybe_log_memory("final merge complete", force=True)


def visualize_single_env():
    if ENV_GEN_VERBOSE:
        print(
            f"Generating one valid '{ENV_TYPE}' environment and demo trajectories...",
            flush=True,
        )
    env, results = gen_single_env_data_aubo()
    if len(results) == 0:
        print(
            "No additional candidate pairs solved; falling back to the demo pair.",
            flush=True,
        )
        selfcc = AuboCollisionSpheres()
        c1, c2 = env.demo_candidates
        results = forward_backward_aubo(c1, c2, env.cuboids, env.cylinders, selfcc)
        if len(results) == 0:
            print("Found no results", flush=True)
            return
    sim = Bullet(gui=True)
    robot = sim.load_robot(BulletAubo)
    sim.load_primitives(env.obstacles)
    target_robot = load_target_wrist_visual(sim)
    target_point_marker = None
    for idx, r in enumerate(results):
        target_point = target_point_for_candidate(env, r.target_candidate)
        if target_point_marker is None:
            target_point_marker = load_target_point_visual(sim, target_point)
        else:
            move_target_point_visual(sim, target_point_marker, target_point)
        robot.marionette(r.start_candidate.config)
        target_robot.marionette(r.target_candidate.config)
        print(
            f"Visualizing case {idx + 1}/{len(results)}: "
            "showing start pose, target wrist3_Link, and target point"
        )
        time.sleep(1.0)

        print("Visualizing global solution")
        print(f"global solution has {len(r.global_solution)} waypoints")
        if len(r.global_solution) != 0:
            for q in r.global_solution:
                robot.marionette(q)
                time.sleep(0.1)

            robot.marionette(r.target_candidate.config)
            time.sleep(0.3)

        print("Visualizing hybrid solution")
        robot.marionette(r.start_candidate.config)
        time.sleep(0.3)
        print(f"Hybrid solution has {len(r.hybrid_solution)} waypoints")
        for q in r.hybrid_solution:
            robot.marionette(q)
            time.sleep(0.1)
        robot.marionette(r.target_candidate.config)
        time.sleep(0.5)


def validate_output_dir(data_dir: str) -> None:
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Output directory does not exist: {data_dir}. "
            "Create it first and bind-mount it into the container."
        )
    if not os.access(data_dir, os.W_OK):
        raise PermissionError(f"Output directory is not writable: {data_dir}")
    output_path = os.path.join(data_dir, "all_data.hdf5")
    if os.path.exists(output_path):
        raise FileExistsError(
            f"Refusing to overwrite existing file: {output_path}. "
            "Delete it or choose an empty output directory."
        )


def main() -> int:
    global START_TIME
    global IS_NEUTRAL
    global ENV_TYPE
    global PLANNER_MODE
    global FABRIC_URDF_PATH
    global NUM_WORKERS
    global MAX_TASKS_PER_CHILD
    global LOG_MEMORY
    global MEMORY_LOG_EVERY
    global ENV_GEN_VERBOSE
    global VERIFY_SOLVABLE_RUNTIME
    global NUM_SCENES
    global NUM_PLANS_PER_SCENE
    global TMP_DATA_DIR
    global FINAL_DATA_DIR
    START_TIME = time.time()
    noOutputHandler()
    np.random.seed()
    random.seed()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "env_type",
        choices=["tabletop", "cubby", "merged-cubby", "dresser"],
    )
    parser.add_argument("--neutral", action="store_true")
    parser.add_argument(
        "--planner-mode",
        choices=[
            "fabric",
            "curobo",
            "global",
        ],
        default="fabric",
        help=(
            "fabric (default): AIT*→RRTConnect global + SE3 EEF plan + Geometric "
            "Fabrics hybrid (same pipeline as Franka, needs lula). "
            "curobo: cuRobo GPU MotionGen only, skipping AIT*→RRTConnect. "
            "global: AIT*→RRTConnect global only, no hybrid expert."
        ),
    )
    parser.add_argument(
        "--fabric-urdf",
        type=str,
        default="",
        help=(
            "Path to the Aubo URDF used by Geometric Fabrics (lula). "
            "Required when --planner-mode is fabric. "
            "Typically the URDF inside the Isaac Sim / cuRobo Docker."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=NUM_WORKERS,
        help="Number of worker processes to use for pipeline generation.",
    )
    parser.add_argument(
        "--max-tasks-per-child",
        type=int,
        default=MAX_TASKS_PER_CHILD,
        help="Recycle each worker after this many scenes to limit memory growth.",
    )
    parser.add_argument(
        "--log-memory",
        action="store_true",
        help="Print lightweight RSS memory logs while the pipeline runs.",
    )
    parser.add_argument(
        "--memory-log-every",
        type=int,
        default=MEMORY_LOG_EVERY,
        help="When --log-memory is set, log every N completed scenes.",
    )
    subparsers = parser.add_subparsers(dest="run_type", required=True)
    run_full = subparsers.add_parser("full-pipeline")
    run_full.add_argument("data_dir", type=str)
    run_full.add_argument("--num-scenes", type=int, default=NUM_SCENES)
    run_full.add_argument("--plans-per-scene", type=int, default=NUM_PLANS_PER_SCENE)
    test_pipeline = subparsers.add_parser("test-pipeline")
    test_pipeline.add_argument("data_dir", type=str)
    test_pipeline.add_argument("--num-scenes", type=int, default=10)
    test_pipeline.add_argument("--plans-per-scene", type=int, default=4)
    subparsers.add_parser(
        "test-environment",
        help="Generates a few trajectories for a single environment and visualizes them with PyBullet",
    )
    visualize_cases_parser = subparsers.add_parser(
        "visualize-cases",
        help="Generates several random environments and visualizes them one by one with PyBullet",
    )
    visualize_cases_parser.add_argument("--num-cases", type=int, default=3)

    args = parser.parse_args()

    IS_NEUTRAL = args.neutral
    ENV_TYPE = args.env_type
    PLANNER_MODE = args.planner_mode
    if args.fabric_urdf:
        FABRIC_URDF_PATH = args.fabric_urdf
    NUM_WORKERS = max(1, args.num_workers)
    MAX_TASKS_PER_CHILD = max(1, args.max_tasks_per_child)
    LOG_MEMORY = args.log_memory
    MEMORY_LOG_EVERY = max(1, args.memory_log_every)
    ENV_GEN_VERBOSE = args.run_type in ["test-environment", "visualize-cases"]
    VERIFY_SOLVABLE_RUNTIME = 10.0

    if args.run_type in ["full-pipeline", "test-pipeline"]:
        NUM_SCENES = args.num_scenes
        NUM_PLANS_PER_SCENE = args.plans_per_scene

    if args.run_type == "test-environment":
        visualize_single_env()
    elif args.run_type == "visualize-cases":
        visualize_cases(args.num_cases)
    else:
        TMP_DATA_DIR = f"/tmp/tmp_data_aubu_{uuid.uuid4()}/"
        os.mkdir(TMP_DATA_DIR)
        FINAL_DATA_DIR = args.data_dir
        validate_output_dir(FINAL_DATA_DIR)
        gen()
    return 0


if __name__ == "__main__":
    exit_code = 0
    try:
        exit_code = main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        exit_code = 130
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        # os._exit avoids the Boost.Python destructor crash (double-free on
        # exit) that occurs when using the pip ompl package.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
