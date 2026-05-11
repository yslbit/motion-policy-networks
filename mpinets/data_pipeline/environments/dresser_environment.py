# MIT License
#
# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES, University of Washington. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from geometrout.primitive import Cuboid, CuboidArray, Cylinder, CylinderArray
from geometrout.transform import SE3
from robofin.bullet import Bullet, BulletAubo, BulletFranka
from robofin.robots import FrankaRobot, FrankaRealRobot
from robofin.robots_aubo import AuboRobot
from robofin.collision import FrankaSelfCollisionChecker
from robofin.collision_aubo import AuboCollisionSpheres
import yourdfpy
import trimesh.transformations as tra
import trimesh
import numpy as np

import re
import time
import random

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Dict, Sequence, Union

from mpinets.data_pipeline.environments.base_environment import (
    TaskOrientedCandidate,
    NeutralCandidate,
    Environment,
    radius_sample,
)


@dataclass
class DresserCandidate(TaskOrientedCandidate):
    """
    Represents a configuration, its end-effector pose, and some metadata about the dresser
    (i.e. which drawer it belongs to and the free space inside that drawer).
    """

    drawer_idx: int
    support_volume: Cuboid


@dataclass
class SupportSurface:
    polygon: trimesh.path.polygons.Polygon
    facet_index: int
    node_name: str
    transform: np.ndarray


@dataclass
class Container:
    geometry: trimesh.Trimesh
    node_name: str
    transform: np.ndarray
    support_surface: Optional[SupportSurface] = None


class DresserEnvironment(Environment):
    def __init__(self, max_reach: float = Environment.FRANKA_MAX_REACH):
        super().__init__()
        self.max_reach = max_reach
        self._reach_scale = max_reach / self.FRANKA_MAX_REACH
        self.demo_candidates = []

    def _uses_aubo(self, selfcc: Union[FrankaSelfCollisionChecker, AuboCollisionSpheres]) -> bool:
        return isinstance(selfcc, AuboCollisionSpheres)

    def _uses_compact_workspace_profile(self) -> bool:
        return self.max_reach < (self.FRANKA_MAX_REACH - 1e-6)

    def _same_level_rejection_probability(
        self, selfcc: Union[FrankaSelfCollisionChecker, AuboCollisionSpheres]
    ) -> float:
        # Aubo's limited reach means fewer valid drawer pairs exist;
        # rejecting same-level pairs is wasteful.
        return 0.0 if self._uses_aubo(selfcc) else 0.7

    def _drawer_pose_sample_count(
        self, selfcc: Union[FrankaSelfCollisionChecker, AuboCollisionSpheres]
    ) -> int:
        return 400 if self._uses_aubo(selfcc) else 100

    def _drawer_vertical_offset(
        self, selfcc: Union[FrankaSelfCollisionChecker, AuboCollisionSpheres]
    ) -> float:
        # AUBO uses wrist3_Link directly as the task frame, so it does not need as
        # much "inside the drawer" clearance as Franka's gripper frame.
        # Keep this small: the vertical_offset directly shrinks the support volume
        # height, and the wrist3 sphere (r=0.049) needs min_dim >= 0.098 to fit.
        return 0.01 if self._uses_aubo(selfcc) else 0.05

    def _drawer_theta_radius(
        self, selfcc: Union[FrankaSelfCollisionChecker, AuboCollisionSpheres]
    ) -> float:
        return (np.pi / 6) if self._uses_aubo(selfcc) else (np.pi / 4)

    def _start_pose_attempts(
        self, selfcc: Union[FrankaSelfCollisionChecker, AuboCollisionSpheres]
    ) -> int:
        return 3 if self._uses_aubo(selfcc) else 1

    def _aubo_drawer_config_sample_count(self) -> int:
        return 1000

    def _aubo_drawer_task_offset(self) -> float:
        # Aubo i3H has no gripper: wrist3_Link IS the TCP.
        # Any non-zero offset shifts wrist3 away from the sampled task point,
        # which in the compact-profile dresser (depth ~12 cm) pushes wrist3
        # outside the drawer ~30-40% of the time.
        return 0.0

    def _aubo_drawer_task_tolerance(self) -> float:
        # Allow the wrist3 sphere to protrude by up to 1.5 cm.
        # The obstacle collision check still prevents physical collisions;
        # this tolerance just relaxes the "fully inside support volume" rule
        # so that the usable safe zone is large enough to find solutions.
        return 0.015

    def _aubo_drawer_fallback_tolerance(self) -> float:
        # Slightly more lenient fallback when no perfect pose is found.
        return 0.02

    def _aubo_effective_reach(self) -> float:
        # Leave ~15% margin for wrist orientation freedom on a 6-DOF arm.
        return self.max_reach * 0.85

    def _aubo_pose_from_task_point(
        self,
        task_point: np.ndarray,
        x_dir: np.ndarray,
        y_dir: np.ndarray,
        z_dir: np.ndarray,
    ) -> SE3:
        return SE3.from_unit_axes(
            origin=task_point + self._aubo_drawer_task_offset() * x_dir,
            x=x_dir,
            y=y_dir,
            z=z_dir,
        )

    def _aubo_task_point_from_pose(self, pose: SE3) -> np.ndarray:
        return (
            np.asarray(pose.xyz, dtype=np.float64)
            - self._aubo_drawer_task_offset() * pose.matrix[:3, 0]
        )

    def target_point_from_pose(self, pose: SE3) -> np.ndarray:
        return self._aubo_task_point_from_pose(pose)

    def _aubo_support_volume_can_fit_eef(
        self,
        selfcc: AuboCollisionSpheres,
        support_volume: Cuboid,
    ) -> bool:
        # The current Aubo EEF model is a single wrist3 sphere, but using the
        # maximum radius keeps this safe if more EEF spheres are added later.
        max_radius = float(np.max(selfcc.cspheres["wrist3_Link"].radii))
        return float(np.min(support_volume.dims)) >= (2.0 * max_radius)

    def _aubo_eef_support_clearance(
        self,
        pose: SE3,
        selfcc: AuboCollisionSpheres,
        support_volume: Cuboid,
    ) -> float:
        info = selfcc.eef_csphere_info(pose, frame=self._eef_frame(selfcc))
        return float(np.max(support_volume.sdf(info.centers) + info.radii))

    def _support_volume_reachable(self, support_volume: Cuboid) -> bool:
        center = np.asarray(support_volume.center)
        return float(np.linalg.norm(center)) <= self._aubo_effective_reach()

    def _normalize_direction(self, direction: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            raise ValueError("Cannot normalize a near-zero direction vector")
        return direction / norm

    def _drawer_axes(self, drawer_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        drawer_node = f"drawer_{drawer_idx}"
        if drawer_node not in self.scene.graph.nodes:
            drawer_node = f"dresser/{drawer_node}"
        drawer_transform = self.scene.graph.get(drawer_node)[0]
        front_dir = np.asarray(drawer_transform[:3, 1], dtype=np.float64)
        up_dir = np.asarray(drawer_transform[:3, 2], dtype=np.float64)
        front_dir = self._normalize_direction(front_dir)
        up_dir = self._normalize_direction(up_dir)
        if np.dot(up_dir, np.array([0.0, 0.0, 1.0])) < 0.0:
            up_dir = -up_dir
        outward_hint = np.asarray(drawer_transform[:3, 3], dtype=np.float64) - np.asarray(
            self.dresser_asset.centroid,
            dtype=np.float64,
        )
        outward_hint[2] = 0.0
        if np.linalg.norm(outward_hint) > 1e-8 and np.dot(front_dir, outward_hint) < 0.0:
            front_dir = -front_dir
        right_dir = self._normalize_direction(np.cross(front_dir, up_dir))
        return right_dir, front_dir, up_dir

    def _aubo_front_entry_pose_and_config(
        self,
        sim: Bullet,
        arm: BulletAubo,
        selfcc: AuboCollisionSpheres,
        support_volume: Cuboid,
        primitive_arrays: List[Any],
        drawer_idx: int,
    ) -> Tuple[Optional[SE3], Optional[np.ndarray]]:
        right_dir, front_dir, up_dir = self._drawer_axes(drawer_idx)
        samples = support_volume.sample_volume(self._drawer_pose_sample_count(selfcc))
        relative = samples - np.asarray(support_volume.center)
        front_scores = relative @ front_dir
        side_scores = np.abs(relative @ right_dir)
        height_scores = np.abs(relative @ up_dir)
        front_cutoff = -0.10 * float(support_volume.dims[1])
        preferred = samples[front_scores >= front_cutoff]
        if len(preferred) == 0:
            preferred = samples
            preferred_relative = relative
            preferred_front_scores = front_scores
        else:
            preferred_relative = preferred - np.asarray(support_volume.center)
            preferred_front_scores = preferred_relative @ front_dir
        preferred_side_scores = np.abs(preferred_relative @ right_dir)
        preferred_height_scores = np.abs(preferred_relative @ up_dir)
        order = np.lexsort(
            (
                preferred_height_scores,
                preferred_side_scores,
                -preferred_front_scores,
            )
        )
        preferred = preferred[order]
        down_dir = -up_dir
        base_side_dir = self._normalize_direction(np.cross(down_dir, front_dir))
        best_pose = None
        best_config = None
        best_clearance = np.inf
        for task_point in preferred:
            for roll in (-0.35, -0.15, 0.0, 0.15, 0.35):
                side_dir = (
                    np.cos(roll) * base_side_dir + np.sin(roll) * down_dir
                )
                side_dir = self._normalize_direction(side_dir)
                z_dir = self._normalize_direction(np.cross(front_dir, side_dir))
                pose = self._aubo_pose_from_task_point(
                    task_point,
                    front_dir,
                    side_dir,
                    z_dir,
                )
                eef_clearance = self._aubo_eef_support_clearance(
                    pose, selfcc, support_volume
                )
                if eef_clearance < best_clearance:
                    best_clearance = eef_clearance
                    best_pose = pose
                if eef_clearance > self._aubo_drawer_task_tolerance():
                    continue
                if self._eef_in_collision(pose, selfcc, primitive_arrays, arm):
                    continue
                config = self._collision_free_ik(pose, selfcc, primitive_arrays, arm)
                if config is None:
                    continue
                arm.marionette(config)
                if sim.in_collision(arm, check_self=True):
                    continue
                return pose, config
        if (
            best_pose is not None
            and best_clearance <= self._aubo_drawer_fallback_tolerance()
        ):
            config = self._collision_free_ik(best_pose, selfcc, primitive_arrays, arm)
            if config is not None:
                arm.marionette(config)
                if not sim.in_collision(arm, check_self=True):
                    return best_pose, config
        return None, None

    def _aubo_config_sample_pose_and_config(
        self,
        sim: Bullet,
        arm: BulletAubo,
        selfcc: AuboCollisionSpheres,
        support_volume: Cuboid,
        primitive_arrays: List[Any],
    ) -> Tuple[Optional[SE3], Optional[np.ndarray]]:
        joint_limits = AuboRobot.constants.JOINT_LIMITS
        neutral = AuboRobot.constants.NEUTRAL
        best_pose = None
        best_config = None
        best_clearance = np.inf
        for ii in range(self._aubo_drawer_config_sample_count()):
            if ii < int(0.7 * self._aubo_drawer_config_sample_count()):
                config = np.clip(
                    neutral + np.random.normal(0.0, 0.45, 6),
                    joint_limits[:, 0],
                    joint_limits[:, 1],
                )
            else:
                config = np.random.uniform(
                    joint_limits[:, 0],
                    joint_limits[:, 1],
                )
            if selfcc.has_self_collision(config):
                continue
            arm.marionette(config)
            if sim.in_collision(arm, check_self=True):
                continue
            pose = AuboRobot.fk(config, eff_frame=self._eef_frame(selfcc))
            eef_clearance = self._aubo_eef_support_clearance(
                pose, selfcc, support_volume
            )
            if eef_clearance < best_clearance:
                best_clearance = eef_clearance
                best_pose = pose
                best_config = np.copy(config)
            if eef_clearance > self._aubo_drawer_task_tolerance():
                continue
            if self._eef_in_collision(pose, selfcc, primitive_arrays, arm):
                continue
            return pose, config
        if (
            best_pose is not None
            and best_clearance <= self._aubo_drawer_fallback_tolerance()
            and not self._eef_in_collision(best_pose, selfcc, primitive_arrays, arm)
        ):
            return best_pose, best_config
        return None, None

    def _aubo_topdown_pose_and_config(
        self,
        sim: Bullet,
        arm: BulletAubo,
        selfcc: AuboCollisionSpheres,
        support_volume: Cuboid,
        primitive_arrays: List[Any],
    ) -> Tuple[Optional[SE3], Optional[np.ndarray]]:
        """Top-down approach: wrist3_Link points straight down into the drawer.

        This mirrors how Franka handles drawers and is often more reachable
        than the front-entry approach for a short 6-DOF arm.
        """
        samples = support_volume.sample_volume(self._drawer_pose_sample_count(selfcc))
        # Sort by distance from base (closer = more likely reachable)
        dists = np.linalg.norm(samples, axis=1)
        order = np.argsort(dists)
        samples = samples[order]

        for sample in samples:
            for theta in (0.0, 0.3, -0.3, 0.6, -0.6):
                x = np.array([np.cos(theta), np.sin(theta), 0.0], dtype=np.float64)
                z = np.array([0.0, 0.0, -1.0], dtype=np.float64)
                y = np.cross(z, x)
                pose = self._aubo_pose_from_task_point(sample, x, y, z)
                if (
                    self._aubo_eef_support_clearance(pose, selfcc, support_volume)
                    > self._aubo_drawer_task_tolerance()
                ):
                    continue
                if self._eef_in_collision(pose, selfcc, primitive_arrays, arm):
                    continue
                config = self._collision_free_ik(pose, selfcc, primitive_arrays, arm)
                if config is None:
                    continue
                arm.marionette(config)
                if sim.in_collision(arm, check_self=True):
                    continue
                return pose, config
        return None, None

    def _eef_frame(
        self, selfcc: Union[FrankaSelfCollisionChecker, AuboCollisionSpheres]
    ) -> str:
        return "wrist3_Link" if self._uses_aubo(selfcc) else "right_gripper"

    def _default_prismatic_joint(self, arm: Union[BulletFranka, BulletAubo]) -> float:
        return getattr(arm, "default_prismatic_value", 0.025)

    def _build_primitive_arrays(self) -> List[Any]:
        primitive_arrays: List[Any] = []
        cuboids = self.cuboids
        cylinders = self.cylinders
        if cuboids:
            primitive_arrays.append(CuboidArray(cuboids))
        if cylinders:
            primitive_arrays.append(CylinderArray(cylinders))
        return primitive_arrays

    def _load_planning_arm(
        self, sim: Bullet, selfcc: Union[FrankaSelfCollisionChecker, AuboCollisionSpheres]
    ) -> Union[BulletFranka, BulletAubo]:
        if self._uses_aubo(selfcc):
            return sim.load_robot(BulletAubo)
        return sim.load_robot(FrankaRobot)

    def _has_self_collision(
        self,
        selfcc: Union[FrankaSelfCollisionChecker, AuboCollisionSpheres],
        config: np.ndarray,
        arm: Union[BulletFranka, BulletAubo],
    ) -> bool:
        if self._uses_aubo(selfcc):
            return selfcc.has_self_collision(config)
        return selfcc.has_self_collision(config, self._default_prismatic_joint(arm))

    def _eef_in_collision(
        self,
        pose: SE3,
        selfcc: Union[FrankaSelfCollisionChecker, AuboCollisionSpheres],
        primitive_arrays: List[Any],
        arm: Union[BulletFranka, BulletAubo],
    ) -> bool:
        if self._uses_aubo(selfcc):
            return selfcc.aubo_eef_collides_fast(
                pose,
                primitive_arrays,
                frame=self._eef_frame(selfcc),
            )
        return selfcc.franka_eef_collides_fast(
            pose,
            self._default_prismatic_joint(arm),
            primitive_arrays,
            frame=self._eef_frame(selfcc),
        )

    def _collision_free_ik(
        self,
        pose: SE3,
        selfcc: Union[FrankaSelfCollisionChecker, AuboCollisionSpheres],
        primitive_arrays: List[Any],
        arm: Union[BulletFranka, BulletAubo],
    ) -> Optional[np.ndarray]:
        if self._uses_aubo(selfcc):
            return AuboRobot.collision_free_ik(
                pose,
                selfcc,
                primitive_arrays,
                eff_frame=self._eef_frame(selfcc),
                retries=1000,
            )
        return FrankaRealRobot.collision_free_ik(
            pose,
            self._default_prismatic_joint(arm),
            selfcc,
            primitive_arrays,
            eff_frame=self._eef_frame(selfcc),
            retries=1000,
        )

    def _random_neutral_config(
        self, selfcc: Union[FrankaSelfCollisionChecker, AuboCollisionSpheres]
    ) -> np.ndarray:
        if self._uses_aubo(selfcc):
            return AuboRobot.random_neutral()
        return FrankaRealRobot.random_neutral(method="uniform")

    def _forward_kinematics(
        self,
        config: np.ndarray,
        selfcc: Union[FrankaSelfCollisionChecker, AuboCollisionSpheres],
    ) -> SE3:
        if self._uses_aubo(selfcc):
            return AuboRobot.fk(config, eff_frame=self._eef_frame(selfcc))
        return FrankaRealRobot.fk(config, eff_frame=self._eef_frame(selfcc))


    def _gen(self, selfcc: Union[FrankaSelfCollisionChecker, AuboCollisionSpheres]) -> bool:
        self.dresser_asset = self._gen_dresser()

        self.joint_names = self.dresser_asset._model.actuated_joint_names

        if len(self.joint_names) < 2:
            self.demo_candidates = []
            return False

        self.scene = self.dresser_asset.as_trimesh_scene(namespace="dresser")
        self.scene_containers = {}

        cnt = 0
        while True:
            res = self._label_containment(geom_ids=f".*drawer_{cnt}_.*")
            if len(res) == 0:
                break

            self.scene_containers[f"volume_{cnt}"] = res

            cnt += 1

        if len(self.scene_containers) == 0:
            self.demo_candidates = []
            return False

        indices = list(range(len(self.joint_names)))
        random.shuffle(indices)

        sim = Bullet(gui=False)
        arm = self._load_planning_arm(sim, selfcc)

        (
            start_pose,
            start_q,
            start_support_volume,
            target_pose,
            target_q,
            target_support_volume,
        ) = (None, None, None, None, None, None)
        for i in range(len(indices)):
            idx_start = indices[i]
            self.open_drawer(idx_start)
            res = self._label_containment()
            self.scene_containers["volume"] = res
            sim.load_primitives(self.obstacles)
            primitive_arrays = self._build_primitive_arrays()
            start_support_volume = self.get_support_volume(
                idx_start,
                vertical_offset=self._drawer_vertical_offset(selfcc),
            )
            if start_support_volume is None:
                sim.clear_all_obstacles()
                self.close_drawer(idx_start)
                continue
            if self._uses_aubo(selfcc):
                if not self._support_volume_reachable(start_support_volume):
                    sim.clear_all_obstacles()
                    self.close_drawer(idx_start)
                    continue
                if not self._aubo_support_volume_can_fit_eef(selfcc, start_support_volume):
                    sim.clear_all_obstacles()
                    self.close_drawer(idx_start)
                    continue
            for _ in range(self._start_pose_attempts(selfcc)):
                start_pose, start_q = self.random_pose_and_config(
                    sim,
                    arm,
                    selfcc,
                    start_support_volume,
                    primitive_arrays,
                    drawer_idx=idx_start,
                )
                sim.clear_all_obstacles()
                if start_pose is None or start_q is None:
                    continue
                for j in range(i + 1, len(indices)):
                    idx_target = indices[j]
                    self.open_drawer(idx_target)
                    res = self._label_containment()
                    self.scene_containers["volume"] = res
                    sim.load_primitives(self.obstacles)
                    primitive_arrays = self._build_primitive_arrays()
                    # Check whether the previous solution is still valid
                    arm.marionette(start_q)
                    if sim.in_collision(arm):
                        sim.clear_all_obstacles()
                        self.close_drawer(idx_target)
                        continue
                    target_support_volume = self.get_support_volume(
                        idx_target,
                        vertical_offset=self._drawer_vertical_offset(selfcc),
                    )
                    if target_support_volume is None:
                        sim.clear_all_obstacles()
                        self.close_drawer(idx_target)
                        continue
                    if self._uses_aubo(selfcc):
                        if not self._support_volume_reachable(target_support_volume):
                            sim.clear_all_obstacles()
                            self.close_drawer(idx_target)
                            continue
                        if not self._aubo_support_volume_can_fit_eef(
                            selfcc, target_support_volume
                        ):
                            sim.clear_all_obstacles()
                            self.close_drawer(idx_target)
                            continue

                    if np.isclose(
                        np.max(start_support_volume.corners, axis=0)[2],
                        np.max(target_support_volume.corners, axis=0)[2],
                        atol=0.05,
                    ) and np.random.rand() < self._same_level_rejection_probability(selfcc):
                        sim.clear_all_obstacles()
                        self.close_drawer(idx_target)
                        continue

                    (
                        target_pose,
                        target_q,
                    ) = self.random_pose_and_config(
                        sim,
                        arm,
                        selfcc,
                        target_support_volume,
                        primitive_arrays,
                        drawer_idx=idx_target,
                    )
                    sim.clear_all_obstacles()
                    if target_pose is not None and target_q is not None:
                        break
                    self.close_drawer(idx_target)
                if target_pose is not None and target_q is not None:
                    break
            if target_pose is not None and target_q is not None:
                break
            self.close_drawer(idx_start)

        if start_q is None or target_q is None:
            self.demo_candidates = []
            return False
        self.demo_candidates = [
            DresserCandidate(
                pose=start_pose,
                config=start_q,
                drawer_idx=idx_start,
                support_volume=start_support_volume,
                negative_volumes=[target_support_volume],
            ),
            DresserCandidate(
                pose=target_pose,
                config=target_q,
                drawer_idx=idx_target,
                support_volume=target_support_volume,
                negative_volumes=[start_support_volume],
            ),
        ]
        return True

    def _gen_dresser(self):
        s = self._reach_scale
        for attempt in range(20 if self._uses_compact_workspace_profile() else 1):
            if self._uses_compact_workspace_profile():
                # Geometry for compact-reach robots (Aubo i3H, reach≈0.72m).
                #
                # KEY INSIGHT: with rotation≈π/2 the dresser body-Y axis maps to
                # world -X.  Drawers therefore open TOWARD the robot (in -X).
                # With a deep dresser the open drawer extends all the way back to
                # x≈0, forcing the arm to reach "under itself" — scene collision
                # is unavoidable.
                #
                # Fix: shorten the depth so the open drawer sits in the accessible
                # space IN FRONT of the dresser, not back near the robot base:
                #   dresser centre x = 0.55
                #   depth           = 0.18  →  front face at x ≈ 0.55 - 0.09 = 0.46
                #   drawer travel   = 0.18 * 0.9 = 0.162
                #   open-drawer interior: x = 0.46 – 0.162 = 0.30  to  0.46
                #   support volume centre x ≈ 0.38  (well within 0.72 m reach)
                #
                # The arm approaches the open-drawer space from the robot side
                # WITHOUT having to thread through the dresser body.
                width, depth, height = (
                    radius_sample(0.45, 0.05),
                    radius_sample(0.18, 0.02),
                    radius_sample(0.42, 0.04),
                )
                x, y, rotation = (
                    radius_sample(0.55, 0.04),
                    radius_sample(0.0, 0.05),
                    radius_sample(np.pi / 2, np.pi / 12),
                )
            else:
                width, depth, height = (
                    radius_sample(1.0, 0.2),
                    radius_sample(0.3, 0.1),
                    radius_sample(0.7, 0.15),
                )
                x, y, rotation = (
                    radius_sample(0.65, 0.1),
                    radius_sample(0.0, 0.1),
                    radius_sample(np.pi / 2, np.pi / 3),
                )
            transform = tra.euler_matrix(0, 0, rotation)
            base_z = 0.0
            transform[:3, 3] = np.array([x, y, base_z])
            dresser_kwargs = dict(
                width=width,
                depth=depth,
                height=height,
                transform=transform,
            )
            if self._uses_compact_workspace_profile():
                # Favor 2-3 larger drawers so each support volume can fit the
                # full wrist3 collision sphere instead of only the frame origin.
                dresser_kwargs.update(
                    split_prob=0.38, split_decay=0.58, min_split_size=0.20
                )
            dresser = Dresser(**dresser_kwargs)
            min_drawers = 2
            max_drawers = 3 if self._uses_compact_workspace_profile() else None
            num_drawers = len(dresser._model.actuated_joint_names)
            if num_drawers >= min_drawers and (
                max_drawers is None or num_drawers <= max_drawers
            ):
                return dresser
        return dresser

    def _compute_support_polyhedra(
        self,
        support_surface: SupportSurface,
        mesh: trimesh.Trimesh,
        gravity: np.ndarray,
        ray_cast_count: int,
        min_volume: float,
        distance_above_support: float,
        max_height: float,
        erosion_distance: float,
    ) -> Tuple[bool, Optional[trimesh.Trimesh]]:
        """
        See documentation for _get_support_polyhedra. Computes support polyhedra for a single polygon

        Returns:
            bool: If support_polygon is a support polyhedra
            trimesh.Trimesh: Corresponding support polyhedra
        """
        # for each support polygon, sample raycasts to determine maximum height of extrusion in direction fo gravity
        pts = trimesh.path.polygons.sample(
            support_surface.polygon, count=ray_cast_count
        )

        if len(pts) == 0:
            return False, None

        pts3d_local = np.column_stack([pts, distance_above_support * np.ones(len(pts))])
        T = (
            self.scene.graph.get(support_surface.node_name)[0]
            @ support_surface.transform
        )
        pts3d = trimesh.transform_points(points=pts3d_local, matrix=T)

        intersections, ray_ids, _ = mesh.ray.intersects_location(
            pts3d, np.array(len(pts) * [list(-gravity)]), multiple_hits=False
        )
        # if no intersection occurs we don't deem this a support polyhedra (e.g. top of shelf or table)
        if len(intersections) > 0:
            distances = np.linalg.norm((intersections - pts3d[ray_ids]), axis=1)
            min_distance = np.min(distances)

            assert min_distance >= 0

            if (
                min_distance >= trimesh.constants.tol.merge
                and min_distance <= max_height
            ):
                if support_surface.polygon.geom_type == "MultiPolygon":
                    return False, None

                if (min_distance - erosion_distance) > 0:
                    inscribing_polyhedra = trimesh.creation.extrude_polygon(
                        support_surface.polygon, min_distance - erosion_distance
                    )
                else:
                    return False, None

                if inscribing_polyhedra.volume >= min_volume:
                    return True, inscribing_polyhedra

        return False, None

    def _label_containment(self, geom_ids: Optional[str] = None) -> List[Container]:
        """Search for volumes in the asset.

        Args:
            geom_ids (Optional, str): Regular expression of all valid geometries to be included in the search. Or None if all should be included. Defaults to None.

        Returns:
            list[Container]: Volume data.
        """
        min_area = 0.01
        min_volume = 0.00001
        gravity = np.array([0, 0, -1.0])
        gravity_tolerance = 0.1
        erosion_distance = 0.02
        distance_above_support = 0.001

        support_surfaces = []

        geometry_names = list(self.scene.geometry.keys())
        if geom_ids is not None:
            x = re.compile(geom_ids)
            geometry_names = list(filter(x.search, geometry_names))

        support_meshes = [self.scene.geometry[name] for name in geometry_names]
        support_meshes_node_names = [
            self.scene.graph.geometry_nodes[g][0] for g in geometry_names
        ]

        if len(support_meshes) == 0:
            return []

        for obj_mesh, obj_node_name in zip(support_meshes, support_meshes_node_names):
            # rotate gravity vector into mesh coordinates
            mesh_transform, _ = self.scene.graph.get(obj_node_name)
            local_gravity = mesh_transform[:3, :3].T @ gravity

            # get all facets that are aligned with -local_gravity and bigger than min_area
            support_facet_indices = [
                idx
                for idx in np.argsort(obj_mesh.facets_area)
                if np.isclose(
                    obj_mesh.facets_normal[idx].dot(-local_gravity),
                    1.0,
                    atol=gravity_tolerance,
                )
                and obj_mesh.facets_area[idx] > min_area
            ]

            for index in support_facet_indices:
                normal = obj_mesh.facets_normal[index]
                origin = obj_mesh.facets_origin[index]

                facet_T = trimesh.geometry.plane_transform(origin, normal)
                facet_T_inv = trimesh.transformations.inverse_matrix(facet_T)
                vertices = trimesh.transform_points(obj_mesh.vertices, facet_T)[:, :2]

                # find boundary edges for the facet
                edges = obj_mesh.edges_sorted.reshape((-1, 6))[
                    obj_mesh.facets[index]
                ].reshape((-1, 2))
                group = trimesh.grouping.group_rows(edges, require_count=1)

                # run the polygon conversion
                polygons = trimesh.path.polygons.edges_to_polygons(
                    edges=edges[group], vertices=vertices
                )

                for polygon in polygons:
                    if polygon.geom_type == "MultiPolygon":
                        polys = list(polygon)
                    else:
                        polys = [polygon]

                    for poly in polys:
                        poly = poly.buffer(-erosion_distance)

                        if not poly.is_empty and poly.area > min_area:
                            support_surfaces.append(
                                SupportSurface(
                                    poly,
                                    index,
                                    obj_node_name,
                                    facet_T_inv,
                                )
                            )

        if len(support_surfaces) == 0:
            print("Warning! No support polygons selected.")
            return []

        support_data = []

        scene_mesh = self.scene.dump(concatenate=True)

        for support_surface in support_surfaces:
            (
                is_support_polyhedra,
                inscribing_polyhedra,
            ) = self._compute_support_polyhedra(
                support_surface=support_surface,
                mesh=scene_mesh,
                gravity=gravity,
                ray_cast_count=10,
                min_volume=min_volume,
                distance_above_support=distance_above_support,
                max_height=10.0,
                erosion_distance=erosion_distance,
            )
            if is_support_polyhedra:
                support_data.append(
                    Container(
                        geometry=inscribing_polyhedra,
                        node_name=support_surface.node_name,
                        transform=support_surface.transform,
                        support_surface=support_surface,
                    )
                )

        if len(support_data) == 0:
            print(f"No containers found.")

        return support_data

    def open_drawer(self, i: int):
        """Opens the i'th drawer of the dresser.

        Args:
            i (int): Index of the drawer that will be opened.
        """
        joint_name = self.joint_names[i]
        configuration = self.dresser_asset._model.joint_map[joint_name].limit.upper
        self.dresser_asset.update_config({joint_name: configuration})

        self.dresser_asset.update_transformations(self.scene)

    def close_drawer(self, i: int):
        """Closes the i'th drawer of the dresser.

        Args:
            i (int): Index of the drawer that will be closed.
        """
        joint_name = self.joint_names[i]
        configuration = self.dresser_asset._model.joint_map[joint_name].limit.lower
        self.dresser_asset.update_config({joint_name: configuration})

        self.dresser_asset.update_transformations(self.scene)

    def get_support_volume(self, idx: int, vertical_offset: float = 0.05) -> Cuboid:
        """Returns the volume describing the i'th drawer of the dresser.

        Args:
            idx (int): Index of the drawer.
            vertical_offset (float, optional): Volume is translated and reduced in z direction by this offset. Defaults to 0.05.

        Returns:
            Cuboid: Volume represented as a box.
        """
        try:
            volume = self.scene_containers[f"volume_{idx}"][0]
        except (KeyError, IndexError) as e:
            return None

        T = self.scene.graph.get(volume.node_name)[0] @ volume.transform
        center_pose = np.eye(4)
        # Use the actual AABB center, not extents/2.  The inscribing
        # polyhedra is created by extrude_polygon in the facet's local frame
        # where the polygon is NOT necessarily anchored at the origin.
        # extents/2 only equals the center when bounds[0] == [0,0,0].
        center_pose[:3, 3] = np.asarray(volume.geometry.bounds).mean(axis=0)
        center_pose = T @ center_pose
        dims = volume.geometry.extents
        # Ensures the the gripper is always at least 5 centimeters inside
        # the drawer
        # This is okay to do because these boxes are always oriented up
        dims = [*dims[:2], dims[2] - vertical_offset]
        center = [
            center_pose[0, 3],
            center_pose[1, 3],
            center_pose[2, 3] - vertical_offset / 2,
        ]
        pose = SE3.from_matrix(center_pose)

        return Cuboid(
            center=center,
            dims=dims,
            quaternion=pose.so3.wxyz,
        )

    def random_pose_and_config(
        self,
        sim: Bullet,
        arm: Union[BulletFranka, BulletAubo],
        selfcc: Union[FrankaSelfCollisionChecker, AuboCollisionSpheres],
        support_volume: Cuboid,
        primitive_arrays: Optional[List[Any]] = None,
        drawer_idx: Optional[int] = None,
    ) -> Tuple[Optional[SE3], Optional[np.ndarray]]:
        if primitive_arrays is None:
            primitive_arrays = self._build_primitive_arrays()
        if self._uses_aubo(selfcc):
            if drawer_idx is not None:
                # Strategy 1: front-entry approach (wrist points into drawer)
                pose, config = self._aubo_front_entry_pose_and_config(
                    sim, arm, selfcc, support_volume, primitive_arrays, drawer_idx,
                )
                if pose is not None and config is not None:
                    return pose, config
                # Strategy 2: top-down approach (wrist points straight down)
                pose, config = self._aubo_topdown_pose_and_config(
                    sim, arm, selfcc, support_volume, primitive_arrays,
                )
                if pose is not None and config is not None:
                    return pose, config
            # Strategy 3: random config sampling (last resort)
            return self._aubo_config_sample_pose_and_config(
                sim, arm, selfcc, support_volume, primitive_arrays,
            )
        samples = support_volume.sample_volume(self._drawer_pose_sample_count(selfcc))

        pose, q = None, None
        for sample in samples:
            theta = radius_sample(0.0, self._drawer_theta_radius(selfcc))
            x = np.array([np.cos(theta), np.sin(theta), 0.0], dtype=np.float64)
            z = np.array([0.0, 0.0, -1.0], dtype=np.float64)
            y = np.cross(z, x)
            pose = SE3.from_unit_axes(
                origin=sample,
                x=x,
                y=y,
                z=z,
            )
            if self._eef_in_collision(pose, selfcc, primitive_arrays, arm):
                pose = None
                continue
            q = self._collision_free_ik(pose, selfcc, primitive_arrays, arm)
            if q is None:
                pose = None
                continue
            arm.marionette(q)
            if sim.in_collision(arm, check_self=True):
                pose, q = None, None
                continue
            break
        return pose, q

    def _gen_neutral_candidates(
        self, how_many: int, selfcc: Union[FrankaSelfCollisionChecker, AuboCollisionSpheres]
    ) -> List[NeutralCandidate]:
        sim = Bullet(gui=False)
        arm = self._load_planning_arm(sim, selfcc)
        sim.load_primitives(self.obstacles)
        primitive_arrays = self._build_primitive_arrays()
        candidates: List[NeutralCandidate] = []
        for _ in range(how_many * 50):
            if len(candidates) >= how_many:
                break
            sample = self._random_neutral_config(selfcc)
            arm.marionette(sample)
            if not (
                sim.in_collision(arm, check_self=True)
                or self._has_self_collision(selfcc, sample, arm)
            ):
                pose = self._forward_kinematics(sample, selfcc)
                if not self._eef_in_collision(pose, selfcc, primitive_arrays, arm):
                    candidates.append(
                        NeutralCandidate(
                            config=sample,
                            pose=pose,
                            negative_volumes=[
                                x.support_volume for x in self.demo_candidates
                            ],
                        )
                    )
        return candidates

    def _gen_additional_candidate_sets(
        self, how_many: int, selfcc: Union[FrankaSelfCollisionChecker, AuboCollisionSpheres]
    ) -> List[List[TaskOrientedCandidate]]:
        """
        Creates additional candidates, where the candidates correspond to the support volumes
        of the environment's generated candidates (created by the `gen` function)

        :param how_many int: How many candidates to generate in each support volume (the result is guaranteed
                             to match this number or the function will run forever)
        :param selfcc Union[FrankaSelfCollisionChecker, AuboCollisionSpheres]: Checks for self
                                                  collisions using robot-specific sphere models.
        :rtype List[List[TaskOrientedCandidate]]: A pair of candidate sets, where each has `how_many`
                                      candidates that matches the corresponding support volume
                                      for the respective element in `self.demo_candidates`
        """
        start_support = self.demo_candidates[0].support_volume
        target_support = self.demo_candidates[1].support_volume
        candidate_sets: List[List[TaskOrientedCandidate]] = []

        sim = Bullet(gui=False)
        arm = self._load_planning_arm(sim, selfcc)
        sim.load_primitives(self.obstacles)
        primitive_arrays = self._build_primitive_arrays()

        for idx, _ in enumerate(self.demo_candidates):
            candidate_set: List[TaskOrientedCandidate] = []
            ii = 0
            while ii < how_many:
                pose, q = self.random_pose_and_config(
                    sim,
                    arm,
                    selfcc,
                    self.demo_candidates[idx].support_volume,
                    primitive_arrays,
                    drawer_idx=self.demo_candidates[idx].drawer_idx,
                )
                if pose is not None and q is not None:
                    candidate_set.append(
                        DresserCandidate(
                            pose=pose,
                            config=q,
                            drawer_idx=self.demo_candidates[idx].drawer_idx,
                            support_volume=self.demo_candidates[idx].support_volume,
                            negative_volumes=self.demo_candidates[idx].negative_volumes,
                        )
                    )
                    ii += 1
            candidate_sets.append(candidate_set)
        return candidate_sets

    @property
    def obstacles(self) -> List[Cuboid]:
        """
        Returns all cuboids in the scene (and the scene is entirely made of cuboids)

        :rtype List[Cuboid]: The list of cuboids in this scene
        """
        return self.cuboids

    @property
    def cuboids(self):
        """
        Returns all cuboids in the scene (and the scene is entirely made of cuboids)

        :rtype List[Cuboid]: The list of cuboids in this scene
        """
        cuboids = []
        for node in self.scene.graph.nodes_geometry:
            transform, geom_name = self.scene.graph[node]
            geom = self.scene.geometry[geom_name]
            metadata = self.scene.geometry[geom_name].metadata
            if isinstance(geom, trimesh.primitives.Box):
                box_offset = tra.translation_matrix(geom.centroid)
                transform = transform @ box_offset
                pose = SE3.from_matrix(transform)
                c = Cuboid(
                    center=pose.xyz,
                    quaternion=pose.so3.wxyz,
                    dims=geom.extents,
                )
                cuboids.append(c)
        # Add a ground plane for compact-workspace robots (e.g. Aubo i3H)
        # so that arm-ground collisions are detected during data generation.
        if self._uses_compact_workspace_profile():
            ground_thickness = 0.02
            cuboids.append(
                Cuboid(
                    center=[0.0, 0.0, -ground_thickness / 2],
                    dims=[4.0, 4.0, ground_thickness],
                    quaternion=[1.0, 0.0, 0.0, 0.0],
                )
            )
        return cuboids

    @property
    def cylinders(self) -> List[Cylinder]:
        """
        Returns an empty list because there are no cylinders in this scene, but left in
        to conform to the standard

        :rtype List[Cuboid]: The list of cuboids in this scene
        """
        return []


class Dresser:
    def __init__(
        self,
        width: float,
        depth: float,
        height: float,
        transform: np.ndarray,
        split_prob: float = 0.7,
        split_decay: float = 0.8,
        min_split_size: float = 0.3,
    ):
        """Procedural dresser generator based on recursive splitting of compartments.

        Args:
            width (float): Width of dresser.
            depth (float): Depth of dresser.
            height (float): Height of dresser.
            split_prob (float, optional): The probability of splitting a compartment into two. Defaults to 0.7.
            split_decay (float, optional): The decay rate of the splitting probability for each level of recursion. Defaults to 0.8.
            min_split_size (float, optional): Minimum dimension for a compartment to be eligible for splitting. Defaults to 0.3.
        """
        self.width = width
        self.depth = depth
        self.height = height
        self.min_split_size = min_split_size

        self.num_drawers = 0

        name = "dresser"
        self._dresser = yourdfpy.Robot(name=name)

        self.body_name = f"{name}_body"
        self._add_body(
            name=self.body_name,
            width=width,
            depth=depth,
            height=height,
        )

        # run recursive splits
        self._split(
            x=0,
            y=0,
            width=self.width,
            height=self.height,
            prob=split_prob,
            decay=split_decay,
        )

        self._fname = ".urdf"
        self._scale = 1.0
        self._origin = np.eye(4)

        self._raw_model = yourdfpy.URDF(
            robot=self._dresser,
            build_scene_graph=True,
            load_meshes=True,
            build_collision_scene_graph=True,
            load_collision_meshes=True,
            force_mesh=False,
            force_collision_mesh=False,
        )
        self._model = self._raw_model

        self._configuration = np.zeros(len(self._model.actuated_joint_names))

        self._update_bounds()
        self._origin = (
            tra.inverse_matrix(
                tra.translation_matrix(
                    [self.centroid[0], self.centroid[1], self.bounds[0, 2]]
                )
            )
            @ transform
        )
        self._update_bounds()

    def update_transformations(self, scene: trimesh.Scene):
        """Helper function to update the transformations of a scene graph according to the current configuration of the dresser.

        Args:
            scene (trimesh.Scene): Scene graph to be updated.
        """
        object_scene = self.as_trimesh_scene(namespace="dresser")
        for edge in object_scene.graph.to_edgelist():
            scene.graph.update(
                frame_from=edge[0],
                frame_to=edge[1],
                matrix=edge[2]["matrix"],
                extras=scene.graph.transforms.edge_data[edge[0], edge[1]].get("extras"),
            )

    def update_config(self, joint_dict: Dict[str, float]):
        """Update configuration of articulated parts of dresser.

        Args:
            joint_dict (dict[str, float]): Mapping of joint names and values.
        """
        self._model.update_cfg(joint_dict)
        self._configuration = self._model.cfg.copy()

    def _get_reference_frame(self) -> np.ndarray:
        """Return translation relative to the mesh's boundaries.

        Args:
            mesh (trimesh.Trimesh): Mesh.
            x (str): "bottom" or "centroid"
            y (str): "bottom" or "centroid"
            z (str): "bottom" or "centroid"

        Returns:
            np.ndarray: 4x4 homogeneous matrix.
        """
        translation = [self.centroid[0], self.centroid[1], self.bounds[0, 2]]
        return tra.translation_matrix(translation)

    def _merge_trimesh_scenes(
        self, scene1: trimesh.Scene, scene2: trimesh.Scene
    ) -> trimesh.Scene:
        """Merge two or more scenes. Uses trimesh.append_scenes internally but in addition finds common nodes by checking their transforms.

        Args:
            *scenes (*trimesh.Scene): A variable number of trimesh scenes

        Returns:
            trimesh.Scene: The resulting merged scene.
        """
        base_frame = scene1.graph.base_frame

        # Nodes with the same name are potentially representing the same location
        potentially_shared_nodes = set(scene1.graph.transforms.nodes) & set(
            scene2.graph.transforms.nodes
        )

        # Ensure that the poses of both nodes are identical
        shared_nodes = list(
            {
                shared_node
                for shared_node in potentially_shared_nodes
                if np.allclose(
                    scene1.graph.get(shared_node)[0], scene2.graph.get(shared_node)[0]
                )
            }
        )

        return trimesh.scene.scene.append_scenes(
            [scene1, scene2], common=shared_nodes, base_frame=base_frame
        )

    def _update_bounds(self):
        """Update bounds of the dresser"""
        mesh = self.as_trimesh_scene(namespace="dresser")

        self.extents = mesh.extents
        self.bounds = mesh.bounds

        # used for assigning frames relative to centroid or CoM
        self.centroid = mesh.centroid

    def _scaled_mesh(self, mesh, orientation=None) -> trimesh.Trimesh:
        """Scale mesh.

        Args:
            mesh (trimesh.Trimesh): Mesh to scale.
            orientation (np.ndarray, optional): 4x4 homogeneous matrix. Defaults to None.

        Returns:
            trimesh.Trimesh: Scaled mesh.
        """
        if mesh is None:
            return None

        try:
            scaled_mesh = mesh.copy(include_cache=True)
        except TypeError:
            # trimesh primitives in newer releases no longer accept include_cache.
            scaled_mesh = mesh.copy()

        my_scale = self._scale
        if orientation is not None:
            try:
                my_scale = np.abs(orientation[:3, :3].T @ self._scale)
            except Exception as _:
                my_scale = self._scale

        scaled_mesh.apply_scale(my_scale)

        return scaled_mesh

    def _scaled_transform(self, transform: np.ndarray) -> np.ndarray:
        """Scale transform

        Args:
            transform (np.ndarray): 4x4 homogeneous transform.

        Returns:
            np.ndarray: 4x4 scaled homogenous transform
        """
        scaled_transform = np.copy(transform)

        my_scale = self._scale
        scaled_transform[:3, 3] *= my_scale
        return scaled_transform

    def as_trimesh_scene(self, namespace: str) -> trimesh.Scene:
        """Return dresser as a trimesh.Scene graph. Always uses the collision geometry from the URDF

        Args:
            namespace (str): Name prefix added to all nodes in the scene graph.

        Returns:
            trimesh.Scene: Scene representing the dresser.
        """
        s = trimesh.Scene(base_frame=namespace)

        s.metadata["joint_names"] = []
        s.metadata["joint_configuration"] = []
        s.metadata["articulations"] = {}

        self._model.update_cfg(self._configuration)

        urdf_scene = self._model.collision_scene

        # copy nodes and edges from original scene graph
        # and change identifiers by prepending a namespace
        edges = []
        for a, b, attr in urdf_scene.graph.to_edgelist():
            if "geometry" in attr:
                attr["geometry"] = f"{namespace}/{attr['geometry']}"

            if "matrix" in attr:
                attr["matrix"] = self._scaled_transform(np.array(attr["matrix"]))

            # rename nodes with additional namespace
            edges.append((f"{namespace}/{a}", f"{namespace}/{b}", attr))

        # add base link
        edges.append(
            (
                s.graph.base_frame,
                f"{namespace}/{urdf_scene.graph.base_frame}",
                {
                    "matrix": self._origin.tolist(),
                    "extras": {
                        "joint": {
                            "name": f"{namespace}/origin_joint",
                            "type": "fixed",
                        }
                    },
                },
            )
        )

        # copy geometries
        geometry = {}
        for k, v in urdf_scene.geometry.items():
            geometry[f"{namespace}/{k}"] = self._scaled_mesh(v)

            # add metadata about whether this is collision or visual geometry
            geometry[f"{namespace}/{k}"].metadata["layer"] = (
                "collision"
                if k in self._model.collision_scene.geometry.keys()
                else "visual"
            )

        s.graph.from_edgelist(edges, strict=True)
        s.geometry.update(geometry)

        # extract articulation information
        for joint in self._model.robot.joints:
            parent_node_name = f"{namespace}/{joint.parent}"
            node_name = f"{namespace}/{joint.child}"

            # add articulation data as edge attributes
            s.graph.transforms.edge_data[(parent_node_name, node_name)].update(
                {
                    "extras": {
                        "joint": {
                            "name": f"{namespace}/{joint.name}",
                            "type": joint.type,
                            "axis": joint.axis.tolist()
                            if not joint.type == "fixed"
                            else [1.0, 0, 0],
                            "limit_velocity": getattr(joint.limit, "velocity", 10.0),
                            "limit_effort": getattr(joint.limit, "effort", 10.0),
                            "limit_lower": getattr(joint.limit, "lower", -10.0),
                            "limit_upper": getattr(joint.limit, "upper", 10.0),
                        }
                    }
                }
            )

            if joint.name in self._model.actuated_joint_names:
                s.metadata["joint_names"].append(f"{namespace}/{joint.name}")
                s.metadata["joint_configuration"] = np.append(
                    s.metadata["joint_configuration"], 0.0
                )
                s.metadata["articulations"][(parent_node_name, node_name)] = {
                    "name": f"{namespace}/{joint.name}",
                    "type": joint.type,
                    "axis": joint.axis.tolist()
                    if not joint.type == "fixed"
                    else [1.0, 0, 0],
                    "limit_velocity": getattr(joint.limit, "velocity", 10.0),
                    "limit_effort": getattr(joint.limit, "effort", 10.0),
                    "limit_lower": getattr(joint.limit, "lower", -10.0),
                    "limit_upper": getattr(joint.limit, "upper", 10.0),
                }

                # add default information for export
                if (
                    s.metadata["articulations"][(parent_node_name, node_name)][
                        "limit_velocity"
                    ]
                    is None
                ):
                    s.metadata["articulations"][(parent_node_name, node_name)][
                        "limit_velocity"
                    ] = 10.0
                if (
                    s.metadata["articulations"][(parent_node_name, node_name)][
                        "limit_effort"
                    ]
                    is None
                ):
                    s.metadata["articulations"][(parent_node_name, node_name)][
                        "limit_effort"
                    ] = 10.0
                if (
                    s.metadata["articulations"][(parent_node_name, node_name)][
                        "limit_lower"
                    ]
                    is None
                ):
                    s.metadata["articulations"][(parent_node_name, node_name)][
                        "limit_lower"
                    ] = -10.0
                if (
                    s.metadata["articulations"][(parent_node_name, node_name)][
                        "limit_upper"
                    ]
                    is None
                ):
                    s.metadata["articulations"][(parent_node_name, node_name)][
                        "limit_upper"
                    ] = 10.0

        return s

    def _split(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        prob: float = 0.8,
        decay: float = 0.9,
    ):
        """Recursive dresser splitting.

        Args:
            x (float): x-coordinate of 2D coordinates of dresser front.
            y (float): y-coordinate of 2D coordinates of dresser front.
            width (float): Width of 2D dresser front.
            height (float): Height of 2D dresser front.
            prob (float, optional): Probabiliy of splitting further. Defaults to 0.8.
            decay (float, optional): Decay rate of splitting probability. Defaults to 0.9.
        """
        min_sz = self.min_split_size

        # check split probability
        rand = np.random.random()
        do_split = rand < prob

        # minimum size for splitting
        if width < min_sz and height < min_sz:
            do_split = False

        if not do_split:
            self._add_drawer(
                parent=self.body_name,
                x=x,
                y=y,
                width=width,
                height=height,
                drawer_depth=self.depth * 0.9,
            )
            return

        # decay the split probability
        prob_new = prob * decay

        # calculate wall position
        w_xyz = self._local_to_body(
            x + width / 2.0,
            y + height / 2.0,
        )
        w_xyz[1] -= self.depth / 2.0

        # decide on vertical/horizontal split
        vertical_split = np.random.random() < 0.5

        # minimum size for splitting vert/horiz
        if width < min_sz:
            vertical_split = False
        if height < min_sz:
            vertical_split = True

        # wrap call to make sure that we pass all params to the lower level
        def run_split(x, y, width, height, prob):
            self._split(
                x=x,
                y=y,
                width=width,
                height=height,
                prob=prob,
                decay=decay,
            )

        wall_thickness = 0.01
        if vertical_split:
            self._add_wall(
                link=self._dresser.links[0],
                origin=tra.translation_matrix(w_xyz),
                size=(wall_thickness, self.depth, height),
            )

            new_w = width / 2.0 - wall_thickness / 2.0
            r_left = (x, y, new_w, height)
            r_right = (x + width / 2.0 + wall_thickness / 2.0, y, new_w, height)

            run_split(
                x=r_left[0],
                y=r_left[1],
                width=r_left[2],
                height=r_left[3],
                prob=prob_new,
            )
            run_split(
                x=r_right[0],
                y=r_right[1],
                width=r_right[2],
                height=r_right[3],
                prob=prob_new,
            )
        else:  # horizontal split
            self._add_wall(
                link=self._dresser.links[0],
                origin=tra.translation_matrix(w_xyz),
                size=(width, self.depth, wall_thickness),
            )

            new_h = height / 2.0 - wall_thickness / 2.0
            r_top = (x, y, width, new_h)
            r_bot = (x, y + height / 2.0 + wall_thickness / 2, width, new_h)

            run_split(
                x=r_top[0],
                y=r_top[1],
                width=r_top[2],
                height=r_top[3],
                prob=prob_new,
            )
            run_split(
                x=r_bot[0],
                y=r_bot[1],
                width=r_bot[2],
                height=r_bot[3],
                prob=prob_new,
            )

    def _local_to_body(self, x: float, y: float) -> List[float]:
        """Converts local rectangle coordinates into dresser body coordinates
        The local coordinate frame has x pointing to the right and y to the bottom.

        Args:
            x (float): x-coordinate of local coordinates.
            y (float): y-coordinate of local coordinates.

        Returns:
            list (3,): 3D coordinates in dresser reference frame.
        """

        bx = self.width / 2.0 - x
        by = self.depth / 2.0
        bz = self.height - y
        return [bx, by, bz]

    def _create_box_visual(
        self, size: Sequence[float], origin: np.ndarray, name: str
    ) -> yourdfpy.Visual:
        """Create visual URDF element with box geometry.

        Args:
            size (list): 3D size of box.
            origin (np.ndarray): 4x4 homogenous matrix of box pose.
            name (str, optional): Name of visual element. Defaults to None.
            material (urdfpy.Material, optional): Material. Defaults to None.

        Returns:
            yourdfpy.Visual: Visual element.
        """
        return yourdfpy.Visual(
            name=name,
            geometry=yourdfpy.Geometry(box=yourdfpy.Box(size=size)),
            origin=origin,
            material=None,
        )

    def _create_box_collision(
        self, size: Sequence[float], origin: np.ndarray, name: str
    ) -> yourdfpy.Collision:
        """Create collision URDF element with box geometry.

        Args:
            size (list): 3D size of box.
            origin (np.ndarray): 4x4 homogenous matrix of box pose.
            name (str, optional): Name of collision element. Defaults to None.

        Returns:
            yourdfpy.Collision: Collision element.
        """
        return yourdfpy.Collision(
            name=name,
            geometry=yourdfpy.Geometry(box=yourdfpy.Box(size=size)),
            origin=origin,
        )

    def _add_body(self, name: str, width: float, depth: float, height: float):
        """Generate and add a dresser body link to the URDF.
        Top and Bottom boards are the same size as width and depth.
        The side boards cover the Bottom and Top boards on the sides.

        Args:
            name (str): Name of the link element.
            width (float): Width of interior of body link.
            depth (float): Depth of interior of body link.
            height (float): Height of interior of body link.
        """
        wall_thickness = 0.01
        boxes = [
            {
                "origin": tra.translation_matrix([0, 0, -wall_thickness / 2]), # 生成一个单位旋转矩阵的转换矩阵，pos为所填参数
                "size": (width, depth, wall_thickness),
            },
            {
                "origin": tra.translation_matrix([0, 0, height + wall_thickness / 2]),
                "size": (width, depth, wall_thickness),
            },
            # sideboards
            {
                "origin": tra.translation_matrix(
                    [width / 2 + wall_thickness / 2, 0, height / 2]
                ),
                "size": (wall_thickness, depth, height + 2 * wall_thickness),
            },
            {
                "origin": tra.translation_matrix(
                    [-width / 2 - wall_thickness / 2, 0, height / 2]
                ),
                "size": (wall_thickness, depth, height + 2 * wall_thickness),
            },
            # backboard
            {
                "origin": tra.translation_matrix(
                    [0, -depth / 2.0 + wall_thickness / 2.0, height / 2]
                ),
                "size": (
                    width + 2 * wall_thickness,
                    wall_thickness,
                    height + 2 * wall_thickness,
                ),
            },
        ]

        self._dresser.links.append(yourdfpy.Link(name=name))

        for i, board in enumerate(boxes):
            inertial = yourdfpy.Inertial(mass=0.1, inertia=np.eye(3), origin=np.eye(4))
            link_name = f"{name}_board_{i}"
            link = yourdfpy.Link(
                name=link_name,
                inertial=inertial,
                visuals=[
                    self._create_box_visual(
                        name=f"{name}_board_{i}",
                        origin=tra.identity_matrix(),
                        size=board["size"],
                    )
                ],
                collisions=[
                    self._create_box_collision(
                        name=f"{name}_board_{i}",
                        origin=tra.identity_matrix(),
                        size=board["size"],
                    )
                ],
            )

            joint = self._create_fixed_joint(
                parent=name,
                child=link_name,
                origin=board["origin"],
            )

            self._dresser.joints.append(joint)
            self._dresser.links.append(link)

    def _create_fixed_joint(
        self, parent: str, child: str, origin: np.ndarray
    ) -> yourdfpy.Joint:
        """Create a URDF joint element for a fixed joint.

        Args:
            parent (str): Name of parent link.
            child (str): Name of child link.
            origin (np.ndarray): 4x4 homogeneous matrix for joint pose.

        Returns:
            yourdfpy.Joint: Joint element.
        """

        return yourdfpy.Joint(
            name=parent + "_to_" + child,
            type="fixed",
            parent=parent,
            child=child,
            origin=origin,
        )

    def _create_prismatic_joint(
        self,
        parent: str,
        child: str,
        origin: np.ndarray,
        axis: np.ndarray,
        lower: float,
        upper: float,
    ) -> yourdfpy.Joint:
        """Create a URDF joint element for a prismatic joint.

        Args:
            parent (str): Name of parent link.
            child (str): Name of child link.
            origin (np.ndarray): 4x4 homogeneous matrix for joint pose.
            axis (tuple, optional): Joint axis.
            lower (float, optional): Lower joint limit. Defaults to 0.0.
            upper (float, optional): Upper joint limit. Defaults to 0.4.
            damping (float, optional): Joint damping. Defaults to 0.01.
            friction (float, optional): Joint friction. Defaults to 0.01.

        Returns:
            yourdfpy.Joint: Joint element.
        """
        return yourdfpy.Joint(
            name=parent + "_to_" + child,
            parent=parent,
            child=child,
            type="prismatic",
            origin=origin,
            axis=axis,
            dynamics=yourdfpy.Dynamics(damping=0.01, friction=0.01),
            limit=yourdfpy.Limit(effort=1000.0, lower=lower, upper=upper, velocity=1.0),
        )

    def _add_drawer(
        self,
        parent: str,
        width: float,
        height: float,
        x: float,
        y: float,
        drawer_depth: float,
    ):
        """Add a drawer with a prismatic joint.

        Args:
            parent (str): Name of parent link of prismatic joint.
            width (float): Width of drawer front.
            height (float): Height of drawer front.
            x (float, optional): Local x-coordinate. Defaults to 0.0.
            y (float, optional): Local y-coordinate. Defaults to 0.0.
            drawer_depth (float, optional): Depth of drawer part that goes inside dresser (depth without front board thickness). Defaults to 0.5.
        """
        name = "drawer_" + str(self.num_drawers)
        self.num_drawers += 1

        frontboard_thickness = 0.019
        wall_thickness = 0.004

        boxes = [
            # front
            {
                "origin": tra.translation_matrix((0, frontboard_thickness / 2, 0)),
                "size": (width, frontboard_thickness, height),
            },
            # bottom
            {
                "origin": tra.translation_matrix(
                    (
                        0,
                        -(drawer_depth - wall_thickness) / 2,
                        (wall_thickness - height) / 2,
                    )
                ),
                "size": (
                    width - 2 * wall_thickness,
                    drawer_depth - wall_thickness,
                    wall_thickness,
                ),
            },
            # left
            {
                "origin": tra.translation_matrix(
                    (
                        (width - wall_thickness) / 2,
                        (wall_thickness - drawer_depth) / 2,
                        0,
                    )
                ),
                "size": (wall_thickness, drawer_depth - wall_thickness, height),
            },
            # right
            {
                "origin": tra.translation_matrix(
                    (
                        (wall_thickness - width) / 2,
                        (wall_thickness - drawer_depth) / 2,
                        0,
                    )
                ),
                "size": (wall_thickness, drawer_depth - wall_thickness, height),
            },
            # back
            {
                "origin": tra.translation_matrix(
                    (0, -drawer_depth + wall_thickness / 2, 0)
                ),
                "size": (width, wall_thickness, height),
            },
        ]

        self._dresser.links.append(yourdfpy.Link(name=name))

        for i, board in enumerate(boxes):
            inertial = yourdfpy.Inertial(mass=0.1, inertia=np.eye(3), origin=np.eye(4))
            link_name = f"{name}_board_{i}"
            link = yourdfpy.Link(
                name=link_name,
                inertial=inertial,
                visuals=[
                    self._create_box_visual(
                        name=f"{name}_board_{i}",
                        origin=tra.identity_matrix(),
                        size=board["size"],
                    )
                ],
                collisions=[
                    self._create_box_collision(
                        name=f"{name}_board_{i}",
                        origin=tra.identity_matrix(),
                        size=board["size"],
                    )
                ],
            )

            joint = self._create_fixed_joint(
                # name=f"{name}_fixed_joint_{i}",
                parent=name,
                child=link_name,
                origin=board["origin"],
            )

            self._dresser.joints.append(joint)
            self._dresser.links.append(link)

        # create prismatic joint
        d_xyz = self._local_to_body(
            x + width / 2,
            y + height / 2,
        )
        self._dresser.joints.append(
            self._create_prismatic_joint(
                parent=parent,
                child=name,
                origin=tra.translation_matrix(d_xyz),
                axis=np.array([0, 1, 0]),
                lower=0.0,
                upper=drawer_depth * 0.9,
            )
        )

    def _add_wall(self, link: yourdfpy.Link, origin: np.ndarray, size: Sequence[float]):
        """Add URDF elements that represent visual and collision geometries for wall.

        Args:
            link (urdfpy.Link): URDF link to which this wall geometry will be added.
            origin (np.ndarray): 4x4 homogenous matrix of wall pose.
            size (list): 3D size of box representing wall.
        """
        new_link = yourdfpy.Link(
            name=f"{link.name}_geometry",
            visuals=[
                self._create_box_visual(
                    name=f"{link.name}_wall", size=size, origin=origin
                )
            ],
            collisions=[
                self._create_box_collision(
                    name=f"{link.name}_wall", size=size, origin=origin
                )
            ],
        )
        joint = self._create_fixed_joint(
            parent=link.name,
            child=new_link.name,
            origin=tra.identity_matrix(),
        )

        self._dresser.joints.append(joint)
        self._dresser.links.append(new_link)
