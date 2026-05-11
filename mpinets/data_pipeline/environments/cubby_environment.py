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

from dataclasses import dataclass
import random
from typing import Any, List, Optional, Tuple, Union

from geometrout.primitive import Cuboid, CuboidArray, Cylinder, CylinderArray
from geometrout.transform import SE3
from robofin.bullet import Bullet, BulletAubo, BulletFranka
from robofin.robots import FrankaRobot, FrankaRealRobot
from robofin.robots_aubo import AuboRobot
from robofin.collision import FrankaSelfCollisionChecker
from robofin.collision_aubo import AuboCollisionSpheres
import numpy as np

from mpinets.data_pipeline.environments.base_environment import (
    TaskOrientedCandidate,
    NeutralCandidate,
    Environment,
    radius_sample,
)


@dataclass
class CubbyCandidate(TaskOrientedCandidate):
    """
    Represents a configuration, its end-effector pose, and some metadata about the cubby
    (i.e. which cubby pocket it belongs to and the free space inside that cubby).
    """

    pocket_idx: int
    support_volume: Cuboid


class Cubby:
    """
    The actual cubby construction itself, without any robot info.
    """

    def __init__(self, max_reach: float = Environment.FRANKA_MAX_REACH):
        s = max_reach / Environment.FRANKA_MAX_REACH
        self.cubby_left = radius_sample(0.7 * s, 0.1 * s)
        self.cubby_right = radius_sample(-0.7 * s, 0.1 * s)
        self.cubby_front = radius_sample(0.55 * s, 0.1 * s)
        self.cubby_back = self.cubby_front + radius_sample(0.35 * s, 0.2 * s)
        self.cubby_mid_v_y = radius_sample(0.0, 0.1 * s)
        self.cubby_bottom = radius_sample(0.2, 0.1)
        self.cubby_top = radius_sample(0.7, 0.1)
        self.cubby_mid_h_z = radius_sample(0.45, 0.1)
        self.thickness = radius_sample(0.02, 0.01)
        self.middle_shelf_thickness = self.thickness
        self.center_wall_thickness = self.thickness
        self.in_cabinet_rotation = radius_sample(0, np.pi / 18)

    @property
    def rotation_matrix(self) -> np.ndarray:
        """
        Cubbies are essentially represented as unrotated boxes that are then rotated around
        their central yaw axis by `self.in_cabinet_rotation`. This function produces the
        rotation matrix corresponding to that value and axis.
        cabinet_T_world: Transforms from world frame to cabinet frame (where the cabinet is unrotated and centered at the origin)
        :rtype np.ndarray: The rotation matrix
        """
        cabinet_T_world = np.array(
            [
                [1, 0, 0, -(self.cubby_front + self.cubby_back) / 2],
                [0, 1, 0, -(self.cubby_left + self.cubby_right) / 2],
                [0, 0, 1, -(self.cubby_top + self.cubby_bottom) / 2],
                [0, 0, 0, 1],
            ]
        )
        in_cabinet_rotation = np.array(
            [
                [
                    np.cos(self.in_cabinet_rotation),
                    -np.sin(self.in_cabinet_rotation),
                    0,
                    0,
                ],
                [
                    np.sin(self.in_cabinet_rotation),
                    np.cos(self.in_cabinet_rotation),
                    0,
                    0,
                ],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        world_T_cabinet = np.array(
            [
                [1, 0, 0, (self.cubby_front + self.cubby_back) / 2],
                [0, 1, 0, (self.cubby_left + self.cubby_right) / 2],
                [0, 0, 1, (self.cubby_top + self.cubby_bottom) / 2],
                [0, 0, 0, 1],
            ]
        )
        pivot = np.matmul(
            world_T_cabinet, np.matmul(in_cabinet_rotation, cabinet_T_world)
        )
        return pivot

    def _unrotated_cuboids(self) -> List[Cuboid]:
        """
        Returns the unrotated cuboids that must then be rotated to produce the final cubby.

        :rtype List[Cuboid]: All the cuboids in the cubby
        """
        cuboids = [
            # Back Wall
            Cuboid(
                center=[
                    self.cubby_back,
                    (self.cubby_left + self.cubby_right) / 2,
                    self.cubby_top / 2,
                ],
                dims=[
                    self.thickness,
                    (self.cubby_left - self.cubby_right),
                    self.cubby_top,
                ],
                quaternion=[1, 0, 0, 0],
            ),
            # Bottom Shelf
            Cuboid(
                center=[
                    (self.cubby_front + self.cubby_back) / 2,
                    (self.cubby_left + self.cubby_right) / 2,
                    self.cubby_bottom,
                ],
                dims=[
                    self.cubby_back - self.cubby_front,
                    self.cubby_left - self.cubby_right,
                    self.thickness,
                ],
                quaternion=[1, 0, 0, 0],
            ),
            # Top Shelf
            Cuboid(
                center=[
                    (self.cubby_front + self.cubby_back) / 2,
                    (self.cubby_left + self.cubby_right) / 2,
                    self.cubby_top,
                ],
                dims=[
                    self.cubby_back - self.cubby_front,
                    self.cubby_left - self.cubby_right,
                    self.thickness,
                ],
                quaternion=[1, 0, 0, 0],
            ),
            # Right Wall
            Cuboid(
                center=[
                    (self.cubby_front + self.cubby_back) / 2,
                    self.cubby_right,
                    (self.cubby_top + self.cubby_bottom) / 2,
                ],
                dims=[
                    self.cubby_back - self.cubby_front,
                    self.thickness,
                    (self.cubby_top - self.cubby_bottom) + self.thickness,
                ],
                quaternion=[1, 0, 0, 0],
            ),
            # Left Wall
            Cuboid(
                center=[
                    (self.cubby_front + self.cubby_back) / 2,
                    self.cubby_left,
                    (self.cubby_top + self.cubby_bottom) / 2,
                ],
                dims=[
                    self.cubby_back - self.cubby_front,
                    self.thickness,
                    (self.cubby_top - self.cubby_bottom) + self.thickness,
                ],
                quaternion=[1, 0, 0, 0],
            ),
        ]
        if not np.isclose(self.cubby_mid_v_y, 0.0):
            # Center Wall (vertical)
            cuboids.append(
                Cuboid(
                    center=[
                        (self.cubby_front + self.cubby_back) / 2,
                        self.cubby_mid_v_y,
                        (self.cubby_top + self.cubby_bottom) / 2,
                    ],
                    dims=[
                        self.cubby_back - self.cubby_front,
                        self.center_wall_thickness,
                        self.cubby_top - self.cubby_bottom + self.thickness,
                    ],
                    quaternion=[1, 0, 0, 0],
                )
            )
        if not np.isclose(self.cubby_mid_h_z, 0.0):
            # Middle Shelf
            cuboids.append(
                Cuboid(
                    center=[
                        (self.cubby_front + self.cubby_back) / 2,
                        (self.cubby_left + self.cubby_right) / 2,
                        self.cubby_mid_h_z,
                    ],
                    dims=[
                        self.cubby_back - self.cubby_front,
                        self.cubby_left - self.cubby_right,
                        self.middle_shelf_thickness,
                    ],
                    quaternion=[1, 0, 0, 0],
                )
            )
        return cuboids

    @property
    def cuboids(self) -> List[Cuboid]:
        """
        Returns the cuboids that make up the cubby
        :rtype List[Cuboid]: The cuboids that make up each section of the cubby
        """
        cuboids: List[Cuboid] = []
        for cuboid in self._unrotated_cuboids():
            center = cuboid.center
            new_matrix = np.matmul(
                self.rotation_matrix,
                np.array(
                    [
                        [1, 0, 0, center[0]],
                        [0, 1, 0, center[1]],
                        [0, 0, 1, center[2]],
                        [0, 0, 0, 1],
                    ]
                ),
            )
            pose = SE3.from_matrix(new_matrix)
            cuboids.append(Cuboid(pose.xyz, cuboid.dims, quaternion=pose.so3.wxyz))
        return cuboids

    @property
    def support_volumes(self) -> List[Cuboid]:
        """
        Returns the support volumes inside each of the cubby pockets.
        These support volumes could be tighter to make for more efficient environment
        queries, but right now they include half of the surrounding shelves.

        :rtype List[Cuboid]: The list of support volumes
        """
        if np.isclose(self.center_wall_thickness, 0) and np.isclose(
            self.middle_shelf_thickness, 0
        ):
            centers = [
                np.array(
                    [
                        self.cubby_front + self.cubby_back,
                        self.cubby_left + self.cubby_right,
                        self.cubby_top + self.cubby_bottom,
                    ]
                )
                / 2
            ]
            dims = [
                np.array(
                    [
                        self.cubby_back - self.cubby_front,
                        self.cubby_left - self.cubby_right,
                        self.cubby_top - self.cubby_bottom,
                    ]
                )
            ]
        elif np.isclose(self.center_wall_thickness, 0):
            centers = [
                np.array(
                    [
                        (self.cubby_front + self.cubby_back) / 2,
                        (self.cubby_left + self.cubby_right) / 2,
                        (self.cubby_top + self.cubby_mid_h_z) / 2,
                    ]
                ),
                np.array(
                    [
                        (self.cubby_front + self.cubby_back) / 2,
                        (self.cubby_left + self.cubby_right) / 2,
                        (self.cubby_bottom + self.cubby_mid_h_z) / 2,
                    ]
                ),
            ]
            dims = [
                np.array(
                    [
                        (self.cubby_back - self.cubby_front),
                        (self.cubby_left - self.cubby_right),
                        (self.cubby_top - self.cubby_mid_h_z),
                    ]
                ),
                np.array(
                    [
                        (self.cubby_back - self.cubby_front),
                        (self.cubby_left - self.cubby_right),
                        (self.cubby_mid_h_z - self.cubby_bottom),
                    ]
                ),
            ]
        elif np.isclose(self.middle_shelf_thickness, 0):
            centers = [
                np.array(
                    [
                        (self.cubby_front + self.cubby_back) / 2,
                        (self.cubby_left + self.cubby_mid_v_y) / 2,
                        (self.cubby_top + self.cubby_bottom) / 2,
                    ]
                ),
                np.array(
                    [
                        (self.cubby_front + self.cubby_back) / 2,
                        (self.cubby_mid_v_y + self.cubby_right) / 2,
                        (self.cubby_top + self.cubby_bottom) / 2,
                    ]
                ),
            ]
            dims = [
                np.array(
                    [
                        (self.cubby_back - self.cubby_front),
                        (self.cubby_left - self.cubby_mid_v_y),
                        (self.cubby_top - self.cubby_bottom),
                    ]
                ),
                np.array(
                    [
                        (self.cubby_back - self.cubby_front),
                        (self.cubby_mid_v_y - self.cubby_right),
                        (self.cubby_top - self.cubby_bottom),
                    ]
                ),
            ]
        else:
            centers = [
                np.array(
                    [
                        (self.cubby_front + self.cubby_back) / 2,
                        (self.cubby_left + self.cubby_mid_v_y) / 2,
                        (self.cubby_top + self.cubby_mid_h_z) / 2,
                    ]
                ),
                np.array(
                    [
                        (self.cubby_front + self.cubby_back) / 2,
                        (self.cubby_mid_v_y + self.cubby_right) / 2,
                        (self.cubby_top + self.cubby_mid_h_z) / 2,
                    ]
                ),
                np.array(
                    [
                        (self.cubby_front + self.cubby_back) / 2,
                        (self.cubby_left + self.cubby_mid_v_y) / 2,
                        (self.cubby_bottom + self.cubby_mid_h_z) / 2,
                    ]
                ),
                np.array(
                    [
                        (self.cubby_front + self.cubby_back) / 2,
                        (self.cubby_mid_v_y + self.cubby_right) / 2,
                        (self.cubby_bottom + self.cubby_mid_h_z) / 2,
                    ]
                ),
            ]
            dims = [
                np.array(
                    [
                        (self.cubby_back - self.cubby_front),
                        (self.cubby_left - self.cubby_mid_v_y),
                        (self.cubby_top - self.cubby_mid_h_z),
                    ]
                ),
                np.array(
                    [
                        (self.cubby_back - self.cubby_front),
                        (self.cubby_mid_v_y - self.cubby_right),
                        (self.cubby_top - self.cubby_mid_h_z),
                    ]
                ),
                np.array(
                    [
                        (self.cubby_back - self.cubby_front),
                        (self.cubby_left - self.cubby_mid_v_y),
                        (self.cubby_mid_h_z - self.cubby_bottom),
                    ]
                ),
                np.array(
                    [
                        (self.cubby_back - self.cubby_front),
                        (self.cubby_mid_v_y - self.cubby_right),
                        (self.cubby_mid_h_z - self.cubby_bottom),
                    ]
                ),
            ]

        volumes = []
        for c, d in zip(centers, dims):
            unrotated_pose = np.eye(4)
            unrotated_pose[:3, 3] = c
            pose = SE3.from_matrix(np.matmul(self.rotation_matrix, unrotated_pose))
            volumes.append(Cuboid(center=pose.xyz, dims=d, quaternion=pose.so3.wxyz))
        return volumes


class CubbyEnvironment(Environment):
    def __init__(self, max_reach: float = Environment.FRANKA_MAX_REACH):
        super().__init__()
        self.max_reach = max_reach
        self.demo_candidates = []

    def gen_scene(self, **kwargs) -> bool:
        self.cubby = Cubby(max_reach=self.max_reach)
        self.generated = True
        return True

    def _uses_aubo(self, selfcc: Union[FrankaSelfCollisionChecker, AuboCollisionSpheres]) -> bool:
        return isinstance(selfcc, AuboCollisionSpheres)

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
        """
        Generates an environment and a pair of start/end candidates

        :param selfcc Union[FrankaSelfCollisionChecker, AuboCollisionSpheres]: Checks for self collisions using spheres that
                                                  mimic the internal Franka/Aubo collision checker.
        :rtype bool: Whether the environment was successfully generated
        """
        self.cubby = Cubby(max_reach=self.max_reach)
        support_idxs = np.arange(len(self.cubby.support_volumes))
        random.shuffle(support_idxs)
        supports = self.cubby.support_volumes

        sim = Bullet(gui=False)
        sim.load_primitives(self.cubby.cuboids)
        arm = self._load_planning_arm(sim, selfcc)
        primitive_arrays = self._build_primitive_arrays()

        (
            start_pose,
            start_q,
            start_support_volume,
            target_pose,
            target_q,
            target_support_volume,
        ) = (None, None, None, None, None, None)

        for ii, idx in enumerate(support_idxs):
            start_support_volume = supports[idx]
            start_pose, start_q = self.random_pose_and_config(
                sim,
                arm,
                selfcc,
                start_support_volume,
                primitive_arrays,
            )
            if start_pose is None or start_q is None:
                continue
            for jdx in support_idxs[ii + 1 :]:
                target_support_volume = supports[jdx]
                target_pose, target_q = self.random_pose_and_config(
                    sim,
                    arm,
                    selfcc,
                    target_support_volume,
                    primitive_arrays,
                )
                if target_pose is not None and target_q is not None:
                    break
            if target_pose is not None and target_q is not None:
                break

        if start_q is None or target_q is None:
            self.demo_candidates = None
            return False
        self.demo_candidates = (
            CubbyCandidate(
                pose=start_pose,
                config=start_q,
                pocket_idx=idx,
                support_volume=start_support_volume,
                negative_volumes=[s for ii, s in enumerate(supports) if ii != idx],
            ),
            CubbyCandidate(
                pose=target_pose,
                config=target_q,
                pocket_idx=jdx,
                support_volume=target_support_volume,
                negative_volumes=[s for ii, s in enumerate(supports) if ii != jdx],
            ),
        )
        return True

    def random_pose_and_config(
        self,
        sim: Bullet,
        arm: Union[BulletFranka, BulletAubo],
        selfcc: Union[FrankaSelfCollisionChecker, AuboCollisionSpheres],
        support_volume: Cuboid,
        primitive_arrays: Optional[List[Any]] = None,
    ) -> Tuple[Optional[SE3], Optional[np.ndarray]]:
        """
        Creates a random end effector pose in the desired support volume and solves for
        collision free IK.

        :param sim Bullet: A simulator already loaded with the obstacles
        :param arm Union[BulletFranka, BulletAubo]: The simulated arm used for final collision
                                                    validation
        :param selfcc Union[FrankaSelfCollisionChecker, AuboCollisionSpheres]: Checks for self
                                                                               collisions using
                                                                               robot-specific
                                                                               sphere models
        :param support_volume Cuboid: The desired support volume to search inside
        :rtype Tuple[Optional[SE3], Optional[np.ndarray]]: Returns a valid pose and IK solution if
                                                          one is found. Otherwise, returns None, None
        """
        if primitive_arrays is None:
            primitive_arrays = self._build_primitive_arrays()
        samples = support_volume.sample_volume(100)
        pose, q = None, None
        for sample in samples:
            theta = radius_sample(0, np.pi / 4)
            z = np.array([np.cos(theta), np.sin(theta), 0.0], dtype=np.float64)
            x = np.array([0.0, 0.0, -1.0], dtype=np.float64)
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
        """
        Generates a set of neutral candidates (all collision free)

        :param how_many int: How many to generate ideally--can be less if there are a lot of failures
        :param selfcc Union[FrankaSelfCollisionChecker, AuboCollisionSpheres]: Checks for self
                                                  collisions using robot-specific sphere models.
        :rtype List[NeutralCandidate]: A list of neutral candidates
        """
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
                            negative_volumes=self.cubby.support_volumes,
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
        candidate_sets = []

        sim = Bullet(gui=False)
        arm = self._load_planning_arm(sim, selfcc)
        sim.load_primitives(self.obstacles)
        primitive_arrays = self._build_primitive_arrays()

        for idx, candidate in enumerate(self.demo_candidates):
            candidate_set: List[TaskOrientedCandidate] = []
            ii = 0
            while ii < how_many:
                pose, q = self.random_pose_and_config(
                    sim,
                    arm,
                    selfcc,
                    candidate.support_volume,
                    primitive_arrays,
                )
                if pose is not None and q is not None:
                    candidate_set.append(
                        CubbyCandidate(
                            pose=pose,
                            config=q,
                            pocket_idx=candidate.pocket_idx,
                            support_volume=candidate.support_volume,
                            negative_volumes=candidate.negative_volumes,
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
        return self.cubby.cuboids

    @property
    def cuboids(self) -> List[Cuboid]:
        """
        Returns all cuboids in the scene (and the scene is entirely made of cuboids)

        :rtype List[Cuboid]: The list of cuboids in this scene
        """
        return self.cubby.cuboids

    @property
    def cylinders(self) -> List[Cylinder]:
        """
        Returns an empty list because there are no cylinders in this scene, but left in
        to conform to the standard

        :rtype List[Cuboid]: The list of cuboids in this scene
        """
        return []


class MergedCubbyEnvironment(CubbyEnvironment):
    """
    This class first creates a standard 2x2 cubby and then merges the cubby holes necessary
    to create an unobstructed path between the start and goal.
    """

    def gen(self, selfcc: Union[FrankaSelfCollisionChecker, AuboCollisionSpheres]) -> bool:
        """
        Generates an environment and a pair of start/end candidates

        :param selfcc Union[FrankaSelfCollisionChecker, AuboCollisionSpheres]: Checks for self
                                                  collisions using robot-specific sphere models.
        :rtype bool: Whether the environment was successfully generated
        """
        success = super().gen(selfcc)
        if not success:
            return False
        assert len(self.cubby.support_volumes) == 4
        start, target = self.demo_candidates[0], self.demo_candidates[1]
        if (start.pocket_idx in [0, 1] and target.pocket_idx in [2, 3]) or (
            target.pocket_idx in [0, 1] and start.pocket_idx in [2, 3]
        ):
            self.cubby.middle_shelf_thickness = 0.0
        if (start.pocket_idx in [0, 2] and target.pocket_idx in [1, 3]) or (
            target.pocket_idx in [0, 2] and start.pocket_idx in [1, 3]
        ):
            self.cubby.center_wall_thickness = 0.0
        new_supports = self.cubby.support_volumes
        start_support_idx, target_support_idx = -1, -1
        for ii, s in enumerate(new_supports):
            if s.sdf(start.pose.xyz) < 0:
                start_support_idx = ii
            if s.sdf(target.pose.xyz) < 0:
                target_support_idx = ii
        assert start_support_idx == target_support_idx
        start.support_volume, start.pocket_idx = (
            new_supports[start_support_idx],
            start_support_idx,
        )
        target.support_volume, target.pocket_idx = (
            new_supports[target_support_idx],
            target_support_idx,
        )
        self.demo_candidates = (start, target)
        return success
