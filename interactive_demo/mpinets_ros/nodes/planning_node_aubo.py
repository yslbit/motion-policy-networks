#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES, University of Washington. All rights reserved.
#
# Aubo i3 adaptation of planning_node.py
# Changes from original:
#   - Uses AuboRobot (6-DOF) instead of FrankaRealRobot
#   - Uses TorchAuboSampler instead of FrankaSampler
#   - Uses normalize/unnormalize_aubo_joints instead of Franka versions
#   - Publishes to aubo ROS driver topic (configurable via ROS param)
#   - base_frame = "base_link"
#   - No gripper logic

import torch
from mpinets.model import MotionPolicyNetwork
from robofin.robots_aubo import AuboRobot
from robofin.samplers_aubo import TorchAuboSampler
import numpy as np
from mpinets.utils_aubo import normalize_aubo_joints, unnormalize_aubo_joints
from mpinets_msgs.msg import PlanningProblem
from sensor_msgs.msg import PointCloud2, PointField
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Header
import time
import trimesh.transformations as tra
from functools import partial
from geometrout.transform import SE3
import argparse
from typing import List, Tuple, Any

import rospy

NUM_ROBOT_POINTS = 2048
NUM_OBSTACLE_POINTS = 4096
NUM_TARGET_POINTS = 128
MAX_ROLLOUT_LENGTH = 75

JOINT_NAMES = [
    "shoulder_joint",
    "upperArm_joint",
    "foreArm_joint",
    "wrist1_joint",
    "wrist2_joint",
    "wrist3_joint",
]


class Planner:
    @torch.no_grad()
    def __init__(self, mdl_file: str):
        self.mdl = MotionPolicyNetwork.load_from_checkpoint(mdl_file).cuda().eval()
        self.fk_sampler = TorchAuboSampler(device="cuda:0")

    @torch.no_grad()
    def target_point_cloud(self, pose: SE3) -> torch.Tensor:
        target_points = self.fk_sampler.sample_end_effector(
            torch.as_tensor(pose.matrix).float().cuda().unsqueeze(0),
            num_points=NUM_TARGET_POINTS,
        )
        return target_points

    @torch.no_grad()
    def plan(
        self, q0: np.ndarray, target_pose: SE3, obstacle_pc: np.ndarray
    ) -> Tuple[bool, List[List[float]]]:
        assert obstacle_pc.shape == (NUM_OBSTACLE_POINTS, 3)
        obstacle_points = torch.as_tensor(obstacle_pc).cuda()
        target_points = self.target_point_cloud(target_pose).squeeze()
        assert np.all(AuboRobot.constants.JOINT_LIMITS[:, 0] <= q0), \
            "Configuration is outside of feasible limits"
        assert np.all(q0 <= AuboRobot.constants.JOINT_LIMITS[:, 1]), \
            "Configuration is outside of feasible limits"
        q = torch.as_tensor(q0).cuda().unsqueeze(0).float()
        robot_points = self.fk_sampler.sample(q, NUM_ROBOT_POINTS)
        point_cloud = torch.cat(
            (
                torch.zeros(NUM_ROBOT_POINTS, 4),
                torch.ones(NUM_OBSTACLE_POINTS, 4),
                2 * torch.ones(NUM_TARGET_POINTS, 4),
            ),
            dim=0,
        ).cuda()
        point_cloud[:NUM_ROBOT_POINTS, :3] = robot_points.squeeze(0).float()
        point_cloud[NUM_ROBOT_POINTS:NUM_ROBOT_POINTS + NUM_OBSTACLE_POINTS, :3] = (
            obstacle_points.float()
        )
        point_cloud[NUM_ROBOT_POINTS + NUM_OBSTACLE_POINTS:, :3] = target_points.float()
        point_cloud = point_cloud.unsqueeze(0)

        trajectory = [q]
        q_norm = normalize_aubo_joints(q)
        success = False
        for _ in range(MAX_ROLLOUT_LENGTH):
            q_norm = torch.clamp(q_norm + self.mdl(point_cloud, q_norm), min=-1, max=1)
            qt = unnormalize_aubo_joints(q_norm).type_as(q)
            trajectory.append(qt)
            eff_pose = AuboRobot.fk(
                qt.squeeze().detach().cpu().numpy(), eff_frame="wrist3_Link"
            )
            if (
                np.linalg.norm(eff_pose._xyz - target_pose._xyz) < 0.01
                and np.abs(
                    np.degrees(
                        (eff_pose.so3._quat * target_pose.so3._quat.conjugate).radians
                    )
                ) < 15
            ):
                success = True
                break
            robot_points = self.fk_sampler.sample(qt, NUM_ROBOT_POINTS)
            point_cloud[:, :NUM_ROBOT_POINTS, :3] = robot_points
        return success, [q.squeeze().cpu().numpy().tolist() for q in trajectory]


class PlanningNode:
    def __init__(self):
        rospy.init_node("mpinets_planning_node_aubo")
        time.sleep(1)

        self.planner = None
        self.base_frame = "base_link"
        self.planning_problem_subscriber = rospy.Subscriber(
            "/mpinets/planning_problem",
            PlanningProblem,
            self.plan_callback,
            queue_size=1,
        )
        self.full_point_cloud_publisher = rospy.Publisher(
            "/mpinets/full_point_cloud", PointCloud2, queue_size=2
        )
        # Topic for aubo_robot ROS driver - configurable via param
        aubo_traj_topic = rospy.get_param(
            "/mpinets_planning_node_aubo/trajectory_topic",
            "/aubo_driver/joint_trajectory",
        )
        self.plan_publisher = rospy.Publisher(
            aubo_traj_topic, JointTrajectory, queue_size=1
        )
        rospy.loginfo("Loading data")
        self.load_point_cloud_data(
            rospy.get_param("/mpinets_planning_node_aubo/point_cloud_path")
        )
        rospy.loginfo("Data loaded")
        rospy.loginfo("Loading model")
        self.planner = Planner(rospy.get_param("/mpinets_planning_node_aubo/mdl_path"))
        rospy.loginfo("Model loaded")
        rospy.loginfo("System ready")

    @staticmethod
    def clean_point_cloud(
        xyz: np.ndarray, rgba: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Workspace filter for Aubo i3 (adjust bounds to match your setup)
        task_mask = np.logical_and.reduce(
            (
                xyz[:, 0] > 0.1,
                xyz[:, 0] < 1.2,
                xyz[:, 1] > -0.6,
                xyz[:, 1] < 0.6,
                xyz[:, 2] > -0.05,
                xyz[:, 2] < 0.8,
            )
        )
        xyz = xyz[task_mask]
        rgba = rgba[task_mask]
        random_mask = np.random.choice(len(xyz), size=NUM_OBSTACLE_POINTS, replace=False)
        return xyz[random_mask], rgba[random_mask]

    def load_point_cloud_data(self, path: str):
        observation_data = np.load(path, allow_pickle=True).item()
        full_pc = tra.transform_points(
            observation_data["pc"], observation_data["camera_pose"]
        )
        no_robot_mask = (
            observation_data["label_map"]["robot"] != observation_data["pc_label"]
        )
        scene_pc = full_pc[no_robot_mask]
        scene_colors = observation_data["pc_color"][no_robot_mask] / 255.0
        scene_colors = np.concatenate(
            (scene_colors, np.ones((len(scene_colors), 1))), axis=1
        )
        rospy.Timer(
            rospy.Duration(1.0),
            partial(self.publish_point_cloud_data, scene_pc, scene_colors),
        )
        self.full_scene_pc = scene_pc
        self.full_scene_colors = scene_colors

    def publish_point_cloud_data(self, points: np.ndarray, colors: np.ndarray, _: Any):
        ros_dtype = PointField.FLOAT32
        dtype = np.float32
        itemsize = np.dtype(dtype).itemsize
        colors[:, -1] = 0.5
        data = np.concatenate((points, colors), axis=1).astype(dtype).tobytes()
        fields = [
            PointField(name=n, offset=i * itemsize, datatype=ros_dtype, count=1)
            for i, n in enumerate("xyzrgba")
        ]
        header = Header(frame_id=self.base_frame, stamp=rospy.Time.now())
        msg = PointCloud2(
            header=header,
            height=1,
            width=points.shape[0],
            is_dense=False,
            is_bigendian=False,
            fields=fields,
            point_step=(itemsize * 7),
            row_step=(itemsize * 7 * points.shape[0]),
            data=data,
        )
        self.full_point_cloud_publisher.publish(msg)

    def plan_callback(self, msg: PlanningProblem):
        q0 = np.asarray(msg.q0.position)
        target = SE3(
            xyz=[
                msg.target.transform.translation.x,
                msg.target.transform.translation.y,
                msg.target.transform.translation.z,
            ],
            quaternion=[
                msg.target.transform.rotation.w,
                msg.target.transform.rotation.x,
                msg.target.transform.rotation.y,
                msg.target.transform.rotation.z,
            ],
        )
        scene_pc, scene_colors = self.clean_point_cloud(
            self.full_scene_pc, self.full_scene_colors
        )
        if self.planner is None:
            rospy.logwarn("Model is not yet loaded")
            return
        rospy.loginfo("Attempting to plan")
        success, plan = self.planner.plan(q0, target, scene_pc)
        rospy.loginfo(f"Planning succeeded: {success}")
        joint_trajectory = JointTrajectory()
        joint_trajectory.header.stamp = rospy.Time.now()
        joint_trajectory.header.frame_id = self.base_frame
        joint_trajectory.joint_names = JOINT_NAMES
        for ii, q in enumerate(plan):
            point = JointTrajectoryPoint(
                time_from_start=rospy.Duration.from_sec(0.12 * ii)
            )
            for qi in q:
                point.positions.append(qi)
            joint_trajectory.points.append(point)
        rospy.loginfo("Planning solution published")
        self.plan_publisher.publish(joint_trajectory)


if __name__ == "__main__":
    PlanningNode()
    rospy.spin()
