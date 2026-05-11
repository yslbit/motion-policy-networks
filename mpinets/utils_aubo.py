from typing import Union, Tuple

import numpy as np
import torch

from robofin.robot_constants_aubo import AuboConstants


def normalize_aubo_joints(
    batch_trajectory: Union[np.ndarray, torch.Tensor],
    limits: Tuple[float, float] = (-1, 1),
) -> Union[np.ndarray, torch.Tensor]:
    """
    Normalizes 6D Aubo i3 joint angles to [-1, 1] (or specified limits).

    :param batch_trajectory: shape [..., 6]
    :param limits: output range
    """
    if isinstance(batch_trajectory, torch.Tensor):
        joint_limits = torch.as_tensor(AuboConstants.JOINT_LIMITS).type_as(batch_trajectory)
        return (batch_trajectory - joint_limits[:, 0]) / (
            joint_limits[:, 1] - joint_limits[:, 0]
        ) * (limits[1] - limits[0]) + limits[0]
    elif isinstance(batch_trajectory, np.ndarray):
        joint_limits = AuboConstants.JOINT_LIMITS
        return (batch_trajectory - joint_limits[:, 0]) / (
            joint_limits[:, 1] - joint_limits[:, 0]
        ) * (limits[1] - limits[0]) + limits[0]
    else:
        raise NotImplementedError("Only torch.Tensor and np.ndarray implemented")


def unnormalize_aubo_joints(
    batch_trajectory: Union[np.ndarray, torch.Tensor],
    limits: Tuple[float, float] = (-1, 1),
) -> Union[np.ndarray, torch.Tensor]:
    """
    Unnormalizes 6D Aubo i3 joint angles from [-1, 1] back to joint limits.

    :param batch_trajectory: shape [..., 6], values in `limits`
    :param limits: input range
    """
    if isinstance(batch_trajectory, torch.Tensor):
        joint_limits = torch.as_tensor(AuboConstants.JOINT_LIMITS).type_as(batch_trajectory)
        limit_range = joint_limits[:, 1] - joint_limits[:, 0]
        lower = joint_limits[:, 0]
        for _ in range(batch_trajectory.ndim - 1):
            limit_range = limit_range.unsqueeze(0)
            lower = lower.unsqueeze(0)
        return (batch_trajectory - limits[0]) * limit_range / (limits[1] - limits[0]) + lower
    elif isinstance(batch_trajectory, np.ndarray):
        joint_limits = AuboConstants.JOINT_LIMITS
        limit_range = joint_limits[:, 1] - joint_limits[:, 0]
        lower = joint_limits[:, 0]
        for _ in range(batch_trajectory.ndim - 1):
            limit_range = limit_range[np.newaxis, ...]
            lower = lower[np.newaxis, ...]
        return (batch_trajectory - limits[0]) * limit_range / (limits[1] - limits[0]) + lower
    else:
        raise NotImplementedError("Only torch.Tensor and np.ndarray implemented")
