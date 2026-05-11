"""
Aubo i3H-specific loss functions for Motion Policy Networks training.
"""
from typing import Tuple
from mpinets.geometry import TorchCuboids, TorchCylinders
import torch.nn.functional as F
import torch
from robofin.samplers_aubo import TorchAuboSampler
from mpinets.utils_aubo import unnormalize_aubo_joints


def point_match_loss(input_pc: torch.Tensor, target_pc: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(input_pc, target_pc, reduction="mean") + F.l1_loss(
        input_pc, target_pc, reduction="mean"
    )


def collision_loss(
    input_pc: torch.Tensor,
    cuboid_centers: torch.Tensor,
    cuboid_dims: torch.Tensor,
    cuboid_quaternions: torch.Tensor,
    cylinder_centers: torch.Tensor,
    cylinder_radii: torch.Tensor,
    cylinder_heights: torch.Tensor,
    cylinder_quaternions: torch.Tensor,
) -> torch.Tensor:
    cuboids = TorchCuboids(cuboid_centers, cuboid_dims, cuboid_quaternions)
    cylinders = TorchCylinders(
        cylinder_centers, cylinder_radii, cylinder_heights, cylinder_quaternions
    )
    sdf_values = torch.minimum(cuboids.sdf(input_pc), cylinders.sdf(input_pc))
    return F.hinge_embedding_loss(
        sdf_values,
        -torch.ones_like(sdf_values),
        margin=0.03,
        reduction="mean",
    )


class AuboCollisionAndBCLossContainer:
    """
    Loss container for Aubo i3H.  Mirrors CollisionAndBCLossContainer but
    uses TorchAuboSampler (6-DOF, no gripper) instead of FrankaSampler.
    """

    def __init__(self):
        self.fk_sampler = None
        self.num_points = 1024

    def __call__(
        self,
        input_normalized: torch.Tensor,
        cuboid_centers: torch.Tensor,
        cuboid_dims: torch.Tensor,
        cuboid_quaternions: torch.Tensor,
        cylinder_centers: torch.Tensor,
        cylinder_radii: torch.Tensor,
        cylinder_heights: torch.Tensor,
        cylinder_quaternions: torch.Tensor,
        target_normalized: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.fk_sampler is None:
            self.fk_sampler = TorchAuboSampler(
                num_robot_points=self.num_points,
                use_cache=True,
                device=str(input_normalized.device),
            )

        # Run all FK and loss computation in float32; AMP autocast would turn FK
        # matrix multiplications into float16, breaking robofin's dtype assertions.
        with torch.cuda.amp.autocast(enabled=False):
            q_in = unnormalize_aubo_joints(input_normalized.float())
            q_tgt = unnormalize_aubo_joints(target_normalized.float())

            # The sampler was initialized with num_robot_points=self.num_points,
            # so calling sample() without num_points preserves fixed point order.
            input_pc = self.fk_sampler.sample(q_in)[..., :3]
            target_pc = self.fk_sampler.sample(q_tgt)[..., :3]

            col_loss = collision_loss(
                input_pc,
                cuboid_centers.float(),
                cuboid_dims.float(),
                cuboid_quaternions.float(),
                cylinder_centers.float(),
                cylinder_radii.float(),
                cylinder_heights.float(),
                cylinder_quaternions.float(),
            )
            pm_loss = point_match_loss(input_pc, target_pc)

        return col_loss, pm_loss
