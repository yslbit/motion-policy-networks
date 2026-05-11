"""
Aubo i3H-specific Motion Policy Network (6-DOF, no gripper).

Architecture is identical to MotionPolicyNetwork except:
  - Input / output joints = 6 (Aubo) instead of 7 (Franka)
  - Uses TorchAuboSampler and TorchAuboCollisionSpheres for validation
  - Uses unnormalize_aubo_joints for rollout
"""

import torch
from torch import nn
import pytorch_lightning as pl
from pointnet2_ops.pointnet2_modules import PointnetSAModule

from mpinets.loss_aubo import AuboCollisionAndBCLossContainer
from mpinets.utils_aubo import unnormalize_aubo_joints
from mpinets.geometry import TorchCuboids, TorchCylinders
from typing import List, Dict, Callable, Optional


# ---------------------------------------------------------------------------
# PointNet encoder  (unchanged from original)
# ---------------------------------------------------------------------------

class MPiNetsPointNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=512,
                radius=0.05,
                nsample=128,
                mlp=[1, 64, 64, 64],
                bn=False,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.3,
                nsample=128,
                mlp=[64, 128, 128, 256],
                bn=False,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(mlp=[256, 512, 512, 1024], bn=False)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 4096),
            nn.GroupNorm(16, 4096),
            nn.LeakyReLU(inplace=True),
            nn.Linear(4096, 2048),
            nn.GroupNorm(16, 2048),
            nn.LeakyReLU(inplace=True),
            nn.Linear(2048, 2048),
        )

    @staticmethod
    def _break_up_pc(pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous()
        return xyz, features

    def forward(self, pointcloud):
        xyz, features = self._break_up_pc(pointcloud)
        for module in self.SA_modules:
            xyz, features = module(xyz, features)
        return self.fc_layer(features.squeeze(-1))


# ---------------------------------------------------------------------------
# 6-DOF policy network
# ---------------------------------------------------------------------------

class AuboMotionPolicyNetwork(pl.LightningModule):
    """6-DOF variant of MotionPolicyNetwork for the Aubo i3H."""

    DOF = 6

    def __init__(self):
        super().__init__()
        self.point_cloud_encoder = MPiNetsPointNet()
        self.feature_encoder = nn.Sequential(
            nn.Linear(self.DOF, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2048 + 64, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, self.DOF),
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, xyz: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        pc_encoding = self.point_cloud_encoder(xyz)
        feature_encoding = self.feature_encoder(q)
        x = torch.cat((pc_encoding, feature_encoding), dim=1)
        return self.decoder(x)


class AuboTrainingMotionPolicyNetwork(AuboMotionPolicyNetwork):
    """Training wrapper for the Aubo 6-DOF policy network."""

    def __init__(
        self,
        num_robot_points: int,
        point_match_loss_weight: float,
        collision_loss_weight: float,
        val_rollout_steps: int = 69,
    ):
        super().__init__()
        self.num_robot_points = num_robot_points
        self.point_match_loss_weight = point_match_loss_weight
        self.collision_loss_weight = collision_loss_weight
        self.val_rollout_steps = val_rollout_steps
        self.fk_sampler = None
        self.collision_sampler = None
        self.loss_fun = AuboCollisionAndBCLossContainer()

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def rollout(
        self,
        batch: Dict[str, torch.Tensor],
        rollout_length: int,
        sampler: Callable[[torch.Tensor], torch.Tensor],
        unnormalize: bool = False,
    ) -> List[torch.Tensor]:
        xyz, q = batch["xyz"], batch["configuration"]
        if q.ndim == 1:
            xyz = xyz.unsqueeze(0)
            q = q.unsqueeze(0)

        q_unnorm = unnormalize_aubo_joints(q)
        assert isinstance(q_unnorm, torch.Tensor)
        trajectory = [q_unnorm if unnormalize else q]

        for _ in range(rollout_length):
            q = torch.clamp(q + self(xyz, q), min=-1, max=1)
            q_unnorm = unnormalize_aubo_joints(q).type_as(q)
            trajectory.append(q_unnorm if unnormalize else q)
            samples = sampler(q_unnorm).type_as(xyz)
            xyz[:, : samples.shape[1], :3] = samples

        return trajectory

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        xyz, q = batch["xyz"], batch["configuration"]
        pred_delta = self(xyz, q)
        y_hat = torch.clamp(q + pred_delta, min=-1, max=1)
        collision_loss, point_match_loss = self.loss_fun(
            y_hat,
            batch["cuboid_centers"],
            batch["cuboid_dims"],
            batch["cuboid_quats"],
            batch["cylinder_centers"],
            batch["cylinder_radii"],
            batch["cylinder_heights"],
            batch["cylinder_quats"],
            batch["supervision"],
        )
        self.log("point_match_loss", point_match_loss)
        self.log("collision_loss", collision_loss)

        expert_delta = batch["supervision"] - q
        self.log("pred_delta_norm", torch.linalg.norm(pred_delta, dim=1).mean())
        self.log("expert_delta_norm", torch.linalg.norm(expert_delta, dim=1).mean())
        self.log(
            "effective_pred_delta_norm",
            torch.linalg.norm(y_hat - q, dim=1).mean(),
        )
        self.log(
            "one_step_joint_error",
            torch.linalg.norm(y_hat - batch["supervision"], dim=1).mean(),
        )
        self.log(
            "clamped_fraction",
            ((q + pred_delta < -1) | (q + pred_delta > 1)).float().mean(),
        )

        val_loss = (
            self.point_match_loss_weight * point_match_loss
            + self.collision_loss_weight * collision_loss
        )
        self.log("val_loss", val_loss)
        return val_loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def sample(self, q: torch.Tensor) -> torch.Tensor:
        """Sample a surface point cloud from the robot at config q."""
        assert self.fk_sampler is not None
        # AMP autocast turns FK matrix-multiplications to float16; disable it here
        # so the sampler always runs in float32.
        with torch.cuda.amp.autocast(enabled=False):
            pc = self.fk_sampler.sample(q.float(), self.num_robot_points)  # (B, N, 4)
        return pc[..., :3]  # (B, N, 3)

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        from robofin.samplers_aubo import TorchAuboSampler
        from robofin.collision_aubo import TorchAuboCollisionSpheres

        if self.fk_sampler is None:
            self.fk_sampler = TorchAuboSampler(
                num_robot_points=self.num_robot_points,
                use_cache=True,
                device=str(self.device),
            )
        if self.collision_sampler is None:
            self.collision_sampler = TorchAuboCollisionSpheres(device=str(self.device))

        rollout = self.rollout(batch, self.val_rollout_steps, self.sample, unnormalize=True)

        # End-effector position error
        with torch.cuda.amp.autocast(enabled=False):
            eff = self.fk_sampler.end_effector_pose(rollout[-1].float())  # (B, 4, 4)
        position_error = torch.linalg.vector_norm(
            eff[:, :3, 3] - batch["target_position"].float(), dim=1
        )
        self.log("avg_target_error", torch.mean(position_error))

        # Collision rate
        cuboids = TorchCuboids(
            batch["cuboid_centers"],
            batch["cuboid_dims"],
            batch["cuboid_quats"],
        )
        cylinders = TorchCylinders(
            batch["cylinder_centers"],
            batch["cylinder_radii"],
            batch["cylinder_heights"],
            batch["cylinder_quats"],
        )

        B = batch["cuboid_centers"].size(0)
        rollout_tensor = torch.stack(rollout, dim=1)  # (B, val_rollout_steps+1, DOF)
        assert rollout_tensor.shape == (B, self.val_rollout_steps + 1, self.DOF)
        T = self.val_rollout_steps + 1
        rollout_flat = rollout_tensor.reshape(-1, self.DOF)  # (B*T, 6)

        cspheres = self.collision_sampler.csphere_info(rollout_flat)
        # centers: (B*T, N_spheres, 3) → (B, T, N_spheres, 3)
        N_spheres = cspheres.centers.shape[1]
        sphere_sequence = cspheres.centers.reshape(B, T, N_spheres, 3)
        radii = cspheres.radii[0]  # (N_spheres,) - same for all configs

        sdf_values = torch.minimum(
            cuboids.sdf_sequence(sphere_sequence),
            cylinders.sdf_sequence(sphere_sequence),
        )  # (B, T, N_spheres)

        # A trajectory has a collision if any sphere at any timestep penetrates any obstacle
        radius_collisions = sdf_values < radii.unsqueeze(0).unsqueeze(0)  # (B, T, N_spheres)
        has_collision = torch.any(radius_collisions.reshape(B, -1), dim=1)  # (B,)
        collision_rate = torch.mean(has_collision.float())
        self.log("collision_rate", collision_rate)

        # Success: reached target and no collision
        success = (~has_collision) & (position_error < 0.05)
        self.log("success_rate", torch.mean(success.float()))

        return position_error.mean()
