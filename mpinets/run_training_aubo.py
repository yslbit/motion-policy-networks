"""
Training script for the Aubo i3H Motion Policy Network.

Usage:
    python run_training_aubo.py jobconfig_aubo.yaml [--test] [--gpus N]

Data directory layout expected:
    <data_dir>/
        train/
            *.hdf5
        val/
            *.hdf5

To split an all_data.hdf5 file into train/val:
    python run_training_aubo.py --split-data /path/to/all_data.hdf5 /path/to/data_dir
"""

from typing import Optional, Dict, Any
from pathlib import Path
import sys
import os
import argparse
import yaml


def setup_trainer(
    gpus: int,
    test: bool,
    should_checkpoint: bool,
    logger,
    checkpoint_interval: int,
    checkpoint_dir: str,
    validation_interval: float,
    limit_val_batches,
    num_sanity_val_steps: int,
    keep_latest_checkpoints: int,
    save_last_checkpoint: bool,
    resume_checkpoint: Optional[str],
):
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint

    class RollingCheckpoint(ModelCheckpoint):
        """Keep only the most recent periodic checkpoints in a directory."""

        def __init__(self, keep_latest: int, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.keep_latest = keep_latest

        def _save_checkpoint(self, trainer, filepath):
            super()._save_checkpoint(trainer, filepath)
            if self.keep_latest < 0:
                return
            checkpoint_dirpath = Path(self.dirpath)
            checkpoints = sorted(
                (
                    path
                    for path in checkpoint_dirpath.glob("*.ckpt")
                    if path.name != "last.ckpt"
                ),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            for stale_checkpoint in checkpoints[self.keep_latest :]:
                try:
                    stale_checkpoint.unlink()
                except FileNotFoundError:
                    pass

    callbacks = []
    if should_checkpoint:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        callbacks.append(
            RollingCheckpoint(
                keep_latest=keep_latest_checkpoints,
                dirpath=checkpoint_dir,
                filename="aubo-mpinets-{epoch:02d}-{val_loss:.4f}",
                save_top_k=-1,
                save_last=save_last_checkpoint,
                every_n_train_steps=checkpoint_interval,
            )
        )
    return pl.Trainer(
        enable_checkpointing=should_checkpoint,
        callbacks=callbacks,
        max_epochs=1 if test else 500,
        gradient_clip_val=1.0,
        gpus=gpus,
        precision=16,
        logger=False if logger is None else logger,
        val_check_interval=1.0 if test else validation_interval,
        limit_val_batches=1 if test else limit_val_batches,
        num_sanity_val_steps=0 if test else num_sanity_val_steps,
        resume_from_checkpoint=resume_checkpoint,
    )


def setup_logger(
    should_log: bool,
    experiment_name: str,
    config_values: Dict[str, Any],
    checkpoint_dir: str,
):
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import WandbLogger

    if not should_log:
        pl.utilities.rank_zero_info("Disabling WandB logging")
        return None

    logger = WandbLogger(
        name=experiment_name,
        project=config_values.get("wandb_project", "mpinets-aubo"),
        save_dir=config_values.get("wandb_save_dir", checkpoint_dir),
        log_model=config_values.get("wandb_log_model", False),
    )
    logger.log_hyperparams(config_values)
    return logger


def split_data(hdf5_path: str, out_dir: str, val_frac: float = 0.1):
    """Split a flat all_data.hdf5 into train/ and val/ sub-directories."""
    import h5py
    import numpy as np
    import shutil

    src = Path(hdf5_path)
    out = Path(out_dir)
    train_dir = out / "train"
    val_dir = out / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(src, "r") as f:
        n = f["global_solutions"].shape[0]
        idx = np.arange(n)
        np.random.shuffle(idx)
        n_val = max(1, int(n * val_frac))
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]

        def write_split(dst: Path, indices):
            sorted_idx = np.sort(indices)
            with h5py.File(dst, "w") as g:
                for key in f.keys():
                    data = f[key][sorted_idx, ...]
                    g.create_dataset(key, data=data)
            print(f"  Wrote {len(indices)} trajectories → {dst}")

        write_split(train_dir / "train.hdf5", train_idx)
        write_split(val_dir / "val.hdf5", val_idx)

    print(f"Split done: {n} total → {len(train_idx)} train, {len(val_idx)} val")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", help="YAML config file")
    parser.add_argument("--test", action="store_true", help="Smoke-test (1 epoch, 1 batch)")
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument(
        "--no-logging",
        action="store_true",
        help="Disable WandB logging",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume training from a specific checkpoint path",
    )
    parser.add_argument(
        "--resume-last",
        action="store_true",
        help="Resume training from <save_checkpoint_dir>/last.ckpt",
    )
    parser.add_argument(
        "--split-data",
        nargs=2,
        metavar=("HDF5_FILE", "OUT_DIR"),
        help="Split all_data.hdf5 into train/val and exit",
    )
    args = parser.parse_args()

    if args.split_data:
        split_data(args.split_data[0], args.split_data[1])
        return

    if args.config is None:
        parser.error("config file required unless --split-data is used")
    if args.resume_last and args.resume_from is not None:
        parser.error("--resume-last and --resume-from cannot be used together")

    # Import heavy deps only when actually training
    from mpinets.data_loader_aubo import AuboDataModule
    from mpinets.model_aubo import AuboTrainingMotionPolicyNetwork

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    try:
        import torch

        matmul_precision = cfg.get("float32_matmul_precision", "high")
        if hasattr(torch, "set_float32_matmul_precision") and matmul_precision:
            torch.set_float32_matmul_precision(matmul_precision)
    except Exception as exc:
        print(f"Could not set float32 matmul precision: {exc}")

    dm_params = cfg["data_module_parameters"]
    dm = AuboDataModule(
        data_dir=dm_params["data_dir"],
        trajectory_key=dm_params.get("trajectory_key", "hybrid_solutions"),
        num_robot_points=cfg["shared_parameters"]["num_robot_points"],
        num_obstacle_points=dm_params["num_obstacle_points"],
        num_target_points=dm_params["num_target_points"],
        random_scale=dm_params.get("random_scale", 0.015),
        batch_size=cfg.get("batch_size", 10),
        num_workers=dm_params.get("num_workers"),
        persistent_workers=dm_params.get("persistent_workers", True),
        prefetch_factor=dm_params.get("prefetch_factor", 2),
    )

    model_params = cfg["training_model_parameters"]
    model = AuboTrainingMotionPolicyNetwork(
        num_robot_points=cfg["shared_parameters"]["num_robot_points"],
        point_match_loss_weight=model_params["point_match_loss_weight"],
        collision_loss_weight=model_params["collision_loss_weight"],
        val_rollout_steps=cfg.get("val_rollout_steps", 69),
    )

    gpus = args.gpus if args.gpus is not None else cfg.get("gpus", 1)
    checkpoint_dir = cfg.get("save_checkpoint_dir", "./checkpoints_aubo")
    logger = setup_logger(
        should_log=(not args.test) and (not args.no_logging),
        experiment_name=cfg.get("experiment_name", "AuboI3TrainingJob"),
        config_values=cfg,
        checkpoint_dir=checkpoint_dir,
    )
    resume_checkpoint = args.resume_from
    if args.resume_last:
        resume_checkpoint = str(Path(checkpoint_dir) / "last.ckpt")
    if resume_checkpoint is not None and not Path(resume_checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {resume_checkpoint}")
    if resume_checkpoint is not None:
        print(f"Resuming training from checkpoint: {resume_checkpoint}")
    if logger is not None and cfg.get("wandb_watch", False):
        logger.watch(model, log="gradients", log_freq=100)
    trainer = setup_trainer(
        gpus=gpus,
        test=args.test,
        should_checkpoint=not args.test,
        logger=logger,
        checkpoint_interval=cfg.get("checkpoint_interval", 6000),
        checkpoint_dir=checkpoint_dir,
        validation_interval=cfg.get("validation_interval", 3000),
        limit_val_batches=cfg.get("limit_val_batches", 1.0),
        num_sanity_val_steps=cfg.get("num_sanity_val_steps", 0),
        keep_latest_checkpoints=cfg.get("keep_latest_checkpoints", 3),
        save_last_checkpoint=cfg.get("save_last_checkpoint", True),
        resume_checkpoint=resume_checkpoint,
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
