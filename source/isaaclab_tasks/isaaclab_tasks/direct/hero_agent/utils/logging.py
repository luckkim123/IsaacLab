# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Logging utilities for Hero Agent environment."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

    from ..hero_agent_env import HeroAgentEnv


def log_episode_metrics(
    extras: dict,
    env_ids: torch.Tensor,
    reset_terminated: torch.Tensor,
    reset_time_outs: torch.Tensor,
    reward_sums: dict[str, torch.Tensor],
    env: HeroAgentEnv,
    robot: Articulation,
    joint_ids: list[int],
    joint_pos_targets: torch.Tensor,
) -> None:
    """Log episode metrics before reset.

    Records reward sums, termination counts, attitude errors, and joint positions
    to extras["log"] for TensorBoard/WandB logging.

    Args:
        extras: Environment extras dictionary to write logs to.
        env_ids: Environment indices being reset.
        reset_terminated: Termination flags for all environments.
        reset_time_outs: Timeout flags for all environments.
        reward_sums: Accumulated rewards per term from RewardManager.
        env: HeroAgentEnv instance for attitude error access.
        robot: Robot articulation for joint position access.
        joint_ids: ALBC joint indices.
        joint_pos_targets: Current joint position targets.
    """
    extras["log"] = {}

    # Reward sums
    for name, value in reward_sums.items():
        extras["log"][f"Episode_Reward/{name}"] = value

    # Termination counts
    extras["log"]["Episode_Termination/terminated"] = torch.count_nonzero(reset_terminated[env_ids]).item()
    extras["log"]["Episode_Termination/time_out"] = torch.count_nonzero(reset_time_outs[env_ids]).item()

    if len(env_ids) == 0:
        return

    # Attitude errors (use cached values to avoid side effects)
    attitude_errors_deg = torch.rad2deg(env._attitude_error[env_ids])
    extras["log"]["Attitude_Error/roll_deg"] = attitude_errors_deg[:, 0].abs().mean().item()
    extras["log"]["Attitude_Error/pitch_deg"] = attitude_errors_deg[:, 1].abs().mean().item()
    extras["log"]["Attitude_Error/total_deg"] = attitude_errors_deg[:, :2].abs().sum(dim=-1).mean().item()

    # Joint positions
    joint_pos = robot.data.joint_pos[env_ids][:, joint_ids]
    extras["log"]["Joint_Position/joint1_rad"] = joint_pos[:, 0].mean().item()
    extras["log"]["Joint_Position/joint2_rad"] = joint_pos[:, 1].mean().item()
    extras["log"]["Joint_Position/target1_rad"] = joint_pos_targets[env_ids, 0].mean().item()
    extras["log"]["Joint_Position/target2_rad"] = joint_pos_targets[env_ids, 1].mean().item()


def log_encoder_metrics(
    writer: Any,
    policy: Any,
    env: Any,
    iteration: int,
    device: str | torch.device,
    logger_type: str = "tensorboard",
) -> None:
    """Log encoder-specific metrics to WandB/TensorBoard.

    This function logs internal encoder states for HORA Phase 1 training monitoring.
    Can be called from any training script using standard OnPolicyRunner.

    Metrics logged:
        - Encoder/z_dim{i}_mean: Per-dimension latent mean
        - Encoder/z_dim{i}_std: Per-dimension latent std
        - Encoder/z_min, z_max, z_range: Latent range statistics
        - Encoder/raw_mean, raw_std: Pre-softplus encoder output
        - Encoder/raw_abs_max: Maximum absolute raw value
        - Encoder/grad_norm: L2 norm of encoder gradients
        - Encoder/variance_ratio: Compression quality metric

    Args:
        writer: TensorBoard SummaryWriter or equivalent logger.
        policy: Policy with encoder attribute (ActorCriticEncoder).
        env: Environment instance with get_observations() method.
        iteration: Current training iteration.
        device: Computation device.
        logger_type: Logger type ("tensorboard" or "wandb").
    """
    if not hasattr(policy, "encoder"):
        return

    with torch.no_grad():
        obs = env.get_observations().to(device)

        # Extract privileged observations
        privileged_key = policy._privileged_key
        privileged = obs[privileged_key]

        # Compute raw encoder output (pre-softplus)
        raw = policy.encoder(privileged)

        # Compute z via softplus + z_min (matches ActorCriticEncoder._encode)
        z = torch.nn.functional.softplus(raw) + policy.z_min

        # --- Z Latent Statistics ---
        z_mean = z.mean(dim=0)
        z_std = z.std(dim=0)

        for i in range(z.shape[-1]):
            writer.add_scalar(f"Encoder/z_dim{i}_mean", z_mean[i].item(), iteration)
            writer.add_scalar(f"Encoder/z_dim{i}_std", z_std[i].item(), iteration)

        # Global z range statistics
        z_min_val = z.min().item()
        z_max_val = z.max().item()
        z_range = z_max_val - z_min_val

        writer.add_scalar("Encoder/z_min", z_min_val, iteration)
        writer.add_scalar("Encoder/z_max", z_max_val, iteration)
        writer.add_scalar("Encoder/z_range", z_range, iteration)

        # --- Raw Output Statistics (pre-softplus) ---
        raw_mean = raw.mean().item()
        raw_std = raw.std().item()
        raw_abs_max = raw.abs().max().item()

        writer.add_scalar("Encoder/raw_mean", raw_mean, iteration)
        writer.add_scalar("Encoder/raw_std", raw_std, iteration)
        writer.add_scalar("Encoder/raw_abs_max", raw_abs_max, iteration)

        # --- Variance Ratio (compression quality) ---
        privileged_var = privileged.var().item() + 1e-8
        z_var = z.var().item()
        variance_ratio = z_var / privileged_var
        writer.add_scalar("Encoder/variance_ratio", variance_ratio, iteration)

        # --- WandB Histograms ---
        if logger_type == "wandb":
            try:
                import wandb

                for i in range(z.shape[-1]):
                    wandb.log({f"Encoder/z_dim{i}_dist": wandb.Histogram(z[:, i].cpu().numpy())}, step=iteration)
            except ImportError:
                pass

    # --- Gradient Norm ---
    encoder_params = list(policy.encoder.parameters())
    if encoder_params and encoder_params[0].grad is not None:
        grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in encoder_params if p.grad is not None) ** 0.5
        writer.add_scalar("Encoder/grad_norm", grad_norm, iteration)


def log_encoder_tdc_metrics(
    writer: Any,
    policy: Any,
    env: Any,
    iteration: int,
    device: str | torch.device,
    logger_type: str = "tensorboard",
) -> None:
    """Log Encoder-TDC integration metrics to WandB/TensorBoard.

    Logs adaptive TDC parameters derived from encoder z and RL actions:
        - TDC/m_hat_roll_mean, m_hat_pitch_mean: Encoder-estimated design inertia
        - TDC/m_hat_roll_std, m_hat_pitch_std: M_hat variation across envs
        - TDC/kp_roll_mean, kp_pitch_mean: Adaptive proportional gains
        - TDC/kd_roll_mean, kd_pitch_mean: Adaptive derivative gains
        - TDC/kp_roll_std, kp_pitch_std, kd_roll_std, kd_pitch_std: Gain variation
        - TDC/z_m_hat_roll, z_m_hat_pitch: Raw z[3:5] before clamping

    Args:
        writer: TensorBoard SummaryWriter or equivalent logger.
        policy: Policy with get_last_z() method (ActorCriticEncoderTDC).
        env: Wrapped environment (will be unwrapped to access TDC state).
        iteration: Current training iteration.
        device: Computation device.
        logger_type: Logger type ("tensorboard" or "wandb").
    """
    # Unwrap to get the actual Isaac Lab env
    raw_env = env
    while hasattr(raw_env, "unwrapped") and raw_env is not raw_env.unwrapped:
        raw_env = raw_env.unwrapped

    if not hasattr(raw_env, "_tdc"):
        return

    tdc = raw_env._tdc

    with torch.no_grad():
        # --- M_hat from encoder z ---
        m_hat = tdc._m_hat  # (num_envs, 2)
        writer.add_scalar("TDC/m_hat_roll_mean", m_hat[:, 0].mean().item(), iteration)
        writer.add_scalar("TDC/m_hat_pitch_mean", m_hat[:, 1].mean().item(), iteration)
        writer.add_scalar("TDC/m_hat_roll_std", m_hat[:, 0].std().item(), iteration)
        writer.add_scalar("TDC/m_hat_pitch_std", m_hat[:, 1].std().item(), iteration)

        # --- Adaptive PD gains ---
        kp = tdc._kp  # (num_envs, 2)
        kd = tdc._kd  # (num_envs, 2)
        writer.add_scalar("TDC/kp_roll_mean", kp[:, 0].mean().item(), iteration)
        writer.add_scalar("TDC/kp_pitch_mean", kp[:, 1].mean().item(), iteration)
        writer.add_scalar("TDC/kd_roll_mean", kd[:, 0].mean().item(), iteration)
        writer.add_scalar("TDC/kd_pitch_mean", kd[:, 1].mean().item(), iteration)
        writer.add_scalar("TDC/kp_roll_std", kp[:, 0].std().item(), iteration)
        writer.add_scalar("TDC/kp_pitch_std", kp[:, 1].std().item(), iteration)
        writer.add_scalar("TDC/kd_roll_std", kd[:, 0].std().item(), iteration)
        writer.add_scalar("TDC/kd_pitch_std", kd[:, 1].std().item(), iteration)

        # --- Raw z[3:5] (pre-clamp M_hat from encoder) ---
        if hasattr(policy, "get_last_z"):
            z = policy.get_last_z()
            if z is not None:
                writer.add_scalar("TDC/z_m_hat_roll", z[:, 3].mean().item(), iteration)
                writer.add_scalar("TDC/z_m_hat_pitch", z[:, 4].mean().item(), iteration)

        # --- WandB Histograms for gain distributions ---
        if logger_type == "wandb":
            try:
                import wandb

                wandb.log(
                    {
                        "TDC/kp_roll_dist": wandb.Histogram(kp[:, 0].cpu().numpy()),
                        "TDC/kp_pitch_dist": wandb.Histogram(kp[:, 1].cpu().numpy()),
                        "TDC/kd_roll_dist": wandb.Histogram(kd[:, 0].cpu().numpy()),
                        "TDC/kd_pitch_dist": wandb.Histogram(kd[:, 1].cpu().numpy()),
                        "TDC/m_hat_roll_dist": wandb.Histogram(m_hat[:, 0].cpu().numpy()),
                        "TDC/m_hat_pitch_dist": wandb.Histogram(m_hat[:, 1].cpu().numpy()),
                    },
                    step=iteration,
                )
            except ImportError:
                pass
