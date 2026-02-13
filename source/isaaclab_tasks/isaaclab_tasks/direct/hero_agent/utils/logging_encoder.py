# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Encoder metrics logging for Hero Agent environment.

Provides per-iteration encoder and Encoder-TDC metrics for WandB/TensorBoard.
Called from EncoderRunner.log and AdaptRunner.learn.
"""

from __future__ import annotations

from typing import Any

import torch

from .env_utils import unwrap_env
from .logging import _ROLL_PITCH, _collect_tensor_stats, _get_wandb, flush_metrics, pearson_r


def log_encoder_metrics(
    writer: Any,
    policy: Any,
    env: Any,
    iteration: int,
    device: str | torch.device,
    logger_type: str = "tensorboard",
    metrics: dict[str, float] | None = None,
) -> None:
    """Log encoder-specific metrics to WandB/TensorBoard.

    This function logs internal encoder states for HORA Phase 1 training monitoring.

    Args:
        writer: TensorBoard SummaryWriter or equivalent logger.
        policy: Policy with encoder attribute (ActorCriticEncoder).
        env: Environment instance with get_observations() method.
        iteration: Current training iteration.
        device: Computation device.
        logger_type: Logger type ("tensorboard" or "wandb").
        metrics: Optional dict to accumulate into. If None, flushes immediately.
    """
    if not hasattr(policy, "encoder"):
        return

    flush_after = metrics is None
    if metrics is None:
        metrics = {}

    # Single forward pass -- cache z as CPU numpy for histogram reuse
    wandb_extras: dict[str, Any] = {}

    with torch.no_grad():
        obs = env.get_observations().to(device)
        privileged = obs[policy._privileged_key]

        # Compute raw encoder output (pre-softplus) and z
        raw = policy.encoder(privileged)
        z = policy._softplus_z(raw)

        # Cache for histograms (avoids second forward pass)
        z_cpu = z.cpu().numpy()

        # Per-dimension z statistics
        z_mean = z.mean(dim=0)
        z_std = z.std(dim=0)
        for i in range(z.shape[-1]):
            metrics[f"Encoder/z_dim{i}_mean"] = z_mean[i].item()
            metrics[f"Encoder/z_dim{i}_std"] = z_std[i].item()

        # Global z range
        z_min_val = z.min().item()
        z_max_val = z.max().item()
        metrics["Encoder/z_min"] = z_min_val
        metrics["Encoder/z_max"] = z_max_val

        # Raw output statistics (pre-softplus)
        metrics["Encoder/raw_mean"] = raw.mean().item()
        metrics["Encoder/raw_std"] = raw.std().item()
        metrics["Encoder/raw_abs_max"] = raw.abs().max().item()

        # Variance ratio (compression quality)
        privileged_var = privileged.var().item() + 1e-8
        metrics["Encoder/variance_ratio"] = z.var().item() / privileged_var

        # z-physics correlation: Pearson R between z dims and privileged obs
        # Privileged layout (24D): [volume(1), cg(3), cb(3), inertia(3)] * 2 bodies + [mass(1), cog(3)]
        if privileged.shape[-1] >= 24 and z.shape[-1] >= 5:
            metrics["Encoder/z3_vs_Ixx_corr"] = pearson_r(z[:, 3], privileged[:, 7]).item()
            metrics["Encoder/z4_vs_Iyy_corr"] = pearson_r(z[:, 4], privileged[:, 8]).item()
            metrics["Encoder/z3_vs_payload_mass_corr"] = pearson_r(z[:, 3], privileged[:, 20]).item()
            metrics["Encoder/z0_vs_volume_corr"] = pearson_r(z[:, 0], privileged[:, 0]).item()

    # Gradient norm (outside no_grad context)
    encoder_params = list(policy.encoder.parameters())
    if encoder_params and encoder_params[0].grad is not None:
        grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in encoder_params if p.grad is not None) ** 0.5
        metrics["Encoder/grad_norm"] = grad_norm

    # Build WandB histograms from cached z_cpu (no second forward pass)
    if logger_type == "wandb":
        wandb = _get_wandb()
        if wandb is not None:
            for i in range(z_cpu.shape[-1]):
                wandb_extras[f"Encoder/z_dim{i}_dist"] = wandb.Histogram(z_cpu[:, i])

    # Flush if we own the metrics dict (no external accumulator)
    if flush_after:
        flush_metrics(writer, metrics, iteration, logger_type, wandb_extras=wandb_extras)


def log_encoder_tdc_metrics(
    writer: Any,
    policy: Any,
    env: Any,
    iteration: int,
    device: str | torch.device,
    logger_type: str = "tensorboard",
    metrics: dict[str, float] | None = None,
) -> None:
    """Log Encoder-TDC integration metrics to WandB/TensorBoard.

    Logs adaptive TDC parameters derived from encoder z and RL actions:
        - TDC/{m_hat,kp,kd}_{roll,pitch}_{mean,std}: Per-axis statistics
        - TDC/z_m_hat_{roll,pitch}: Raw z[3:5] from encoder

    Args:
        writer: TensorBoard SummaryWriter or equivalent logger.
        policy: Policy with get_last_z() method (ActorCriticEncoderTDC).
        env: Wrapped environment (will be unwrapped to access TDC state).
        iteration: Current training iteration.
        device: Computation device.
        logger_type: Logger type ("tensorboard" or "wandb").
        metrics: Optional dict to accumulate into. If None, flushes immediately.
    """
    raw_env = unwrap_env(env)

    if not hasattr(raw_env, "_tdc"):
        return

    tdc = raw_env._tdc

    flush_after = metrics is None
    if metrics is None:
        metrics = {}

    wandb_extras: dict[str, Any] = {}

    with torch.no_grad():
        # M_hat, Kp, Kd -- all (num_envs, 2) with [roll, pitch]
        _collect_tensor_stats(metrics, "TDC/m_hat", tdc._m_hat, _ROLL_PITCH)
        _collect_tensor_stats(metrics, "TDC/kp", tdc._kp, _ROLL_PITCH)
        _collect_tensor_stats(metrics, "TDC/kd", tdc._kd, _ROLL_PITCH)

        # NOTE: TDE compensation diagnostics (u_hat_rms, delta_T_b_rms, etc.)
        # are logged per-episode via log_tdc_diagnostics() in base_env._collect_episode_metrics().
        # Duplicating them here would cause per-iteration vs per-episode value conflicts.

        # Raw z[3:5] (pre-clamp M_hat from encoder)
        if hasattr(policy, "get_last_z"):
            z = policy.get_last_z()
            if z is not None:
                metrics["TDC/z_m_hat_roll"] = z[:, 3].mean().item()
                metrics["TDC/z_m_hat_pitch"] = z[:, 4].mean().item()

        # Build WandB histograms from already-available tensors (no extra computation)
        if logger_type == "wandb":
            wandb = _get_wandb()
            if wandb is not None:
                for name, tensor in [("kp", tdc._kp), ("kd", tdc._kd), ("m_hat", tdc._m_hat)]:
                    for i, axis in enumerate(_ROLL_PITCH):
                        wandb_extras[f"TDC/{name}_{axis}_dist"] = wandb.Histogram(tensor[:, i].cpu().numpy())

    # Flush if we own the metrics dict
    if flush_after:
        flush_metrics(writer, metrics, iteration, logger_type, wandb_extras=wandb_extras)
