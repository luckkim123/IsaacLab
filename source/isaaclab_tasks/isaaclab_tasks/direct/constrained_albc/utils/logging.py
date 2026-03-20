# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Consolidated logging and environment utilities for constrained ALBC.

Provides all TB/WandB metric functions and environment helpers:
    - flush_metrics: Core logging utilities
    - unwrap_env: Environment unwrapping helper
    - log_dr_metrics: Domain randomization parameters
    - log_encoder_metrics: Encoder z health
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from ..albc_env import ALBCEnv

logger = logging.getLogger(__name__)


# =============================================================================
# Core Logging Utilities
# =============================================================================


def flush_metrics(
    writer: Any,
    metrics: dict[str, float],
    step: int,
    logger_type: str = "tensorboard",
    wandb_extras: dict[str, Any] | None = None,
) -> None:
    """Flush accumulated metrics dict to TensorBoard and/or WandB in a single call.

    Scalars are always written via ``writer.add_scalar()`` so that TensorBoard
    records are never skipped -- even when the logger backend is WandB (RSL-RL's
    ``WandbSummaryWriter.add_scalar`` writes to both TB and WandB).

    Non-scalar WandB data (histograms, images, etc.) can be passed via
    *wandb_extras* and will be sent in a single ``wandb.log()`` call together
    with scalars.

    Args:
        writer: TensorBoard SummaryWriter (or RSL-RL WandbSummaryWriter).
        metrics: Dict of {tag: scalar_value} to log.
        step: Training step for x-axis.
        logger_type: Backend type ("tensorboard" or "wandb").
        wandb_extras: Optional dict of non-scalar WandB objects (e.g. Histogram).
            Ignored when logger_type is not "wandb".
    """
    if not metrics and not wandb_extras:
        return

    # Always record scalars via writer.add_scalar (works for both TB and WandB writers)
    for tag, value in metrics.items():
        writer.add_scalar(tag, value, step)

    # Send non-scalar extras (histograms, etc.) directly to WandB
    if logger_type == "wandb" and wandb_extras:
        wandb = _get_wandb()
        if wandb is not None:
            wandb.log(wandb_extras, step=step, commit=False)


def _get_wandb() -> Any | None:
    """Import wandb or return None if unavailable."""
    try:
        import wandb

        return wandb
    except ImportError:
        return None


# =============================================================================
# Environment Utilities
# =============================================================================


def unwrap_env(env: Any) -> Any:
    """Unwrap environment through wrapper chain to get the raw Isaac Lab env.

    Handles the common case where ``gym.Env.unwrapped`` returns ``self``
    (preventing infinite loops).

    Args:
        env: Potentially wrapped environment.

    Returns:
        The innermost (raw) environment.
    """
    raw = env
    while hasattr(raw, "unwrapped") and raw is not raw.unwrapped:
        raw = raw.unwrapped
    return raw


# =============================================================================
# Domain Randomization Metrics (5 essential metrics)
# =============================================================================


def log_dr_metrics(
    extras: dict,
    env: ALBCEnv,
) -> None:
    """Log essential domain randomization parameter statistics.

    Metrics kept (5):
        - DR/buoyancy_force_mean: critical for ALBC torque capacity
        - DR/inertia_roll_mean, DR/inertia_pitch_mean: per-axis rotational dynamics
        - DR/payload_mass_mean: when payload enabled
        - DR/ocean_current_mag_mean: when ocean current enabled

    Args:
        extras: Environment extras dictionary (must have "log" key).
        env: ALBCEnv instance with _hydro, _buoy_hydro, etc.
    """
    log = extras["log"]

    with torch.no_grad():
        hydro = env._hydro
        log["DR/buoyancy_force_mean"] = hydro.buoyancy_force.mean().item()
        inertia = hydro.rigid_body_inertia  # (num_envs, 3) = Ixx, Iyy, Izz
        log["DR/inertia_roll_mean"] = inertia[:, 0].mean().item()
        log["DR/inertia_pitch_mean"] = inertia[:, 1].mean().item()

        # Payload (if enabled)
        if env._payload_mass is not None:
            log["DR/payload_mass_mean"] = env._payload_mass.mean().item()

        # Ocean current (if model has current velocity)
        if hasattr(hydro, "_current_velocity") and hydro._current_velocity is not None:
            current_mag = torch.linalg.norm(hydro._current_velocity[:, :3], dim=-1)
            log["DR/ocean_current_mag_mean"] = current_mag.mean().item()


# =============================================================================
# Encoder Metrics (3 essential metrics)
# =============================================================================


def log_encoder_metrics(
    writer: Any,
    policy: Any,
    env: Any,
    iteration: int,
    device: str | torch.device,
    logger_type: str = "tensorboard",
    metrics: dict[str, float] | None = None,
) -> None:
    """Log essential encoder metrics.

    Metrics kept (3):
        - Encoder/z_mean, z_std: aggregate z health
        - Encoder/grad_norm: training signal

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

    with torch.no_grad():
        obs = env.get_observations().to(device)
        z = policy._encode(obs)

        metrics["Encoder/z_mean"] = z.mean().item()
        metrics["Encoder/z_std"] = z.std().item()
        metrics["Encoder/z_min"] = z.min().item()
        metrics["Encoder/z_max"] = z.max().item()

    # Gradient norm (outside no_grad context)
    encoder_params = list(policy.encoder.parameters())
    if encoder_params and encoder_params[0].grad is not None:
        grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in encoder_params if p.grad is not None) ** 0.5
        metrics["Encoder/grad_norm"] = grad_norm

    if flush_after:
        flush_metrics(writer, metrics, iteration, logger_type)


# =============================================================================
# DR Infeasibility Logging
# =============================================================================

_dr_infeasibility_logger: logging.Logger | None = None


def _get_dr_infeasibility_logger() -> logging.Logger:
    """Get or create DR infeasibility logger (singleton).

    The logger writes to "constrained_albc.dr_infeasibility". Runners can attach
    a FileHandler to this logger to persist infeasibility records to disk.
    """
    global _dr_infeasibility_logger
    if _dr_infeasibility_logger is None:
        _dr_infeasibility_logger = logging.getLogger("constrained_albc.dr_infeasibility")
        _dr_infeasibility_logger.setLevel(logging.INFO)
        if not _dr_infeasibility_logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[DR-Infeasible] %(message)s"))
            _dr_infeasibility_logger.addHandler(handler)
            _dr_infeasibility_logger.propagate = False
    return _dr_infeasibility_logger


def log_dr_infeasibility(
    env_ids: torch.Tensor,
    mean_errors_deg: torch.Tensor,
    hydro: object,
    buoy_hydro: object,
    payload_mass: torch.Tensor | None,
    robot: object,
    albc_joint_ids: list[int],
) -> None:
    """Log DR parameters for physically infeasible environments.

    Called when an environment has high error despite correct arm direction,
    suggesting the DR parameter combination is physically unachievable.

    Args:
        env_ids: Global environment indices flagged as infeasible.
        mean_errors_deg: Mean roll/pitch error in degrees for each env.
        hydro: Main body HydrodynamicsModel.
        buoy_hydro: Buoy HydrodynamicsModel.
        payload_mass: Per-env payload mass tensor or None.
        robot: Robot articulation (for joint positions).
        albc_joint_ids: ALBC joint indices.
    """
    dr_log = _get_dr_infeasibility_logger()
    for i, eid in enumerate(env_ids):
        e = eid.item()
        params = {
            "env_id": e,
            "mean_error_deg": round(mean_errors_deg[i].item(), 2),
            "buoyancy_force": round(hydro.buoyancy_force[e].item(), 3),
            "inertia_rp": [round(v, 4) for v in hydro.rigid_body_inertia[e, :2].tolist()],
            "cob_main": [round(v, 4) for v in hydro.center_of_buoyancy[e].tolist()],
            "cog_main": [round(v, 4) for v in hydro.center_of_gravity[e].tolist()],
            "volume_main": round(hydro.volume[e].item(), 5),
            "volume_buoy": round(buoy_hydro.volume[e].item(), 5),
        }
        if payload_mass is not None:
            params["payload_mass"] = round(payload_mass[e].item(), 3)
        joint_pos = robot.data.joint_pos[e, albc_joint_ids]  # type: ignore[union-attr]
        params["joint_pos"] = [round(v, 3) for v in joint_pos.tolist()]
        dr_log.info("Infeasible DR: %s", params)
