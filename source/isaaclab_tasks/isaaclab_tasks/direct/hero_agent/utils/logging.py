# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Consolidated logging and environment utilities for Hero Agent.

Provides all TB/WandB metric functions and environment helpers:
    - flush_metrics, pearson_r: Core logging utilities
    - unwrap_env, connect_encoder_to_env: Environment unwrapping helpers
    - log_tdc_init, log_tdc_control_state, log_tdc_reset_info: TDC console logging
    - log_tdc_diagnostics: TDC health metrics (4 essential metrics)
    - log_dr_metrics: Domain randomization parameters (4 essential metrics)
    - log_encoder_metrics: Encoder z health (3 essential metrics)
    - log_tdc_controller_metrics: TDC controller state (M_hat, Kp/Kd)

Merged from logging.py, env_utils.py, and metrics.py.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch

from ..controllers.tdc import compute_M_bb

if TYPE_CHECKING:
    from ..base_env import HeroAgentEnv
    from ..controllers.tdc import TDCControllerCfg

logger = logging.getLogger(__name__)

_ROLL_PITCH = ("roll", "pitch")


# =============================================================================
# Core Logging Utilities
# =============================================================================


def pearson_r(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute Pearson correlation coefficient between two 1D tensors."""
    a_c = a - a.mean()
    b_c = b - b.mean()
    return (a_c * b_c).sum() / (a_c.norm() * b_c.norm() + 1e-8)


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


class _WandbTBWriter:
    """Adapter that forwards ``add_scalar`` to both TensorBoard and WandB.

    ``flush_metrics()`` relies on ``writer.add_scalar()`` for all scalar logging.
    When WandB is the logger backend, we still need TensorBoard records, so this
    adapter wraps a real TB SummaryWriter and additionally calls ``wandb.log()``.
    """

    def __init__(self, tb_writer: Any) -> None:
        self._tb = tb_writer

    def add_scalar(self, tag: str, value: Any, global_step: int | None = None, **kw: Any) -> None:
        self._tb.add_scalar(tag, value, global_step, **kw)
        wandb = _get_wandb()
        if wandb is not None:
            wandb.log({tag: value}, step=global_step, commit=False)


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


def connect_encoder_to_env(env: Any, policy: Any, caller_name: str = "Runner") -> None:
    """Wire encoder policy to environment for encoder z access.

    Args:
        env: Wrapped environment (will be unwrapped).
        policy: Policy with encoder (ActorCriticEncoder or subclass).
        caller_name: Name for log message.
    """
    raw_env = unwrap_env(env)
    if hasattr(raw_env, "set_encoder_policy"):
        raw_env.set_encoder_policy(policy)
        logger.info("[%s] Connected encoder policy to env for M_hat extraction.", caller_name)


# =============================================================================
# TDC Console Logging
# =============================================================================

_tdc_logger: logging.Logger | None = None


def _get_tdc_logger() -> logging.Logger:
    """Get or create TDC console logger (singleton)."""
    global _tdc_logger
    if _tdc_logger is None:
        _tdc_logger = logging.getLogger("hero_agent.tdc")
        _tdc_logger.setLevel(logging.INFO)
        if not _tdc_logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[TDC] %(message)s"))
            _tdc_logger.addHandler(handler)
            _tdc_logger.propagate = False
    return _tdc_logger


def log_tdc_init(tdc_cfg: TDCControllerCfg, F_bu: float, tdc_dt: float) -> None:
    """Log TDC controller initialization parameters."""
    tdc_log = _get_tdc_logger()
    tdc_log.info(
        "TDC Controller initialized | M_hat=(%.3f, %.3f) Kp=%.1f Kd=%.1f "
        "F_bu=%.2f h=%.3f dt=%.4f max_jvel=%.1f base=(%.3f,%.3f)",
        *tdc_cfg.m_hat,
        tdc_cfg.kp,
        tdc_cfg.kd,
        F_bu,
        tdc_cfg.h,
        tdc_dt,
        tdc_cfg.max_joint_velocity,
        *tdc_cfg.base_position,
    )


def log_tdc_control_state(
    step_counter: int,
    step_dt: float,
    log_env_id: int,
    roll: torch.Tensor,
    pitch: torch.Tensor,
    ang_vel_body: torch.Tensor,
    p_EE_desired: torch.Tensor,
    joint_pos_targets: torch.Tensor,
    target_euler: torch.Tensor,
) -> None:
    """Log TDC control diagnostics for a single environment."""
    tdc_log = _get_tdc_logger()
    i = log_env_id
    t = step_counter * step_dt
    r_deg = torch.rad2deg(roll[i]).item()
    p_deg = torch.rad2deg(pitch[i]).item()
    err_r = torch.rad2deg(target_euler[i, 0] - roll[i]).item()
    err_p = torch.rad2deg(target_euler[i, 1] - pitch[i]).item()
    pq = ang_vel_body[i, :2]
    ee = p_EE_desired[i]
    jt = joint_pos_targets[i]
    tdc_log.info(
        "t=%5.2fs | roll=%+6.2f pitch=%+6.2f | err_r=%+6.2f err_p=%+6.2f | "
        "p=%.3f q=%.3f | pEE=(%.4f,%.4f) | j=(%.3f,%.3f)",
        t,
        r_deg,
        p_deg,
        err_r,
        err_p,
        pq[0].item(),
        pq[1].item(),
        ee[0].item(),
        ee[1].item(),
        jt[0].item(),
        jt[1].item(),
    )


def log_tdc_reset_info(
    env_ids: torch.Tensor,
    terminated: torch.Tensor,
    time_outs: torch.Tensor,
    root_pos_w: torch.Tensor,
    env_origins: torch.Tensor,
    episode_length_buf: torch.Tensor,
    step_dt: float,
) -> None:
    """Log termination reason before reset."""
    tdc_log = _get_tdc_logger()
    for eid in env_ids:
        i = eid.item()
        reason = "timeout" if time_outs[i] else ("terminated" if terminated[i] else "unknown")
        height = root_pos_w[i, 2].item()
        xy = root_pos_w[i, :2] - env_origins[i, :2]
        dist = torch.linalg.norm(xy).item()
        ep_len = episode_length_buf[i].item()
        tdc_log.info(
            "RESET env=%d | reason=%s | ep_steps=%d (%.1fs) | h=%.2f dist=%.2f",
            i,
            reason,
            ep_len,
            ep_len * step_dt,
            height,
            dist,
        )


# =============================================================================
# TDC Diagnostics (4 essential metrics)
# =============================================================================


def log_tdc_diagnostics(
    log: dict[str, float | torch.Tensor],
    tdc: object,
    env: object | None = None,
) -> None:
    """Log essential TDE compensation diagnostics.

    Metrics kept (4):
        - TDC/pd_torque_rms: PD output magnitude
        - TDC/u_hat_rms: TDE compensation magnitude
        - TDC/tde_to_pd_ratio: TDE health ratio
        - TDC/stability_violated_frac: stability check (requires env)

    Args:
        log: Metrics dict to accumulate into.
        tdc: TDCController instance.
        env: Environment instance (optional, enables stability metrics).
    """
    with torch.no_grad():
        u_hat_rms = tdc.u_hat.norm(dim=-1).mean().item()
        pd_rms = tdc.pd_torque.norm(dim=-1).mean().item()
        log["TDC/pd_torque_rms"] = pd_rms
        log["TDC/u_hat_rms"] = u_hat_rms
        log["TDC/tde_to_pd_ratio"] = u_hat_rms / max(pd_rms, 1e-8)

        # Stability violated fraction (requires env with kinematics + TDC cfg)
        if env is not None and hasattr(env, "_kinematics") and hasattr(env.cfg, "tdc"):
            joint_pos = env._robot.data.joint_pos[:, env._albc_joint_ids]
            p_EE = env._kinematics.forward(joint_pos)
            M_bb = compute_M_bb(
                I_ROV=env._hydro.rigid_body_inertia[:, :2],
                m_A=env._buoy_hydro.added_mass_matrix[:, 1, 1],
                x_bu=p_EE[:, 0],
                y_bu=p_EE[:, 1],
                h=env.cfg.tdc.h,
                m_body=env._buoy_hydro.body_mass,
            )
            M_hat = tdc._m_hat
            ratio = M_bb / M_hat.clamp(min=1e-6)
            stability_norm = (1.0 - ratio).abs().max(dim=-1).values
            log["TDC/stability_violated_frac"] = (stability_norm >= 1.0).float().mean().item()

            # Log active stability gate fraction (only when gate is enabled and affecting rewards)
            if hasattr(env, "_stability_gate_frac"):
                log["TDC/stability_gate_frac"] = env._stability_gate_frac


# =============================================================================
# Domain Randomization Metrics (4 essential metrics)
# =============================================================================


def log_dr_metrics(
    extras: dict,
    env: HeroAgentEnv,
) -> None:
    """Log essential domain randomization parameter statistics.

    Metrics kept (5):
        - DR/buoyancy_force_mean: critical for TDC lambda
        - DR/inertia_roll_mean, DR/inertia_pitch_mean: per-axis TDC stability
        - DR/payload_mass_mean: when payload enabled
        - DR/ocean_current_mag_mean: when ocean current enabled

    Args:
        extras: Environment extras dictionary (must have "log" key).
        env: HeroAgentEnv instance with _hydro, _buoy_hydro, etc.
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
        privileged = obs[policy._privileged_key]
        z = policy._encode(privileged)

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
# TDC Controller Metrics (M_hat, Kp/Kd)
# =============================================================================


def log_tdc_controller_metrics(
    writer: Any,
    env: Any,
    iteration: int,
    logger_type: str = "tensorboard",
    metrics: dict[str, float] | None = None,
) -> None:
    """Log essential TDC integration metrics.

    Metrics kept (6):
        - TDC/m_hat_roll_mean, m_hat_pitch_mean: M_hat health
        - TDC/m_hat_roll_rel_mae, m_hat_pitch_rel_mae: M_hat accuracy
        - TDC/kp_roll_mean, kp_pitch_mean: gain health

    Args:
        writer: TensorBoard SummaryWriter or equivalent logger.
        env: Wrapped environment (will be unwrapped to access TDC state).
        iteration: Current training iteration.
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

    with torch.no_grad():
        # M_hat, Kp, Kd mean per axis
        for i, axis in enumerate(_ROLL_PITCH):
            metrics[f"TDC/m_hat_{axis}_mean"] = tdc._m_hat[:, i].mean().item()
            metrics[f"TDC/kp_{axis}_mean"] = tdc._kp[:, i].mean().item()
            metrics[f"TDC/kd_{axis}_mean"] = tdc._kd[:, i].mean().item()

        # M_hat vs true M_bb: relative MAE only (no correlation, no absolute MAE)
        if hasattr(raw_env, "_kinematics") and hasattr(raw_env, "_buoy_hydro"):
            joint_pos = raw_env._robot.data.joint_pos[:, raw_env._albc_joint_ids]
            p_EE = raw_env._kinematics.forward(joint_pos)
            M_true = compute_M_bb(
                I_ROV=raw_env._hydro.rigid_body_inertia[:, :2],
                m_A=raw_env._buoy_hydro.added_mass_matrix[:, 1, 1],
                x_bu=p_EE[:, 0],
                y_bu=p_EE[:, 1],
                h=raw_env.cfg.tdc.h,
                m_body=raw_env._buoy_hydro.body_mass,
            )
            M_hat = tdc._m_hat
            for i, axis in enumerate(_ROLL_PITCH):
                metrics[f"TDC/m_hat_{axis}_rel_mae"] = (
                    ((M_hat[:, i] - M_true[:, i]) / M_true[:, i].clamp(min=1e-4)).abs().mean().item()
                )

    if flush_after:
        flush_metrics(writer, metrics, iteration, logger_type)
