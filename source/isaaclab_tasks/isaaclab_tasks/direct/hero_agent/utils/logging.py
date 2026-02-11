# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Logging utilities for Hero Agent environment."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

    from ..base_env import HeroAgentEnv
    from ..controllers.tdc import TDCControllerCfg


# -----------------------------------------------------------------------------
# Private helpers
# -----------------------------------------------------------------------------

_ROLL_PITCH = ("roll", "pitch")


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


def _collect_tensor_stats(
    metrics: dict[str, float],
    prefix: str,
    tensor: torch.Tensor,
    dim_names: tuple[str, ...],
) -> None:
    """Collect per-dimension mean/std of a (num_envs, D) tensor into metrics dict.

    Args:
        metrics: Dict to accumulate into.
        prefix: Tag prefix (e.g., "TDC/m_hat").
        tensor: Shape (num_envs, D) tensor.
        dim_names: Names for each dimension (length must equal D).
    """
    for i, name in enumerate(dim_names):
        metrics[f"{prefix}_{name}_mean"] = tensor[:, i].mean().item()
        metrics[f"{prefix}_{name}_std"] = tensor[:, i].std().item()


def _get_wandb() -> Any | None:
    """Import wandb or return None if unavailable."""
    try:
        import wandb

        return wandb
    except ImportError:
        return None


from .env_utils import unwrap_env

# -----------------------------------------------------------------------------
# TDC console logging (called from HeroAgentTDCEnv)
# -----------------------------------------------------------------------------

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
    logger = _get_tdc_logger()
    logger.info(
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
    logger = _get_tdc_logger()
    i = log_env_id
    t = step_counter * step_dt
    r_deg = torch.rad2deg(roll[i]).item()
    p_deg = torch.rad2deg(pitch[i]).item()
    err_r = torch.rad2deg(target_euler[i, 0] - roll[i]).item()
    err_p = torch.rad2deg(target_euler[i, 1] - pitch[i]).item()
    pq = ang_vel_body[i, :2]
    ee = p_EE_desired[i]
    jt = joint_pos_targets[i]
    logger.info(
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
    logger = _get_tdc_logger()
    for eid in env_ids:
        i = eid.item()
        reason = "timeout" if time_outs[i] else ("terminated" if terminated[i] else "unknown")
        height = root_pos_w[i, 2].item()
        xy = root_pos_w[i, :2] - env_origins[i, :2]
        dist = torch.linalg.norm(xy).item()
        ep_len = episode_length_buf[i].item()
        logger.info(
            "RESET env=%d | reason=%s | ep_steps=%d (%.1fs) | h=%.2f dist=%.2f",
            i,
            reason,
            ep_len,
            ep_len * step_dt,
            height,
            dist,
        )


# -----------------------------------------------------------------------------
# DR parameter logging (called from HeroAgentEnv._collect_episode_metrics)
# -----------------------------------------------------------------------------


def log_dr_metrics(
    extras: dict,
    env: HeroAgentEnv,
    robot: Articulation,
    joint_ids: list[int],
) -> None:
    """Log domain randomization parameter statistics to extras["log"].

    Reads per-env DR state from hydrodynamic models, joint parameters, and
    payload config. All values are logged as scalar summaries (mean/std) so
    TensorBoard/WandB can track the distribution of randomized parameters
    across parallel environments.

    Args:
        extras: Environment extras dictionary (must have "log" key).
        env: HeroAgentEnv instance with _hydro, _buoy_hydro, etc.
        robot: Robot articulation for joint parameter access.
        joint_ids: ALBC joint indices for stiffness/damping readout.
    """
    log = extras["log"]

    with torch.no_grad():
        # Main body hydrodynamics
        hydro = env._hydro
        log["DR/volume_mean"] = hydro.volume.mean().item()
        log["DR/water_density_mean"] = hydro.water_density.mean().item()
        log["DR/water_density_std"] = hydro.water_density.std().item()
        log["DR/buoyancy_force_mean"] = hydro.buoyancy_force.mean().item()
        log["DR/cob_z_mean"] = hydro.center_of_buoyancy[:, 2].mean().item()
        log["DR/cog_z_mean"] = hydro.center_of_gravity[:, 2].mean().item()
        log["DR/inertia_mean"] = hydro.rigid_body_inertia.mean().item()

        # Buoy hydrodynamics
        buoy = env._buoy_hydro
        log["DR/buoy_volume_mean"] = buoy.volume.mean().item()
        log["DR/buoy_buoyancy_mean"] = buoy.buoyancy_force.mean().item()

        # Joint actuator gains
        stiffness = robot.data.joint_stiffness[:, joint_ids]
        damping = robot.data.joint_damping[:, joint_ids]
        log["DR/joint_stiffness_mean"] = stiffness.mean().item()
        log["DR/joint_stiffness_std"] = stiffness.std().item()
        log["DR/joint_damping_mean"] = damping.mean().item()

        # Buoyancy force distribution (critical for TDC lambda)
        log["DR/buoyancy_force_std"] = hydro.buoyancy_force.std().item()

        # Payload (if enabled)
        if env._payload_mass is not None:
            log["DR/payload_mass_mean"] = env._payload_mass.mean().item()
            log["DR/payload_mass_std"] = env._payload_mass.std().item()
        if env._payload_cog_offset is not None:
            log["DR/payload_cog_offset_norm_mean"] = torch.linalg.norm(env._payload_cog_offset, dim=-1).mean().item()

        # Ocean current (if model has current velocity)
        if hasattr(hydro, "_current_velocity") and hydro._current_velocity is not None:
            current_mag = torch.linalg.norm(hydro._current_velocity[:, :3], dim=-1)
            log["DR/ocean_current_mag_mean"] = current_mag.mean().item()


# -----------------------------------------------------------------------------
# TDC diagnostics (called from base_env._collect_episode_metrics)
# -----------------------------------------------------------------------------


def log_tdc_diagnostics(
    log: dict[str, float | torch.Tensor],
    tdc: object,
) -> None:
    """Log TDE compensation diagnostics for any env with a TDC controller.

    Records U_hat, delta_T_b, and PD torque magnitudes to assess TDE health.
    Low tde_to_pd_ratio indicates good M_hat accuracy (< 0.5 good, > 1.0 problematic).

    Args:
        log: Metrics dict to accumulate into (e.g. extras["log"]).
        tdc: TDCController instance with u_hat, pd_torque, delta_T_b properties.
    """
    with torch.no_grad():
        u_hat_rms = tdc.u_hat.norm(dim=-1).mean().item()
        pd_rms = tdc.pd_torque.norm(dim=-1).mean().item()
        log["TDC/u_hat_rms"] = u_hat_rms
        log["TDC/u_hat_roll_mean"] = tdc.u_hat[:, 0].mean().item()
        log["TDC/u_hat_pitch_mean"] = tdc.u_hat[:, 1].mean().item()
        log["TDC/delta_T_b_rms"] = tdc.delta_T_b.norm(dim=-1).mean().item()
        log["TDC/pd_torque_rms"] = pd_rms
        log["TDC/tde_to_pd_ratio"] = u_hat_rms / max(pd_rms, 1e-8)


# -----------------------------------------------------------------------------
# Encoder metrics (called from EncoderRunner.log)
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
# Encoder-TDC metrics (called from EncoderRunner.log and AdaptRunner.learn)
# -----------------------------------------------------------------------------


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
