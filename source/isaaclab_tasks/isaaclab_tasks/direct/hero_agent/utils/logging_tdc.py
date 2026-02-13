# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TDC console logging utilities for Hero Agent environment.

Provides formatted console output for TDC controller state, reset events,
and TDE compensation diagnostics. Called from tdc_env.py and base_env.py.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..controllers.tdc import TDCControllerCfg


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
