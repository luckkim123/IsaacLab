# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cost functions for NORBC constrained RL.

Each function returns a per-env cost tensor of shape (num_envs,).
Probabilistic costs return binary {0, 1}; average costs return non-negative scalars.

Cost functions receive the environment instance and optional parameters.
The CostManager passes context kwargs (ee_pos, prev_ee_pos, z, prev_z)
from the environment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..constrained_encoder_tdc_env import HeroAgentConstrainedEncoderTDCEnv


def workspace_cost(
    env: HeroAgentConstrainedEncoderTDCEnv,
    r_max: float = 0.466,
    **_kwargs,
) -> torch.Tensor:
    """Workspace boundary violation: ||P_EE|| > r_max.

    Binary indicator (probabilistic constraint). Returns 1 if the end-effector
    position norm exceeds the maximum reachable radius (L1 + L2 = 0.466m).

    Args:
        env: Environment with _tdc controller (has _p_EE buffer).
        r_max: Maximum reachable radius (sum of link lengths). Default 0.466m.

    Returns:
        Cost tensor of shape (num_envs,), values in {0, 1}.
    """
    p_EE = env._tdc._p_EE_prev  # (num_envs, 2)
    ee_norm = torch.linalg.norm(p_EE, dim=-1)  # (num_envs,)
    return (ee_norm > r_max).float()


def control_smoothness_cost(
    env: HeroAgentConstrainedEncoderTDCEnv,
    **_kwargs,
) -> torch.Tensor:
    """EE position change magnitude: ||P_EE_t - P_EE_{t-1}||.

    Continuous cost (average constraint). Penalizes large jumps in end-effector
    position commands, encouraging smooth control trajectories.

    Returns:
        Cost tensor of shape (num_envs,), non-negative.
    """
    return torch.linalg.norm(env._tdc._p_EE_prev - env._prev_ee_pos, dim=-1)


def inertia_rate_cost(
    env: HeroAgentConstrainedEncoderTDCEnv,
    **_kwargs,
) -> torch.Tensor:
    """Encoder latent change rate: ||z_t - z_{t-1}||.

    Continuous cost (average constraint). Penalizes rapid changes in the encoder
    latent z, which would cause sudden M_hat jumps destabilizing TDC.

    Returns:
        Cost tensor of shape (num_envs,), non-negative.
    """
    if env._encoder_policy is None or env._last_z is None or env._prev_z is None:
        return torch.zeros(env.num_envs, device=env.device)
    return torch.linalg.norm(env._last_z - env._prev_z, dim=-1)


def angular_velocity_cost(
    env: HeroAgentConstrainedEncoderTDCEnv,
    omega_max: float = 1.5,
    **_kwargs,
) -> torch.Tensor:
    """Body angular velocity exceeding limit: max(|p|, |q|) > omega_max.

    Binary indicator (probabilistic). Uses roll/pitch rates from root_ang_vel_b.
    Conservative threshold (1.5 rad/s) vs hard termination (3.14 rad/s).

    Args:
        env: Environment with robot data.
        omega_max: Maximum allowed angular velocity in rad/s. Default 1.5.

    Returns:
        Cost tensor of shape (num_envs,), values in {0, 1}.
    """
    ang_vel = env._robot.data.root_ang_vel_b[:, :2]  # (num_envs, 2) [p, q]
    return (ang_vel.abs().max(dim=-1).values > omega_max).float()


def joint_velocity_cost(
    env: HeroAgentConstrainedEncoderTDCEnv,
    vel_max: float = 1.0,
    **_kwargs,
) -> torch.Tensor:
    """Joint velocity exceeding limit: max(|qdot|) > vel_max.

    Binary indicator (probabilistic). Uses ALBC joint velocities.
    Default 1.0 rad/s (~57 deg/s): ~1.5x the physical peak velocity (~0.65 rad/s)
    given TDC rate limit (2.5 rad/s * 0.02s = 0.05 rad/step) and PD dynamics
    (Kp=200, Kd=10, I~0.16 -> nearly critically damped, peak vel ~0.65 rad/s).

    Args:
        env: Environment with robot data and _albc_joint_ids.
        vel_max: Maximum allowed joint velocity in rad/s. Default 1.0.

    Returns:
        Cost tensor of shape (num_envs,), values in {0, 1}.
    """
    joint_vel = env._robot.data.joint_vel[:, env._albc_joint_ids]  # (num_envs, 2)
    return (joint_vel.abs().max(dim=-1).values > vel_max).float()
