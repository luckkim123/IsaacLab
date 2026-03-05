# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Constraint cost functions for IPO (Interior-point Policy Optimization).

Provides binary indicator cost functions (0 or 1 per step) for the NORBC-style
constrained RL pipeline. Each constraint has a discounted budget D_k.

Constraints:
    0. joint_velocity: 1 if any joint velocity exceeds limit
    1. accumulated_rotation: 1 if any joint accumulated rotation exceeds max
    2. joint_oscillation: 1 if HF component RMS exceeds limit
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

    from ..base_env import HeroAgentEnv


@configclass
class ALBCConstraintCfg:
    """Configuration for ALBC constraint costs in IPO pipeline.

    All costs are binary indicators (0 or 1 per step per env).
    Discounted budget: d_k = D_k / (1 - gamma).
    """

    num_constraints: int = 3

    # Constraint 0: joint velocity limit (rad/s)
    joint_velocity_limit: float = 3.0

    # Constraint 1: accumulated rotation (full rotations before violation)
    max_accumulated_rotations: float = 2.0

    # Constraint 2: joint oscillation HF RMS limit (rad/s)
    oscillation_limit: float = 1.5

    # Per-constraint budgets D_k (probability of violation per step)
    constraint_budgets: tuple[float, ...] = (0.3, 0.05, 0.3)

    # IPO barrier parameters
    barrier_t: float = 1.0
    barrier_t_final: float = 50.0
    barrier_t_schedule_iters: int = 1000

    # Adaptive threshold: d_k^i = max(d_k, J_C_k + alpha * d_k)
    adaptive_threshold_alpha: float = 0.1

    # Cost GAE parameters
    cost_gamma: float = 0.99
    cost_lam: float = 0.95


def joint_velocity_cost(
    _robot: Articulation,
    env: HeroAgentEnv,
    limit: float = 3.0,
) -> torch.Tensor:
    """Binary cost: 1 if any ALBC joint velocity exceeds limit.

    Args:
        env: Environment instance.
        limit: Velocity threshold in rad/s.

    Returns:
        (num_envs,) binary tensor.
    """
    joint_vel = _robot.data.joint_vel[:, env._albc_joint_ids]
    return (joint_vel.abs().max(dim=-1).values > limit).float()


def accumulated_rotation_cost(
    _robot: Articulation,
    env: HeroAgentEnv,
    max_rotations: float = 2.0,
) -> torch.Tensor:
    """Binary cost: 1 if any joint accumulated rotation exceeds max.

    Requires env._accumulated_rotation buffer (initialized in base_env).

    Args:
        env: Environment instance.
        max_rotations: Maximum full rotations (2*pi each) before violation.

    Returns:
        (num_envs,) binary tensor.
    """
    threshold = max_rotations * 2.0 * torch.pi
    return (env._accumulated_rotation.abs().max(dim=-1).values > threshold).float()


def joint_oscillation_cost(
    _robot: Articulation,
    env: HeroAgentEnv,
    limit: float = 1.5,
) -> torch.Tensor:
    """Binary cost: 1 if HF joint velocity RMS exceeds limit.

    Reuses env._ema_joint_vel for the low-frequency component.

    Args:
        env: Environment instance.
        limit: HF RMS threshold in rad/s.

    Returns:
        (num_envs,) binary tensor.
    """
    joint_vel = _robot.data.joint_vel[:, env._albc_joint_ids]
    hf = joint_vel - env._ema_joint_vel
    hf_rms = hf.pow(2).mean(dim=-1).sqrt()
    return (hf_rms > limit).float()


def compute_all_costs(
    robot: Articulation,
    env: HeroAgentEnv,
    cfg: ALBCConstraintCfg,
) -> torch.Tensor:
    """Compute all K constraint costs and stack into (num_envs, K) tensor.

    Args:
        robot: Robot articulation.
        env: Environment instance.
        cfg: Constraint configuration.

    Returns:
        (num_envs, K) binary cost tensor.
    """
    c0 = joint_velocity_cost(robot, env, limit=cfg.joint_velocity_limit)
    c1 = accumulated_rotation_cost(robot, env, max_rotations=cfg.max_accumulated_rotations)
    c2 = joint_oscillation_cost(robot, env, limit=cfg.oscillation_limit)
    return torch.stack([c0, c1, c2], dim=-1)
