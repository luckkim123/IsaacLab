# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Constraint cost functions for IPO (Interior-point Policy Optimization).

Provides cost functions (binary indicator or continuous) for the constrained
RL pipeline. Each constraint has a per-step budget D_k and a cost_type.

Binary constraints output 0/1 per step. Continuous constraints output a
non-negative scalar per step. The budget semantics differ:
    - binary: D_k = probability of violation per step
    - average: D_k = expected cost per step (mean over episode)

Registry pattern: ALBCConstraintCfg holds a list of ConstraintTermCfg,
each referencing a cost function. compute_all_costs() iterates over them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

    from ..base_env import HeroAgentEnv


# =============================================================================
# ConstraintTermCfg: per-term configuration
# =============================================================================


@configclass
class ConstraintTermCfg:
    """Configuration for a single constraint term.

    Attributes:
        func: Cost function (robot, env, **params) -> (num_envs,) tensor.
        params: Keyword arguments forwarded to func.
        budget: Per-step budget D_k.
        cost_type: "binary" (0/1 indicator) or "average" (continuous non-negative).
        name: Logging name. Derived from func.__name__ if empty.
    """

    func: Callable = lambda _r, _e: torch.zeros(1)  # placeholder; overridden per term
    params: dict = {}
    budget: float = 0.1
    cost_type: str = "binary"
    name: str = ""


# =============================================================================
# ALBCConstraintCfg: top-level constraint configuration
# =============================================================================


@configclass
class ALBCConstraintCfg:
    """Configuration for ALBC constraint costs in IPO pipeline.

    Uses a registry pattern: ``terms`` is a list of ConstraintTermCfg.
    ``num_constraints`` and ``constraint_budgets`` are derived properties.
    """

    terms: list[ConstraintTermCfg] = []

    # IPO barrier parameters
    barrier_t: float = 1.0
    barrier_t_final: float = 50.0
    barrier_t_schedule_iters: int = 1000

    # Adaptive threshold: d_k^i = max(d_k, J_C_k + alpha * d_k)
    adaptive_threshold_alpha: float = 0.1

    # Cost GAE parameters
    cost_gamma: float = 0.99
    cost_lam: float = 0.95

    @property
    def num_constraints(self) -> int:
        return len(self.terms)

    @property
    def constraint_budgets(self) -> tuple[float, ...]:
        return tuple(t.budget for t in self.terms)

    @property
    def constraint_names(self) -> tuple[str, ...]:
        return tuple(t.name or t.func.__name__ for t in self.terms)


# =============================================================================
# Cost functions: binary indicators
# =============================================================================


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
    limit: float = 0.6,
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


def attitude_absolute_cost(
    _robot: Articulation,
    env: HeroAgentEnv,
    limit: float = 0.436,
) -> torch.Tensor:
    """Binary cost: 1 if absolute roll or pitch exceeds limit (rad).

    Safety constraint to prevent capsizing. Uses absolute body orientation.

    Args:
        env: Environment instance.
        limit: Maximum absolute roll/pitch in radians (~25 deg default).

    Returns:
        (num_envs,) binary tensor.
    """
    roll, pitch, _ = euler_xyz_from_quat(_robot.data.root_quat_w)
    return (torch.max(roll.abs(), pitch.abs()) > limit).float()


def attitude_error_cost(
    _robot: Articulation,
    env: HeroAgentEnv,
    limit: float = 0.262,
) -> torch.Tensor:
    """Binary cost: 1 if attitude tracking error exceeds limit (rad).

    Tracking quality constraint. Uses env._attitude_error (target-relative).

    Args:
        env: Environment instance.
        limit: Maximum tracking error in radians (~15 deg default).

    Returns:
        (num_envs,) binary tensor.
    """
    err = env._attitude_error[:, :2]
    return (err.abs().max(dim=-1).values > limit).float()


def singularity_cost(
    _robot: Articulation,
    env: HeroAgentEnv,
    sin_g2_limit: float = 0.15,
) -> torch.Tensor:
    """Binary cost: 1 if arm is near kinematic singularity.

    For a 2-link planar arm (L1=L2), singularities occur when the 2nd joint
    angle g2 approaches 0 (full extension) or +-pi (fully folded), both
    corresponding to |sin(g2)| -> 0 where the Jacobian loses rank.

    Args:
        env: Environment instance.
        sin_g2_limit: Threshold on |sin(g2)|. Default 0.15 (~8.6 deg from
            singularity). Below this, DLS damping dominates and EE control
            degrades significantly.

    Returns:
        (num_envs,) binary tensor.
    """
    g2 = _robot.data.joint_pos[:, env._albc_joint_ids[1]]
    return (g2.sin().abs() < sin_g2_limit).float()


# =============================================================================
# Cost functions: continuous (average)
# =============================================================================


def action_smoothness_cost(
    _robot: Articulation,
    env: HeroAgentEnv,
) -> torch.Tensor:
    """Continuous cost: L2 norm of action rate |a(t) - a(t-1)|.

    Args:
        env: Environment instance.

    Returns:
        (num_envs,) non-negative tensor.
    """
    return torch.linalg.norm(env._actions - env._prev_actions, dim=-1)


def angular_velocity_cost(
    _robot: Articulation,
    env: HeroAgentEnv,
) -> torch.Tensor:
    """Continuous cost: L2 norm of roll/pitch angular velocity.

    Args:
        env: Environment instance.

    Returns:
        (num_envs,) non-negative tensor.
    """
    ang_vel_rp = _robot.data.root_ang_vel_b[:, :2]
    return ang_vel_rp.pow(2).sum(dim=-1).sqrt()


# =============================================================================
# compute_all_costs: registry-based dispatch
# =============================================================================


def compute_all_costs(
    robot: Articulation,
    env: HeroAgentEnv,
    cfg: ALBCConstraintCfg,
) -> torch.Tensor:
    """Compute all K constraint costs and stack into (num_envs, K) tensor.

    Iterates over cfg.terms and calls each term's func with its params.

    Args:
        robot: Robot articulation.
        env: Environment instance.
        cfg: Constraint configuration.

    Returns:
        (num_envs, K) cost tensor.
    """
    return torch.stack([t.func(robot, env, **t.params) for t in cfg.terms], dim=-1)
