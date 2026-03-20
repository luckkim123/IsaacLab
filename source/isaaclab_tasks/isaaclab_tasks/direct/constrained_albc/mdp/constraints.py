# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Constraint cost functions for Lagrangian constrained RL.

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

    from ..albc_env import ALBCEnv


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
# Cost functions: per-step costs (binary or continuous)
# =============================================================================


def accumulated_rotation_cost(
    _robot: Articulation,
    env: ALBCEnv,
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


def attitude_absolute_cost(
    _robot: Articulation,
    env: ALBCEnv,
    limit: float = 1.396,
) -> torch.Tensor:
    """Binary cost: 1 if absolute roll or pitch exceeds limit (rad).

    Safety constraint to prevent capsizing. Uses absolute body orientation.
    Acts as a warning zone before episode termination (90 deg).

    Args:
        env: Environment instance.
        limit: Maximum absolute roll/pitch in radians (~80 deg default).

    Returns:
        (num_envs,) binary tensor.
    """
    roll, pitch, _ = euler_xyz_from_quat(_robot.data.root_quat_w)
    return (torch.max(roll.abs(), pitch.abs()) > limit).float()


def effort_limit_cost(
    _robot: Articulation,
    env: ALBCEnv,
    real_limit_scale: float = 1.0,
) -> torch.Tensor:
    """Binary cost: 1 if any ALBC joint computed torque exceeds real motor limit.

    Uses per-env current effort limits (after DR), not a cached scalar.
    This correctly handles envs whose DR'd limit is lower than the default.

    Args:
        env: Environment instance.
        real_limit_scale: Scale applied to per-env effort limit (1.0 = current DR'd limit).

    Returns:
        (num_envs,) binary tensor.
    """
    computed = _robot.data.computed_torque[:, env._albc_joint_ids]
    # Per-env DR'd effort limits: (num_envs, num_joints) -> max across joints -> (num_envs,)
    real_limit = _robot.data.joint_effort_limits[:, env._albc_joint_ids].max(dim=-1).values * real_limit_scale
    return (computed.abs().max(dim=-1).values > real_limit).float()


def joint_velocity_limit_cost(
    _robot: Articulation,
    env: ALBCEnv,
    limit_rad_per_s: float = 4.189,
) -> torch.Tensor:
    """Binary cost: 1 if any ALBC joint velocity exceeds limit.

    Hard velocity limit based on motor specs. 4.189 rad/s = 40 RPM.

    Args:
        env: Environment instance.
        limit_rad_per_s: Maximum joint velocity in rad/s.

    Returns:
        (num_envs,) binary tensor.
    """
    joint_vel = _robot.data.joint_vel[:, env._albc_joint_ids]
    return (joint_vel.abs().max(dim=-1).values > limit_rad_per_s).float()


def overshoot_cost(
    _robot: Articulation,
    env: ALBCEnv,
    threshold: float = 0.035,
) -> torch.Tensor:
    """Binary cost: 1 if attitude error sign flips with magnitude > threshold.

    Detects overshoot: the error crosses zero (sign change on any axis)
    while the current error exceeds threshold. Uses per-axis signed error
    from env._prev_attitude_error_rp (roll/pitch).

    Args:
        env: Environment instance.
        threshold: Minimum error magnitude (rad) to count as overshoot (~2 deg).

    Returns:
        (num_envs,) binary tensor.
    """
    curr = env._attitude_error[:, :2]
    prev = env._prev_attitude_error_rp
    # Per-axis conjunction: sign flip AND magnitude > threshold on the SAME axis
    per_axis = (curr * prev < 0) & (curr.abs() > threshold)
    return per_axis.any(dim=-1).float()


def yaw_velocity_cost(
    _robot: Articulation,
    env: ALBCEnv,
) -> torch.Tensor:
    """Continuous cost: absolute yaw angular velocity (rad/s).

    Average constraint -- budget D_k is the target mean yaw rate.
    Buoyancy control cannot generate Z-axis torque, so yaw velocity
    is purely from disturbances and coupling. Constraining it
    prevents policies from exploiting yaw-coupled motions.

    Returns:
        (num_envs,) non-negative tensor in rad/s.
    """
    return _robot.data.root_ang_vel_b[:, 2].abs()


# =============================================================================
# compute_all_costs: registry-based dispatch
# =============================================================================


def compute_all_costs(
    robot: Articulation,
    env: ALBCEnv,
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
