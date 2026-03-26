# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Constraint cost functions for IPO (Interior-Point Optimization).

Two types following the paper's framework:
    Probabilistic: C_k = I(violation) in {0, 1}, budget = max violation probability
    Average:       C_k = f(s,a,s') in R, budget = max average value

All constraints satisfy: J_Ck(pi) = E[sum gamma^t C_k] <= d_k
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


# --- Configuration ---


@configclass
class ConstraintTermCfg:
    """Single constraint: cost function + per-step budget D_k."""

    func: Callable = lambda _r, _e: torch.zeros(1)
    params: dict = {}
    budget: float = 0.1
    name: str = ""


@configclass
class ALBCConstraintCfg:
    """List of constraint terms for IPO barrier."""

    terms: list[ConstraintTermCfg] = []
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


# --- Probabilistic Constraints (binary indicator) ---


def attitude_limit_cost(
    _robot: Articulation,
    _env: ALBCEnv,
    limit: float = 1.396,
) -> torch.Tensor:
    """C_com: I(max(|roll|, |pitch|) > limit). Maps to paper's c_com (tilt limit)."""
    roll, pitch, _ = euler_xyz_from_quat(_robot.data.root_quat_w)
    return (torch.max(roll.abs(), pitch.abs()) > limit).float()


def torque_limit_cost(
    _robot: Articulation,
    env: ALBCEnv,
    limit_nm: float = 9.5,
) -> torch.Tensor:
    """C_jt: I(any |tau_j| > tau_max). Maps to paper's c_jt (joint torque limit).

    Uses applied_torque (post-actuator-clamp) rather than computed_torque (pre-clamp)
    because computed_torque from the PD controller is unbounded (Kp*error can be 500+ Nm)
    while the physical motor output is limited by the actuator effort_limit.
    """
    applied = _robot.data.applied_torque[:, env._albc_joint_ids]
    return (applied.abs() > limit_nm).any(dim=-1).float()


def velocity_limit_cost(
    _robot: Articulation,
    env: ALBCEnv,
    limit_rad_per_s: float = 4.189,
) -> torch.Tensor:
    """C_jv: I(any |q_dot_j| > q_dot_max). Maps to paper's c_jv (joint velocity limit)."""
    joint_vel = _robot.data.joint_vel[:, env._albc_joint_ids]
    return (joint_vel.abs().max(dim=-1).values > limit_rad_per_s).float()


# --- Average Constraints (continuous) ---


def yaw_velocity_cost(
    _robot: Articulation,
    _env: ALBCEnv,
) -> torch.Tensor:
    """C_ov: |w_z|. Maps to paper's c_ov (undesired orthogonal rotation)."""
    return _robot.data.root_ang_vel_b[:, 2].abs()


# --- Dispatch ---


def compute_all_costs(
    robot: Articulation,
    env: ALBCEnv,
    cfg: ALBCConstraintCfg,
) -> torch.Tensor:
    """Compute all K costs -> (num_envs, K) tensor."""
    return torch.stack([t.func(robot, env, **t.params) for t in cfg.terms], dim=-1)
