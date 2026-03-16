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
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_apply_inverse

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


def joint_velocity_cost(
    _robot: Articulation,
    env: HeroAgentEnv,
) -> torch.Tensor:
    """Continuous cost: max absolute ALBC joint velocity (rad/s).

    Average constraint -- budget D_k is the target mean velocity.

    Args:
        env: Environment instance.

    Returns:
        (num_envs,) non-negative tensor in rad/s.
    """
    joint_vel = _robot.data.joint_vel[:, env._albc_joint_ids]
    return joint_vel.abs().max(dim=-1).values


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
) -> torch.Tensor:
    """Continuous cost: HF joint velocity RMS (rad/s).

    Average constraint -- budget D_k is the target mean HF RMS.
    Reuses env._ema_joint_vel for the low-frequency component.

    Args:
        env: Environment instance.

    Returns:
        (num_envs,) non-negative tensor in rad/s.
    """
    joint_vel = _robot.data.joint_vel[:, env._albc_joint_ids]
    hf = joint_vel - env._ema_joint_vel
    return hf.pow(2).mean(dim=-1).sqrt()


def attitude_absolute_cost(
    _robot: Articulation,
    env: HeroAgentEnv,
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


def effort_limit_cost(
    _robot: Articulation,
    env: HeroAgentEnv,
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


def yaw_velocity_cost(
    _robot: Articulation,
    env: HeroAgentEnv,
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


def cob_cog_alignment_cost(
    _robot: Articulation,
    env: HeroAgentEnv,
) -> torch.Tensor:
    """Continuous cost: lateral XY offset between system CoB and CoG (meters).

    Average constraint -- budget D_k is the target mean offset.
    Large lateral CoB-CoG separation creates a persistent roll/pitch bias
    that the controller must fight against, wasting actuator authority.

    System CoG: mass-weighted (main + buoy + payload).
    System CoB: volume-weighted (main + buoy).

    Returns:
        (num_envs,) non-negative tensor in meters.
    """
    hydro = env._hydro
    buoy_hydro = env._buoy_hydro
    root_quat = _robot.data.root_quat_w
    root_pos = _robot.data.root_pos_w

    # Buoy offset in body frame
    buoy_pos_w = _robot.data.body_pos_w[:, env._buoy_body_id[0]]
    buoy_offset_b = quat_apply_inverse(root_quat, buoy_pos_w - root_pos)

    # Mass-weighted CoG
    m_main = hydro.body_mass  # (num_envs,)
    m_buoy = buoy_hydro.body_mass  # (num_envs,)
    r_cg = m_main.unsqueeze(-1) * hydro.center_of_gravity + m_buoy.unsqueeze(-1) * (
        buoy_offset_b + buoy_hydro.center_of_gravity
    )

    m_total = m_main + m_buoy
    if env._payload_mass is not None:
        gripper_pos_w = _robot.data.body_pos_w[:, env._gripper_body_id[0]]
        gripper_quat = _robot.data.body_quat_w[:, env._gripper_body_id[0]]
        payload_cog_w = gripper_pos_w + quat_apply(
            gripper_quat,
            env._payload_attachment_offset + env._payload_cog_offset,
        )
        payload_cog_b = quat_apply_inverse(root_quat, payload_cog_w - root_pos)
        p_mass = env._payload_mass.squeeze(-1)  # (num_envs,)
        r_cg = r_cg + p_mass.unsqueeze(-1) * payload_cog_b
        m_total = m_total + p_mass
    r_cg = r_cg / m_total.unsqueeze(-1)

    # Volume-weighted CoB
    V_main = hydro.volume  # (num_envs,)
    V_buoy = buoy_hydro.volume
    r_cb = (
        V_main.unsqueeze(-1) * hydro.center_of_buoyancy
        + V_buoy.unsqueeze(-1) * (buoy_offset_b + buoy_hydro.center_of_buoyancy)
    ) / (V_main + V_buoy).unsqueeze(-1)

    return torch.linalg.norm(r_cb[:, :2] - r_cg[:, :2], dim=-1)


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
