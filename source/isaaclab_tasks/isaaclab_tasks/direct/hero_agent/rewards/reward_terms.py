# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward term functions for Hero Agent ALBC environments.

This module provides standalone reward functions following Isaac Lab conventions.
Each function computes a reward component for the given robot state.

Function signature:
    func(robot: Articulation, **params) -> torch.Tensor

where the returned tensor has shape (num_envs,).

Note: Weights and dt scaling are applied by the RewardManager, not here.

ALBC (Active Linear Buoyancy Controller) uses joint-based attitude control
without thrusters, so rewards focus on orientation and joint action costs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab_tasks.direct.hero_agent.tasks.albc_attitude_task import ALBCAttitudeTask


# =============================================================================
# Orientation Reward Terms (for attitude control)
# =============================================================================


def orientation_upright(
    robot: Articulation,
    **kwargs,
) -> torch.Tensor:
    """Upright orientation reward (linear).

    Computes how aligned the robot's body Z-axis is with the world Z-axis.
    Returns value in [0, 1] where 1 = perfectly upright, 0 = upside down.

    Note: This is a gentle reward. For stricter orientation control,
    use orientation_upright_exp which penalizes tilting more aggressively.

    Args:
        robot: Robot articulation for state access.

    Returns:
        Reward values. Shape: (num_envs,).
    """
    num_envs = robot.data.root_pos_w.shape[0]
    device = robot.data.root_pos_w.device

    # World up vector
    up_w = torch.tensor([[0.0, 0.0, 1.0]], device=device).expand(num_envs, -1)

    # Transform to body frame and check alignment
    up_b = quat_apply_inverse(robot.data.root_quat_w, up_w)

    # Map z-component from [-1, 1] to [0, 1]
    return (up_b[:, 2] + 1.0) / 2.0


def orientation_upright_exp(
    robot: Articulation,
    exp_scale: float = 0.5,
    **kwargs,
) -> torch.Tensor:
    """Exponential upright orientation reward.

    Computes: exp(-tilt_angle / exp_scale)

    This reward penalizes tilting more aggressively than linear reward.
    At exp_scale=0.5 (default):
        - 0 deg tilt  -> reward = 1.0
        - 15 deg tilt -> reward = 0.59
        - 30 deg tilt -> reward = 0.35
        - 45 deg tilt -> reward = 0.21

    Args:
        robot: Robot articulation for state access.
        exp_scale: Scale factor for exponential decay (radians).

    Returns:
        Reward values. Shape: (num_envs,).
    """
    num_envs = robot.data.root_pos_w.shape[0]
    device = robot.data.root_pos_w.device

    # World up vector
    up_w = torch.tensor([[0.0, 0.0, 1.0]], device=device).expand(num_envs, -1)

    # Transform to body frame and check alignment
    up_b = quat_apply_inverse(robot.data.root_quat_w, up_w)

    # Compute tilt angle from z-component: cos(tilt) = up_b[:, 2]
    # Clamp to avoid numerical issues with acos
    cos_tilt = up_b[:, 2].clamp(-1.0, 1.0)
    tilt_angle = torch.acos(cos_tilt)  # radians

    return torch.exp(-tilt_angle / exp_scale)


# =============================================================================
# Action Penalty Terms (for smooth joint control)
# =============================================================================


def angular_velocity_penalty(
    robot: Articulation,
    **kwargs,
) -> torch.Tensor:
    """Angular velocity penalty.

    Penalizes high angular velocities in the body frame.
    Returns: sum(omega^2) (positive; weight in config should be negative).

    Args:
        robot: Robot articulation for state access.

    Returns:
        Penalty values (positive). Shape: (num_envs,).
    """
    return torch.sum(torch.square(robot.data.root_ang_vel_b), dim=1)


def action_rate_penalty(
    robot: Articulation,
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Action rate penalty.

    Penalizes rapid changes in action to encourage smooth control.
    Returns: sum((a_t - a_{t-1})^2) (positive; weight in config should be negative).

    Args:
        robot: Robot articulation (unused but required for signature).
        actions: Current actions. Shape: (num_envs, num_actions).
        prev_actions: Previous actions. Shape: (num_envs, num_actions).

    Returns:
        Penalty values (positive). Shape: (num_envs,).
    """
    action_rate = actions - prev_actions
    return torch.sum(torch.square(action_rate), dim=1)


def action_magnitude_penalty(
    robot: Articulation,
    actions: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Action magnitude penalty.

    Penalizes large action values to encourage energy efficiency.
    Returns: sum(a^2) (positive; weight in config should be negative).

    Args:
        robot: Robot articulation (unused but required for signature).
        actions: Current actions. Shape: (num_envs, num_actions).

    Returns:
        Penalty values (positive). Shape: (num_envs,).
    """
    return torch.sum(torch.square(actions), dim=1)


def alive_bonus(
    robot: Articulation,
    **kwargs,
) -> torch.Tensor:
    """Alive bonus.

    Constant reward for surviving each timestep.
    Returns: 1.0 for all environments.

    Args:
        robot: Robot articulation for accessing num_envs and device.

    Returns:
        Constant reward values. Shape: (num_envs,).
    """
    num_envs = robot.data.root_pos_w.shape[0]
    device = robot.data.root_pos_w.device
    return torch.ones(num_envs, device=device)


# =============================================================================
# ALBC (Active Linear Buoyancy Controller) Reward Terms
# =============================================================================


def albc_potential_reward(
    robot: Articulation,
    task: ALBCAttitudeTask,
    scale: float = 8.0,
    **kwargs,
) -> torch.Tensor:
    """Potential-based pose reward for ALBC.

    Computes: scale * exp(-potential)

    This reward encourages the robot to minimize attitude error.
    At zero error (potential=0), reward is maximized at 'scale'.

    Args:
        robot: Robot articulation (unused but required for signature).
        task: ALBCAttitudeTask instance with potential tracking.
        scale: Maximum reward at zero error (default: 8.0).

    Returns:
        Reward values. Shape: (num_envs,).
    """
    return scale * torch.exp(-task.potentials)


def albc_progress_reward(
    robot: Articulation,
    task: ALBCAttitudeTask,
    **kwargs,
) -> torch.Tensor:
    """Progress reward based on potential difference for ALBC.

    Computes: prev_potential - current_potential

    Positive when moving toward target (error decreasing),
    negative when moving away (error increasing).

    Args:
        robot: Robot articulation (unused but required for signature).
        task: ALBCAttitudeTask instance with potential tracking.

    Returns:
        Reward values. Shape: (num_envs,).
    """
    return task.prev_potentials - task.potentials


def albc_action_cost(
    robot: Articulation,
    actions: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Action magnitude cost for ALBC joint control.

    Computes: sum(actions^2, dim=-1)

    Penalizes large joint velocity commands to encourage smooth control.
    Should be used with negative weight in config (e.g., -0.1).

    Args:
        robot: Robot articulation (unused but required for signature).
        actions: Current actions (joint velocity commands). Shape: (num_envs, 2).

    Returns:
        Cost values (positive). Shape: (num_envs,).
    """
    return torch.sum(actions ** 2, dim=-1)


def albc_alive_bonus(
    robot: Articulation,
    **kwargs,
) -> torch.Tensor:
    """Alive bonus for ALBC task.

    Returns constant reward for each timestep the robot is alive.
    From original implementation: alive_reward = 0.5

    Args:
        robot: Robot articulation for accessing num_envs and device.

    Returns:
        Constant reward values. Shape: (num_envs,).
    """
    num_envs = robot.data.root_pos_w.shape[0]
    device = robot.data.root_pos_w.device
    return torch.ones(num_envs, device=device) * 0.5
