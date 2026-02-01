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

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab_tasks.direct.hero_agent.tasks.albc_attitude_task import ALBCAttitudeTask


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
