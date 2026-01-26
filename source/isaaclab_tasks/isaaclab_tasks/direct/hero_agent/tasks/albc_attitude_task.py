# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ALBC (Active Linear Buoyancy Controller) attitude task for UUV environments.

This module implements the ALBC attitude task where the underwater vehicle uses
revolute joints (joint1, joint2) to control buoyancy position for attitude stabilization.
Target attitude is fixed at (0, 0, 0) and the task uses potential-based shaping
for reward computation.

Reference: Original IsaacGym implementation in isaacgym_agent/tasks/heroagent.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat

from .task_base import TaskBase, TaskBaseCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


@configclass
class ALBCAttitudeTaskCfg(TaskBaseCfg):
    """Configuration for the ALBC attitude control task.

    The ALBC task requires the UUV to maintain a target attitude using only
    the active linear buoyancy controller (2 revolute joints). Target attitude
    is typically fixed at (0, 0, 0) for upright stabilization.
    """

    # Target attitude [roll, pitch, yaw] in radians (default: upright)
    target_attitude: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Whether to randomize target attitude on reset (typically False for ALBC)
    randomize_target: bool = False

    # Target attitude randomization range if randomize_target=True [roll, pitch, yaw]
    target_attitude_range: tuple[float, float, float] = (0.3, 0.3, 0.0)


class ALBCAttitudeTask(TaskBase):
    """ALBC attitude control task implementation for UUV environments.

    Uses potential-based shaping for reward computation:
        potential = sqrt(roll_err^2 + pitch_err^2 + yaw_err^2)
        reward = 8*exp(-potential) + (prev_potential - potential) - actions_cost

    The vehicle must achieve and maintain a target orientation (default: upright)
    using only the buoyancy control mechanism (no thrusters).

    Observation Space (task contribution):
        - Attitude error in body frame (3) - roll, pitch, yaw errors

    Attributes:
        potentials: Current potential values for each environment.
        prev_potentials: Previous potential values for progress reward.
    """

    cfg: ALBCAttitudeTaskCfg

    def __init__(
        self,
        cfg: ALBCAttitudeTaskCfg,
        num_envs: int,
        device: str,
    ) -> None:
        """Initialize the ALBC attitude task.

        Args:
            cfg: ALBC attitude task configuration.
            num_envs: Number of parallel environments.
            device: Computation device.
        """
        super().__init__(cfg, num_envs, device)

    def _setup_buffers(self) -> None:
        """Setup ALBC attitude task buffers including potential tracking."""
        # Target Euler angles (roll, pitch, yaw)
        target = torch.tensor(self.cfg.target_attitude, device=self.device)
        self._target_euler = target.unsqueeze(0).expand(self.num_envs, -1).clone()

        # Attitude error buffer
        self._attitude_error = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)

        # Potential-based reward tracking
        self._potentials = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._prev_potentials = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    @property
    def goal_observation_dim(self) -> int:
        """Dimension of goal-related observations (attitude error)."""
        return 3

    @property
    def potentials(self) -> torch.Tensor:
        """Current potential values. Shape: (num_envs,)."""
        return self._potentials

    @property
    def prev_potentials(self) -> torch.Tensor:
        """Previous potential values. Shape: (num_envs,)."""
        return self._prev_potentials

    @property
    def target_euler(self) -> torch.Tensor:
        """Target Euler angles (roll, pitch, yaw). Shape: (num_envs, 3)."""
        return self._target_euler

    @property
    def attitude_error(self) -> torch.Tensor:
        """Current attitude error (roll, pitch, yaw). Shape: (num_envs, 3)."""
        return self._attitude_error

    def get_goal_observations(
        self,
        robot: Articulation,
    ) -> torch.Tensor:
        """Compute attitude error and update potentials.

        Returns the difference between target and current orientation
        as roll, pitch, yaw errors. Also updates the potential values
        for potential-based reward computation.

        Args:
            robot: Robot articulation for pose access.

        Returns:
            Attitude error (target - current). Shape: (num_envs, 3).
        """
        # Get current orientation as Euler angles
        current_quat = robot.data.root_quat_w  # (w, x, y, z)
        roll, pitch, yaw = euler_xyz_from_quat(current_quat)

        # Stack to (num_envs, 3)
        current_euler = torch.stack([roll, pitch, yaw], dim=-1)

        # Compute attitude error: target - current
        self._attitude_error = self._target_euler - current_euler

        # Wrap angles to [-pi, pi] for proper error computation
        self._attitude_error = torch.atan2(
            torch.sin(self._attitude_error), torch.cos(self._attitude_error)
        )

        # Update potentials: save previous and compute new
        self._prev_potentials = self._potentials.clone()

        # Potential = L2 norm of attitude error
        self._potentials = torch.sqrt(
            self._attitude_error[:, 0] ** 2
            + self._attitude_error[:, 1] ** 2
            + self._attitude_error[:, 2] ** 2
        )

        return self._attitude_error

    def reset(
        self,
        env_ids: torch.Tensor,
        env_origins: torch.Tensor,
    ) -> None:
        """Reset target attitudes and potentials for specified environments.

        Args:
            env_ids: Environment indices to reset.
            env_origins: Origin positions for each environment (not used for attitude).
        """
        num_reset = len(env_ids)

        if self.cfg.randomize_target:
            # Sample random target attitude within range
            attitude_range = torch.tensor(self.cfg.target_attitude_range, device=self.device)
            base_attitude = torch.tensor(self.cfg.target_attitude, device=self.device)

            random_offset = (torch.rand(num_reset, 3, device=self.device) * 2 - 1) * attitude_range
            target_euler = base_attitude + random_offset
        else:
            # Use fixed target attitude
            target_euler = torch.tensor(
                self.cfg.target_attitude, device=self.device
            ).unsqueeze(0).expand(num_reset, -1)

        # Store target Euler angles
        self._target_euler[env_ids] = target_euler

        # Reset potentials to prevent reward spikes on reset
        self._potentials[env_ids] = 0.0
        self._prev_potentials[env_ids] = 0.0

    def get_goal_state(self) -> dict[str, torch.Tensor]:
        """Get current goal state for visualization.

        Returns:
            Dictionary with target attitude and potential tensors.
        """
        return {
            "target_euler": self._target_euler,
            "potentials": self._potentials,
            "prev_potentials": self._prev_potentials,
        }
