# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Attitude task manager for Hero Agent environments.

Encapsulates target attitude, error computation, and potential-based reward
tracking. Extracted from HeroAgentEnv to separate task logic from environment
lifecycle.
"""

from __future__ import annotations

import torch

from isaaclab.utils.math import euler_xyz_from_quat

# Only roll and pitch are controllable (buoyancy cannot generate yaw torque)
_ROLL_PITCH_DIMS = 2


class AttitudeTask:
    """Manages attitude target, error computation, and potential-based rewards.

    Handles:
        - Target Euler angle generation (fixed or randomized)
        - Attitude error computation (quaternion -> Euler -> wrapped error)
        - Potential tracking for telescoping progress reward

    The potential is defined as the L2 norm of roll/pitch errors (yaw excluded
    because buoyancy control cannot generate Z-axis torque).
    """

    def __init__(
        self,
        num_envs: int,
        device: str,
        target_attitude: tuple[float, float, float],
        randomize: bool,
        target_range: tuple[float, float, float],
    ) -> None:
        """Initialize attitude task buffers.

        Args:
            num_envs: Number of parallel environments.
            device: Computation device.
            target_attitude: Default [roll, pitch, yaw] target in radians.
            randomize: Whether to randomize targets on reset.
            target_range: Max random offset per axis in radians.
        """
        self._num_envs = num_envs
        self._device = device
        self._randomize = randomize
        self._base_attitude = torch.tensor(target_attitude, device=device)
        self._target_range = torch.tensor(target_range, device=device)

        # Target Euler angles (roll, pitch, yaw)
        self._target_euler = self._base_attitude.unsqueeze(0).expand(num_envs, -1).clone()

        # Attitude error and potential tracking buffers
        self._attitude_error = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)
        self._potentials = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self._prev_potentials = torch.zeros(num_envs, dtype=torch.float32, device=device)

    @property
    def target_euler(self) -> torch.Tensor:
        """Target Euler angles [roll, pitch, yaw]. Shape: (num_envs, 3)."""
        return self._target_euler

    @property
    def attitude_error(self) -> torch.Tensor:
        """Last computed attitude error. Shape: (num_envs, 3)."""
        return self._attitude_error

    @property
    def potentials(self) -> torch.Tensor:
        """Current potential (L2 norm of roll/pitch error). Shape: (num_envs,)."""
        return self._potentials

    @property
    def prev_potentials(self) -> torch.Tensor:
        """Previous step potential. Shape: (num_envs,)."""
        return self._prev_potentials

    def compute_error(
        self,
        quat: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute attitude error from quaternion orientation.

        Args:
            quat: Quaternion orientation (w, x, y, z). Shape: (N, 4).
            env_ids: Environment indices. If None, computes for all envs.

        Returns:
            Attitude error (target - current), wrapped to [-pi, pi]. Shape: (N, 3).
        """
        current_euler = torch.stack(euler_xyz_from_quat(quat), dim=-1)
        target = self._target_euler if env_ids is None else self._target_euler[env_ids]
        error = target - current_euler
        return torch.atan2(torch.sin(error), torch.cos(error))

    def update_potentials(self, quat: torch.Tensor) -> None:
        """Update potential values for reward computation.

        Saves current potential as prev_potential and computes new potential
        from roll/pitch errors. Yaw is excluded because buoyancy control
        cannot generate Z-axis torque.

        Also caches attitude_error so that logging in _reset_idx uses the
        current step's error.

        Args:
            quat: Current root quaternion. Shape: (num_envs, 4).
        """
        self._prev_potentials = self._potentials.clone()
        self._attitude_error = self.compute_error(quat)
        self._potentials = torch.linalg.norm(self._attitude_error[:, :_ROLL_PITCH_DIMS], dim=-1)

    def get_attitude_error(self, quat: torch.Tensor) -> torch.Tensor:
        """Compute and cache attitude error for observations.

        Args:
            quat: Current root quaternion. Shape: (num_envs, 4).

        Returns:
            Attitude error (target - current). Shape: (num_envs, 3).
        """
        self._attitude_error = self.compute_error(quat)
        return self._attitude_error

    def reset_targets(self, env_ids: torch.Tensor) -> None:
        """Reset target attitudes for specified environments.

        Args:
            env_ids: Environment indices to reset.
        """
        num_reset = len(env_ids)

        if self._randomize:
            random_offset = (torch.rand(num_reset, 3, device=self._device) * 2 - 1) * self._target_range
            self._target_euler[env_ids] = self._base_attitude + random_offset
        else:
            self._target_euler[env_ids] = self._base_attitude.unsqueeze(0).expand(num_reset, -1)

        self._potentials[env_ids] = 0.0
        self._prev_potentials[env_ids] = 0.0

    def initialize_potentials(self, env_ids: torch.Tensor, quat: torch.Tensor) -> None:
        """Initialize potential values after robot pose reset.

        Sets both prev_potentials and potentials to the same value to prevent
        spurious progress reward on the first step.

        Args:
            env_ids: Environment indices that were reset.
            quat: Root quaternion for the reset envs. Shape: (len(env_ids), 4).
        """
        attitude_error = self.compute_error(quat, env_ids)
        initial_potential = torch.linalg.norm(attitude_error[:, :_ROLL_PITCH_DIMS], dim=-1)
        self._potentials[env_ids] = initial_potential
        self._prev_potentials[env_ids] = initial_potential
