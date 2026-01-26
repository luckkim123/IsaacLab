# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Thruster model for underwater vehicles.

This module provides a modular thruster system implementation that converts
thruster commands to body forces and torques using an allocation matrix.
"""

from __future__ import annotations

import torch

# Import configuration class from isaaclab_assets
from isaaclab_assets.robots.uuv import ThrusterCfg

# Alias for backward compatibility (ThrusterModelCfg was the old name)
ThrusterModelCfg = ThrusterCfg

__all__ = ["ThrusterModel", "ThrusterCfg", "ThrusterModelCfg"]


class ThrusterModel:
    """Thruster model for underwater vehicles.

    This model implements first-order thruster dynamics with configurable
    time constants and an allocation matrix for converting thruster commands
    to body forces and torques.

    Attributes:
        cfg: Thruster configuration.
        num_envs: Number of parallel environments.
        device: Computation device.
    """

    def __init__(
        self,
        cfg: ThrusterCfg,
        num_envs: int,
        device: str,
        enable_randomization: bool = False,
    ) -> None:
        """Initialize the thruster model.

        Args:
            cfg: Thruster configuration.
            num_envs: Number of parallel environments.
            device: Computation device.
            enable_randomization: Whether to enable per-environment parameter storage.
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device

        # Setup buffers
        self._setup_buffers()

        # Setup allocation matrix
        self._allocation_matrix = torch.tensor(
            cfg.allocation_matrix, dtype=torch.float32, device=device
        )

        # Setup per-environment parameters if randomization enabled
        if enable_randomization:
            self._thrust_coeff = torch.full((num_envs,), cfg.thrust_coefficient, device=device)
            self._time_constant_up = torch.full((num_envs,), cfg.time_constant_up, device=device)
            self._time_constant_down = torch.full((num_envs,), cfg.time_constant_down, device=device)
        else:
            self._thrust_coeff = None
            self._time_constant_up = None
            self._time_constant_down = None

    def _setup_buffers(self) -> None:
        """Setup internal buffers."""
        self._state = torch.zeros(self.num_envs, self.cfg.num_thrusters, device=self.device)
        self._body_forces = torch.zeros(self.num_envs, 3, device=self.device)
        self._body_torques = torch.zeros(self.num_envs, 3, device=self.device)

    @property
    def state(self) -> torch.Tensor:
        """Current thruster state (normalized). Shape: (num_envs, num_thrusters)."""
        return self._state

    @property
    def body_forces(self) -> torch.Tensor:
        """Computed body forces. Shape: (num_envs, 3)."""
        return self._body_forces

    @property
    def body_torques(self) -> torch.Tensor:
        """Computed body torques. Shape: (num_envs, 3)."""
        return self._body_torques

    def apply_dynamics(
        self,
        commands: torch.Tensor,
        dt: float,
    ) -> None:
        """Apply first-order thruster dynamics to commands.

        Args:
            commands: Raw thruster commands. Shape: (num_envs, num_thrusters).
            dt: Time step duration.
        """
        target_state = commands.clamp(-1.0, 1.0)

        if self._thrust_coeff is not None:
            tau_up = self._time_constant_up.unsqueeze(-1)
            tau_down = self._time_constant_down.unsqueeze(-1)
        else:
            tau_up = self.cfg.time_constant_up
            tau_down = self.cfg.time_constant_down

        tau = torch.where(
            target_state > self._state,
            tau_up if isinstance(tau_up, torch.Tensor) else torch.tensor(tau_up, device=self.device),
            tau_down if isinstance(tau_down, torch.Tensor) else torch.tensor(tau_down, device=self.device),
        )

        alpha = dt / tau
        self._state = self._state + alpha * (target_state - self._state)

    def compute_wrench(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute body forces and torques from current thruster state.

        Returns:
            Tuple of (forces, torques) tensors. Each shape: (num_envs, 3).
        """
        if self._thrust_coeff is not None:
            thrust_magnitude = self._state * self._thrust_coeff.unsqueeze(-1)
        else:
            thrust_magnitude = self._state * self.cfg.thrust_coefficient

        thrust_magnitude = thrust_magnitude.clamp(-self.cfg.max_thrust, self.cfg.max_thrust)

        body_wrench = torch.einsum("ij,nj->ni", self._allocation_matrix, thrust_magnitude)

        self._body_forces = body_wrench[:, :3]
        self._body_torques = body_wrench[:, 3:]

        return self._body_forces, self._body_torques

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset thruster state for specified environments.

        Args:
            env_ids: Environment indices to reset.
        """
        self._state[env_ids] = 0.0

    def randomize_parameters(
        self,
        env_ids: torch.Tensor,
        thrust_coeff_scale: tuple[float, float] = (0.8, 1.2),
        time_constant_scale: tuple[float, float] = (0.8, 1.2),
    ) -> None:
        """Randomize thruster parameters for specified environments.

        Args:
            env_ids: Environment indices to randomize.
            thrust_coeff_scale: Scale range for thrust coefficient.
            time_constant_scale: Scale range for time constants.
        """
        if self._thrust_coeff is None:
            return

        num_envs = len(env_ids)

        tc_scale = (
            torch.rand(num_envs, device=self.device) * (thrust_coeff_scale[1] - thrust_coeff_scale[0])
            + thrust_coeff_scale[0]
        )
        self._thrust_coeff[env_ids] = self.cfg.thrust_coefficient * tc_scale

        time_scale = (
            torch.rand(num_envs, device=self.device) * (time_constant_scale[1] - time_constant_scale[0])
            + time_constant_scale[0]
        )
        self._time_constant_up[env_ids] = self.cfg.time_constant_up * time_scale
        self._time_constant_down[env_ids] = self.cfg.time_constant_down * time_scale
