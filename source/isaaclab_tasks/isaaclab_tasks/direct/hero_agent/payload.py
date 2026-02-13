# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Virtual payload physics model for Hero Agent gripper body.

Extracted from HeroAgentEnv to separate payload force computation from
the environment lifecycle. The payload is modeled as a point mass attached
to the gripper body (fixed to base via base_to_gripper joint).
"""

from __future__ import annotations

import torch

from isaaclab.utils.math import quat_apply_inverse


class PayloadPhysics:
    """Virtual payload weight model for gripper body.

    Computes gravitational force and torque from a point mass attached
    at (attachment_offset + cog_offset) in the gripper body frame.
    """

    def __init__(
        self,
        num_envs: int,
        device: str,
        default_mass: float,
        attachment_offset: tuple[float, float, float],
        gravity_vec: tuple[float, float, float],
    ) -> None:
        """Initialize payload buffers.

        Args:
            num_envs: Number of parallel environments.
            device: Computation device.
            default_mass: Default payload mass in kg.
            attachment_offset: Attachment offset in gripper frame (m).
            gravity_vec: Gravity vector from simulation config.
        """
        self._num_envs = num_envs
        self._device = device
        self._default_mass = default_mass

        self._mass = torch.full((num_envs,), default_mass, device=device)
        offset_tensor = torch.tensor(attachment_offset, device=device, dtype=torch.float32)
        self._attachment_offset = offset_tensor.expand(num_envs, -1).clone()
        self._cog_offset = torch.zeros(num_envs, 3, device=device)
        self._gravity_vec = torch.tensor(gravity_vec, device=device, dtype=torch.float32)

    @property
    def mass(self) -> torch.Tensor:
        """Payload mass per environment. Shape: (num_envs,)."""
        return self._mass

    @property
    def attachment_offset(self) -> torch.Tensor:
        """Attachment offset in gripper frame. Shape: (num_envs, 3)."""
        return self._attachment_offset

    @property
    def cog_offset(self) -> torch.Tensor:
        """CoG offset from attachment point. Shape: (num_envs, 3)."""
        return self._cog_offset

    def compute_wrench(self, gripper_quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute payload weight force and torque in gripper body frame.

        Args:
            gripper_quat: Gripper body quaternion (w,x,y,z). Shape: (num_envs, 4).

        Returns:
            Tuple of (forces, torques) in gripper body frame.
        """
        payload_weight_w = self._mass.unsqueeze(-1) * self._gravity_vec
        payload_weight_b = quat_apply_inverse(gripper_quat, payload_weight_w)
        effective_offset = self._attachment_offset + self._cog_offset
        payload_torque_b = torch.cross(effective_offset, payload_weight_b, dim=-1)
        return payload_weight_b, payload_torque_b

    def reset(
        self,
        env_ids: torch.Tensor,
        default_mass: float,
        default_attachment_offset: tuple[float, float, float],
    ) -> None:
        """Reset payload state for specified environments.

        Args:
            env_ids: Environment indices to reset.
            default_mass: Default mass to reset to.
            default_attachment_offset: Default attachment offset.
        """
        self._mass[env_ids] = default_mass
        offset = torch.tensor(default_attachment_offset, device=self._device, dtype=torch.float32)
        self._attachment_offset[env_ids] = offset
        self._cog_offset[env_ids] = 0.0
