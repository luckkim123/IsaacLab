# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ActorCritic with TDC-specific output for RL-ALBC integration.

Extends ActorCriticEncoder to expose the encoder's latent z as M_hat
for the TDC controller. The actor outputs PD gains (4D) instead of
direct joint velocities (2D).

Actor Output (4D):
    [K_p_roll, K_d_roll, K_p_pitch, K_d_pitch]
    All outputs in [-1, 1], scaled to gain ranges by TDC controller.

Reference:
    - IROS 2026: RL-ALBC with TDC-based adaptive attitude control
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .actor_critic_encoder import ActorCriticEncoder

if TYPE_CHECKING:
    from tensordict import TensorDict


class ActorCriticEncoderTDC(ActorCriticEncoder):
    """ActorCriticEncoder variant that exposes z for TDC controller.

    Differences from base ActorCriticEncoder:
        - Stores last computed z in _last_z for external access via last_z property
        - TDC controller reads M_hat = last_z[3:5] (roll/pitch per 6-DOF convention)
    """

    def __init__(
        self,
        obs: Any,
        obs_groups: Any,
        num_actions: int = 4,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if num_actions != 4:
            print(
                f"Warning: ActorCriticEncoderTDC expects num_actions=4 (gains), "
                f"got {num_actions}. Proceeding with provided value."
            )
        super().__init__(obs, obs_groups, num_actions, *args, **kwargs)
        self._last_z: torch.Tensor | None = None

    @property
    def last_z(self) -> torch.Tensor | None:
        """Last computed encoder latent z. TDC uses z[3:5] as M_hat (roll/pitch).

        Returns:
            Latent z tensor of shape (num_envs, encoder_latent_dim), or None.
        """
        return self._last_z

    def _get_combined_obs(self, obs: TensorDict) -> torch.Tensor:
        """Get combined observation, storing z for TDC controller access."""
        policy_obs = obs[self._policy_obs_key]
        z = self._encode(obs[self._privileged_key])
        self._last_z = z
        return torch.cat([policy_obs, z], dim=-1)
