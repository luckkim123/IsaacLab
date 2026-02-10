# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ActorCriticEncoder variant that exposes encoder latent z for TDC integration.

This module extends ActorCriticEncoder to store the last computed z so that
the environment can extract M_hat = z[3:5] for the TDC controller's adaptive
design inertia.

Architecture (same as base ActorCriticEncoder):
    Encoder: privileged (22D) -> MLP -> softplus + z_min -> z (6D)
    Actor:   cat([policy_obs, z]) = 19D -> MLP -> actions (4D: Kp_r, Kp_p, Kd_r, Kd_p)
    Critic:  cat([policy_obs, z]) = 19D -> MLP -> value (1D)

The only difference is that _get_combined_obs() caches z in _last_z, and
get_last_z() provides read access for the environment.

Timing:
    RSL-RL loop: obs = env.get_observations() -> action = policy.act(obs)
                 -> env.step(action) [calls _pre_physics_step]
    So _last_z is computed in act() and available when _pre_physics_step() runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .actor_critic_encoder import ActorCriticEncoder

if TYPE_CHECKING:
    from tensordict import TensorDict


class ActorCriticEncoderTDC(ActorCriticEncoder):
    """ActorCriticEncoder with z exposure for TDC M_hat extraction.

    Overrides _get_combined_obs() to cache the encoder latent z after each
    forward pass. The environment retrieves z via get_last_z() to set
    TDC M_hat = clamp(z[3:5], m_hat_min, m_hat_max).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._last_z: torch.Tensor | None = None

    def _get_combined_obs(self, obs: TensorDict) -> torch.Tensor:
        """Get combined observation and cache z for environment access.

        Both actor and critic use the same combined observation (symmetric design).
        The cached _last_z enables the environment to extract M_hat from the
        encoder output without re-running the encoder.
        """
        policy_obs = obs[self._policy_obs_key]
        z = self._encode(obs[self._privileged_key])
        self._last_z = z
        return torch.cat([policy_obs, z], dim=-1)

    def get_last_z(self) -> torch.Tensor | None:
        """Return the last computed encoder latent z.

        Returns:
            z tensor of shape (num_envs, encoder_latent_dim), or None if
            act() has not been called yet.
        """
        return self._last_z
