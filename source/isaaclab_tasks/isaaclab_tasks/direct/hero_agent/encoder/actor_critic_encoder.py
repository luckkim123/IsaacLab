# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ActorCritic with extrinsics encoder for HORA/RMA training.

This module provides the encoder-based actor-critic networks:
    - ActorCriticEncoder: Base encoder network (Phase 1 teacher training)
    - ActorCriticEncoderTDC: Encoder with z exposure for TDC M_hat extraction

Architecture:
    Encoder: privileged (18D) -> MLP [256, 128, 64] -> z (6D)
    Actor:   cat([policy_obs, z]) = 19D -> MLP [256, 128, 64] -> actions
    Critic:  cat([policy_obs, z]) = 19D -> MLP [256, 128, 64] -> value (1D)

Note: Critic does NOT receive privileged info directly (symmetric with actor).
This forces the encoder to compress useful information into z.

Design choices:
    - 6D latent: general compressed representation of extrinsic parameters.
    - 18D privileged: Buoyancy/geometry parameters (volume, CoG, CoB) per body
      + payload (mass, cog_offset). Inertia and added mass excluded.

Reference:
    - HORA: Heuristic-Free Online Robust Adaptation (Qi et al., 2023)
    - RMA: Rapid Motor Adaptation (Kumar et al., 2021)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, NoReturn

import torch

logger = logging.getLogger(__name__)
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.networks import MLP, EmpiricalNormalization
from torch.distributions import Normal

if TYPE_CHECKING:
    from tensordict import TensorDict


class ActorCriticEncoder(nn.Module):
    """ActorCritic with extrinsics encoder for HORA Phase 1 teacher policy.

    The encoder compresses privileged information into a positive latent vector z
    using softplus: z = softplus(raw_output) + z_min.
    This guarantees z > z_min (positive). Unlike scaled sigmoid, softplus has
    no gradient saturation at large values.

    Gradient flow: During PPO update, stored observations are replayed through
    the full network (encoder + actor/critic), so encoder gradients flow via
    both actor and critic loss backpropagation.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        # Encoder parameters
        policy_obs_dim: int = 13,
        privileged_dim: int = 18,
        encoder_hidden_dims: list[int] | tuple[int, ...] = (256, 128, 64),
        encoder_latent_dim: int = 6,
        encoder_activation: str = "relu",
        encoder_output_activation: str = "softplus",
        z_min: float = 0.01,
        # Actor-Critic parameters
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: list[int] | tuple[int, ...] = (256, 128, 64),
        critic_hidden_dims: list[int] | tuple[int, ...] = (256, 128, 64),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            logger.warning(
                "ActorCriticEncoder.__init__ got unexpected arguments, which will be ignored: %s",
                list(kwargs.keys()),
            )
        super().__init__()

        # Store dimension info
        self.obs_groups = obs_groups
        self.policy_obs_dim = policy_obs_dim
        self.privileged_dim = privileged_dim
        self.encoder_latent_dim = encoder_latent_dim
        self.encoder_output_activation = encoder_output_activation
        self.z_min = z_min
        self.state_dependent_std = state_dependent_std

        # Extract obs key names from obs_groups for direct TensorDict access
        policy_groups = obs_groups["policy"]
        if len(policy_groups) != 2:
            raise ValueError(
                f"ActorCriticEncoder requires exactly 2 obs groups in 'policy', "
                f"got {len(policy_groups)}: {policy_groups}"
            )
        self._policy_obs_key = policy_groups[0]
        self._privileged_key = policy_groups[1]

        # Verify obs dimensions match expected
        policy_obs_shape = obs[self._policy_obs_key].shape
        privileged_shape = obs[self._privileged_key].shape

        if len(policy_obs_shape) != 2 or len(privileged_shape) != 2:
            raise ValueError("ActorCriticEncoder only supports 1D observations (batch, dim).")
        if policy_obs_shape[-1] != policy_obs_dim:
            raise ValueError(f"Policy obs dim {policy_obs_shape[-1]} != expected {policy_obs_dim}")
        if privileged_shape[-1] != privileged_dim:
            raise ValueError(f"Privileged dim {privileged_shape[-1]} != expected {privileged_dim}")

        # Encoder: privileged -> z
        if encoder_output_activation == "tanh":
            self.encoder = MLP(
                privileged_dim,
                encoder_latent_dim,
                list(encoder_hidden_dims),
                encoder_activation,
                last_activation="tanh",
            )
        else:
            self.encoder = MLP(privileged_dim, encoder_latent_dim, list(encoder_hidden_dims), encoder_activation)
        logger.info("Encoder MLP: %s", self.encoder)

        # Actor/Critic input: policy_obs + z (symmetric design)
        # Note: Privileged info is NOT passed to critic to force encoder learning.
        # If critic receives privileged directly, it ignores z and encoder collapses.
        num_combined_obs = policy_obs_dim + encoder_latent_dim

        # Actor
        actor_output = [2, num_actions] if self.state_dependent_std else num_actions
        self.actor = MLP(num_combined_obs, actor_output, list(actor_hidden_dims), activation)
        logger.info("Actor MLP: %s", self.actor)

        # Actor observation normalization (applied to actor input: policy_obs + z)
        self.actor_obs_normalization = actor_obs_normalization
        self.actor_obs_normalizer = (
            EmpiricalNormalization(num_combined_obs) if actor_obs_normalization else nn.Identity()
        )

        # Critic
        self.critic = MLP(num_combined_obs, 1, list(critic_hidden_dims), activation)
        logger.info("Critic MLP: %s", self.critic)

        # Critic observation normalization (applied to critic input: policy_obs + z)
        self.critic_obs_normalization = critic_obs_normalization
        self.critic_obs_normalizer = (
            EmpiricalNormalization(num_combined_obs) if critic_obs_normalization else nn.Identity()
        )

        # Action noise
        self.noise_std_type = noise_std_type
        self._validate_noise_std_type(noise_std_type)
        self._init_noise_params(num_actions, init_noise_std)

        # Action distribution (populated in _update_distribution)
        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    def _validate_noise_std_type(self, noise_std_type: str) -> None:
        """Validate the noise standard deviation type."""
        if noise_std_type not in ("scalar", "log"):
            raise ValueError(f"Unknown noise_std_type: {noise_std_type}. Must be 'scalar' or 'log'.")

    def _init_noise_params(self, num_actions: int, init_noise_std: float) -> None:
        """Initialize noise parameters based on state_dependent_std and noise_std_type."""
        if self.state_dependent_std:
            nn.init.zeros_(self.actor[-2].weight[num_actions:])
            if self.noise_std_type == "scalar":
                init_value = init_noise_std
            else:
                init_value = float(torch.log(torch.tensor(init_noise_std + 1e-7)).item())
            nn.init.constant_(self.actor[-2].bias[num_actions:], init_value)
        elif self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        else:  # log
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))

    def reset(self, _dones: torch.Tensor | None = None) -> None:
        """Reset hidden states. No-op for non-recurrent networks."""
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError("Use act(), act_inference(), or evaluate() instead.")

    @property
    def action_mean(self) -> torch.Tensor:
        assert self.distribution is not None, "Call act() first to initialize distribution"
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        assert self.distribution is not None, "Call act() first to initialize distribution"
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        assert self.distribution is not None, "Call act() first to initialize distribution"
        return self.distribution.entropy().sum(dim=-1)

    # --- Observation processing ---

    def _softplus_z(self, raw: torch.Tensor) -> torch.Tensor:
        """Apply softplus + z_min to guarantee z > z_min (positive latent).

        This is the canonical activation for all encoder/adaptation z outputs.
        Centralizing here ensures consistency across Phase 1 encoder, Phase 2
        adaptation, and the adaptation runner.
        """
        return F.softplus(raw) + self.z_min

    def _encode(self, privileged: torch.Tensor) -> torch.Tensor:
        """Encode privileged info into latent z.

        For softplus mode: z = softplus(encoder_output) + z_min (positive, no saturation).
        For tanh mode: z = tanh(encoder_output) in [-1, 1] (built into MLP last layer).
        """
        if self.encoder_output_activation == "tanh":
            return self.encoder(privileged)
        return self._softplus_z(self.encoder(privileged))

    def _get_combined_obs(self, obs: TensorDict) -> torch.Tensor:
        """Get combined observation: cat([policy_obs, z]).

        Both actor and critic use the same combined observation (symmetric design).
        Critic does NOT receive privileged info directly to force encoder learning.
        """
        policy_obs = obs[self._policy_obs_key]
        z = self._encode(obs[self._privileged_key])
        return torch.cat([policy_obs, z], dim=-1)

    # --- Action distribution ---

    def _compute_std(self, std_or_mean: torch.Tensor, log_std: torch.Tensor | None = None) -> torch.Tensor:
        """Compute action standard deviation based on noise_std_type.

        Args:
            std_or_mean: When state_dependent_std=True, this is the raw std output.
                         When False, this is the action mean (used only for shape).
            log_std: Log standard deviation (for state_dependent_std + log noise type).
        """
        if self.state_dependent_std:
            if self.noise_std_type == "scalar":
                return std_or_mean
            assert log_std is not None, "log_std required for state_dependent_std with log noise_std_type"
            return torch.exp(log_std)
        elif self.noise_std_type == "scalar":
            return self.std.expand_as(std_or_mean)
        else:
            return torch.exp(self.log_std).expand_as(std_or_mean)

    def _update_distribution(self, actor_obs: torch.Tensor) -> None:
        """Update the action distribution given actor observations."""
        if self.state_dependent_std:
            mean_and_std = self.actor(actor_obs)
            mean, std_or_log_std = torch.unbind(mean_and_std, dim=-2)
            std = self._compute_std(std_or_log_std, std_or_log_std)
        else:
            mean = self.actor(actor_obs)
            std = self._compute_std(mean)
        self.distribution = Normal(mean, std)

    # --- Core API ---

    def act(self, obs: TensorDict, **_kwargs: Any) -> torch.Tensor:
        """Sample an action from the policy distribution."""
        actor_obs = self.actor_obs_normalizer(self._get_combined_obs(obs))  # type: ignore[operator]
        self._update_distribution(actor_obs)
        assert self.distribution is not None
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        """Get deterministic action (mean) for inference."""
        actor_obs = self.actor_obs_normalizer(self._get_combined_obs(obs))  # type: ignore[operator]
        output = self.actor(actor_obs)
        return output[..., 0, :] if self.state_dependent_std else output

    def evaluate(self, obs: TensorDict, **_kwargs: Any) -> torch.Tensor:
        """Evaluate the value function for given observations."""
        critic_obs = self.critic_obs_normalizer(self._get_combined_obs(obs))  # type: ignore[operator]
        return self.critic(critic_obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Compute log probability of actions under current distribution."""
        assert self.distribution is not None, "Call act() first to initialize distribution"
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update observation normalization statistics."""
        if self.actor_obs_normalization or self.critic_obs_normalization:
            combined_obs = self._get_combined_obs(obs)
            if self.actor_obs_normalization and hasattr(self.actor_obs_normalizer, "update"):
                self.actor_obs_normalizer.update(combined_obs)  # type: ignore[union-attr]
            if self.critic_obs_normalization and hasattr(self.critic_obs_normalizer, "update"):
                self.critic_obs_normalizer.update(combined_obs)  # type: ignore[union-attr]

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load model parameters.

        Matches the RSL-RL ActorCritic API contract by returning True to indicate
        resumed training. OnPolicyRunner.load() uses this return value to decide
        whether to restore optimizer state and iteration counter.
        """
        super().load_state_dict(state_dict, strict=strict)
        return True


class ActorCriticEncoderTDC(ActorCriticEncoder):
    """ActorCriticEncoder with z exposure for TDC M_hat extraction.

    Overrides _get_combined_obs() to cache the encoder latent z after each
    forward pass. The environment retrieves z via get_last_z() to compute
    TDC M_hat from decomposed z[3:6] + FK joint positions.

    Timing:
        RSL-RL loop: obs = env.get_observations() -> action = policy.act(obs)
                     -> env.step(action) [calls _pre_physics_step]
        So _last_z is computed in act() and available when _pre_physics_step() runs.
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
