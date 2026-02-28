# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ActorCritic with extrinsics encoder for HORA/RMA training.

This module provides the encoder-based actor-critic network:
    - ActorCriticEncoder: Base encoder network (Phase 1 teacher training)

Architecture (symmetric critic):
    Encoder: privileged (28D) -> MLP [256, 128, 64] -> z (13D)
    Actor:   cat([policy_obs, z]) = 26D -> MLP [256, 128, 64] -> actions
    Critic:  cat([policy_obs, z]) = 26D -> MLP [256, 128, 64] -> value (1D)

Both actor and critic see z from the encoder (symmetric design, HORA/RMA standard).
The encoder receives gradient from both actor loss and critic loss, ensuring
the encoder learns meaningful representations even early in training.

Reference:
    - HORA: Heuristic-Free Online Robust Adaptation (Qi et al., 2023)
    - RMA: Rapid Motor Adaptation (Kumar et al., 2021)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, NoReturn

import torch
import torch.nn as nn
from rsl_rl.networks import MLP, EmpiricalNormalization
from torch.distributions import Normal

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from tensordict import TensorDict


class ActorCriticEncoder(nn.Module):
    """ActorCritic with extrinsics encoder for HORA Phase 1 teacher policy.

    The encoder compresses privileged information into a bounded latent vector z.

    Symmetric critic design (HORA/RMA standard):
        Actor:  cat([policy_obs, z]) -- encoder must compress privileged into z
        Critic: cat([policy_obs, z]) -- also sees z, encoder gets gradient from both

    The encoder receives gradient from both actor loss and critic value loss.

    Activation modes:
        - "tanh": z = tanh(raw) in [-1, 1]. Built into MLP last layer (matches HORA original).
        - "sigmoid": z = z_min + sigmoid(raw) * (z_max - z_min). Bounded in [z_min, z_max]. Default.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        # Encoder parameters
        policy_obs_dim: int = 13,
        privileged_dim: int = 32,
        encoder_hidden_dims: list[int] | tuple[int, ...] = (256, 128, 64),
        encoder_latent_dim: int = 13,
        encoder_activation: str = "relu",
        encoder_output_activation: str = "sigmoid",
        encoder_obs_normalization: bool = False,
        z_min: float = 0.01,
        z_max: float = 2.0,
        # Actor-Critic parameters
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: list[int] | tuple[int, ...] = (256, 128, 64),
        critic_hidden_dims: list[int] | tuple[int, ...] = (256, 128, 64),
        activation: str = "elu",
        init_noise_std: float = 1.0,
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
        self.z_max = z_max

        # Encoder input normalization (Welford's online mean/var)
        self.encoder_obs_normalization = encoder_obs_normalization
        self.encoder_obs_normalizer = (
            EmpiricalNormalization(privileged_dim) if encoder_obs_normalization else nn.Identity()
        )

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

        # Initialize last encoder layer for sigmoid activation.
        # bias=0 -> sigmoid(0)=0.5 -> z starts at midpoint of [z_min, z_max].
        # Small weight std ensures initial z is tightly clustered near midpoint,
        # allowing the encoder to gradually learn the appropriate z mapping.
        if encoder_output_activation == "sigmoid":
            last_linear = self.encoder[-1]
            assert isinstance(last_linear, nn.Linear), (
                f"Expected last encoder layer to be Linear, got {type(last_linear)}"
            )
            nn.init.constant_(last_linear.bias, 0.0)
            nn.init.normal_(last_linear.weight, std=0.01)

        logger.info("Encoder MLP: %s", self.encoder)

        # Actor input: policy_obs + z (encoder compressed)
        num_actor_obs = policy_obs_dim + encoder_latent_dim

        # Actor
        self.actor = MLP(num_actor_obs, num_actions, list(actor_hidden_dims), activation)
        logger.info("Actor MLP: %s", self.actor)

        # Actor observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs) if actor_obs_normalization else nn.Identity()

        # Critic input: policy_obs + z (symmetric, shares encoder with actor)
        num_critic_obs = policy_obs_dim + encoder_latent_dim
        self.critic = MLP(num_critic_obs, 1, list(critic_hidden_dims), activation)
        logger.info("Critic MLP (symmetric, %dD input): %s", num_critic_obs, self.critic)

        # Critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        self.critic_obs_normalizer = (
            EmpiricalNormalization(num_critic_obs) if critic_obs_normalization else nn.Identity()
        )

        # Action noise (log parameterization: std = exp(log_std))
        self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))

        # Action distribution (populated in _update_distribution)
        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

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

    def _activate_z(self, raw: torch.Tensor) -> torch.Tensor:
        """Apply output activation to raw encoder/adaptation output.

        This is the canonical activation for all encoder/adaptation z outputs.
        Centralizing here ensures consistency across Phase 1 encoder, Phase 2
        adaptation, and the adaptation runner.

        Modes:
            tanh: tanh(raw) in [-1, 1]. Default (matches HORA).
            sigmoid: z_min + sigmoid(raw) * (z_max - z_min). Bounded.
        """
        if self.encoder_output_activation == "tanh":
            return torch.tanh(raw)
        return self.z_min + torch.sigmoid(raw) * (self.z_max - self.z_min)

    def _encode(self, privileged: torch.Tensor) -> torch.Tensor:
        """Encode privileged info into latent z.

        For tanh mode: z = tanh(raw) in [-1, 1] (built into MLP last layer).
        For sigmoid mode: z = z_min + sigmoid(raw) * (z_max - z_min). Bounded.

        Privileged obs is normalized before the encoder MLP when
        encoder_obs_normalization is enabled.
        """
        normalized = self.encoder_obs_normalizer(privileged)
        if self.encoder_output_activation == "tanh":
            return self.encoder(normalized)
        return self._activate_z(self.encoder(normalized))

    def _get_combined_obs(self, obs: TensorDict) -> torch.Tensor:
        """Combined observation: cat([policy_obs, z_from_encoder]).

        Symmetric design: both actor and critic see the same encoder z.
        """
        policy_obs = obs[self._policy_obs_key]
        z = self._encode(obs[self._privileged_key])
        return torch.cat([policy_obs, z], dim=-1)

    # --- Action distribution ---

    def _update_distribution(self, actor_obs: torch.Tensor) -> None:
        """Update the action distribution given actor observations."""
        mean = self.actor(actor_obs)
        std = torch.exp(self.log_std).expand_as(mean)
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
        return self.actor(actor_obs)

    @torch.no_grad()
    def act_with_z_hat(self, obs: TensorDict, z_hat: torch.Tensor) -> torch.Tensor:
        """Get deterministic action using a pre-computed z_hat (detached).

        Avoids duplicating actor internals in the AdaptRunner by routing
        through the same normalizer and actor MLP used by act_inference().

        Args:
            obs: Observation dict containing policy obs.
            z_hat: Pre-computed latent estimate (will be detached internally).

        Returns:
            Clamped deterministic actions. Shape: (N, num_actions).
        """
        policy_obs = obs[self._policy_obs_key]
        combined_obs = torch.cat([policy_obs, z_hat.detach()], dim=-1)
        actor_obs = self.actor_obs_normalizer(combined_obs)  # type: ignore[operator]
        return self.actor(actor_obs).clamp(-1.0, 1.0)

    def evaluate(self, obs: TensorDict, **_kwargs: Any) -> torch.Tensor:
        """Evaluate the value function for given observations (symmetric, uses encoder z)."""
        critic_obs = self.critic_obs_normalizer(self._get_combined_obs(obs))  # type: ignore[operator]
        return self.critic(critic_obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Compute log probability of actions under current distribution."""
        assert self.distribution is not None, "Call act() first to initialize distribution"
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update observation normalization statistics."""
        if self.encoder_obs_normalization and hasattr(self.encoder_obs_normalizer, "update"):
            self.encoder_obs_normalizer.update(obs[self._privileged_key])  # type: ignore[union-attr]
        combined = self._get_combined_obs(obs)
        if self.actor_obs_normalization and hasattr(self.actor_obs_normalizer, "update"):
            self.actor_obs_normalizer.update(combined)  # type: ignore[union-attr]
        if self.critic_obs_normalization and hasattr(self.critic_obs_normalizer, "update"):
            self.critic_obs_normalizer.update(combined)  # type: ignore[union-attr]

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load model parameters with backward compatibility.

        Returns True to indicate resumed training (RSL-RL API contract).
        """
        if self.encoder_obs_normalization:
            prefix = "encoder_obs_normalizer."
            if not any(k.startswith(prefix) for k in state_dict):
                logger.info("Old checkpoint: injecting default encoder_obs_normalizer state.")
                for k, v in self.encoder_obs_normalizer.state_dict().items():
                    state_dict[prefix + k] = v

        # Filter out unknown keys from old checkpoints (state_dependent_std, noise_std_type, etc.)
        current_keys = set(self.state_dict().keys())
        filtered = {k: v for k, v in state_dict.items() if k in current_keys}
        if len(filtered) < len(state_dict):
            dropped = set(state_dict.keys()) - current_keys
            logger.info("Dropped %d unknown checkpoint keys: %s", len(dropped), dropped)

        # Warn if essential keys are missing (model would silently use random weights)
        missing = current_keys - set(filtered.keys())
        essential_prefixes = ("encoder.", "actor.", "critic.", "log_std")
        missing_essential = {k for k in missing if any(k.startswith(p) for p in essential_prefixes)}
        if missing_essential:
            logger.warning("Missing %d essential keys in checkpoint: %s", len(missing_essential), missing_essential)

        super().load_state_dict(filtered, strict=False)
        return True
