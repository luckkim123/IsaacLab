# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ActorCritic with extrinsics encoder for HORA/RMA training.

This module provides the encoder-based actor-critic network:
    - ActorCriticEncoder: Base encoder network (Phase 1 teacher training)

Architecture (symmetric critic):
    Encoder: privileged (26D) -> MLP [256, 128, 64] -> z (13D)
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

logger = logging.getLogger(__name__)
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.networks import MLP, EmpiricalNormalization
from torch.distributions import Normal

if TYPE_CHECKING:
    from tensordict import TensorDict


class ActorCriticEncoder(nn.Module):
    """ActorCritic with extrinsics encoder for HORA Phase 1 teacher policy.

    The encoder compresses privileged information into a bounded latent vector z.

    Symmetric critic design (HORA/RMA standard):
        Actor:  cat([policy_obs, z]) -- encoder must compress privileged into z
        Critic: cat([policy_obs, z]) -- also sees z, encoder gets gradient from both

    The encoder receives gradient from both actor loss and critic value loss.
    This ensures the encoder learns meaningful z representations, unlike the
    asymmetric design where the critic bypasses the encoder (causing encoder death).

    Activation modes:
        - "tanh": z = tanh(raw) in [-1, 1]. Built into MLP last layer. Default (matches HORA).
        - "sigmoid": z = z_min + sigmoid(raw) * (z_max - z_min). Bounded in [z_min, z_max].
        - "softplus": z = softplus(raw) + z_min. Legacy, can collapse to z_min.
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
        self.z_max = z_max
        self.state_dependent_std = state_dependent_std

        # Encoder input normalization (Welford's online mean/var)
        # Fixes privileged obs scale mismatch (volume ~0.01 vs body_mass ~10).
        # Only updates stats in training mode; Phase 2 (frozen encoder) is safe.
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

        # Initialize last encoder layer bias so activation starts in a
        # region with good gradient signal, regardless of random seed.
        if encoder_output_activation == "sigmoid":
            last_linear = self.encoder[-1]
            assert isinstance(last_linear, nn.Linear), (
                f"Expected last encoder layer to be Linear, got {type(last_linear)}"
            )
            # Bias init: logit((nominal - z_min) / (z_max - z_min)) so z starts
            # near mid-range (~0.5 * (z_max - z_min) + z_min). sigmoid(0) = 0.5.
            # Small weights + zero bias => sigmoid output starts near 0.5.
            nn.init.constant_(last_linear.bias, 0.0)
            nn.init.normal_(last_linear.weight, std=0.01)
        elif encoder_output_activation == "softplus":
            last_linear = self.encoder[-1]
            assert isinstance(last_linear, nn.Linear), (
                f"Expected last encoder layer to be Linear, got {type(last_linear)}"
            )
            nn.init.constant_(last_linear.bias, 0.5)

        logger.info("Encoder MLP: %s", self.encoder)

        # Actor input: policy_obs + z (encoder compressed)
        num_actor_obs = policy_obs_dim + encoder_latent_dim

        # Actor
        actor_output = [2, num_actions] if self.state_dependent_std else num_actions
        self.actor = MLP(num_actor_obs, actor_output, list(actor_hidden_dims), activation)
        logger.info("Actor MLP: %s", self.actor)

        # Actor observation normalization (applied to actor input: policy_obs + z)
        self.actor_obs_normalization = actor_obs_normalization
        self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs) if actor_obs_normalization else nn.Identity()

        # Critic input: policy_obs + z (symmetric, shares encoder with actor)
        num_critic_obs = policy_obs_dim + encoder_latent_dim
        self.critic = MLP(num_critic_obs, 1, list(critic_hidden_dims), activation)
        logger.info("Critic MLP (symmetric, %dD input): %s", num_critic_obs, self.critic)

        # Critic observation normalization (applied to critic input: policy_obs + z)
        self.critic_obs_normalization = critic_obs_normalization
        self.critic_obs_normalizer = (
            EmpiricalNormalization(num_critic_obs) if critic_obs_normalization else nn.Identity()
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

    def _activate_z(self, raw: torch.Tensor) -> torch.Tensor:
        """Apply output activation to raw encoder/adaptation output.

        This is the canonical activation for all encoder/adaptation z outputs.
        Centralizing here ensures consistency across Phase 1 encoder, Phase 2
        adaptation, and the adaptation runner.

        Modes:
            tanh: tanh(raw) in [-1, 1]. Default (matches HORA).
            sigmoid: z_min + sigmoid(raw) * (z_max - z_min). Bounded.
            softplus: softplus(raw) + z_min. Legacy, can collapse.
        """
        if self.encoder_output_activation == "tanh":
            return torch.tanh(raw)
        if self.encoder_output_activation == "sigmoid":
            return self.z_min + torch.sigmoid(raw) * (self.z_max - self.z_min)
        return F.softplus(raw) + self.z_min

    def _encode(self, privileged: torch.Tensor) -> torch.Tensor:
        """Encode privileged info into latent z.

        For tanh mode: z = tanh(raw) in [-1, 1] (built into MLP last layer). Default.
        For sigmoid mode: z = z_min + sigmoid(raw) * (z_max - z_min). Bounded.
        For softplus mode: z = softplus(raw) + z_min. Legacy, can collapse.

        Privileged obs is normalized before the encoder MLP when
        encoder_obs_normalization is enabled (fixes 1000x scale mismatch).
        """
        normalized = self.encoder_obs_normalizer(privileged)
        if self.encoder_output_activation == "tanh":
            return self.encoder(normalized)
        return self._activate_z(self.encoder(normalized))

    def _get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        """Actor observation: cat([policy_obs, z_from_encoder]).

        The actor sees privileged info only through the encoder's compressed
        latent z, forcing the encoder to learn useful representations.
        """
        policy_obs = obs[self._policy_obs_key]
        z = self._encode(obs[self._privileged_key])
        return torch.cat([policy_obs, z], dim=-1)

    def _get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        """Critic observation: cat([policy_obs, z_from_encoder]).

        Symmetric design: critic sees z (same as actor), so encoder receives
        gradient from both actor loss and critic loss. This is the standard
        HORA/RMA architecture.
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
        actor_obs = self.actor_obs_normalizer(self._get_actor_obs(obs))  # type: ignore[operator]
        self._update_distribution(actor_obs)
        assert self.distribution is not None
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        """Get deterministic action (mean) for inference."""
        actor_obs = self.actor_obs_normalizer(self._get_actor_obs(obs))  # type: ignore[operator]
        output = self.actor(actor_obs)
        return output[..., 0, :] if self.state_dependent_std else output

    def evaluate(self, obs: TensorDict, **_kwargs: Any) -> torch.Tensor:
        """Evaluate the value function for given observations (symmetric, uses encoder z)."""
        critic_obs = self.critic_obs_normalizer(self._get_critic_obs(obs))  # type: ignore[operator]
        return self.critic(critic_obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Compute log probability of actions under current distribution."""
        assert self.distribution is not None, "Call act() first to initialize distribution"
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update observation normalization statistics.

        Updates encoder, actor, and critic normalizers independently.
        Encoder normalizer uses raw privileged obs.
        Actor normalizer uses policy_obs + z (encoder output).
        Critic normalizer uses policy_obs + z (symmetric, same as actor).
        """
        if self.encoder_obs_normalization and hasattr(self.encoder_obs_normalizer, "update"):
            self.encoder_obs_normalizer.update(obs[self._privileged_key])  # type: ignore[union-attr]
        if self.actor_obs_normalization and hasattr(self.actor_obs_normalizer, "update"):
            self.actor_obs_normalizer.update(self._get_actor_obs(obs))  # type: ignore[union-attr]
        if self.critic_obs_normalization and hasattr(self.critic_obs_normalizer, "update"):
            self.critic_obs_normalizer.update(self._get_critic_obs(obs))  # type: ignore[union-attr]

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load model parameters with backward compatibility.

        Matches the RSL-RL ActorCritic API contract by returning True to indicate
        resumed training. OnPolicyRunner.load() uses this return value to decide
        whether to restore optimizer state and iteration counter.

        Backward compat:
            - Injects default encoder_obs_normalizer state for pre-normalization checkpoints.
            - Rebuilds critic MLP when input dim differs (e.g., symmetric vs asymmetric critic).
              Critic is unused during inference (act_inference), so this is safe for eval.
        """
        if self.encoder_obs_normalization:
            prefix = "encoder_obs_normalizer."
            if not any(k.startswith(prefix) for k in state_dict):
                logger.info("Old checkpoint: injecting default encoder_obs_normalizer state.")
                for k, v in self.encoder_obs_normalizer.state_dict().items():
                    state_dict[prefix + k] = v

        # Handle critic dimension mismatch (e.g., old privileged-only critic vs new asymmetric)
        ckpt_critic_weight = state_dict.get("critic.0.weight")
        if ckpt_critic_weight is not None:
            ckpt_critic_in = ckpt_critic_weight.shape[1]
            current_critic_in = self.critic[0].weight.shape[1]
            if ckpt_critic_in != current_critic_in:
                logger.warning(
                    "Critic input dim mismatch: checkpoint=%d, current=%d. "
                    "Rebuilding critic to match checkpoint (safe for inference).",
                    ckpt_critic_in,
                    current_critic_in,
                )
                # Infer hidden dims from checkpoint weight shapes
                critic_weight_keys = sorted(
                    k for k in state_dict if k.startswith("critic.") and k.endswith(".weight")
                )
                hidden_dims = [state_dict[k].shape[0] for k in critic_weight_keys[:-1]]
                self.critic = MLP(ckpt_critic_in, 1, hidden_dims, "elu")

                # Rebuild critic_obs_normalizer if enabled
                if self.critic_obs_normalization:
                    self.critic_obs_normalizer = EmpiricalNormalization(ckpt_critic_in)

        super().load_state_dict(state_dict, strict=strict)
        return True
