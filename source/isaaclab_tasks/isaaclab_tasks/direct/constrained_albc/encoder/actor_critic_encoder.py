# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ActorCritic with extrinsics encoder for HORA/RMA training.

This module provides the encoder-based actor-critic network:
    - ActorCriticEncoder: Base encoder network (Phase 1 teacher training)

Architecture:
    Encoder: cat([policy_obs, hist_flat, privileged]) = 280D -> MLP -> softsign -> z (13D)
    Actor:   cat([policy_obs, hist_flat, z]) = 266D -> MLP -> actions
    Critic:  cat([policy_obs, hist_flat, privileged]) = 280D -> MLP -> value (1D)

    The encoder receives policy_obs + proprioception history + privileged info,
    producing a time-varying z that encodes both dynamic state and DR parameters.
    This matches NORBC/ANYmal/RMA where encoder input includes dynamic privileged
    quantities (body velocity, contact forces, terrain), ensuring z changes every
    timestep and the policy naturally develops z-dependency.

    proprio_hist (N, 30, 8) is flattened to (N, 240) and concatenated directly.
    No embedding module -- the encoder/actor/critic MLPs learn from raw history.

Reference:
    - HORA: Heuristic-Free Online Robust Adaptation (Qi et al., 2023)
    - RMA: Rapid Motor Adaptation (Kumar et al., 2021)
    - NORBC: Neural Online Robust Boundary Controller (Kim et al., 2024)
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

    The encoder compresses dynamic state + privileged info into bounded latent z.
    Encoder input includes policy_obs, proprioception history, and privileged info,
    producing a time-varying z (changes every timestep). This matches NORBC/ANYmal
    where encoder input contains dynamic quantities, ensuring strong policy-encoder
    coupling via natural z-dependency.

    Architecture:
        Encoder: cat([policy_obs, hist_flat, privileged]) = 280D -> MLP -> softsign -> z (13D)
        Actor:   cat([policy_obs, hist_flat, z]) = 266D -> MLP -> actions
        Critic:  cat([policy_obs, hist_flat, privileged]) = 280D -> MLP -> value (asymmetric)

    Encoder gradient flows only from actor loss (critic doesn't use z).
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        # Encoder parameters
        policy_obs_dim: int = 13,
        privileged_dim: int = 19,
        encoder_hidden_dims: list[int] | tuple[int, ...] = (256, 128, 64),
        encoder_latent_dim: int = 13,
        encoder_activation: str = "elu",
        encoder_obs_normalization: bool = False,
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
        # --- Parse obs_groups: require exactly 3 keys [policy_obs, privileged, proprio_hist] ---
        policy_groups = obs_groups["policy"]
        if len(policy_groups) != 3:
            raise ValueError(
                f"ActorCriticEncoder requires exactly 3 obs groups in 'policy' "
                f"[policy_obs, privileged, proprio_hist], got {len(policy_groups)}: {policy_groups}"
            )
        self._policy_obs_key = policy_groups[0]
        self._privileged_key = policy_groups[1]
        self._proprio_hist_key = policy_groups[2]

        # --- Verify obs dimensions ---
        policy_obs_shape = obs[self._policy_obs_key].shape
        privileged_shape = obs[self._privileged_key].shape
        proprio_hist_shape = obs[self._proprio_hist_key].shape

        if len(policy_obs_shape) != 2 or len(privileged_shape) != 2:
            raise ValueError("ActorCriticEncoder only supports 1D observations (batch, dim).")
        if policy_obs_shape[-1] != policy_obs_dim:
            raise ValueError(f"Policy obs dim {policy_obs_shape[-1]} != expected {policy_obs_dim}")
        if privileged_shape[-1] != privileged_dim:
            raise ValueError(f"Privileged dim {privileged_shape[-1]} != expected {privileged_dim}")
        if len(proprio_hist_shape) != 3:
            raise ValueError(f"proprio_hist must be 3D (batch, history, features), got shape {proprio_hist_shape}")

        self._hist_flat_dim = proprio_hist_shape[1] * proprio_hist_shape[2]
        logger.info(
            "History: (%d, %d) -> flatten -> %dD (raw concat, no embedding)",
            proprio_hist_shape[1],
            proprio_hist_shape[2],
            self._hist_flat_dim,
        )

        # --- Encoder MLP: cat([policy_obs, hist_flat, privileged]) -> softsign -> z ---
        # Dynamic input (policy_obs + history) ensures z varies every timestep,
        # matching NORBC/ANYmal encoder design where input includes dynamic state.
        encoder_input_dim = policy_obs_dim + self._hist_flat_dim + privileged_dim
        self.encoder_input_dim = encoder_input_dim

        self.encoder_obs_normalization = encoder_obs_normalization
        self.encoder_obs_normalizer = (
            EmpiricalNormalization(encoder_input_dim) if encoder_obs_normalization else nn.Identity()
        )

        self.encoder = MLP(
            encoder_input_dim,
            encoder_latent_dim,
            list(encoder_hidden_dims),
            encoder_activation,
            last_activation=None,
        )
        logger.info("Encoder MLP (%dD input): %s", encoder_input_dim, self.encoder)

        # --- Actor MLP: [policy_obs, hist_flat, z] -> actions ---
        num_actor_obs = policy_obs_dim + self._hist_flat_dim + encoder_latent_dim

        self.actor = MLP(num_actor_obs, num_actions, list(actor_hidden_dims), activation)
        logger.info("Actor MLP (%dD input): %s", num_actor_obs, self.actor)

        self.actor_obs_normalization = actor_obs_normalization
        self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs) if actor_obs_normalization else nn.Identity()

        # --- Critic MLP: [policy_obs, hist_flat, privileged] -> value (asymmetric) ---
        num_critic_obs = policy_obs_dim + self._hist_flat_dim + privileged_dim
        self.num_critic_obs = num_critic_obs
        self.critic = MLP(num_critic_obs, 1, list(critic_hidden_dims), activation)
        logger.info("Critic MLP (history-asymmetric, %dD input): %s", num_critic_obs, self.critic)

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

    def _get_hist_flat(self, obs: TensorDict) -> torch.Tensor:
        """Get flattened proprioception history from observations.

        Returns:
            Flattened history. Shape: (N, H*D) e.g. (N, 240).
        """
        return obs[self._proprio_hist_key].flatten(start_dim=1)

    def _encode_from_parts(
        self, policy_obs: torch.Tensor, hist_flat: torch.Tensor, privileged: torch.Tensor
    ) -> torch.Tensor:
        """Core encoding: cat([policy_obs, hist_flat, privileged]) -> softsign -> z."""
        encoder_input = torch.cat([policy_obs, hist_flat, privileged], dim=-1)
        x = self.encoder(self.encoder_obs_normalizer(encoder_input))
        return torch.nn.functional.softsign(x)

    def _encode(self, obs: TensorDict) -> torch.Tensor:
        """Encode dynamic state + privileged info into latent z.

        encoder(normalize(cat([policy_obs, hist_flat, privileged]))) -> softsign -> z in (-1, 1)

        Dynamic input (policy_obs, hist_flat) ensures z varies every timestep,
        matching NORBC encoder design. Static privileged info (DR params) provides
        ground-truth environment parameters for the information bottleneck.
        """
        return self._encode_from_parts(
            obs[self._policy_obs_key], self._get_hist_flat(obs), obs[self._privileged_key]
        )

    def _get_combined_obs(self, obs: TensorDict) -> torch.Tensor:
        """Combined observation for actor: cat([policy_obs, hist_flat, z])."""
        policy_obs = obs[self._policy_obs_key]
        hist_flat = self._get_hist_flat(obs)
        z = self._encode_from_parts(policy_obs, hist_flat, obs[self._privileged_key])
        return torch.cat([policy_obs, hist_flat, z], dim=-1)

    def _get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        """Critic observation: cat([policy_obs, hist_flat, privileged])."""
        return torch.cat([obs[self._policy_obs_key], self._get_hist_flat(obs), obs[self._privileged_key]], dim=-1)

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

    def evaluate(self, obs: TensorDict, **_kwargs: Any) -> torch.Tensor:
        """Evaluate the value function for given observations."""
        critic_obs = self.critic_obs_normalizer(self._get_critic_obs(obs))  # type: ignore[operator]
        return self.critic(critic_obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Compute log probability of actions under current distribution."""
        assert self.distribution is not None, "Call act() first to initialize distribution"
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update observation normalization statistics."""
        if self.encoder_obs_normalization:
            policy_obs = obs[self._policy_obs_key]
            hist_flat = self._get_hist_flat(obs)
            encoder_input = torch.cat([policy_obs, hist_flat, obs[self._privileged_key]], dim=-1)
            self.encoder_obs_normalizer.update(encoder_input)  # type: ignore[union-attr]
        if self.actor_obs_normalization:
            with torch.no_grad():
                combined = self._get_combined_obs(obs)
            self.actor_obs_normalizer.update(combined)  # type: ignore[union-attr]
        if self.critic_obs_normalization:
            critic_input = self._get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_input)  # type: ignore[union-attr]

    def _handle_dim_mismatch(self, state_dict: dict, prefix: str) -> None:
        """Reinitialize module if checkpoint input dimension doesn't match current model."""
        weight_keys = sorted(
            [k for k in state_dict if k.startswith(prefix) and k.endswith(".weight")],
            key=lambda k: int(k.removeprefix(prefix).split(".")[0]),
        )
        if not weight_keys:
            return
        first_key = weight_keys[0]
        ckpt_input_dim = state_dict[first_key].shape[1]

        if prefix == "encoder.":
            current_module = self.encoder
            expected_dim = self.encoder_input_dim
        elif prefix == "critic.":
            current_module = self.critic
            expected_dim = self.num_critic_obs
        elif prefix == "cost_critic." and hasattr(self, "cost_critic"):
            current_module = self.cost_critic
            expected_dim = self.num_critic_obs
        else:
            return

        if ckpt_input_dim != expected_dim:
            logger.warning(
                "%s input dim mismatch (checkpoint %dD vs model %dD), reinitializing.",
                prefix.rstrip("."),
                ckpt_input_dim,
                expected_dim,
            )
            for k, v in current_module.state_dict().items():
                state_dict[prefix + k] = v

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load model parameters with backward compatibility.

        Handles missing keys, dimension mismatches, and old checkpoints.
        Returns True to indicate resumed training (RSL-RL API contract).
        """
        if self.encoder_obs_normalization:
            prefix = "encoder_obs_normalizer."
            has_keys = any(k.startswith(prefix) for k in state_dict)
            if not has_keys:
                logger.info("Old checkpoint: injecting default encoder_obs_normalizer state.")
                for k, v in self.encoder_obs_normalizer.state_dict().items():
                    state_dict[prefix + k] = v
            else:
                # Check for dimension mismatch (e.g. old 27D -> new 280D)
                mean_key = prefix + "mean"
                if mean_key in state_dict and state_dict[mean_key].shape[-1] != self.encoder_input_dim:
                    logger.warning(
                        "encoder_obs_normalizer dim mismatch (%dD vs %dD), reinitializing.",
                        state_dict[mean_key].shape[-1],
                        self.encoder_input_dim,
                    )
                    for k, v in self.encoder_obs_normalizer.state_dict().items():
                        state_dict[prefix + k] = v

        # Detect input dimension mismatches (encoder: input dim change, critic: symmetric <-> asymmetric)
        self._handle_dim_mismatch(state_dict, "encoder.")
        self._handle_dim_mismatch(state_dict, "critic.")

        # Filter out unknown keys from old checkpoints
        current_keys = set(self.state_dict().keys())
        filtered = {k: v for k, v in state_dict.items() if k in current_keys}
        if len(filtered) < len(state_dict):
            dropped = set(state_dict.keys()) - current_keys
            logger.info("Dropped %d unknown checkpoint keys: %s", len(dropped), dropped)

        # Warn if essential keys are missing
        missing = current_keys - set(filtered.keys())
        essential_prefixes = ("encoder.", "actor.", "critic.", "log_std")
        missing_essential = {k for k in missing if any(k.startswith(p) for p in essential_prefixes)}
        if missing_essential:
            logger.warning("Missing %d essential keys in checkpoint: %s", len(missing_essential), missing_essential)

        super().load_state_dict(filtered, strict=False)
        return True
