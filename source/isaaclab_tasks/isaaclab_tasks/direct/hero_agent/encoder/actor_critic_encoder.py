# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ActorCritic with extrinsics encoder for HORA/RMA training.

This module provides the encoder-based actor-critic network:
    - ActorCriticEncoder: Base encoder network (Phase 1 teacher training)

Architecture (with history, default for encoder training):
    Encoder: cat([policy_obs, hist_flat, privileged]) = 272D -> MLP -> tanh -> z (13D)
    Actor:   cat([policy_obs, hist_flat, z]) = 266D -> MLP -> actions
    Critic:  cat([policy_obs, hist_flat, privileged]) = 272D -> MLP -> value (1D)

    proprio_hist (N, 30, 8) is flattened to (N, 240) and concatenated directly.
    No embedding module -- the encoder/actor/critic MLPs learn from raw history.

Architecture (without history, legacy / backward compatible):
    Encoder: privileged (19D) -> MLP -> tanh -> z (13D)
    Actor:   cat([policy_obs, z]) = 26D -> MLP -> actions
    Critic:  cat([policy_obs, z]) or cat([policy_obs, privileged_raw]) -> MLP -> value

History mode is activated when obs_groups has 3 keys (proprio_hist present).

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

    The encoder compresses privileged information into a bounded latent vector z.
    When history is enabled (obs_groups has 3 keys), flattened proprioception
    history is concatenated to encoder, actor, and critic inputs.

    History mode (obs_groups has 3 keys):
        - Encoder: cat([policy_obs, hist_flat, privileged]) -> z
        - Actor: cat([policy_obs, hist_flat, z]) -> actions (no privileged)
        - Critic: cat([policy_obs, hist_flat, privileged]) -> value (inherently asymmetric)
        Encoder gradient comes only from actor loss (critic doesn't use z).

    Legacy mode (obs_groups has 2 keys):
        - Symmetric: Actor and Critic both see cat([policy_obs, z]).
        - Asymmetric (NORBC): Actor sees z, Critic sees raw privileged.
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
        encoder_output_activation: str = "tanh",
        encoder_obs_normalization: bool = False,
        z_min: float = 0.01,
        z_max: float = 2.0,
        z_bounds_coef: float = 0.1,
        z_bounds_soft_bound: float = 0.9,
        # Actor-Critic parameters
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: list[int] | tuple[int, ...] = (256, 128, 64),
        critic_hidden_dims: list[int] | tuple[int, ...] = (256, 128, 64),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        # Asymmetric critic (NORBC): critic sees raw privileged instead of encoder z
        asymmetric_critic: bool = False,
        # History parameters (kept for ActorCriticEncoderAdapt compatibility)
        proprio_history_len: int = 30,
        proprio_feature_dim: int = 8,
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
        self.asymmetric_critic = asymmetric_critic
        self.encoder_output_activation = encoder_output_activation
        self.z_min = z_min
        self.z_max = z_max
        self.z_bounds_coef = z_bounds_coef
        self.z_bounds_soft_bound = z_bounds_soft_bound

        # Store for subclass use (ActorCriticEncoderAdapt)
        self._proprio_history_len = proprio_history_len
        self._proprio_feature_dim = proprio_feature_dim

        # Last encoded z (retained with grad for z_bounds_loss computation in PPO)
        self._last_z: torch.Tensor | None = None

        # --- Parse obs_groups: extract key names ---
        policy_groups = obs_groups["policy"]
        if len(policy_groups) < 2:
            raise ValueError(
                f"ActorCriticEncoder requires at least 2 obs groups in 'policy', "
                f"got {len(policy_groups)}: {policy_groups}"
            )
        self._policy_obs_key = policy_groups[0]
        self._privileged_key = policy_groups[1]

        # Optional history key (3rd element in obs_groups)
        self._has_history = len(policy_groups) > 2
        if self._has_history:
            self._proprio_hist_key = policy_groups[2]
        else:
            self._proprio_hist_key = None

        # --- Verify obs dimensions ---
        policy_obs_shape = obs[self._policy_obs_key].shape
        privileged_shape = obs[self._privileged_key].shape

        if len(policy_obs_shape) != 2 or len(privileged_shape) != 2:
            raise ValueError("ActorCriticEncoder only supports 1D observations (batch, dim).")
        if policy_obs_shape[-1] != policy_obs_dim:
            raise ValueError(f"Policy obs dim {policy_obs_shape[-1]} != expected {policy_obs_dim}")
        if privileged_shape[-1] != privileged_dim:
            raise ValueError(f"Privileged dim {privileged_shape[-1]} != expected {privileged_dim}")

        # --- Compute history flat dimension ---
        if self._has_history:
            proprio_hist_shape = obs[self._proprio_hist_key].shape
            if len(proprio_hist_shape) != 3:
                raise ValueError(f"proprio_hist must be 3D (batch, history, features), got shape {proprio_hist_shape}")
            self._hist_flat_dim = proprio_hist_shape[1] * proprio_hist_shape[2]
            logger.info(
                "History: (%d, %d) -> flatten -> %dD (raw concat, no embedding)",
                proprio_hist_shape[1],
                proprio_hist_shape[2],
                self._hist_flat_dim,
            )
        else:
            self._hist_flat_dim = 0

        # --- Encoder MLP ---
        # With history: encoder sees [policy_obs, hist_flat, privileged]
        # Without history: encoder sees [privileged] only
        if self._has_history:
            encoder_input_dim = policy_obs_dim + self._hist_flat_dim + privileged_dim
        else:
            encoder_input_dim = privileged_dim

        # Encoder input normalization (Welford's online mean/var)
        self.encoder_obs_normalization = encoder_obs_normalization
        self.encoder_obs_normalizer = (
            EmpiricalNormalization(encoder_input_dim) if encoder_obs_normalization else nn.Identity()
        )

        if encoder_output_activation == "tanh":
            self.encoder = MLP(
                encoder_input_dim,
                encoder_latent_dim,
                list(encoder_hidden_dims),
                encoder_activation,
                last_activation="tanh",
            )
        else:
            self.encoder = MLP(encoder_input_dim, encoder_latent_dim, list(encoder_hidden_dims), encoder_activation)

        # Initialize last encoder layer for sigmoid activation
        if encoder_output_activation == "sigmoid":
            last_linear = self.encoder[-1]
            assert isinstance(last_linear, nn.Linear), (
                f"Expected last encoder layer to be Linear, got {type(last_linear)}"
            )
            nn.init.constant_(last_linear.bias, 0.0)
            nn.init.normal_(last_linear.weight, std=0.01)

        logger.info("Encoder MLP (%dD input): %s", encoder_input_dim, self.encoder)

        # --- Actor MLP ---
        # With history: actor sees [policy_obs, hist_flat, z]
        # Without history: actor sees [policy_obs, z]
        if self._has_history:
            num_actor_obs = policy_obs_dim + self._hist_flat_dim + encoder_latent_dim
        else:
            num_actor_obs = policy_obs_dim + encoder_latent_dim

        self.actor = MLP(num_actor_obs, num_actions, list(actor_hidden_dims), activation)
        logger.info("Actor MLP (%dD input): %s", num_actor_obs, self.actor)

        self.actor_obs_normalization = actor_obs_normalization
        self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs) if actor_obs_normalization else nn.Identity()

        # --- Critic MLP ---
        # With history: critic sees [policy_obs, hist_flat, privileged] (inherently asymmetric)
        # Without history + asymmetric: [policy_obs, privileged]
        # Without history + symmetric: [policy_obs, z]
        if self._has_history:
            num_critic_obs = policy_obs_dim + self._hist_flat_dim + privileged_dim
        elif asymmetric_critic:
            num_critic_obs = policy_obs_dim + privileged_dim
        else:
            num_critic_obs = policy_obs_dim + encoder_latent_dim
        self.num_critic_obs = num_critic_obs
        self.critic = MLP(num_critic_obs, 1, list(critic_hidden_dims), activation)
        if self._has_history:
            mode_str = "history-asymmetric"
        elif asymmetric_critic:
            mode_str = "asymmetric"
        else:
            mode_str = "symmetric"
        logger.info("Critic MLP (%s, %dD input): %s", mode_str, num_critic_obs, self.critic)

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

    def _get_hist_flat(self, obs: TensorDict) -> torch.Tensor:
        """Get flattened proprioception history from observations.

        Args:
            obs: TensorDict containing proprio_hist key.

        Returns:
            Flattened history. Shape: (N, H*D) e.g. (N, 240).
        """
        return obs[self._proprio_hist_key].flatten(start_dim=1)

    def _encode(self, obs: TensorDict, *, store_z: bool = False) -> torch.Tensor:
        """Encode into latent z.

        With history: encoder(cat([policy_obs, hist_flat, privileged])) -> z
        Without history: encoder(privileged) -> z

        Args:
            obs: TensorDict with policy, privileged, and optionally proprio_hist.
            store_z: If True, stores z as _last_z for z_bounds_loss computation.
        """
        privileged = obs[self._privileged_key]
        if self._has_history:
            policy_obs = obs[self._policy_obs_key]
            hist_flat = self._get_hist_flat(obs)
            encoder_input = torch.cat([policy_obs, hist_flat, privileged], dim=-1)
        else:
            encoder_input = privileged

        normalized = self.encoder_obs_normalizer(encoder_input)
        if self.encoder_output_activation == "tanh":
            z = self.encoder(normalized)
        else:
            z = self._activate_z(self.encoder(normalized))
        if store_z:
            self._last_z = z
        return z

    def z_bounds_loss(self) -> torch.Tensor:
        """Compute soft quadratic penalty when |z| exceeds z_bounds_soft_bound.

        Follows the same principle as HORA's bounds_loss on action means:
        penalize outputs that approach activation saturation boundaries.

        Returns zero tensor if z_bounds_coef is 0 or _last_z is not available.
        """
        if self.z_bounds_coef <= 0.0 or self._last_z is None:
            return torch.tensor(0.0, device=self._last_z.device if self._last_z is not None else "cpu")
        z = self._last_z
        excess = torch.clamp_min(z.abs() - self.z_bounds_soft_bound, 0.0)
        return self.z_bounds_coef * excess.pow(2).sum(dim=-1).mean()

    def _get_combined_obs(self, obs: TensorDict, *, store_z: bool = False) -> torch.Tensor:
        """Combined observation for actor.

        With history: cat([policy_obs, hist_flat, z])
        Without history: cat([policy_obs, z])
        """
        policy_obs = obs[self._policy_obs_key]
        z = self._encode(obs, store_z=store_z)
        if self._has_history:
            hist_flat = self._get_hist_flat(obs)
            return torch.cat([policy_obs, hist_flat, z], dim=-1)
        return torch.cat([policy_obs, z], dim=-1)

    def _get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        """Critic observation.

        With history: cat([policy_obs, hist_flat, privileged]) -- inherently asymmetric.
        Without history + asymmetric: cat([policy_obs, privileged]).
        Without history + symmetric: cat([policy_obs, z]).
        """
        if self._has_history:
            policy_obs = obs[self._policy_obs_key]
            hist_flat = self._get_hist_flat(obs)
            privileged_raw = obs[self._privileged_key]
            return torch.cat([policy_obs, hist_flat, privileged_raw], dim=-1)
        if not self.asymmetric_critic:
            return self._get_combined_obs(obs)
        policy_obs = obs[self._policy_obs_key]
        privileged_raw = obs[self._privileged_key]
        return torch.cat([policy_obs, privileged_raw], dim=-1)

    # --- Action distribution ---

    def _update_distribution(self, actor_obs: torch.Tensor) -> None:
        """Update the action distribution given actor observations."""
        mean = self.actor(actor_obs)
        std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    # --- Core API ---

    def act(self, obs: TensorDict, **_kwargs: Any) -> torch.Tensor:
        """Sample an action from the policy distribution."""
        actor_obs = self.actor_obs_normalizer(self._get_combined_obs(obs, store_z=True))  # type: ignore[operator]
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

        Used by AdaptRunner to generate actions with z_hat from adapt_tconv.
        """
        policy_obs = obs[self._policy_obs_key]
        if self._has_history:
            hist_flat = self._get_hist_flat(obs)
            combined_obs = torch.cat([policy_obs, hist_flat, z_hat.detach()], dim=-1)
        else:
            combined_obs = torch.cat([policy_obs, z_hat.detach()], dim=-1)
        actor_obs = self.actor_obs_normalizer(combined_obs)  # type: ignore[operator]
        return self.actor(actor_obs).clamp(-1.0, 1.0)

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
        # Encoder input normalizer
        if self.encoder_obs_normalization and hasattr(self.encoder_obs_normalizer, "update"):
            if self._has_history:
                policy_obs = obs[self._policy_obs_key]
                hist_flat = self._get_hist_flat(obs)
                privileged = obs[self._privileged_key]
                encoder_input = torch.cat([policy_obs, hist_flat, privileged], dim=-1)
            else:
                encoder_input = obs[self._privileged_key]
            self.encoder_obs_normalizer.update(encoder_input)  # type: ignore[union-attr]

        combined = self._get_combined_obs(obs)
        if self.actor_obs_normalization and hasattr(self.actor_obs_normalizer, "update"):
            self.actor_obs_normalizer.update(combined)  # type: ignore[union-attr]
        if self.critic_obs_normalization and hasattr(self.critic_obs_normalizer, "update"):
            critic_input = self._get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_input)  # type: ignore[union-attr]

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load model parameters with backward compatibility.

        Handles missing keys, dimension mismatches, and old checkpoints.
        Returns True to indicate resumed training (RSL-RL API contract).
        """
        if self.encoder_obs_normalization:
            prefix = "encoder_obs_normalizer."
            if not any(k.startswith(prefix) for k in state_dict):
                logger.info("Old checkpoint: injecting default encoder_obs_normalizer state.")
                for k, v in self.encoder_obs_normalizer.state_dict().items():
                    state_dict[prefix + k] = v

        # Detect critic input dimension mismatch (symmetric <-> asymmetric checkpoint)
        self._handle_critic_dim_mismatch(state_dict, "critic.")

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

    def _handle_critic_dim_mismatch(self, state_dict: dict, prefix: str) -> None:
        """Reinitialize critic if checkpoint input dimension doesn't match current model."""
        weight_keys = sorted(
            [k for k in state_dict if k.startswith(prefix) and k.endswith(".weight")],
            key=lambda k: int(k.removeprefix(prefix).split(".")[0]),
        )
        if not weight_keys:
            return
        first_key = weight_keys[0]
        ckpt_input_dim = state_dict[first_key].shape[1]

        if prefix == "critic.":
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
