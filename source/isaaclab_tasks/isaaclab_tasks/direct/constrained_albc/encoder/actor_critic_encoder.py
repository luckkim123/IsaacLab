# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teacher policy network: MLP Encoder + MLP Actor + MLP Critic.

Two modes of operation:
    Separate (default): Encoder, Actor, and Critic are independent MLPs.
    Shared backbone (HORA-style): Actor and Critic share an MLP backbone with
        linear heads, enabling value gradient flow to the encoder.

Architecture (separate mode, HORA-style normalization):
    Encoder: p_t (privileged) -> normalize -> MLP -> softsign -> z (latent)
    Actor:   cat([normalize(o_t), z]) -> MLP -> actions (Gaussian policy)
    Critic:  cat([o_t, p_t]) -> MLP -> value (asymmetric)

    Encoder input normalization modes:
      - Static min-max (HORA-style): (2*x - upper - lower) / (upper - lower) -> [-1, 1]
        Deterministic, no running stats, no z drift. Preferred.
      - EmpiricalNormalization: Running mean/std (legacy, causes z drift -> KL spike).
      - None: Raw p_t input.

    Actor normalization: Only o_t (+ proprio_hist if present) via EmpiricalNorm.
    z is kept raw since softsign already bounds it to (-1, 1).

Architecture (shared backbone mode):
    Encoder: p_t (privileged) -> normalize -> MLP -> softsign -> z (latent)
    Backbone: cat([normalize(o_t), z]) -> shared MLP -> features
    Action head: features -> Linear -> actions
    Value head:  features -> Linear -> value
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, NoReturn

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.networks import MLP, EmpiricalNormalization
from torch.distributions import Normal

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from tensordict import TensorDict


class ActorCriticEncoder(nn.Module):
    """Teacher policy with encoder for privileged-to-latent compression.

    Supports two architectures:
        shared_backbone=False (default): Separate actor/critic MLPs. Critic uses
            privileged info directly (asymmetric). Encoder gradient comes only
            from actor loss.
        shared_backbone=True (HORA-style): Single backbone MLP with linear heads
            for action and value. Both actor and critic losses flow gradient
            through the encoder, providing a stronger learning signal.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        # Encoder
        policy_obs_dim: int = 14,
        privileged_dim: int = 23,
        proprio_hist_dim: int = 0,
        encoder_hidden_dims: list[int] | tuple[int, ...] = (256, 128, 64),
        encoder_latent_dim: int = 13,
        encoder_activation: str = "elu",
        encoder_obs_normalization: bool = False,
        encoder_obs_lower: list[float] | None = None,
        encoder_obs_upper: list[float] | None = None,
        # Actor-Critic
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: list[int] | tuple[int, ...] = (256, 128, 64),
        critic_hidden_dims: list[int] | tuple[int, ...] = (512, 256, 128),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "log",
        # Shared backbone (HORA-style): actor+critic share MLP, value gradient to encoder
        shared_backbone: bool = False,
        # Symmetric critic: critic sees cat([o_t, hist]) instead of cat([o_t, p_t])
        symmetric_critic: bool = False,
        # Action clamping: clamp sampled actions to [-1, 1]
        clamp_actions: bool = True,
        # z bounds loss: soft quadratic penalty when |z| > soft_bound
        z_bounds_coef: float = 0.0,
        z_bounds_soft_bound: float = 0.85,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            logger.warning("ActorCriticEncoder ignoring unexpected kwargs: %s", list(kwargs.keys()))
        super().__init__()

        # Store dimensions
        self.obs_groups = obs_groups
        self.policy_obs_dim = policy_obs_dim
        self.privileged_dim = privileged_dim
        self.encoder_latent_dim = encoder_latent_dim
        self.proprio_hist_dim = proprio_hist_dim
        self.shared_backbone_mode = shared_backbone
        self.symmetric_critic = symmetric_critic
        self.clamp_actions = clamp_actions
        self.z_bounds_coef = z_bounds_coef
        self.z_bounds_soft_bound = z_bounds_soft_bound
        self._last_z: torch.Tensor | None = None

        # Parse obs_groups: require [policy_obs, privileged, (optional) proprio_hist]
        policy_groups = obs_groups["policy"]
        if len(policy_groups) < 2:
            raise ValueError(
                f"ActorCriticEncoder requires at least 2 obs groups in 'policy' "
                f"[policy_obs, privileged], got {len(policy_groups)}: {policy_groups}"
            )
        self._policy_obs_key = policy_groups[0]
        self._privileged_key = policy_groups[1]
        self._proprio_hist_key = policy_groups[2] if len(policy_groups) > 2 else None

        # Verify dimensions
        if obs[self._policy_obs_key].shape[-1] != policy_obs_dim:
            raise ValueError(f"Policy obs dim {obs[self._policy_obs_key].shape[-1]} != expected {policy_obs_dim}")
        if obs[self._privileged_key].shape[-1] != privileged_dim:
            raise ValueError(f"Privileged dim {obs[self._privileged_key].shape[-1]} != expected {privileged_dim}")

        # --- Encoder: p_t -> normalize -> MLP -> softsign -> z ---
        self._has_static_enc_norm = encoder_obs_lower is not None and encoder_obs_upper is not None
        if self._has_static_enc_norm:
            # Static min-max normalization (HORA-style): deterministic, no running stats
            lower = torch.tensor(encoder_obs_lower, dtype=torch.float32)
            upper = torch.tensor(encoder_obs_upper, dtype=torch.float32)
            if lower.shape[0] != privileged_dim or upper.shape[0] != privileged_dim:
                raise ValueError(
                    f"encoder_obs_lower/upper dim {lower.shape[0]}/{upper.shape[0]} != privileged_dim {privileged_dim}"
                )
            self.register_buffer("_enc_obs_lower", lower)
            self.register_buffer("_enc_obs_upper", upper)
            self.encoder_obs_normalizer = nn.Identity()
            self.encoder_obs_normalization = False
            logger.info("Encoder normalization: static min-max (HORA-style) -> [-1, 1]")
        else:
            self.encoder_obs_normalization = encoder_obs_normalization
            self.encoder_obs_normalizer = (
                EmpiricalNormalization(privileged_dim) if encoder_obs_normalization else nn.Identity()
            )
            logger.info("Encoder normalization: %s", "EmpiricalNorm" if encoder_obs_normalization else "none")
        self.encoder = MLP(privileged_dim, encoder_latent_dim, list(encoder_hidden_dims), encoder_activation)
        logger.info("Encoder: %dD -> %s -> softsign -> %dD", privileged_dim, encoder_hidden_dims, encoder_latent_dim)

        # Actor input dimension (shared between modes)
        num_actor_obs = policy_obs_dim + proprio_hist_dim + encoder_latent_dim
        # Normalizer covers only o_t + hist (excludes z which is already bounded by softsign).
        # HORA-style: normalize observations, keep encoder latent raw.
        num_actor_obs_norm = policy_obs_dim + proprio_hist_dim
        self._num_actor_obs_norm = num_actor_obs_norm

        if shared_backbone:
            # --- Shared backbone: cat([normalize(o_t, hist), z]) -> backbone -> features -> heads ---
            bb_dims = list(critic_hidden_dims)
            feature_dim = bb_dims[-1]
            self.actor_obs_normalization = actor_obs_normalization
            self.actor_obs_normalizer = (
                EmpiricalNormalization(num_actor_obs_norm) if actor_obs_normalization else nn.Identity()
            )
            self.backbone = MLP(
                num_actor_obs, feature_dim, bb_dims[:-1], activation, last_activation=activation
            )
            self.action_head = nn.Linear(feature_dim, num_actions)
            self.value_head = nn.Linear(feature_dim, 1)
            nn.init.zeros_(self.action_head.bias)
            nn.init.zeros_(self.value_head.bias)
            logger.info(
                "Shared backbone: %dD -> %s -> %dD features -> action(%dD) + value(1D)",
                num_actor_obs, bb_dims, feature_dim, num_actions,
            )
            # Compatibility attrs
            self.num_critic_obs = num_actor_obs
            self.critic_obs_normalization = False
        else:
            # --- Separate actor and critic (original mode) ---
            self.actor_obs_normalization = actor_obs_normalization
            self.actor_obs_normalizer = (
                EmpiricalNormalization(num_actor_obs_norm) if actor_obs_normalization else nn.Identity()
            )
            self.actor = MLP(num_actor_obs, num_actions, list(actor_hidden_dims), activation)
            logger.info(
                "Actor: %dD (obs=%d+hist=%d+z=%d) -> %s -> %dD",
                num_actor_obs, policy_obs_dim, proprio_hist_dim,
                encoder_latent_dim, actor_hidden_dims, num_actions,
            )

            if symmetric_critic:
                num_critic_obs = policy_obs_dim + proprio_hist_dim
            else:
                num_critic_obs = policy_obs_dim + privileged_dim
            self.num_critic_obs = num_critic_obs
            self.critic_obs_normalization = critic_obs_normalization
            self.critic_obs_normalizer = (
                EmpiricalNormalization(num_critic_obs) if critic_obs_normalization else nn.Identity()
            )
            self.critic = MLP(num_critic_obs, 1, list(critic_hidden_dims), activation)
            critic_type = "symmetric (o_t+hist)" if symmetric_critic else "asymmetric (o_t+p_t)"
            logger.info("Critic [%s]: %dD -> %s -> 1D", critic_type, num_critic_obs, critic_hidden_dims)

        # Action noise (Gaussian policy)
        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        else:
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    def reset(self, _dones: torch.Tensor | None = None) -> None:
        """No-op for non-recurrent networks."""
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError("Use act(), act_inference(), or evaluate().")

    @property
    def action_mean(self) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.entropy().sum(dim=-1)

    # --- Observation processing ---

    def _encode(self, obs: TensorDict) -> torch.Tensor:
        """Encode privileged info into latent z: p_t -> normalize -> MLP -> softsign -> z in (-1, 1)."""
        p_t = obs[self._privileged_key]
        if self._has_static_enc_norm:
            # Static min-max: [lower, upper] -> [-1, 1] (HORA-style, deterministic)
            p_t = (2.0 * p_t - self._enc_obs_upper - self._enc_obs_lower) / (
                self._enc_obs_upper - self._enc_obs_lower
            )
        else:
            p_t = self.encoder_obs_normalizer(p_t)
        z = F.softsign(self.encoder(p_t))
        self._last_z = z
        return z

    def _get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        """Actor observation: cat([normalize(o_t, hist), z_raw]).

        HORA-style: only o_t (+ hist if present) is normalized via EmpiricalNorm.
        z is kept raw since softsign already bounds it to (-1, 1), and normalizing
        non-stationary encoder output with running stats causes KL instability.
        """
        o_t = obs[self._policy_obs_key]
        z = self._encode(obs)
        if self._proprio_hist_key is not None and self._proprio_hist_key in obs:
            hist = obs[self._proprio_hist_key]  # Already flat (N, T*F) from env
            obs_part = torch.cat([o_t, hist], dim=-1)
        else:
            obs_part = o_t
        obs_normed = self.actor_obs_normalizer(obs_part)
        return torch.cat([obs_normed, z], dim=-1)

    def _get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        """Critic observation (separate mode).

        Asymmetric (default): cat([o_t, p_t]) -- privileged info for accurate values.
        Symmetric: cat([o_t, hist]) -- same info as actor (minus z).
        """
        if self.symmetric_critic:
            o_t = obs[self._policy_obs_key]
            if self._proprio_hist_key is not None and self._proprio_hist_key in obs:
                return torch.cat([o_t, obs[self._proprio_hist_key]], dim=-1)
            return o_t
        return torch.cat([obs[self._policy_obs_key], obs[self._privileged_key]], dim=-1)

    # --- Action distribution ---

    def _update_distribution(self, mean: torch.Tensor) -> None:
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(torch.nan_to_num(self.log_std, nan=0.0).clamp(-10.0, 5.0)).expand_as(mean)
        self.distribution = Normal(mean, std)

    # --- Core API ---

    def act(self, obs: TensorDict, **_kwargs: Any) -> torch.Tensor:
        """Sample action from Gaussian policy."""
        actor_obs = self._get_actor_obs(obs)  # normalization applied inside
        if self.shared_backbone_mode:
            features = self.backbone(actor_obs)
            mean = self.action_head(features)
        else:
            mean = self.actor(actor_obs)
        self._update_distribution(mean)
        assert self.distribution is not None
        sample = self.distribution.sample()
        return sample.clamp(-1.0, 1.0) if self.clamp_actions else sample

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        """Deterministic action (mean)."""
        actor_obs = self._get_actor_obs(obs)  # normalization applied inside
        if self.shared_backbone_mode:
            features = self.backbone(actor_obs)
            mean = self.action_head(features)
        else:
            mean = self.actor(actor_obs)
        return mean.clamp(-1.0, 1.0) if self.clamp_actions else mean

    def evaluate(self, obs: TensorDict, **_kwargs: Any) -> torch.Tensor:
        """Evaluate value function.

        In shared backbone mode, value gradient flows through encoder (HORA-style).
        In separate mode, critic uses privileged info directly (no encoder gradient).
        """
        if self.shared_backbone_mode:
            # Backbone uses cat([o_t, z]) -- encoder gradient from value loss
            actor_obs = self._get_actor_obs(obs)  # normalization applied inside
            features = self.backbone(actor_obs)
            return self.value_head(features)
        else:
            critic_obs = self.critic_obs_normalizer(self._get_critic_obs(obs))
            return self.critic(critic_obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Log probability of actions under current distribution."""
        assert self.distribution is not None
        return self.distribution.log_prob(actions).sum(dim=-1)

    def z_bounds_loss(self) -> torch.Tensor:
        """Soft quadratic penalty when |z| exceeds soft_bound. Prevents saturation."""
        if self.z_bounds_coef == 0.0 or self._last_z is None:
            device = self.std.device if self.noise_std_type == "scalar" else self.log_std.device
            return torch.tensor(0.0, device=device)
        excess = torch.clamp_min(self._last_z.abs() - self.z_bounds_soft_bound, 0.0)
        return self.z_bounds_coef * excess.pow(2).sum(dim=-1).mean()

    def update_normalization(self, obs: TensorDict) -> None:
        """Update observation normalization running statistics.

        Static min-max encoder normalization has no running stats (no-op).
        Actor normalizer updates only on o_t (+ hist) dimensions, excluding z.
        """
        if self.encoder_obs_normalization and not self._has_static_enc_norm:
            self.encoder_obs_normalizer.update(obs[self._privileged_key])
        if self.actor_obs_normalization:
            with torch.no_grad():
                o_t = obs[self._policy_obs_key]
                if self._proprio_hist_key is not None and self._proprio_hist_key in obs:
                    hist = obs[self._proprio_hist_key]
                    obs_part = torch.cat([o_t, hist], dim=-1)
                else:
                    obs_part = o_t
            self.actor_obs_normalizer.update(obs_part)
        if not self.shared_backbone_mode and self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self._get_critic_obs(obs))

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load model parameters. Returns True (RSL-RL API contract)."""
        if self.encoder_obs_normalization and not self._has_static_enc_norm:
            prefix = "encoder_obs_normalizer."
            if not any(k.startswith(prefix) for k in state_dict):
                logger.info("Checkpoint missing encoder_obs_normalizer; injecting defaults.")
                for k, v in self.encoder_obs_normalizer.state_dict().items():
                    state_dict[prefix + k] = v
        # Migrate actor_obs_normalizer from old (full o_t+z) to new (o_t only) dimension
        if self.actor_obs_normalization:
            norm_prefix = "actor_obs_normalizer."
            mean_key = norm_prefix + "_mean"
            if mean_key in state_dict:
                old_dim = state_dict[mean_key].shape[-1]
                new_dim = self._num_actor_obs_norm
                if old_dim != new_dim:
                    logger.info(
                        "Actor obs normalizer dim mismatch (%d -> %d); resetting to defaults.",
                        old_dim, new_dim,
                    )
                    for k, v in self.actor_obs_normalizer.state_dict().items():
                        state_dict[norm_prefix + k] = v
        super().load_state_dict(state_dict, strict=False)
        return True
