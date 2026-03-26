# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teacher policy network: MLP Encoder + MLP Actor + MLP Critic.

Following the paper's teacher-student framework (teacher only):
    Encoder: p_t (privileged) -> MLP -> softsign -> z (latent)
    Actor:   cat([o_t, z]) -> MLP -> actions (Gaussian policy)
    Critic:  cat([o_t, p_t]) -> MLP -> value (asymmetric)

The encoder receives only privileged information (DR parameters) and compresses
it into a bounded latent z in (-1, 1). The actor combines policy observations
with the latent z to produce actions. The critic uses full information
(policy obs + privileged) for value estimation.

No proprioception history is used in the teacher -- history processing
(GRU encoder) is reserved for the student policy (future implementation).
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
    """Teacher policy: MLP encoder compresses privileged info into latent z.

    Architecture:
        Encoder: p_t -> MLP -> softsign -> z in (-1, 1)
        Actor:   cat([o_t, z]) -> MLP -> actions
        Critic:  cat([o_t, p_t]) -> MLP -> value (asymmetric)
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
        # Actor-Critic
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: list[int] | tuple[int, ...] = (256, 128, 64),
        critic_hidden_dims: list[int] | tuple[int, ...] = (512, 256, 128),
        activation: str = "elu",
        init_noise_std: float = 1.0,
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

        # --- Encoder: p_t -> softsign -> z ---
        self.encoder_obs_normalization = encoder_obs_normalization
        self.encoder_obs_normalizer = (
            EmpiricalNormalization(privileged_dim) if encoder_obs_normalization else nn.Identity()
        )
        self.encoder = MLP(privileged_dim, encoder_latent_dim, list(encoder_hidden_dims), encoder_activation)
        logger.info("Encoder: %dD -> %s -> softsign -> %dD", privileged_dim, encoder_hidden_dims, encoder_latent_dim)

        # --- Actor: cat([o_t, (hist_flat,) z]) -> actions ---
        num_actor_obs = policy_obs_dim + proprio_hist_dim + encoder_latent_dim
        self.actor_obs_normalization = actor_obs_normalization
        self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs) if actor_obs_normalization else nn.Identity()
        self.actor = MLP(num_actor_obs, num_actions, list(actor_hidden_dims), activation)
        logger.info("Actor: %dD (obs=%d+hist=%d+z=%d) -> %s -> %dD", num_actor_obs, policy_obs_dim, proprio_hist_dim, encoder_latent_dim, actor_hidden_dims, num_actions)

        # --- Critic: cat([o_t, p_t]) -> value (asymmetric) ---
        num_critic_obs = policy_obs_dim + privileged_dim
        self.num_critic_obs = num_critic_obs
        self.critic_obs_normalization = critic_obs_normalization
        self.critic_obs_normalizer = (
            EmpiricalNormalization(num_critic_obs) if critic_obs_normalization else nn.Identity()
        )
        self.critic = MLP(num_critic_obs, 1, list(critic_hidden_dims), activation)
        logger.info("Critic: %dD -> %s -> 1D", num_critic_obs, critic_hidden_dims)

        # Action noise (Gaussian policy with learned log_std)
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
        """Encode privileged info into latent z: p_t -> MLP -> softsign -> z in (-1, 1)."""
        p_t = obs[self._privileged_key]
        return torch.nn.functional.softsign(self.encoder(self.encoder_obs_normalizer(p_t)))

    def _get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        """Actor observation: cat([o_t, (hist_flat,) z])."""
        o_t = obs[self._policy_obs_key]
        z = self._encode(obs)
        if self._proprio_hist_key is not None and self._proprio_hist_key in obs:
            hist = obs[self._proprio_hist_key]  # Already flat (N, T*F) from env
            return torch.cat([o_t, hist, z], dim=-1)
        return torch.cat([o_t, z], dim=-1)

    def _get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        """Critic observation: cat([o_t, p_t]) (asymmetric, uses privileged directly)."""
        return torch.cat([obs[self._policy_obs_key], obs[self._privileged_key]], dim=-1)

    # --- Action distribution ---

    def _update_distribution(self, actor_obs: torch.Tensor) -> None:
        mean = self.actor(actor_obs)
        std = torch.exp(torch.nan_to_num(self.log_std, nan=0.0).clamp(-10.0, 5.0)).expand_as(mean)
        self.distribution = Normal(mean, std)

    # --- Core API ---

    def act(self, obs: TensorDict, **_kwargs: Any) -> torch.Tensor:
        """Sample action from Gaussian policy, clipped to [-1, 1]."""
        actor_obs = self.actor_obs_normalizer(self._get_actor_obs(obs))
        self._update_distribution(actor_obs)
        assert self.distribution is not None
        return self.distribution.sample().clamp(-1.0, 1.0)

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        """Deterministic action (mean), clipped to [-1, 1]."""
        actor_obs = self.actor_obs_normalizer(self._get_actor_obs(obs))
        return self.actor(actor_obs).clamp(-1.0, 1.0)

    def evaluate(self, obs: TensorDict, **_kwargs: Any) -> torch.Tensor:
        """Evaluate reward value function."""
        critic_obs = self.critic_obs_normalizer(self._get_critic_obs(obs))
        return self.critic(critic_obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Log probability of actions under current distribution."""
        assert self.distribution is not None
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update observation normalization running statistics."""
        if self.encoder_obs_normalization:
            self.encoder_obs_normalizer.update(obs[self._privileged_key])
        if self.actor_obs_normalization:
            with torch.no_grad():
                actor_obs = self._get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self._get_critic_obs(obs))

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load model parameters. Returns True (RSL-RL API contract)."""
        if self.encoder_obs_normalization:
            prefix = "encoder_obs_normalizer."
            if not any(k.startswith(prefix) for k in state_dict):
                logger.info("Checkpoint missing encoder_obs_normalizer; injecting defaults.")
                for k, v in self.encoder_obs_normalizer.state_dict().items():
                    state_dict[prefix + k] = v
        super().load_state_dict(state_dict, strict=False)
        return True
