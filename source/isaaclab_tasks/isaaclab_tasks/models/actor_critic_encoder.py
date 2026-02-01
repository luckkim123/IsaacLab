# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ActorCritic with extrinsics encoder for HORA/RMA Phase 1 teacher training.

This module implements a custom ActorCritic network that compresses privileged
hydrodynamic parameters (64D) into a low-dimensional latent z (2D) via a
learned encoder with scaled sigmoid output. The latent is concatenated with
policy observations for the actor, while the critic receives full information.

Architecture:
    Encoder: privileged (64D) -> MLP -> scaled_sigmoid -> z (2D)
    Actor:   cat([policy_obs, z]) = 15D -> MLP -> actions (2D)
    Critic:  cat([policy_obs, z, privileged]) = 79D -> MLP -> value (1D)

Reference:
    - HORA: Heuristic-Free Online Robust Adaptation (Qi et al., 2023)
    - RMA: Rapid Motor Adaptation (Kumar et al., 2021)
"""

from __future__ import annotations

from typing import Any, NoReturn

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.networks import MLP, EmpiricalNormalization


class ActorCriticEncoder(nn.Module):
    """ActorCritic with extrinsics encoder for HORA Phase 1 teacher policy.

    The encoder compresses privileged information into a bounded latent vector z
    using scaled sigmoid: z = z_min + (z_max - z_min) * sigmoid(raw_output).
    This guarantees z > 0 and bounded, compatible with TDC's M_hat = diag(z).

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
        privileged_dim: int = 64,
        encoder_hidden_dims: list[int] | tuple[int] = [64, 32],
        encoder_latent_dim: int = 2,
        encoder_activation: str = "elu",
        z_min: float = 0.1,
        z_max: float = 5.0,
        # Actor-Critic parameters
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: list[int] | tuple[int] = [64, 64],
        critic_hidden_dims: list[int] | tuple[int] = [64, 64],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCriticEncoder.__init__ got unexpected arguments, "
                "which will be ignored: " + str([key for key in kwargs])
            )
        super().__init__()

        # Store dimension info
        self.obs_groups = obs_groups
        self.policy_obs_dim = policy_obs_dim
        self.privileged_dim = privileged_dim
        self.encoder_latent_dim = encoder_latent_dim
        self.z_min = z_min
        self.z_max = z_max
        self.state_dependent_std = state_dependent_std

        # Extract obs key names from obs_groups for direct TensorDict access
        assert len(obs_groups["policy"]) == 2, (
            f"ActorCriticEncoder requires exactly 2 obs groups in 'policy', "
            f"got {len(obs_groups['policy'])}: {obs_groups['policy']}"
        )
        self._policy_obs_key = obs_groups["policy"][0]
        self._privileged_key = obs_groups["policy"][1]

        # Verify obs dimensions match expected
        assert len(obs[self._policy_obs_key].shape) == 2, (
            "ActorCriticEncoder only supports 1D observations."
        )
        assert len(obs[self._privileged_key].shape) == 2, (
            "ActorCriticEncoder only supports 1D observations."
        )
        assert obs[self._policy_obs_key].shape[-1] == policy_obs_dim, (
            f"Policy obs dim {obs[self._policy_obs_key].shape[-1]} != expected {policy_obs_dim}"
        )
        assert obs[self._privileged_key].shape[-1] == privileged_dim, (
            f"Privileged dim {obs[self._privileged_key].shape[-1]} != expected {privileged_dim}"
        )

        # Encoder: privileged -> z
        self.encoder = MLP(privileged_dim, encoder_latent_dim, list(encoder_hidden_dims), encoder_activation)
        print(f"Encoder MLP: {self.encoder}")

        # Actor input: policy_obs + z
        num_actor_obs = policy_obs_dim + encoder_latent_dim

        # Critic input: policy_obs + z + privileged (asymmetric)
        num_critic_obs = policy_obs_dim + encoder_latent_dim + privileged_dim

        # Actor
        if self.state_dependent_std:
            self.actor = MLP(num_actor_obs, [2, num_actions], list(actor_hidden_dims), activation)
        else:
            self.actor = MLP(num_actor_obs, num_actions, list(actor_hidden_dims), activation)
        print(f"Actor MLP: {self.actor}")

        # Actor observation normalization (applied to actor input: policy_obs + z)
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = nn.Identity()

        # Critic
        self.critic = MLP(num_critic_obs, 1, list(critic_hidden_dims), activation)
        print(f"Critic MLP: {self.critic}")

        # Critic observation normalization (applied to critic input: policy_obs + z + privileged)
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = nn.Identity()

        # Action noise
        self.noise_std_type = noise_std_type
        if self.state_dependent_std:
            nn.init.zeros_(self.actor[-2].weight[num_actions:])
            if self.noise_std_type == "scalar":
                nn.init.constant_(self.actor[-2].bias[num_actions:], init_noise_std)
            elif self.noise_std_type == "log":
                nn.init.constant_(
                    self.actor[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                )
            else:
                raise ValueError(f"Unknown noise_std_type: {self.noise_std_type}")
        else:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown noise_std_type: {self.noise_std_type}")

        # Action distribution (populated in _update_distribution)
        self.distribution = None
        Normal.set_default_validate_args(False)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    # --- Observation processing ---

    def _encode(self, privileged: torch.Tensor) -> torch.Tensor:
        """Encode privileged info into bounded latent z via scaled sigmoid.

        z = z_min + (z_max - z_min) * sigmoid(encoder_output)
        Guarantees z in [z_min, z_max], differentiable everywhere.
        """
        raw = self.encoder(privileged)
        z = self.z_min + (self.z_max - self.z_min) * torch.sigmoid(raw)
        return z

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        """Get actor observation: cat([policy_obs, z]) -> 15D."""
        policy_obs = obs[self._policy_obs_key]
        z = self._encode(obs[self._privileged_key])
        return torch.cat([policy_obs, z], dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        """Get critic observation: cat([policy_obs, z, privileged]) -> 79D."""
        policy_obs = obs[self._policy_obs_key]
        privileged = obs[self._privileged_key]
        z = self._encode(privileged)
        return torch.cat([policy_obs, z, privileged], dim=-1)

    # --- Action distribution ---

    def _update_distribution(self, actor_obs: torch.Tensor) -> None:
        if self.state_dependent_std:
            mean_and_std = self.actor(actor_obs)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)
            else:
                raise ValueError(f"Unknown noise_std_type: {self.noise_std_type}")
        else:
            mean = self.actor(actor_obs)
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(f"Unknown noise_std_type: {self.noise_std_type}")
        self.distribution = Normal(mean, std)

    # --- Core API ---

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        self._update_distribution(actor_obs)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        if self.state_dependent_std:
            return self.actor(actor_obs)[..., 0, :]
        else:
            return self.actor(actor_obs)

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        critic_obs = self.get_critic_obs(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        return self.critic(critic_obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)
