# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ActorCritic with extrinsics encoder for HORA/RMA Phase 1 teacher training.

This module implements a custom ActorCritic network that compresses privileged
hydrodynamic parameters (22D) into a low-dimensional latent z (6D) via a
learned encoder with softplus output.

Architecture:
    Encoder: privileged (22D) -> MLP -> softplus + z_min -> z (6D)
    Actor:   cat([policy_obs, z]) = 19D -> MLP -> actions (2D)
    Critic:  cat([policy_obs, z]) = 19D -> MLP -> value (1D)

Note: Critic does NOT receive privileged info directly (symmetric with actor).
This forces the encoder to compress useful information into z. If critic had
direct access to privileged info, it would ignore z and encoder would collapse
to a constant output.

Design choices:
    - Softplus instead of scaled sigmoid: Avoids gradient saturation at bounds.
      z = softplus(raw) + z_min guarantees z > z_min with no upper bound.
    - 6D latent: Matches 6-DOF convention [surge,sway,heave,roll,pitch,yaw].
      TDC uses z[3:5] as M_hat for roll/pitch inertia estimation.
    - 22D privileged (not 64D): Core parameters only (mass, volume, CoG, CoB,
      inertia) without damping/added_mass which don't affect attitude dynamics.

Reference:
    - HORA: Heuristic-Free Online Robust Adaptation (Qi et al., 2023)
    - RMA: Rapid Motor Adaptation (Kumar et al., 2021)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NoReturn

import torch
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
    This guarantees z > z_min (positive), compatible with TDC's M_hat = z[3:5].
    Unlike scaled sigmoid, softplus has no gradient saturation at large values.

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
        privileged_dim: int = 22,
        encoder_hidden_dims: list[int] | tuple[int, ...] = (64, 32),
        encoder_latent_dim: int = 6,
        encoder_activation: str = "elu",
        z_min: float = 0.1,
        # Actor-Critic parameters
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: list[int] | tuple[int, ...] = (64, 64),
        critic_hidden_dims: list[int] | tuple[int, ...] = (64, 64),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            print(f"ActorCriticEncoder.__init__ got unexpected arguments, which will be ignored: {list(kwargs.keys())}")
        super().__init__()

        # Store dimension info
        self.obs_groups = obs_groups
        self.policy_obs_dim = policy_obs_dim
        self.privileged_dim = privileged_dim
        self.encoder_latent_dim = encoder_latent_dim
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
        self.encoder = MLP(privileged_dim, encoder_latent_dim, list(encoder_hidden_dims), encoder_activation)
        print(f"Encoder MLP: {self.encoder}")

        # Actor/Critic input: policy_obs + z (symmetric design)
        # Note: Privileged info is NOT passed to critic to force encoder learning.
        # If critic receives privileged directly, it ignores z and encoder collapses.
        num_combined_obs = policy_obs_dim + encoder_latent_dim

        # Actor
        actor_output = [2, num_actions] if self.state_dependent_std else num_actions
        self.actor = MLP(num_combined_obs, actor_output, list(actor_hidden_dims), activation)
        print(f"Actor MLP: {self.actor}")

        # Actor observation normalization (applied to actor input: policy_obs + z)
        self.actor_obs_normalization = actor_obs_normalization
        self.actor_obs_normalizer = (
            EmpiricalNormalization(num_combined_obs) if actor_obs_normalization else nn.Identity()
        )

        # Critic
        self.critic = MLP(num_combined_obs, 1, list(critic_hidden_dims), activation)
        print(f"Critic MLP: {self.critic}")

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

    def _encode(self, privileged: torch.Tensor) -> torch.Tensor:
        """Encode privileged info into positive latent z via softplus.

        z = softplus(encoder_output) + z_min
        Guarantees z > z_min, no gradient saturation at large values.
        """
        return F.softplus(self.encoder(privileged)) + self.z_min

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
