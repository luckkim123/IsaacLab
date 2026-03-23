# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ActorCritic with extrinsics encoder for HORA/RMA training.

This module provides the encoder-based actor-critic network:
    - ActorCriticEncoder: Base encoder network (Phase 1 teacher training)

Architecture:
    Encoder: privileged (23D) -> MLP -> softsign -> z (13D)
    Actor:   cat([policy_obs, hist_flat, z]) = 266D -> MLP -> actions
    Critic:  cat([policy_obs, hist_flat, privileged]) = 276D -> MLP -> value (1D)

    The encoder takes ONLY privileged info as input (HORA Phase 1 style).
    This forces the actor to use z for DR-specific adaptation, since z is the
    only path through which privileged information reaches the actor.

    proprio_hist (N, 30, 8) is flattened to (N, 240) and concatenated directly.
    No embedding module -- the actor/critic MLPs learn from raw history.

Reference:
    - HORA: Heuristic-Free Online Robust Adaptation (Qi et al., 2023)
    - RMA: Rapid Motor Adaptation (Kumar et al., 2021)
    - NORBC: Neural Online Robust Boundary Controller (Kim et al., 2024)
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, NoReturn

import torch
import torch.nn as nn
from rsl_rl.networks import MLP, EmpiricalNormalization
from torch.distributions import Normal

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from tensordict import TensorDict


class _FixedNormalization(nn.Module):
    """Fixed (x - mean) / std normalization using pre-computed statistics."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("_mean", mean.unsqueeze(0))
        self.register_buffer("_std", std.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / self._std


class ActorCriticEncoder(nn.Module):
    """ActorCritic with extrinsics encoder for HORA Phase 1 teacher policy.

    The encoder compresses privileged information into a bounded latent z (softsign).
    Encoder input is privileged-only (HORA Phase 1 style), ensuring z encodes
    DR parameters rather than redundant policy_obs/history information.

    Architecture:
        Encoder: privileged (23D) -> MLP -> softsign -> z (13D)
        Actor:   cat([policy_obs, hist_flat, z]) = 266D -> MLP -> actions
        Critic:  cat([policy_obs, hist_flat, privileged]) = 276D -> MLP -> value (asymmetric)

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

        # --- Encoder MLP: privileged -> softsign -> z (HORA Phase 1 style) ---
        encoder_input_dim = privileged_dim

        self.encoder_obs_normalization = encoder_obs_normalization
        if encoder_obs_normalization:
            self.encoder_obs_normalizer = self._build_fixed_encoder_normalizer(encoder_input_dim)
        else:
            self.encoder_obs_normalizer = nn.Identity()

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

    @staticmethod
    def _build_fixed_encoder_normalizer(dim: int) -> nn.Module:
        """Build fixed normalization for 23D privileged encoder input.

        Mean and std computed analytically from DR config distributions.
        For uniform U(a,b): mean = (a+b)/2, std = (b-a)/sqrt(12).
        For scaled values (base * U(lo,hi)): mean = base*(lo+hi)/2, std = base*(hi-lo)/sqrt(12).
        For disk-uniform (radius R): mean = 0, std = R/2.

        Privileged obs order (23D):
            [0-2]   main hydro: volume, CoG_z, CoB_z
            [3-5]   buoy hydro: volume, CoG_z, CoB_z
            [6-7]   main inertia: Ixx, Iyy
            [8-9]   buoy inertia: Ixx, Iyy
            [10-13] payload: mass, cog_x, cog_y, cog_z
            [14]    main added mass surge
            [15]    joint stiffness (Kp)
            [16]    joint damping (Kd)
            [17]    joint effort limit
            [18-19] main linear damping roll, pitch
            [20-21] main quadratic damping roll, pitch
            [22]    main body mass
        """
        s12 = math.sqrt(12.0)

        # fmt: off
        mean = torch.tensor([
            0.009,                     # [0]  main volume: 0.009 * mean(U(0.9,1.1)) = 0.009
            -0.05,                     # [1]  main CoG_z: -0.05 + mean(U(-0.02,0.02)) = -0.05
            0.0,                       # [2]  main CoB_z: 0.0 + mean(U(-0.02,0.02)) = 0.0
            0.00268,                   # [3]  buoy volume: 0.00268 * mean(U(0.9,1.1)) = 0.00268
            0.059,                     # [4]  buoy CoG_z: 0.059 + mean(U(-0.02,0.02)) = 0.059
            0.059,                     # [5]  buoy CoB_z: 0.059 + mean(U(-0.02,0.02)) = 0.059
            0.0994 * 1.025,            # [6]  main Ixx: 0.0994 * mean(U(0.75,1.3))
            0.0994 * 1.025,            # [7]  main Iyy: same
            0.00278 * 1.025,           # [8]  buoy Ixx: 0.00278 * mean(U(0.75,1.3))
            0.00278 * 1.025,           # [9]  buoy Iyy: same
            0.5,                       # [10] payload mass: mean(U(0,1)) = 0.5
            0.0,                       # [11] payload cog_x: disk-uniform, mean=0
            0.0,                       # [12] payload cog_y: disk-uniform, mean=0
            -0.015,                    # [13] payload cog_z: mean(U(-0.03,0)) = -0.015
            8.0,                       # [14] main added mass surge: 8.0 * mean(U(0.85,1.15))
            80.0,                      # [15] joint stiffness: mean(U(40,120)) = 80
            2.75,                      # [16] joint damping: mean(U(0.5,5.0)) = 2.75
            8.075,                     # [17] effort limit: 9.5 * mean(U(0.7,1.0)) = 8.075
            0.3,                       # [18] main lin_damp roll: 0.3 * mean(U(0.5,1.5))
            0.3,                       # [19] main lin_damp pitch: same
            1.0,                       # [20] main quad_damp roll: 1.0 * mean(U(0.5,1.5))
            1.0,                       # [21] main quad_damp pitch: same
            9.18,                      # [22] body mass: 9.18 * mean(U(0.9,1.1))
        ])

        std = torch.tensor([
            0.009 * 0.2 / s12,         # [0]  main volume
            0.04 / s12,                # [1]  main CoG_z offset range 0.04
            0.04 / s12,                # [2]  main CoB_z offset range 0.04
            0.00268 * 0.2 / s12,       # [3]  buoy volume
            0.04 / s12,                # [4]  buoy CoG_z
            0.04 / s12,                # [5]  buoy CoB_z
            0.0994 * 0.55 / s12,       # [6]  main Ixx (scale range 0.55)
            0.0994 * 0.55 / s12,       # [7]  main Iyy
            0.00278 * 0.55 / s12,      # [8]  buoy Ixx
            0.00278 * 0.55 / s12,      # [9]  buoy Iyy
            1.0 / s12,                 # [10] payload mass (range 1.0)
            0.05,                      # [11] payload cog_x: disk R=0.1, std=R/2
            0.05,                      # [12] payload cog_y: disk R=0.1, std=R/2
            0.03 / s12,                # [13] payload cog_z (range 0.03)
            8.0 * 0.3 / s12,           # [14] main added mass surge (scale range 0.3)
            80.0 / s12,                # [15] joint stiffness (range 80)
            4.5 / s12,                 # [16] joint damping (range 4.5)
            9.5 * 0.3 / s12,           # [17] effort limit (scale range 0.3)
            0.3 * 1.0 / s12,           # [18] main lin_damp roll (scale range 1.0)
            0.3 * 1.0 / s12,           # [19] main lin_damp pitch
            1.0 * 1.0 / s12,           # [20] main quad_damp roll (scale range 1.0)
            1.0 * 1.0 / s12,           # [21] main quad_damp pitch
            9.18 * 0.2 / s12,          # [22] body mass (scale range 0.2)
        ])
        # fmt: on

        if dim != 23:
            logger.warning(
                "Fixed encoder normalizer expects 23D privileged obs, got %d. Falling back to EmpiricalNormalization.",
                dim,
            )
            return EmpiricalNormalization(dim)

        normalizer = _FixedNormalization(mean, std)
        logger.info("Encoder using fixed normalization (23D, analytical DR stats)")
        return normalizer

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

    def _encode(self, obs: TensorDict) -> torch.Tensor:
        """Encode privileged info into latent z.

        encoder(normalize(privileged)) -> softsign -> z in (-1, 1)
        """
        encoder_input = obs[self._privileged_key]
        x = self.encoder(self.encoder_obs_normalizer(encoder_input))
        return torch.nn.functional.softsign(x)

    def _get_combined_obs(self, obs: TensorDict) -> torch.Tensor:
        """Combined observation for actor: cat([policy_obs, hist_flat, z])."""
        policy_obs = obs[self._policy_obs_key]
        hist_flat = self._get_hist_flat(obs)
        z = self._encode(obs)
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
        # Encoder uses fixed normalization (no update needed).
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
            expected_dim = self.privileged_dim
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
            if not any(k.startswith(prefix) for k in state_dict):
                logger.info("Old checkpoint: injecting default encoder_obs_normalizer state.")
                for k, v in self.encoder_obs_normalizer.state_dict().items():
                    state_dict[prefix + k] = v

        # Detect input dimension mismatches (encoder: privileged-only change, critic: symmetric <-> asymmetric)
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
