# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 2 adaptation network: temporal convolution replaces encoder for z estimation.

This module provides:
    - ProprioAdaptTConv: Temporal conv network (proprio history -> z_hat)
    - ActorCriticEncoderAdapt: Full network with frozen base + trainable adapt (base RL)

Architecture:
    ProprioAdaptTConv:
        Input: (N, H, D) proprioception history (D=8: roll, pitch, p, q, joint_pos, prev_actions)
        -> channel_transform: per-timestep MLP (D -> 32 -> 32)
        -> temporal_aggregation: 3x Conv1d (H -> 3 time steps)
        -> low_dim_proj: Linear(32*3 -> output_dim)
        -> raw output (activation applied externally)

    ActorCriticEncoderAdapt:
        Inherits ActorCriticEncoder directly (base RL, NOT TDC chain).
        Overrides _get_combined_obs() to use adapt_tconv(proprio_hist) instead
        of _encode(privileged). z_hat uses _activate_z() (matching Phase 1).
        evaluate() overridden to use encoder z for critic (symmetric).

Reference:
    HORA ProprioAdaptTConv (references/hora/hora/algo/models/models.py)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from rsl_rl.networks import EmpiricalNormalization

from .actor_critic_encoder import ActorCriticEncoder

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from tensordict import TensorDict


def _compute_conv_output_len(length: int, kernels: list[int], strides: list[int]) -> int:
    """Compute output temporal length after a sequence of Conv1d layers."""
    for k, s in zip(kernels, strides):
        length = (length - k) // s + 1
    return length


class ProprioAdaptTConv(nn.Module):
    """Temporal convolution adaptation module: proprioception history -> z_hat.

    Processes a sliding window of proprioceptive features through:
    1. Per-timestep channel transform (MLP)
    2. Temporal aggregation (1D convolutions)
    3. Low-dimensional projection to latent z

    Uses H=30 with kernels=[9,5,5], strides=[2,1,1] -> temporal output 3.
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 32,
        output_dim: int = 13,
        history_len: int = 30,
    ):
        super().__init__()

        kernels = [9, 5, 5]
        strides = [2, 1, 1]

        final_time_steps = _compute_conv_output_len(history_len, kernels, strides)
        if final_time_steps < 1:
            raise ValueError(
                f"Conv kernels {kernels} with strides {strides} produce 0 output "
                f"for history_len={history_len}. Use smaller kernels."
            )

        self.channel_transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        layers: list[nn.Module] = []
        for k, s in zip(kernels, strides):
            layers.append(nn.Conv1d(hidden_dim, hidden_dim, k, stride=s))
            layers.append(nn.ReLU(inplace=True))
        self.temporal_aggregation = nn.Sequential(*layers)

        self.low_dim_proj = nn.Linear(hidden_dim * final_time_steps, output_dim)

        # Match Phase 1 encoder init: output near zero -> _activate_z(~0) = midpoint of [z_min, z_max].
        nn.init.constant_(self.low_dim_proj.bias, 0.0)
        nn.init.normal_(self.low_dim_proj.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (N, H, input_dim) -> (N, output_dim).

        Args:
            x: Proprioception history. Shape: (N, H, input_dim).

        Returns:
            Raw latent vector (before activation). Shape: (N, output_dim).
        """
        x = self.channel_transform(x)  # (N, H, hidden_dim)
        x = x.permute(0, 2, 1)  # (N, hidden_dim, H) for Conv1d
        x = self.temporal_aggregation(x)  # (N, hidden_dim, T_final)
        return self.low_dim_proj(x.flatten(1))  # (N, output_dim)


class ActorCriticEncoderAdapt(ActorCriticEncoder):
    """Phase 2 adaptation network: adapt_tconv replaces encoder for z estimation.

    Inherits ActorCriticEncoder directly (base RL pipeline, NOT TDC chain).

    During Phase 2 supervised training:
        - adapt_tconv is trainable (L2 loss only)
        - _get_combined_obs() uses z_hat from adapt_tconv (not z from encoder)
        - z_hat is DETACHED before actor input: PPO gradient does NOT reach adapt_tconv
        - AdaptRunner recomputes z_hat independently for L2 loss gradient
        - evaluate() overridden: critic sees z via encoder (symmetric design)

    z_hat activation: matches Phase 1 encoder (via _activate_z).

    The frozen encoder remains available via compute_z_gt() for computing
    the supervision target during training.
    """

    def __init__(
        self,
        *args,
        proprio_history_len: int = 30,
        proprio_feature_dim: int = 8,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.adapt_tconv = ProprioAdaptTConv(
            input_dim=proprio_feature_dim,
            hidden_dim=32,
            output_dim=self.encoder_latent_dim,
            history_len=proprio_history_len,
        )

        self._proprio_hist_key = "proprio_hist"

        # Proprioception history normalizer (train mode: online updates, eval: frozen).
        self.hist_normalizer = EmpiricalNormalization(proprio_feature_dim)

    def compute_z_hat(self, obs: TensorDict) -> torch.Tensor:
        """Canonical z_hat computation: normalize -> adapt_tconv -> activate.

        Both _get_combined_obs() and AdaptRunner's L2 loss use this method,
        ensuring normalization is always applied (train and inference).

        Args:
            obs: Observation dict containing proprio_hist key.

        Returns:
            z_hat: Activated latent estimate. Shape: (N, encoder_latent_dim).
        """
        proprio_hist = obs[self._proprio_hist_key]
        # EmpiricalNormalization expects 2D; reshape 3D -> 2D -> 3D
        N, H, D = proprio_hist.shape
        flat = proprio_hist.reshape(N * H, D)
        flat_norm = self.hist_normalizer(flat)
        proprio_hist_norm = flat_norm.reshape(N, H, D)
        z_hat_raw = self.adapt_tconv(proprio_hist_norm)
        return self._activate_z(z_hat_raw)

    def _get_combined_obs(self, obs: TensorDict) -> torch.Tensor:
        """Actor obs: use z_hat from adaptation module instead of z from encoder.

        z_hat is DETACHED before actor input so PPO gradient does not
        interfere with L2 loss supervision. AdaptRunner recomputes z_hat
        independently for L2 gradient flow.
        """
        policy_obs = obs[self._policy_obs_key]
        z_hat = self.compute_z_hat(obs)
        return torch.cat([policy_obs, z_hat.detach()], dim=-1)

    def evaluate(self, obs: TensorDict, **_kwargs: Any) -> torch.Tensor:
        """Critic uses encoder z (symmetric design), not z_hat from adaptation."""
        policy_obs = obs[self._policy_obs_key]
        z = self._encode(obs[self._privileged_key])
        critic_obs = torch.cat([policy_obs, z], dim=-1)
        critic_obs = self.critic_obs_normalizer(critic_obs)  # type: ignore[operator]
        return self.critic(critic_obs)

    def compute_z_gt(self, obs: TensorDict) -> torch.Tensor:
        """Compute ground truth z from frozen Phase 1 encoder."""
        with torch.no_grad():
            return self._encode(obs[self._privileged_key])

    def get_adapt_parameters(self):
        """Return only adaptation module parameters (for optimizer)."""
        return self.adapt_tconv.parameters()

    def freeze_base(self):
        """Freeze all weights except adapt_tconv."""
        for name, param in self.named_parameters():
            if "adapt_tconv" not in name:
                param.requires_grad = False
