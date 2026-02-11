# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 2 adaptation network: temporal convolution replaces encoder for z estimation.

This module provides:
    - ProprioAdaptTConv: Temporal conv network (proprio history -> z_hat)
    - ActorCriticEncoderTDCAdapt: Full network with frozen base + trainable adapt

Architecture:
    ProprioAdaptTConv:
        Input: (N, H, D) proprioception history (D=12 for ALBC)
        -> channel_transform: per-timestep MLP (D -> 32 -> 32)
        -> temporal_aggregation: 3x Conv1d (H -> 3 time steps)
        -> low_dim_proj: Linear(32*3 -> 6)
        -> softplus + z_min: guarantees z > z_min (matching encoder output space)

    ActorCriticEncoderTDCAdapt:
        Overrides _get_combined_obs() to use adapt_tconv(proprio_hist) instead
        of _encode(privileged). The frozen encoder is still available via
        compute_z_gt() for supervised training.

Reference:
    HORA ProprioAdaptTConv (references/hora/hora/algo/models/models.py)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from .actor_critic_encoder import ActorCriticEncoderTDC

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

    The Conv1d architecture reduces temporal dimension:
        H=30 -> Conv(k=9,s=2) -> 11 -> Conv(k=5,s=1) -> 7 -> Conv(k=5,s=1) -> 3
    """

    KERNELS = [9, 5, 5]
    STRIDES = [2, 1, 1]

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 32,
        output_dim: int = 6,
        history_len: int = 30,
    ):
        super().__init__()

        self.channel_transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.temporal_aggregation = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, self.KERNELS[0], stride=self.STRIDES[0]),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, self.KERNELS[1], stride=self.STRIDES[1]),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, self.KERNELS[2], stride=self.STRIDES[2]),
            nn.ReLU(inplace=True),
        )

        final_time_steps = _compute_conv_output_len(history_len, self.KERNELS, self.STRIDES)
        self.low_dim_proj = nn.Linear(hidden_dim * final_time_steps, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (N, H, input_dim) -> (N, output_dim).

        Args:
            x: Proprioception history. Shape: (N, H, input_dim).

        Returns:
            Raw latent vector (before softplus). Shape: (N, output_dim).
        """
        x = self.channel_transform(x)  # (N, H, hidden_dim)
        x = x.permute(0, 2, 1)  # (N, hidden_dim, H) for Conv1d
        x = self.temporal_aggregation(x)  # (N, hidden_dim, T_final)
        return self.low_dim_proj(x.flatten(1))  # (N, output_dim)


class ActorCriticEncoderTDCAdapt(ActorCriticEncoderTDC):
    """Phase 2 network: adaptation module replaces encoder for z estimation.

    During Phase 2 training:
        - adapt_tconv is trainable (supervised L2 loss against frozen encoder z)
        - All other weights (encoder, actor, critic) are frozen
        - _get_combined_obs() uses z_hat from adapt_tconv (not z from encoder)
        - get_last_z() transparently returns z_hat for env M_hat extraction

    The frozen encoder remains available via compute_z_gt() for computing
    the supervision target during training.
    """

    def __init__(self, *args, proprio_history_len: int = 30, proprio_feature_dim: int = 12, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.adapt_tconv = ProprioAdaptTConv(
            input_dim=proprio_feature_dim,
            hidden_dim=32,
            output_dim=self.encoder_latent_dim,
            history_len=proprio_history_len,
        )

        self._proprio_hist_key = "proprio_hist"
        self._last_z_hat: torch.Tensor | None = None

    def _get_combined_obs(self, obs: TensorDict) -> torch.Tensor:
        """Use z_hat from adaptation module instead of z from encoder.

        The adapt_tconv processes proprioception history to produce z_hat,
        which replaces the encoder output. softplus + z_min maps the raw
        output to the same positive space as the Phase 1 encoder.
        """
        policy_obs = obs[self._policy_obs_key]
        z_hat_raw = self.adapt_tconv(obs[self._proprio_hist_key])
        z_hat = self._softplus_z(z_hat_raw)
        self._last_z = z_hat  # For env M_hat extraction via get_last_z()
        self._last_z_hat = z_hat  # For loss computation in runner
        return torch.cat([policy_obs, z_hat], dim=-1)

    def compute_z_gt(self, obs: TensorDict) -> torch.Tensor:
        """Compute ground truth z from frozen Phase 1 encoder.

        Args:
            obs: Observation dict containing privileged info.

        Returns:
            z_gt: Ground truth latent from encoder. Shape: (N, encoder_latent_dim).
        """
        with torch.no_grad():
            return self._encode(obs[self._privileged_key])

    def get_adapt_parameters(self):
        """Return only adaptation module parameters (for optimizer)."""
        return self.adapt_tconv.parameters()

    def freeze_base(self):
        """Freeze all weights except adapt_tconv.

        After calling this, only adapt_tconv.parameters() have requires_grad=True.
        """
        for name, param in self.named_parameters():
            if "adapt_tconv" not in name:
                param.requires_grad = False
