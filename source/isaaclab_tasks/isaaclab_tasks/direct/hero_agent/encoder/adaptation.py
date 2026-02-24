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
        Overrides _get_actor_obs() to use adapt_tconv(proprio_hist) instead
        of _encode(privileged). z_hat uses _activate_z() (tanh, matching Phase 1).
        _get_critic_obs() is inherited from parent (symmetric, uses encoder z).

        The frozen encoder is still available via compute_z_gt() for
        supervised training.

Reference:
    HORA ProprioAdaptTConv (references/hora/hora/algo/models/models.py)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

from .actor_critic_encoder import ActorCriticEncoder

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

    Conv1d kernel/stride presets by history_len:
        H=30: kernels=[9,5,5], strides=[2,1,1] -> 11 -> 7 -> 3
        H=15: kernels=[3,3,3], strides=[2,1,1] -> 7 -> 5 -> 3
    """

    # Default kernels/strides for H=30 (original HORA)
    DEFAULT_KERNELS = [9, 5, 5]
    DEFAULT_STRIDES = [2, 1, 1]

    # Presets for common history lengths
    _PRESETS: dict[int, tuple[list[int], list[int]]] = {
        30: ([9, 5, 5], [2, 1, 1]),  # -> 11 -> 7 -> 3
        15: ([3, 3, 3], [2, 1, 1]),  # -> 7 -> 5 -> 3
    }

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 32,
        output_dim: int = 13,
        history_len: int = 30,
        conv_kernels: list[int] | None = None,
        conv_strides: list[int] | None = None,
    ):
        super().__init__()

        # Auto-select kernels/strides from preset if not explicitly provided
        if conv_kernels is None or conv_strides is None:
            if history_len in self._PRESETS:
                kernels, strides = self._PRESETS[history_len]
            else:
                logger.warning(
                    "No conv preset for history_len=%d. Falling back to H=30 defaults "
                    "(kernels=%s, strides=%s). This may produce suboptimal temporal aggregation.",
                    history_len,
                    self.DEFAULT_KERNELS,
                    self.DEFAULT_STRIDES,
                )
                kernels, strides = self.DEFAULT_KERNELS, self.DEFAULT_STRIDES
        else:
            kernels, strides = conv_kernels, conv_strides

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
        - _get_actor_obs() uses z_hat from adapt_tconv (not z from encoder)
        - z_hat is DETACHED before actor input: PPO gradient does NOT reach adapt_tconv
        - AdaptRunner recomputes z_hat independently for L2 loss gradient
        - _get_critic_obs() inherited: critic sees z (symmetric)

    z_hat activation: matches Phase 1 encoder (tanh by default).

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

    def _get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        """Actor obs: use z_hat from adaptation module instead of z from encoder.

        z_hat activation matches Phase 1 encoder (tanh via _activate_z).
        z_hat is DETACHED before actor input so PPO gradient does not
        interfere with L2 loss supervision. AdaptRunner recomputes z_hat
        independently for L2 gradient flow.

        _get_critic_obs() is NOT overridden -- inherited from parent,
        critic sees z via encoder (symmetric design).
        """
        policy_obs = obs[self._policy_obs_key]
        z_hat_raw = self.adapt_tconv(obs[self._proprio_hist_key])
        z_hat = self._activate_z(z_hat_raw)
        return torch.cat([policy_obs, z_hat.detach()], dim=-1)

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
