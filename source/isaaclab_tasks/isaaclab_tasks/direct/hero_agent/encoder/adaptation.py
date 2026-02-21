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
        -> low_dim_proj: Linear(32*3 -> output_dim)
        -> raw output (activation applied externally)

    ActorCriticEncoderTDCAdapt:
        Overrides _get_combined_obs() to use adapt_tconv(proprio_hist) instead
        of _encode(privileged). z_hat activation uses sigmoid scaling with
        per-dim min/max ranges for bounded positive output.

        The frozen encoder is still available via compute_z_gt() for
        supervised training.

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
        input_dim: int = 12,
        hidden_dim: int = 32,
        output_dim: int = 6,
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


class ActorCriticEncoderTDCAdapt(ActorCriticEncoderTDC):
    """Phase 2 / single-phase network: adaptation module replaces encoder for z estimation.

    During single-phase training:
        - adapt_tconv is trainable (aux MSE loss only)
        - _get_combined_obs() uses z_hat from adapt_tconv (not z from encoder)
        - z_hat is DETACHED before actor/critic: PPO gradient does NOT reach adapt_tconv
        - _last_z stores the non-detached z_hat for aux loss gradient
        - get_last_z() transparently returns z_hat for env M_hat extraction

    Gradient source for adapt_tconv:
        - Aux MSE loss only (z_hat vs z_true from privileged obs)
        - PPO gradient is blocked by detach to prevent interference

    z_hat activation: sigmoid scaling with per-dim [min, max] ranges.
    sigmoid(0) = 0.5 -> midpoint of each range at initialization.
    Unlike softplus, sigmoid naturally bounds output and avoids the
    vanishing gradient collapse that caused z_hat to stick at z_min.

    The frozen encoder remains available via compute_z_gt() for computing
    the supervision target during training.
    """

    # Default ranges for 3D decomposed output [m_A, I_roll, I_pitch]
    _DEFAULT_Z_HAT_RANGES = [(0.01, 0.5), (0.005, 0.3), (0.005, 0.3)]
    # Nominal physical parameter values for bias initialization
    # m_A ~ 0.08 (buoy added mass), I_roll ~ 0.04 (Ixx), I_pitch ~ 0.05 (Iyy)
    _DEFAULT_Z_HAT_NOMINAL = [0.08, 0.04, 0.05]

    def __init__(
        self,
        *args,
        proprio_history_len: int = 30,
        proprio_feature_dim: int = 12,
        z_hat_ranges: list[tuple[float, float]] | None = None,
        z_hat_nominal: list[float] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.adapt_tconv = ProprioAdaptTConv(
            input_dim=proprio_feature_dim,
            hidden_dim=32,
            output_dim=self.encoder_latent_dim,
            history_len=proprio_history_len,
        )

        # Sigmoid output ranges: z_hat = min + sigmoid(raw) * (max - min)
        ranges = z_hat_ranges if z_hat_ranges is not None else self._DEFAULT_Z_HAT_RANGES
        if len(ranges) != self.encoder_latent_dim:
            raise ValueError(
                f"z_hat_ranges length ({len(ranges)}) must match "
                f"encoder_latent_dim ({self.encoder_latent_dim})"
            )
        z_min = torch.tensor([r[0] for r in ranges], dtype=torch.float32)
        z_max = torch.tensor([r[1] for r in ranges], dtype=torch.float32)
        self.register_buffer("_z_hat_min", z_min)
        self.register_buffer("_z_hat_max", z_max)

        # Initialize low_dim_proj bias to logit of nominal values so that
        # initial z_hat ≈ nominal physical parameters (not range midpoint).
        # Weight scaled down so bias dominates initial output.
        nominal = z_hat_nominal if z_hat_nominal is not None else self._DEFAULT_Z_HAT_NOMINAL
        nominal_t = torch.tensor(nominal, dtype=torch.float32)
        # sigmoid^{-1}(p) = log(p / (1-p))  where p = (nominal - z_min) / (z_max - z_min)
        p = (nominal_t - z_min) / (z_max - z_min)
        p = p.clamp(0.01, 0.99)  # numerical safety
        logit_bias = torch.log(p / (1.0 - p))
        with torch.no_grad():
            self.adapt_tconv.low_dim_proj.bias.copy_(logit_bias)
            # Scale down weights so initial output ≈ bias (network learns deviations)
            self.adapt_tconv.low_dim_proj.weight.mul_(0.01)

        self._proprio_hist_key = "proprio_hist"
        self._last_z_hat: torch.Tensor | None = None

    def _get_combined_obs(self, obs: TensorDict) -> torch.Tensor:
        """Use z_hat from adaptation module instead of z from encoder.

        The adapt_tconv processes proprioception history to produce z_hat,
        which replaces the encoder output.

        Activation: sigmoid scaling with per-dim [min, max] ranges.
        z_hat = z_min + sigmoid(raw) * (z_max - z_min).
        At initialization, low_dim_proj bias is set to logit(nominal)
        so z_hat starts near nominal physical parameter values
        (not the range midpoint).

        z_hat is DETACHED before actor/critic input so PPO gradient does
        not interfere with aux loss supervision. _last_z stores the
        non-detached z_hat for aux MSE gradient to flow through.
        """
        policy_obs = obs[self._policy_obs_key]
        z_hat_raw = self.adapt_tconv(obs[self._proprio_hist_key])
        z_hat = self._z_hat_min + torch.sigmoid(z_hat_raw) * (self._z_hat_max - self._z_hat_min)
        self._last_z = z_hat  # Non-detached: aux loss gradient flows through here
        self._last_z_hat = z_hat
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
