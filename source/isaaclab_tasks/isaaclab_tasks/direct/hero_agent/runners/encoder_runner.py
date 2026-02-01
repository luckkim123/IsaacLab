# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom OnPolicyRunner with encoder-specific WandB logging for Hero Agent ALBC.

This module provides HeroAgentEncoderRunner, a subclass of OnPolicyRunner that
adds comprehensive logging of ActorCriticEncoder internal states to WandB.
"""

from __future__ import annotations

import torch
from rsl_rl.runners import OnPolicyRunner


class HeroAgentEncoderRunner(OnPolicyRunner):
    """OnPolicyRunner with encoder-specific metrics logging for HORA Phase 1 training.

    Extends the base runner to log:
    - Latent z statistics (per-dimension mean/std, range, distribution)
    - Raw encoder output statistics (pre-sigmoid, saturation fraction)
    - Encoder gradient norms
    - Compression quality (variance ratio)
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Check if policy has an encoder
        self._has_encoder = hasattr(self.alg.policy, "encoder")
        if self._has_encoder:
            print("[HeroAgentEncoderRunner] Encoder detected. Encoder metrics logging enabled.")
        else:
            print("[HeroAgentEncoderRunner] No encoder detected. Using standard logging only.")

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        """Extended log method that adds encoder-specific metrics.

        Args:
            locs: Local variables from the training loop
            width: Terminal output width
            pad: Padding for log formatting
        """
        # Call parent log method first
        super().log(locs, width, pad)

        # Log encoder metrics if encoder exists and logging is enabled
        if self._has_encoder and self.log_dir is not None and not self.disable_logs:
            self._log_encoder_metrics(locs["it"])

    def _log_encoder_metrics(self, iteration: int) -> None:
        """Log encoder-specific metrics to WandB/TensorBoard.

        Metrics logged:
        - Encoder/z_dim{i}_mean: Per-dimension latent mean
        - Encoder/z_dim{i}_std: Per-dimension latent std
        - Encoder/z_min, z_max, z_range: Latent range statistics
        - Encoder/raw_mean, raw_std: Pre-sigmoid encoder output
        - Encoder/raw_abs_max: Maximum absolute raw value (saturation indicator)
        - Encoder/saturation_fraction: Fraction of saturated outputs
        - Encoder/grad_norm: L2 norm of encoder gradients
        - Encoder/variance_ratio: Compression quality metric

        Args:
            iteration: Current training iteration
        """
        policy = self.alg.policy
        device = self.device

        # Get current observations from environment
        with torch.no_grad():
            obs = self.env.get_observations().to(device)

            # Extract privileged observations
            privileged_key = policy._privileged_key
            privileged = obs[privileged_key]

            # Compute raw encoder output (pre-sigmoid)
            raw = policy.encoder(privileged)

            # Compute z (post-sigmoid, scaled)
            z = policy.z_min + (policy.z_max - policy.z_min) * torch.sigmoid(raw)

            # --- Z Latent Statistics ---
            z_mean = z.mean(dim=0)  # Per-dimension mean
            z_std = z.std(dim=0)    # Per-dimension std

            for i in range(z.shape[-1]):
                self.writer.add_scalar(f"Encoder/z_dim{i}_mean", z_mean[i].item(), iteration)
                self.writer.add_scalar(f"Encoder/z_dim{i}_std", z_std[i].item(), iteration)

            # Global z range statistics
            z_min_val = z.min().item()
            z_max_val = z.max().item()
            z_range = z_max_val - z_min_val

            self.writer.add_scalar("Encoder/z_min", z_min_val, iteration)
            self.writer.add_scalar("Encoder/z_max", z_max_val, iteration)
            self.writer.add_scalar("Encoder/z_range", z_range, iteration)

            # --- Raw Output Statistics (pre-sigmoid) ---
            raw_mean = raw.mean().item()
            raw_std = raw.std().item()
            raw_abs_max = raw.abs().max().item()

            self.writer.add_scalar("Encoder/raw_mean", raw_mean, iteration)
            self.writer.add_scalar("Encoder/raw_std", raw_std, iteration)
            self.writer.add_scalar("Encoder/raw_abs_max", raw_abs_max, iteration)

            # Saturation fraction: |raw| > 4.0 means sigmoid is ~0.98 or ~0.02
            saturation_threshold = 4.0
            saturation_fraction = (raw.abs() > saturation_threshold).float().mean().item()
            self.writer.add_scalar("Encoder/saturation_fraction", saturation_fraction, iteration)

            # --- Variance Ratio (compression quality) ---
            # Higher ratio means encoder preserves more information variance
            privileged_var = privileged.var().item() + 1e-8  # Avoid division by zero
            z_var = z.var().item()
            variance_ratio = z_var / privileged_var
            self.writer.add_scalar("Encoder/variance_ratio", variance_ratio, iteration)

            # --- WandB Histograms (only for wandb logger) ---
            if self.logger_type == "wandb":
                try:
                    import wandb
                    for i in range(z.shape[-1]):
                        wandb.log({f"Encoder/z_dim{i}_dist": wandb.Histogram(z[:, i].cpu().numpy())}, step=iteration)
                except ImportError:
                    pass  # wandb not available

        # --- Gradient Norm (computed separately, needs grad context) ---
        # Only compute if encoder has parameters with gradients
        encoder_params = list(policy.encoder.parameters())
        if encoder_params and encoder_params[0].grad is not None:
            grad_norm = 0.0
            for p in encoder_params:
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            self.writer.add_scalar("Encoder/grad_norm", grad_norm, iteration)
