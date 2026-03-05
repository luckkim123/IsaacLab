# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""BaseRunner with encoder-specific metrics logging for HORA Phase 1 training.

This module provides EncoderRunner, a subclass of BaseRunner that adds
comprehensive logging of ActorCriticEncoder internal states to WandB/TensorBoard.

The runner detects if the policy has an encoder attribute and automatically
logs encoder-specific metrics (z latent statistics, gradient norms, etc.)
without requiring any configuration changes.

Usage:
    The runner is automatically selected by train.py when the policy class_name
    starts with "ActorCriticEncoder". No manual configuration is needed.
"""

from __future__ import annotations

import logging
import math

from ..utils.logging import connect_encoder_to_env, log_encoder_metrics, log_tdc_controller_metrics
from .base_runner import BaseRunner

logger = logging.getLogger(__name__)


class EncoderRunner(BaseRunner):
    """BaseRunner with encoder-specific metrics logging and encoder LR scheduling.

    Inherits DORAEMON DR scheduling from BaseRunner.
    Adds HORA Phase 1 encoder metrics and cosine LR decay for the encoder.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._has_encoder = hasattr(self.alg.policy, "encoder")

        if self._has_encoder:
            logger.info("[EncoderRunner] Encoder detected. Encoder metrics logging enabled.")
            connect_encoder_to_env(self.env, self.alg.policy, "EncoderRunner")
        else:
            logger.info("[EncoderRunner] No encoder detected. Using standard logging only.")

        # Encoder LR schedule config (from runner cfg)
        self._enc_warmup_frac = self.cfg.get("encoder_lr_warmup_frac", 0.2)
        self._enc_lr_min_ratio = self.cfg.get("encoder_lr_min_ratio", 0.1)

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        """Extended log method that adds encoder-specific metrics.

        Calls BaseRunner.log() first (handles DORAEMON DR update),
        then adds encoder metrics if an encoder is present.

        Args:
            locs: Local variables from the learn() training loop.
            width: Terminal output width for formatting.
            pad: Padding for log formatting.
        """
        super().log(locs, width, pad)

        iteration = locs["it"]

        # Encoder LR cosine decay after warmup
        if self._has_encoder and hasattr(self.alg, "_has_encoder_params") and self.alg._has_encoder_params:
            self._update_encoder_lr(iteration, locs["tot_iter"])

        # Log encoder metrics if encoder exists and logging is enabled
        if self._has_encoder and self.log_dir is not None and not self.disable_logs:
            log_encoder_metrics(
                writer=self.writer,
                policy=self.alg.policy,
                env=self.env,
                iteration=iteration,
                device=self.device,
                logger_type=self.logger_type,
            )
            # Log TDC-specific metrics (M_hat, Kp/Kd) if env has TDC
            log_tdc_controller_metrics(
                writer=self.writer,
                env=self.env,
                iteration=iteration,
                logger_type=self.logger_type,
            )

    def _update_encoder_lr(self, iteration: int, total_iterations: int) -> None:
        """Apply cosine decay to encoder LR after warmup period.

        Schedule:
            - Warmup (0 ~ warmup_frac * total): encoder LR stays at initial value
            - Decay (warmup ~ total): cosine anneal to initial * min_ratio
        """
        warmup_end = int(self._enc_warmup_frac * total_iterations)
        initial_lr = self.alg.encoder_lr

        if iteration <= warmup_end:
            enc_lr = initial_lr
        else:
            progress = (iteration - warmup_end) / max(total_iterations - warmup_end, 1)
            progress = min(progress, 1.0)
            enc_lr = initial_lr * (
                self._enc_lr_min_ratio + (1 - self._enc_lr_min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
            )

        self.alg.optimizer.param_groups[1]["lr"] = enc_lr

        # Log encoder LR
        if self.log_dir is not None and not self.disable_logs:
            self.writer.add_scalar("Loss/encoder_lr", enc_lr, iteration)
