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

from ..utils.logging import connect_encoder_to_env, log_encoder_metrics, log_encoder_tdc_metrics
from .base_runner import BaseRunner

logger = logging.getLogger(__name__)


class EncoderRunner(BaseRunner):
    """BaseRunner with encoder-specific metrics logging.

    Inherits all training enhancements from BaseRunner (adaptive entropy,
    noise floor, DR/reward curriculum). Adds HORA Phase 1 encoder metrics.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._has_encoder = hasattr(self.alg.policy, "encoder")

        if self._has_encoder:
            logger.info("[EncoderRunner] Encoder detected. Encoder metrics logging enabled.")
        else:
            logger.info("[EncoderRunner] No encoder detected. Using standard logging only.")

        # Wire up encoder policy to env for TDC M_hat extraction
        if self._has_encoder:
            connect_encoder_to_env(self.env, self.alg.policy, "EncoderRunner")

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        """Extended log method that adds encoder-specific metrics.

        Calls BaseRunner.log() first (handles curriculum + adaptive entropy),
        then adds encoder metrics if an encoder is present.

        Args:
            locs: Local variables from the learn() training loop.
            width: Terminal output width for formatting.
            pad: Padding for log formatting.
        """
        super().log(locs, width, pad)

        # Log encoder metrics if encoder exists and logging is enabled
        if self._has_encoder and self.log_dir is not None and not self.disable_logs:
            log_encoder_metrics(
                writer=self.writer,
                policy=self.alg.policy,
                env=self.env,
                iteration=locs["it"],
                device=self.device,
                logger_type=self.logger_type,
            )
            # Log TDC-specific metrics (M_hat, adaptive gains) if env has TDC
            log_encoder_tdc_metrics(
                writer=self.writer,
                env=self.env,
                iteration=locs["it"],
                logger_type=self.logger_type,
            )
