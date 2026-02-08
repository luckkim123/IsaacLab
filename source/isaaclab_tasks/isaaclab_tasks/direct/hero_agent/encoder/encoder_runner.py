# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OnPolicyRunner with encoder-specific metrics logging for HORA Phase 1 training.

This module provides EncoderRunner, a subclass of OnPolicyRunner that adds
comprehensive logging of ActorCriticEncoder internal states to WandB/TensorBoard.

The runner detects if the policy has an encoder attribute and automatically
logs encoder-specific metrics (z latent statistics, gradient norms, etc.)
without requiring any configuration changes.

Usage:
    The runner is automatically selected by train.py when the policy class_name
    starts with "ActorCriticEncoder". No manual configuration is needed.
"""

from __future__ import annotations

from rsl_rl.runners import OnPolicyRunner

from ..utils.logging import log_encoder_metrics


class EncoderRunner(OnPolicyRunner):
    """OnPolicyRunner with encoder-specific metrics logging.

    Extends the base runner to log HORA Phase 1 encoder internal states
    (z latent, gradients, etc.).

    The encoder metrics are logged via the reusable log_encoder_metrics()
    function from utils/logging.py, ensuring consistency with the current
    ActorCriticEncoder implementation (softplus activation).
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the encoder runner.

        Args:
            *args: Positional arguments passed to OnPolicyRunner.
            **kwargs: Keyword arguments passed to OnPolicyRunner.
        """
        super().__init__(*args, **kwargs)

        self._has_encoder = hasattr(self.alg.policy, "encoder")

        if self._has_encoder:
            print("[EncoderRunner] Encoder detected. Encoder metrics logging enabled.")
        else:
            print("[EncoderRunner] No encoder detected. Using standard logging only.")

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        """Extended log method that adds encoder-specific metrics.

        Calls the parent log() method first to maintain all standard logging,
        then adds encoder metrics if an encoder is present.

        Args:
            locs: Local variables from the learn() training loop.
            width: Terminal output width for formatting.
            pad: Padding for log formatting.
        """
        # Call parent log method first (handles all standard logging)
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
