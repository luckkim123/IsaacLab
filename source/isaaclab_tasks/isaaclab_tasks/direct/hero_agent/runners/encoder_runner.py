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

import logging

from rsl_rl.runners import OnPolicyRunner

from ..utils.logging import connect_encoder_to_env, log_encoder_metrics, log_encoder_tdc_metrics, unwrap_env

logger = logging.getLogger(__name__)


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
            logger.info("[EncoderRunner] Encoder detected. Encoder metrics logging enabled.")
        else:
            logger.info("[EncoderRunner] No encoder detected. Using standard logging only.")

        # Wire up encoder policy to env for TDC M_hat extraction
        if self._has_encoder:
            connect_encoder_to_env(self.env, self.alg.policy, "EncoderRunner")

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

        # Update reward curriculum after logging so logged data matches the
        # curriculum that produced it (avoids 1-interval offset)
        iteration = locs["it"]
        raw_env = unwrap_env(self.env)
        if hasattr(raw_env, "_reward_manager") and hasattr(raw_env.cfg, "reward"):
            raw_env._reward_manager.update_curriculum(iteration, raw_env.cfg.reward.curriculum_end_iter)

        # Update DR curriculum (ramp perturbation/inertia/mass ranges)
        if hasattr(raw_env, "update_dr_curriculum"):
            raw_env.update_dr_curriculum(iteration)

        # Log DR curriculum progress
        if (
            hasattr(raw_env, "_dr_curriculum_cfg")
            and raw_env._dr_curriculum_cfg is not None
            and self.log_dir is not None
            and not self.disable_logs
        ):
            dr_cur = raw_env._dr_curriculum_cfg
            progress = min(1.0, iteration / max(1, dr_cur.end_iter))
            rand = raw_env.cfg.randomization
            self.writer.add_scalar("DR_Curriculum/progress", progress, iteration)
            self.writer.add_scalar("DR_Curriculum/perturbation_force_max", rand.perturbation_force_range[1], iteration)
            self.writer.add_scalar("DR_Curriculum/inertia_scale_hi", rand.inertia_scale[1], iteration)
            self.writer.add_scalar("DR_Curriculum/payload_mass_max", rand.payload_mass_range[1], iteration)
            self.writer.add_scalar("DR_Curriculum/cog_offset_z_max", rand.cog_offset_z[1], iteration)

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
