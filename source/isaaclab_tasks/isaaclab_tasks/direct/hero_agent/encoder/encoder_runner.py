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

For TDC environments, the runner wraps alg.act() to pass encoder z to the
environment via set_encoder_z() after each policy forward pass, avoiding
the need to copy-paste the upstream learn() method.

Usage:
    The runner is automatically selected by train.py when the policy class_name
    starts with "ActorCriticEncoder". No manual configuration is needed.
"""

from __future__ import annotations

import torch
from rsl_rl.runners import OnPolicyRunner

from ..utils.logging import log_encoder_metrics


class EncoderRunner(OnPolicyRunner):
    """OnPolicyRunner with encoder-specific metrics logging and TDC integration.

    Extends the base runner to:
        1. Log HORA Phase 1 encoder internal states (z latent, gradients, etc.)
        2. Pass encoder z to TDC environment for M_hat estimation

    The encoder metrics are logged via the reusable log_encoder_metrics()
    function from utils/logging.py, ensuring consistency with the current
    ActorCriticEncoder implementation (softplus activation).

    For TDC environments (HeroAgentTDCEnv), the runner automatically calls
    env.set_encoder_z(policy.last_z) after each policy forward pass to
    provide the inertia estimate to the TDC controller.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the encoder runner.

        Args:
            *args: Positional arguments passed to OnPolicyRunner.
            **kwargs: Keyword arguments passed to OnPolicyRunner.
        """
        super().__init__(*args, **kwargs)

        # Check if policy has an encoder attribute
        self._has_encoder = hasattr(self.alg.policy, "encoder")

        # Check if policy exposes last_z (for TDC integration)
        self._has_last_z = hasattr(self.alg.policy, "last_z")

        # Check if unwrapped env has set_encoder_z (TDC environment)
        self._unwrapped_env = self._get_unwrapped_env()
        self._is_tdc_env = hasattr(self._unwrapped_env, "set_encoder_z")

        if self._has_encoder:
            print("[EncoderRunner] Encoder detected. Encoder metrics logging enabled.")
        else:
            print("[EncoderRunner] No encoder detected. Using standard logging only.")

        if self._is_tdc_env and self._has_last_z:
            print("[EncoderRunner] TDC environment detected. Encoder z -> TDC integration enabled.")
            self._wrap_alg_act()

    def _wrap_alg_act(self) -> None:
        """Wrap self.alg.act() to pass encoder z to TDC env after each forward pass.

        This avoids overriding the entire learn() method (~125 lines) just to insert
        a single call after act(). The wrapped method is behaviorally identical:
        z is passed to the TDC environment after every policy forward pass.
        """
        original_act = self.alg.act

        def act_with_z_passing(obs: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            actions = original_act(obs, *args, **kwargs)
            self._pass_encoder_z_to_env()
            return actions

        self.alg.act = act_with_z_passing

    def _get_unwrapped_env(self):
        """Get the unwrapped Isaac Lab environment from the wrapper chain."""
        env = self.env
        while hasattr(env, "unwrapped"):
            inner = env.unwrapped
            if inner is env:
                break
            env = inner
        return env

    def _pass_encoder_z_to_env(self) -> None:
        """Pass encoder z from policy to TDC environment.

        This should be called after policy.act() to provide M_hat to TDC controller.
        """
        if self._is_tdc_env and self._has_last_z:
            last_z = self.alg.policy.last_z
            if last_z is not None:
                self._unwrapped_env.set_encoder_z(last_z)

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
