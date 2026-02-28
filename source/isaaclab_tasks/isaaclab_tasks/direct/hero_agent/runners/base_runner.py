# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OnPolicyRunner with DORAEMON DR scheduling for Hero Agent.

This module provides BaseRunner, a subclass of OnPolicyRunner that:
    - Triggers DORAEMON distribution updates each iteration via log()
    - Noise std floor: safety net preventing exploration collapse
    - Logs all metrics (DORAEMON) to TensorBoard/WandB

Usage:
    Registered as runner for Base and other non-encoder Hero Agent envs.
    EncoderRunner inherits from this class to add encoder-specific metrics.
"""

from __future__ import annotations

import logging

import torch
from rsl_rl.runners import OnPolicyRunner

from ..utils.logging import unwrap_env

logger = logging.getLogger(__name__)


class BaseRunner(OnPolicyRunner):
    """OnPolicyRunner with DORAEMON DR scheduling.

    Provides DORAEMON integration: each iteration, log() calls doraemon.step()
    which updates the Beta DR distribution based on episode success statistics.

    EncoderRunner subclass adds encoder-specific metrics logging.
    """

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Reset environments before training so initial DR samples come from DORAEMON."""
        raw_env = unwrap_env(self.env)
        raw_env._reward_manager.set_max_iterations(num_learning_iterations)
        self.env.reset()
        super().learn(num_learning_iterations, init_at_random_ep_len)

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        """Extended log with DORAEMON update.

        Args:
            locs: Local variables from the learn() training loop.
            width: Terminal output width for formatting.
            pad: Padding for log formatting.
        """
        super().log(locs, width, pad)

        # Noise std floor: always active, prevents exploration collapse under DR
        self._apply_noise_floor()

        iteration = locs["it"]
        raw_env = unwrap_env(self.env)

        # Penalty curriculum: linearly ramp penalty scale
        raw_env._reward_manager.update_curriculum(iteration)

        # DORAEMON: update DR distribution based on episode statistics
        if hasattr(raw_env, "_doraemon") and raw_env._doraemon is not None:
            metrics = raw_env._doraemon.step()
            if self.log_dir is not None and not self.disable_logs:
                for key, value in metrics.items():
                    self.writer.add_scalar(f"DORAEMON/{key}", value, iteration)

    def _apply_noise_floor(self) -> None:
        """Clamp action noise std to minimum floor.

        Prevents exploration collapse under domain randomization.
        Always active regardless of other settings.
        """
        min_std = 0.1
        if hasattr(self.alg.policy, "log_std"):
            if not hasattr(self, "_cached_min_log_std"):
                self._cached_min_log_std = torch.log(torch.tensor(min_std, device=self.device))
            self.alg.policy.log_std.data.clamp_(min=self._cached_min_log_std)
