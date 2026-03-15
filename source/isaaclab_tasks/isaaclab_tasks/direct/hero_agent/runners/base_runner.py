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
import os

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
        if hasattr(self.alg, "set_max_iterations"):
            self.alg.set_max_iterations(num_learning_iterations)
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

    def save(self, path: str, infos: dict | None = None) -> None:
        """Save model checkpoint and DORAEMON state.

        DORAEMON state is saved as a separate ``doraemon_state.pt`` file alongside
        the model checkpoint to avoid modifying the RSL-RL checkpoint format.
        """
        super().save(path, infos)
        raw_env = unwrap_env(self.env)
        if hasattr(raw_env, "_doraemon") and raw_env._doraemon is not None:
            doraemon_path = os.path.join(os.path.dirname(path), "doraemon_state.pt")
            torch.save(raw_env._doraemon.state_dict(), doraemon_path)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        """Load model checkpoint and restore DORAEMON state if available.

        Looks for ``doraemon_state.pt`` in the same directory as the model
        checkpoint.  Missing file is silently ignored (backward compat with
        old checkpoints that lack DORAEMON state).
        """
        infos = super().load(path, load_optimizer, map_location)
        raw_env = unwrap_env(self.env)
        if hasattr(raw_env, "_doraemon") and raw_env._doraemon is not None:
            doraemon_path = os.path.join(os.path.dirname(path), "doraemon_state.pt")
            if os.path.exists(doraemon_path):
                state = torch.load(doraemon_path, map_location=self.device, weights_only=False)
                raw_env._doraemon.load_state_dict(state)
                logger.info("[DORAEMON] Loaded distribution state from %s", doraemon_path)
        return infos

    def _apply_noise_floor(self) -> None:
        """Clamp action noise std to [min, max] range.

        Floor prevents exploration collapse under domain randomization.
        Ceiling prevents entropy explosion when constraint pressure is weak.
        Always active regardless of other settings.
        """
        min_std = 0.25
        max_std = 2.0
        if hasattr(self.alg.policy, "log_std"):
            if not hasattr(self, "_cached_min_log_std"):
                self._cached_min_log_std = torch.log(torch.tensor(min_std, device=self.device))
                self._cached_max_log_std = torch.log(torch.tensor(max_std, device=self.device))
            self.alg.policy.log_std.data.clamp_(min=self._cached_min_log_std, max=self._cached_max_log_std)
