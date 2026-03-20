# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OnPolicyRunner with DORAEMON DR scheduling for constrained ALBC.

This module provides BaseRunner, a subclass of OnPolicyRunner that:
    - Triggers DORAEMON distribution updates each iteration via log()
    - Noise std floor: safety net preventing exploration collapse
    - Logs all metrics (DORAEMON) to TensorBoard/WandB

Usage:
    Registered as runner for Base and other non-encoder constrained ALBC envs.
    EncoderRunner inherits from this class to add encoder-specific metrics.
"""

from __future__ import annotations

import logging
import os

import torch
from rsl_rl.runners import OnPolicyRunner

from ..utils.logging import flush_metrics, unwrap_env

logger = logging.getLogger(__name__)


class BaseRunner(OnPolicyRunner):
    """OnPolicyRunner with DORAEMON DR scheduling.

    Provides DORAEMON integration: each iteration, log() calls doraemon.step()
    which updates the Beta DR distribution based on episode success statistics.

    EncoderRunner subclass adds encoder-specific metrics logging.
    """

    @property
    def _doraemon(self):
        """Return the DORAEMON scheduler from the unwrapped env, or None."""
        raw_env = unwrap_env(self.env)
        dora = getattr(raw_env, "_doraemon", None)
        return dora

    @property
    def _should_log(self) -> bool:
        """Whether logging is active (log_dir set and logs not disabled)."""
        return self.log_dir is not None and not self.disable_logs

    @staticmethod
    def _save_aux_state(path: str, name: str, state: dict) -> None:
        """Save auxiliary state dict alongside a model checkpoint."""
        aux_path = os.path.join(os.path.dirname(path), name)
        torch.save(state, aux_path)

    @staticmethod
    def _load_aux_state(path: str, name: str, device: str) -> dict | None:
        """Load auxiliary state dict from alongside a model checkpoint, or None."""
        aux_path = os.path.join(os.path.dirname(path), name)
        if os.path.exists(aux_path):
            return torch.load(aux_path, map_location=device, weights_only=False)
        return None

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

        # Penalty curriculum: linearly ramp penalty scale
        unwrap_env(self.env)._reward_manager.update_curriculum(iteration)

        # DORAEMON: update DR distribution based on episode statistics
        doraemon = self._doraemon
        if doraemon is not None:
            metrics = doraemon.step()
            if self._should_log:
                flush_metrics(
                    self.writer,
                    {f"DORAEMON/{k}": v for k, v in metrics.items()},
                    iteration,
                    self.logger_type,
                )

    def save(self, path: str, infos: dict | None = None) -> None:
        """Save model checkpoint and DORAEMON state.

        DORAEMON state is saved as a separate ``doraemon_state.pt`` file alongside
        the model checkpoint to avoid modifying the RSL-RL checkpoint format.
        """
        super().save(path, infos)
        doraemon = self._doraemon
        if doraemon is not None:
            self._save_aux_state(path, "doraemon_state.pt", doraemon.state_dict())

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        """Load model checkpoint and restore DORAEMON state if available.

        Looks for ``doraemon_state.pt`` in the same directory as the model
        checkpoint.  Missing file is silently ignored (backward compat with
        old checkpoints that lack DORAEMON state).
        """
        infos = super().load(path, load_optimizer, map_location)
        doraemon = self._doraemon
        if doraemon is not None:
            state = self._load_aux_state(path, "doraemon_state.pt", self.device)
            if state is not None:
                doraemon.load_state_dict(state)
                logger.info("[DORAEMON] Loaded distribution state from checkpoint")
        return infos

    def _apply_noise_floor(self) -> None:
        """Clamp action noise std to minimum for numerical safety.

        Floor prevents log_prob divergence when std approaches zero.
        No ceiling: reward gradient alone controls variance (cost gradient
        is detached from std in ConstraintTRPO, and entropy_coef=0).
        """
        min_std = 0.18
        if hasattr(self.alg.policy, "log_std"):
            if not hasattr(self, "_cached_min_log_std"):
                self._cached_min_log_std = torch.log(torch.tensor(min_std, device=self.device))
            self.alg.policy.log_std.data.clamp_(min=self._cached_min_log_std)
