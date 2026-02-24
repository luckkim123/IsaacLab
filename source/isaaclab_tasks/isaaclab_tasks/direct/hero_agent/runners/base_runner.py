# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OnPolicyRunner with DORAEMON DR scheduling and adaptive entropy for Hero Agent.

This module provides BaseRunner, a subclass of OnPolicyRunner that:
    - Triggers DORAEMON distribution updates each iteration via log()
    - Adaptive entropy: dual EMA reward tracking + noise-std reactive (axPPO-inspired)
    - Noise std floor: safety net preventing exploration collapse
    - Logs all metrics (DORAEMON, entropy) to TensorBoard/WandB

Usage:
    Registered as runner for TDE-Base and other non-encoder Hero Agent envs.
    EncoderRunner inherits from this class to add encoder-specific metrics.
"""

from __future__ import annotations

import logging
import statistics

import torch
from rsl_rl.runners import OnPolicyRunner

from ..utils.logging import unwrap_env

logger = logging.getLogger(__name__)


class BaseRunner(OnPolicyRunner):
    """OnPolicyRunner with DORAEMON DR scheduling and adaptive entropy.

    Provides DORAEMON integration: each iteration, log() calls doraemon.step()
    which updates the Beta DR distribution based on episode success statistics.
    Also provides adaptive entropy (dual EMA + noise-std reactive) to prevent
    exploration collapse under domain randomization.

    EncoderRunner subclass adds encoder-specific metrics logging.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._setup_adaptive_entropy()

    def _setup_adaptive_entropy(self) -> None:
        """Initialize adaptive entropy from runner config (cfg dict)."""
        self._adaptive_entropy_cfg = {
            "enable": self.cfg.get("adaptive_entropy", False),
            "base": self.cfg.get("entropy_base", self.alg.entropy_coef),
            "scale": self.cfg.get("entropy_scale", 15.0),
            "min": self.cfg.get("entropy_min", 0.001),
            "max": self.cfg.get("entropy_max", 0.05),
            "fast_alpha": self.cfg.get("entropy_fast_alpha", 0.1),
            "slow_alpha": self.cfg.get("entropy_slow_alpha", 0.01),
            "std_target": self.cfg.get("entropy_std_target", 0.5),
        }
        self._reward_ema_fast: float | None = None
        self._reward_ema_slow: float | None = None
        self._entropy_warmup_iters = 50

        if self._adaptive_entropy_cfg["enable"]:
            logger.info(
                "[BaseRunner] Adaptive entropy enabled: base=%.4f, scale=%.1f, range=[%.4f, %.4f]",
                self._adaptive_entropy_cfg["base"],
                self._adaptive_entropy_cfg["scale"],
                self._adaptive_entropy_cfg["min"],
                self._adaptive_entropy_cfg["max"],
            )

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Reset environments before training so initial DR samples come from DORAEMON."""
        self.env.reset()
        super().learn(num_learning_iterations, init_at_random_ep_len)

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        """Extended log with DORAEMON update and adaptive entropy.

        Args:
            locs: Local variables from the learn() training loop.
            width: Terminal output width for formatting.
            pad: Padding for log formatting.
        """
        super().log(locs, width, pad)

        # Noise std floor: always active, prevents exploration collapse under DR
        self._apply_noise_floor(locs)

        # Adaptive entropy: adjust entropy_coef based on reward trajectory
        if self._adaptive_entropy_cfg["enable"]:
            self._update_adaptive_entropy(locs)

        iteration = locs["it"]
        raw_env = unwrap_env(self.env)

        # Sigma annealing: tighten tracking sigma over training
        if hasattr(raw_env, "_reward_manager"):
            new_sigma = raw_env._reward_manager.update_sigma(iteration, raw_env.cfg.reward)
            if new_sigma is not None and self.log_dir is not None and not self.disable_logs:
                self.writer.add_scalar("Reward/tracking_sigma", new_sigma, iteration)

        # DORAEMON: update DR distribution based on episode statistics
        if hasattr(raw_env, "_doraemon") and raw_env._doraemon is not None:
            metrics = raw_env._doraemon.step()
            if self.log_dir is not None and not self.disable_logs:
                for key, value in metrics.items():
                    self.writer.add_scalar(f"DORAEMON/{key}", value, iteration)

    def _apply_noise_floor(self, locs: dict) -> None:
        """Clamp action noise std to minimum floor, independent of adaptive entropy.

        Prevents exploration collapse under domain randomization.
        Always active regardless of adaptive_entropy setting.
        """
        min_std = 0.1
        if hasattr(self.alg.policy, "log_std"):
            min_log_std = torch.log(torch.tensor(min_std, device=self.device))
            self.alg.policy.log_std.data.clamp_(min=min_log_std)

        iteration = locs["it"]
        if self.log_dir is not None and not self.disable_logs:
            current_std = self.alg.policy.action_std.mean().item()
            self.writer.add_scalar("Entropy/noise_std_floor_active", float(current_std <= min_std + 1e-4), iteration)

    def _update_adaptive_entropy(self, locs: dict) -> None:
        """Update entropy_coef using combined noise-std and reward-reactive signals.

        Two independent signals, max wins:
            1. Noise-std reactive (PROACTIVE): When mean action std drops below
               std_target, entropy boost grows proportionally to resist collapse.
            2. Reward-reactive (REACTIVE): When fast EMA drops below slow EMA,
               entropy increases to escape local optima during DR curriculum ramp.

        Args:
            locs: Local variables from the training loop.
        """
        cfg = self._adaptive_entropy_cfg

        # Signal 1: Noise-std reactive
        current_std = self.alg.policy.action_std.mean().item()
        std_target = cfg["std_target"]
        std_boost = max(0.0, (std_target - current_std) / std_target) if std_target > 0 else 0.0

        # Signal 2: Reward-reactive (dual EMA)
        rewbuffer = locs.get("rewbuffer", [])
        drop = 0.0
        if len(rewbuffer) > 0:
            mean_reward = statistics.mean(rewbuffer)

            if self._reward_ema_fast is None:
                self._reward_ema_fast = mean_reward
                self._reward_ema_slow = mean_reward
            else:
                ema_fast = (1 - cfg["fast_alpha"]) * self._reward_ema_fast + cfg["fast_alpha"] * mean_reward
                ema_slow = (1 - cfg["slow_alpha"]) * self._reward_ema_slow + cfg["slow_alpha"] * mean_reward
                self._reward_ema_fast = ema_fast
                self._reward_ema_slow = ema_slow

                if locs["it"] >= self._entropy_warmup_iters:
                    denominator = max(abs(ema_slow), 1e-6)
                    drop = max(0.0, (ema_slow - ema_fast) / denominator)

        # Combined: stronger signal wins
        boost = max(std_boost, drop)
        new_coef = cfg["base"] * (1.0 + cfg["scale"] * boost)
        self.alg.entropy_coef = max(cfg["min"], min(cfg["max"], new_coef))

        # Log all signals
        iteration = locs["it"]
        if self.log_dir is not None and not self.disable_logs:
            self.writer.add_scalar("Entropy/entropy_coef", self.alg.entropy_coef, iteration)
            self.writer.add_scalar("Entropy/std_boost", std_boost, iteration)
            self.writer.add_scalar("Entropy/reward_drop", drop, iteration)
            self.writer.add_scalar("Entropy/boost", boost, iteration)
            if self._reward_ema_fast is not None:
                self.writer.add_scalar("Entropy/reward_ema_fast", self._reward_ema_fast, iteration)
                self.writer.add_scalar("Entropy/reward_ema_slow", self._reward_ema_slow, iteration)
