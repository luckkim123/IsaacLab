# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OnPolicyRunner with training enhancements shared across all Hero Agent envs.

This module provides BaseRunner, a subclass of OnPolicyRunner that adds:
    - DR curriculum: apply start values before first rollout, ramp each iteration
    - Reward curriculum: sigma/weight annealing synced with DR progress
    - Adaptive entropy: dual EMA reward tracking + noise-std reactive (axPPO-inspired)
    - Noise std floor: safety net preventing exploration collapse

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
    """OnPolicyRunner with DR/reward curriculum, adaptive entropy, and noise floor.

    Provides all training enhancements that were previously only available in
    EncoderRunner. Encoder-specific metrics logging is in the EncoderRunner subclass.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Adaptive entropy: dual EMA reward tracking (axPPO-inspired).
        self._adaptive_entropy_cfg = {
            "enable": self.cfg.get("adaptive_entropy", True),
            "base": self.cfg.get("entropy_base", self.alg.entropy_coef),
            "scale": self.cfg.get("entropy_scale", 5.0),
            "min": self.cfg.get("entropy_min", 0.001),
            "max": self.cfg.get("entropy_max", 0.05),
            "fast_alpha": self.cfg.get("entropy_fast_alpha", 0.1),
            "slow_alpha": self.cfg.get("entropy_slow_alpha", 0.01),
            "std_target": self.cfg.get("entropy_std_target", 0.7),
        }
        self._reward_ema_fast: float | None = None
        self._reward_ema_slow: float | None = None
        self._entropy_warmup_iters = 10

        if self._adaptive_entropy_cfg["enable"]:
            logger.info(
                "[BaseRunner] Adaptive entropy enabled: base=%.4f, scale=%.1f, range=[%.4f, %.4f]",
                self._adaptive_entropy_cfg["base"],
                self._adaptive_entropy_cfg["scale"],
                self._adaptive_entropy_cfg["min"],
                self._adaptive_entropy_cfg["max"],
            )

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Apply DR curriculum start values before the first rollout.

        Two problems solved:
            1. OnPolicyRunner.learn() updates curriculum inside log(), which runs AFTER
               each iteration's rollout. Without this override, the first iteration
               runs with full DR ranges.
            2. The initial env reset (during env.__init__) samples DR from the full
               config before curriculum exists. We force a reset here so all envs
               re-sample from the curriculum start values.
        """
        raw_env = unwrap_env(self.env)
        if hasattr(raw_env, "update_dr_curriculum"):
            raw_env.update_dr_curriculum(0)
            self.env.reset()
        super().learn(num_learning_iterations, init_at_random_ep_len)

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        """Extended log with curriculum updates and adaptive entropy.

        Args:
            locs: Local variables from the learn() training loop.
            width: Terminal output width for formatting.
            pad: Padding for log formatting.
        """
        super().log(locs, width, pad)

        # Adaptive entropy: adjust entropy_coef based on reward trajectory
        if self._adaptive_entropy_cfg["enable"]:
            self._update_adaptive_entropy(locs)

        iteration = locs["it"]
        raw_env = unwrap_env(self.env)

        # Update reward curriculum
        if hasattr(raw_env, "_reward_manager") and hasattr(raw_env.cfg, "reward"):
            end_iter = raw_env.cfg.reward.curriculum_end_iter
            if end_iter is None and hasattr(raw_env.cfg, "dr_curriculum"):
                end_iter = raw_env.cfg.dr_curriculum.end_iter
            raw_env._reward_manager.update_curriculum(iteration, end_iter or 500)

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

        # Direct noise_std floor: prevent exploration collapse during DR ramp.
        min_std = 0.25
        if hasattr(self.alg.policy, "log_std"):
            min_log_std = torch.log(torch.tensor(min_std, device=self.device))
            self.alg.policy.log_std.data.clamp_(min=min_log_std)

        # Log all signals
        iteration = locs["it"]
        if self.log_dir is not None and not self.disable_logs:
            self.writer.add_scalar("Entropy/entropy_coef", self.alg.entropy_coef, iteration)
            self.writer.add_scalar("Entropy/std_boost", std_boost, iteration)
            self.writer.add_scalar("Entropy/reward_drop", drop, iteration)
            self.writer.add_scalar("Entropy/boost", boost, iteration)
            self.writer.add_scalar("Entropy/noise_std_floor_active", float(current_std < min_std), iteration)
            if self._reward_ema_fast is not None:
                self.writer.add_scalar("Entropy/reward_ema_fast", self._reward_ema_fast, iteration)
                self.writer.add_scalar("Entropy/reward_ema_slow", self._reward_ema_slow, iteration)
