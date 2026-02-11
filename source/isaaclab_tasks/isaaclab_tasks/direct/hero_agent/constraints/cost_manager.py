# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cost Manager for constrained RL training.

Follows the RewardManager pattern (from mdp/rewards.py): computes per-step costs
for each constraint, tracks episode sums, and provides reset/logging support.

Unlike RewardManager, costs are NOT aggregated into a single scalar -- each
constraint maintains its own cost signal for separate GAE computation and
barrier loss evaluation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .config import ConstraintCfg

if TYPE_CHECKING:
    from ..constrained_encoder_tdc_env import HeroAgentConstrainedEncoderTDCEnv


class CostManager:
    """Manages K cost functions for constrained RL.

    Each constraint produces a per-env cost tensor. The manager tracks episode
    sums for logging and provides a combined cost tensor (num_envs, K) for
    the constrained PPO algorithm.
    """

    def __init__(
        self,
        constraints: list[ConstraintCfg],
        num_envs: int,
        device: str,
    ) -> None:
        self.num_envs = num_envs
        self.device = device
        self.num_constraints = len(constraints)

        self._names: list[str] = []
        self._cfgs: list[ConstraintCfg] = []

        for cfg in constraints:
            if cfg.cost_func is not None:
                self._names.append(cfg.name)
                self._cfgs.append(cfg)

        self.num_constraints = len(self._names)

        # Episode sum tracking per constraint
        self._episode_sums: dict[str, torch.Tensor] = {
            name: torch.zeros(num_envs, device=device) for name in self._names
        }

    @property
    def constraint_names(self) -> list[str]:
        """Names of active constraints."""
        return self._names

    @property
    def episode_sums(self) -> dict[str, torch.Tensor]:
        """Episode cost sums per constraint."""
        return self._episode_sums

    def compute(self, env: HeroAgentConstrainedEncoderTDCEnv) -> torch.Tensor:
        """Compute per-step costs for all constraints.

        Args:
            env: The constrained environment instance.

        Returns:
            Cost tensor of shape (num_envs, K) where K = num_constraints.
        """
        costs = torch.zeros(self.num_envs, self.num_constraints, device=self.device)

        for i, (name, cfg) in enumerate(zip(self._names, self._cfgs)):
            cost_val = cfg.cost_func(env, **cfg.params)
            costs[:, i] = cost_val
            self._episode_sums[name] += cost_val

        return costs

    def reset(self, env_ids: torch.Tensor) -> dict[str, float]:
        """Reset episode sums for specified environments.

        Args:
            env_ids: Environment indices to reset.

        Returns:
            Mean episode cost sums before reset (for logging).
        """
        sums_before = {}
        for name in self._names:
            sums_before[name] = self._episode_sums[name][env_ids].mean().item()
            self._episode_sums[name][env_ids] = 0.0
        return sums_before
