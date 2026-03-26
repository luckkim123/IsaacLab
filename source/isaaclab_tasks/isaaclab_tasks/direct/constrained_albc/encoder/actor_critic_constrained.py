# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standard ActorCritic + cost critic head (no encoder).

For ablation testing: barrier constraints without encoder complexity.
"""

from __future__ import annotations

from typing import Any

import torch
from rsl_rl.modules import ActorCritic
from rsl_rl.networks import MLP
from tensordict import TensorDict


class ActorCriticConstrained(ActorCritic):
    """ActorCritic with an additional cost value head for constraint RL."""

    def __init__(self, *args: Any, num_constraints: int = 4, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.num_constraints = num_constraints

        # Cost critic shares input with reward critic (same obs groups)
        num_critic_obs = 0
        for obs_group in self.obs_groups["critic"]:
            # Infer from existing critic input dim
            pass
        # Get dim from the critic's first layer
        num_critic_obs = self.critic[0].in_features

        self.cost_critic = MLP(num_critic_obs, num_constraints, [256, 128, 64], "elu")

    def evaluate_costs(self, obs: TensorDict) -> torch.Tensor:
        """Evaluate per-constraint cost values."""
        critic_obs = self.get_critic_obs(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        return self.cost_critic(critic_obs)
