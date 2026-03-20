# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ActorCriticEncoder with multi-head cost value function for constrained RL (IPO).

Extends ActorCriticEncoder with a cost critic head that predicts per-constraint
cost values V_C_k(s) for K constraints. The cost critic uses the same asymmetric
input path as the reward critic: cat([policy_obs, hist_flat, privileged]).

Architecture:
    Encoder:     cat([policy_obs, hist_flat, privileged]) -> MLP -> tanh -> z (13D)
    Actor:       cat([policy_obs, hist_flat, z]) -> MLP -> actions
    Critic:      cat([policy_obs, hist_flat, privileged]) -> MLP -> value (1D)
    Cost Critic: cat([policy_obs, hist_flat, privileged]) -> MLP -> cost values (K)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from rsl_rl.networks import MLP

from .actor_critic_encoder import ActorCriticEncoder

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from tensordict import TensorDict


class ActorCriticEncoderConstrained(ActorCriticEncoder):
    """ActorCriticEncoder extended with cost value heads for IPO constraints.

    The cost critic outputs K values (one per constraint), used by
    ConstraintTRPO for cost GAE and barrier loss computation.
    """

    def __init__(
        self,
        *args: Any,
        num_constraints: int = 3,
        cost_critic_hidden_dims: list[int] | tuple[int, ...] = (256, 128, 64),
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.num_constraints = num_constraints

        # Cost critic: same input dimension as reward critic (asymmetric)
        num_critic_obs = self.num_critic_obs
        self.cost_critic = MLP(
            num_critic_obs,
            num_constraints,
            list(cost_critic_hidden_dims),
            "elu",
        )
        logger.info(
            "Cost critic MLP (history-asymmetric, %dD input, %d outputs): %s",
            num_critic_obs,
            num_constraints,
            self.cost_critic,
        )

    def evaluate_costs(self, obs: TensorDict) -> torch.Tensor:
        """Evaluate cost value function for all K constraints.

        Uses the same asymmetric critic input: cat([policy_obs, hist_flat, privileged]).

        Returns:
            Cost values (softplus activated). Shape: (batch, K).
        """
        critic_obs = self.critic_obs_normalizer(self._get_critic_obs(obs))  # type: ignore[operator]
        return F.softplus(self.cost_critic(critic_obs))

