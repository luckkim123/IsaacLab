# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ActorCriticEncoder with multi-head cost value function for constrained RL (IPO).

Extends ActorCriticEncoder with a cost critic head that predicts per-constraint
cost values V_C_k(s) for K constraints. The cost critic shares the same input
(policy_obs + encoder z) as the reward critic.

Architecture:
    Encoder:     privileged (19D) -> MLP -> tanh -> z (13D)
    Actor:       cat([policy_obs, z]) = 26D -> MLP -> actions
    Critic:      cat([policy_obs, z]) = 26D -> MLP -> value (1D)
    Cost Critic: cat([policy_obs, z]) = 26D -> MLP -> cost values (K)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
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

        # Cost critic: same input as reward critic (policy_obs + z), K outputs
        num_critic_obs = self.policy_obs_dim + self.encoder_latent_dim
        self.cost_critic = MLP(
            num_critic_obs,
            num_constraints,
            list(cost_critic_hidden_dims),
            "elu",
        )
        logger.info(
            "Cost critic MLP (%dD input, %d outputs): %s",
            num_critic_obs,
            num_constraints,
            self.cost_critic,
        )

    def evaluate_costs(self, obs: TensorDict) -> torch.Tensor:
        """Evaluate cost value function for all K constraints.

        Args:
            obs: TensorDict with policy and privileged observations.

        Returns:
            Cost values. Shape: (batch, K).
        """
        critic_obs = self.critic_obs_normalizer(self._get_combined_obs(obs))  # type: ignore[operator]
        return self.cost_critic(critic_obs)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load with backward compatibility for checkpoints without cost_critic."""
        cost_prefix = "cost_critic."
        has_cost_keys = any(k.startswith(cost_prefix) for k in state_dict)

        if not has_cost_keys:
            logger.info("Checkpoint lacks cost_critic keys; using random initialization.")
            for k, v in self.cost_critic.state_dict().items():
                state_dict[cost_prefix + k] = v

        return super().load_state_dict(state_dict, strict=strict)
