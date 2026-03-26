# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Constrained teacher policy with shared-backbone multi-head value network.

Extends ActorCriticEncoder with a shared value backbone for both reward and
cost value estimation, following the paper's multi-head cost critic design:
    Shared Backbone: cat([o_t, p_t]) -> MLP -> features
    Reward Head:     features -> 1D (reward value)
    Cost Head:       features -> KD (per-constraint cost values)

This reduces parameters compared to separate critic networks (~1/K savings
on the backbone) while sharing learned representations across value functions.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from rsl_rl.networks import MLP

from .actor_critic_encoder import ActorCriticEncoder

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from tensordict import TensorDict


class ActorCriticEncoderConstrained(ActorCriticEncoder):
    """Teacher policy with shared-backbone multi-head value function.

    Overrides the parent's separate critic with a shared backbone that feeds
    both reward and cost value heads. The cost head outputs K values
    (one per constraint) used by the TRPO + IPO algorithm for cost GAE.
    """

    def __init__(
        self,
        *args: Any,
        num_constraints: int = 4,
        cost_critic_hidden_dims: list[int] | tuple[int, ...] = (512, 256, 128, 64),
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_constraints = num_constraints

        # Replace parent's critic with shared backbone + heads.
        # The backbone hidden dims come from cost_critic_hidden_dims.
        backbone_output_dim = cost_critic_hidden_dims[-1]
        backbone_hidden = list(cost_critic_hidden_dims[:-1])

        # Delete parent's separate critic MLP
        del self.critic

        # Shared backbone: cat([o_t, p_t]) -> features
        self.value_backbone = MLP(
            self.num_critic_obs,
            backbone_output_dim,
            backbone_hidden,
            "elu",
        )

        # Reward head: features -> 1D
        self.reward_head = nn.Linear(backbone_output_dim, 1)

        # Cost head: features -> K (one per constraint)
        self.cost_head = nn.Linear(backbone_output_dim, num_constraints)

        logger.info(
            "Shared value backbone: %dD -> %s -> %dD, reward_head(1), cost_head(%d)",
            self.num_critic_obs,
            cost_critic_hidden_dims,
            backbone_output_dim,
            num_constraints,
        )

    def _get_value_features(self, obs: TensorDict) -> torch.Tensor:
        """Compute shared backbone features from critic observations."""
        critic_obs = self.critic_obs_normalizer(self._get_critic_obs(obs))
        return self.value_backbone(critic_obs)

    def evaluate(self, obs: TensorDict, **_kwargs: Any) -> torch.Tensor:
        """Evaluate reward value function via shared backbone + reward head."""
        features = self._get_value_features(obs)
        return self.reward_head(features)

    def evaluate_costs(self, obs: TensorDict) -> torch.Tensor:
        """Evaluate per-constraint cost values via shared backbone + cost head.

        Returns:
            Cost value predictions. Shape: (batch, K).
        """
        features = self._get_value_features(obs)
        return self.cost_head(features)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load model parameters, injecting defaults for missing value network keys."""
        # Handle checkpoint from before shared backbone (separate critic + cost_critic)
        cost_prefix = "cost_head."
        backbone_prefix = "value_backbone."
        has_new_keys = any(k.startswith(backbone_prefix) for k in state_dict)

        if not has_new_keys:
            logger.info("Checkpoint lacks shared value backbone; using random initialization.")
            # Remove old critic keys if present
            old_keys = [k for k in state_dict if k.startswith("critic.") or k.startswith("cost_critic.")]
            for k in old_keys:
                del state_dict[k]
            # Inject new backbone + head keys
            for k, v in self.value_backbone.state_dict().items():
                state_dict[backbone_prefix + k] = v
            for k, v in self.reward_head.state_dict().items():
                state_dict["reward_head." + k] = v
            for k, v in self.cost_head.state_dict().items():
                state_dict[cost_prefix + k] = v

        return super().load_state_dict(state_dict, strict=strict)
