# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ActorCriticConstrained: Encoder-TDC with multi-head cost value function.

Extends ActorCriticEncoderTDC to add K cost value heads that share the critic
backbone. Each head outputs a scalar V_Ck(s) for one constraint, enabling
per-constraint cost return estimation for the log-barrier loss.

Architecture:
    Shared critic backbone: policy_obs + z -> hidden -> hidden_features
    Value head:     hidden_features -> V(s)    (1D, inherited)
    Cost head k:    hidden_features -> V_Ck(s) (1D, per constraint)

Parameter efficiency: Only K small linear layers are added.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from rsl_rl.networks import MLP

from .actor_critic_encoder import ActorCriticEncoderTDC

if TYPE_CHECKING:
    from tensordict import TensorDict


class ActorCriticConstrained(ActorCriticEncoderTDC):
    """ActorCriticEncoderTDC extended with per-constraint cost value heads.

    The cost heads share the critic's hidden layers and only add small
    output projections. This is parameter-efficient and ensures the
    cost value functions benefit from the same representation learning
    as the reward value function.
    """

    def __init__(
        self,
        *args,
        num_constraints: int = 5,
        cost_hidden_dims: list[int] | tuple[int, ...] = (64,),
        **kwargs: Any,
    ) -> None:
        """Initialize with cost value heads.

        Args:
            *args: Positional arguments for ActorCriticEncoderTDC.
            num_constraints: Number of constraint cost heads (K).
            cost_hidden_dims: Hidden layer dims for cost heads.
            **kwargs: Keyword arguments for ActorCriticEncoderTDC.
        """
        # Pop our custom kwargs before passing to parent
        super().__init__(*args, **kwargs)

        self.num_constraints = num_constraints

        # Derive input dim from critic (same combined obs: policy_obs + z)
        combined_dim = self.policy_obs_dim + self.encoder_latent_dim

        # K independent cost value heads (small MLPs)
        self.cost_heads = nn.ModuleList()
        for _ in range(num_constraints):
            head = MLP(combined_dim, 1, list(cost_hidden_dims), "elu")
            self.cost_heads.append(head)

    def evaluate_costs(self, obs: TensorDict, **_kwargs: Any) -> torch.Tensor:
        """Evaluate all cost value functions V_Ck(s).

        Uses the same combined observation as the critic (policy_obs + z).

        Args:
            obs: Observation TensorDict with policy and privileged keys.

        Returns:
            Cost values tensor of shape (batch, K).
        """
        # Reuse the combined obs pipeline from parent (includes z caching)
        combined = self.critic_obs_normalizer(self._get_combined_obs(obs))

        cost_values = []
        for head in self.cost_heads:
            cv = head(combined)  # (batch, 1)
            cost_values.append(cv.squeeze(-1))

        return torch.stack(cost_values, dim=-1)  # (batch, K)
