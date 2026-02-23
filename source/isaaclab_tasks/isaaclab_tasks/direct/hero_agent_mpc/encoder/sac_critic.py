# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Twin Q-Network critic for SAC-MPC.

Symmetric design: actor and critic see the same observations.

Twin Q-networks (Haarnoja et al. 2018):
    Q1, Q2 = independent MLPs with same architecture
    Target: min(Q1, Q2) to prevent value overestimation
"""

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.networks import MLP


class TwinQNetwork(nn.Module):
    """Twin Q-networks for SAC.

    Input: cat([obs_parts..., action]) -> scalar Q-value.
    The critic observes the same obs keys as the actor (symmetric).

    Two independent Q-networks trained with the same target but different
    random seeds to reduce overestimation bias.
    """

    def __init__(
        self,
        policy_obs_dim: int = 13,
        privileged_dim: int = 26,
        action_dim: int = 2,
        hidden_dims: list[int] | None = None,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        self.privileged_dim = privileged_dim
        hidden = hidden_dims if hidden_dims is not None else [256, 256]
        input_dim = policy_obs_dim + privileged_dim + action_dim

        # Independent LayerNorms per Q-network to preserve twin independence.
        # elementwise_affine=False: no trainable weight/bias to avoid
        # online-target parameter lag under Polyak averaging (tau=0.002).
        self.norm_q1 = nn.LayerNorm(input_dim, elementwise_affine=False)
        self.norm_q2 = nn.LayerNorm(input_dim, elementwise_affine=False)

        self.q1 = MLP(input_dim, 1, hidden, activation)
        self.q2 = MLP(input_dim, 1, hidden, activation)

    def forward(
        self,
        policy_obs: torch.Tensor,
        privileged: torch.Tensor | None,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute both Q-values.

        Args:
            policy_obs: Policy observations. Shape: (N, policy_obs_dim).
            privileged: Privileged observations. Shape: (N, privileged_dim).
                None when privileged_dim=0.
            action: Actions. Shape: (N, action_dim).

        Returns:
            q1_value: Q1 prediction. Shape: (N, 1).
            q2_value: Q2 prediction. Shape: (N, 1).
        """
        parts = [policy_obs]
        if privileged is not None:
            parts.append(privileged)
        parts.append(action)
        x_raw = torch.cat(parts, dim=-1)
        return self.q1(self.norm_q1(x_raw)), self.q2(self.norm_q2(x_raw))

