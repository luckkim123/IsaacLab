# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""JIT-scriptable deployment module for Hero Agent Encoder-TDC Phase 3.

This module packages the adaptation network (adapt_tconv), actor, and
normalizers into a single nn.Module suitable for torch.jit.script export.
The exported model produces physical-unit outputs that a C++ TDC controller
can consume directly:

    forward(policy_obs, proprio_hist) -> (kp, kd, m_hat)

All scaling (sigmoid for gains, softplus + z_min for z) is baked into the
module so the consumer does not need to replicate any Python-side
post-processing.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeroAgentDeployModule(nn.Module):
    """Deployment-ready module: adapt_tconv + actor + normalizers.

    Combines the Phase 2 adaptation network with the frozen Phase 1 actor
    into a single scriptable module. Gain sigmoid scaling is applied
    internally so outputs are in physical units.

    Inputs:
        policy_obs: (N, 13) -- attitude errors, angular velocities, etc.
        proprio_hist: (N, H, D) -- proprioception history buffer

    Outputs:
        kp: (N, 2) -- proportional gains in [kp_min, kp_max]
        kd: (N, 2) -- derivative gains in [kd_min, kd_max]
        m_hat: (N, 2) -- estimated design inertia from encoder z[3:5]
    """

    def __init__(
        self,
        adapt_tconv: nn.Module,
        actor: nn.Module,
        actor_obs_normalizer: nn.Module,
        hist_normalizer: nn.Module,
        z_min: float = 0.1,
        kp_min: float = 10.0,
        kp_max: float = 100.0,
        kd_min: float = 2.0,
        kd_max: float = 30.0,
    ):
        super().__init__()

        # Deep copy to decouple from training graph
        self.adapt_tconv = copy.deepcopy(adapt_tconv)
        self.actor = copy.deepcopy(actor)
        self.actor_obs_normalizer = copy.deepcopy(actor_obs_normalizer)
        self.hist_normalizer = copy.deepcopy(hist_normalizer)

        # Register all config values as buffers for JIT compatibility
        self.register_buffer("z_min", torch.tensor(z_min))
        self.register_buffer("kp_min", torch.tensor(kp_min))
        self.register_buffer("kp_max", torch.tensor(kp_max))
        self.register_buffer("kd_min", torch.tensor(kd_min))
        self.register_buffer("kd_max", torch.tensor(kd_max))

        # Freeze everything
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(
        self, policy_obs: torch.Tensor, proprio_hist: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the full deploy inference pipeline.

        Args:
            policy_obs: Policy observation. Shape: (N, 13).
            proprio_hist: Proprioception history. Shape: (N, H, D).

        Returns:
            Tuple of (kp, kd, m_hat):
                kp: Proportional gains. Shape: (N, 2).
                kd: Derivative gains. Shape: (N, 2).
                m_hat: Design inertia estimate. Shape: (N, 2).
        """
        # 1. Normalize proprioception history
        hist_norm = self.hist_normalizer(proprio_hist)

        # 2. Adaptation: proprio_hist -> z_hat (softplus + z_min)
        z_hat_raw = self.adapt_tconv(hist_norm)
        z_hat = F.softplus(z_hat_raw) + self.z_min

        # 3. M_hat from z[3:5]
        m_hat = z_hat[:, 3:5]

        # 4. Actor: [policy_obs, z_hat] -> raw 4D actions
        combined_obs = torch.cat([policy_obs, z_hat], dim=-1)
        actor_obs = self.actor_obs_normalizer(combined_obs)
        raw_actions = self.actor(actor_obs)

        # 5. Sigmoid scaling to physical gain units
        kp = self.kp_min + torch.sigmoid(raw_actions[:, :2]) * (self.kp_max - self.kp_min)
        kd = self.kd_min + torch.sigmoid(raw_actions[:, 2:]) * (self.kd_max - self.kd_min)

        return kp, kd, m_hat
