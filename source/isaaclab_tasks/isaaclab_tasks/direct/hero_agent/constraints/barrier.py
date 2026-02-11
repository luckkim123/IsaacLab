# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Log-barrier and adaptive thresholding for NORBC-style constrained RL.

Implements Interior-point Policy Optimization (IPO) barrier functions:
    - Log-barrier: -log(d_k - J_Ck) / t  (NORBC Eq. 10)
    - Adaptive thresholding: d_k^{i+1} = max(d_k, J_Ck + alpha * d_k)  (NORBC Eq. 11)

The barrier loss is added to the PPO loss to keep cost returns within their
respective budgets. Adaptive thresholding relaxes constraints when the current
policy is infeasible, then tightens as the policy improves.
"""

from __future__ import annotations

import torch

from .config import ConstraintCfg


class LogBarrierManager:
    """Manages K constraints with log-barrier loss and adaptive thresholding.

    For each constraint k:
        - Original limit: D_k (per-step budget from ConstraintCfg)
        - Discounted limit: d_k = D_k / (1 - gamma)  (infinite horizon, NORBC Eq. 8)
        - Adaptive limit: d_k^i >= d_k (relaxed when infeasible, NORBC Eq. 11)
        - Barrier loss: -log(d_k^i - J_Ck) / t  (NORBC Eq. 10)
    """

    def __init__(
        self,
        constraints: list[ConstraintCfg],
        t: float = 10.0,
        alpha: float = 0.5,
        gamma: float = 0.99,
        device: str = "cpu",
    ) -> None:
        """Initialize the log-barrier manager.

        Args:
            constraints: List of constraint configurations.
            t: Barrier steepness parameter. Higher = harder enforcement.
            alpha: Adaptive threshold expansion factor.
            gamma: Cost discount factor for computing d_k = D_k / (1 - gamma).
            device: Computation device.
        """
        self.t = t
        self.alpha = alpha
        self.gamma = gamma
        self.device = device
        self.num_constraints = len(constraints)

        # Discounted limits: d_k = D_k / (1 - gamma)
        limits = [cfg.limit_D / (1.0 - gamma) for cfg in constraints]
        self._d_k = torch.tensor(limits, dtype=torch.float32, device=device)

        # Adaptive limits start at original d_k (tightest feasible)
        self._d_k_adaptive = self._d_k.clone()

        # Track constraint names for logging
        self._names = [cfg.name for cfg in constraints]

    @property
    def adaptive_limits(self) -> torch.Tensor:
        """Current adaptive limits (K,)."""
        return self._d_k_adaptive

    @property
    def original_limits(self) -> torch.Tensor:
        """Original discounted limits (K,)."""
        return self._d_k

    @property
    def constraint_names(self) -> list[str]:
        return self._names

    def compute_barrier_loss(self, cost_returns: torch.Tensor) -> torch.Tensor:
        """Compute log-barrier loss over all constraints.

        barrier_loss = -sum_k log(d_k^i - mean(J_Ck)) / t

        The loss approaches infinity as cost returns approach the adaptive limit,
        creating a soft wall that keeps the policy feasible.

        Args:
            cost_returns: Per-env cost returns, shape (batch, K).

        Returns:
            Scalar barrier loss (negative for minimization).
        """
        # Mean cost return across batch for each constraint
        mean_cost_returns = cost_returns.mean(dim=0)  # (K,)

        # Slack: d_k^i - J_Ck (must be positive for log to be defined)
        slack = self._d_k_adaptive - mean_cost_returns
        # Clamp slack to avoid log(0) or log(negative)
        slack = torch.clamp(slack, min=1e-8)

        # Log-barrier: -sum log(slack) / t
        barrier = -torch.log(slack).sum() / self.t
        return barrier

    def update_thresholds(self, cost_returns: torch.Tensor) -> None:
        """Update adaptive thresholds based on current policy performance.

        d_k^{i+1} = max(d_k, J_Ck(pi_i) + alpha * d_k)  (NORBC Eq. 11)

        When the current policy violates a constraint (J_Ck > d_k), the
        adaptive limit is relaxed to J_Ck + alpha * d_k, giving the policy
        room to gradually become feasible. As the policy improves,
        the limit tightens back toward d_k.

        Args:
            cost_returns: Per-env cost returns, shape (batch, K).
        """
        with torch.no_grad():
            mean_cost_returns = cost_returns.mean(dim=0)  # (K,)
            # Eq. 11: max(d_k, J_Ck + alpha * d_k)
            expanded = mean_cost_returns + self.alpha * self._d_k
            self._d_k_adaptive = torch.max(self._d_k, expanded)

    def get_metrics(self) -> dict[str, float]:
        """Return per-constraint metrics for logging.

        Returns:
            Dict with adaptive limits and slack ratios.
            Note: d_original is omitted (constant config value, not useful for monitoring).
        """
        metrics = {}
        for i, name in enumerate(self._names):
            metrics[f"Constraint/{name}_d_adaptive"] = self._d_k_adaptive[i].item()
            ratio = self._d_k_adaptive[i].item() / (self._d_k[i].item() + 1e-8)
            metrics[f"Constraint/{name}_slack_ratio"] = ratio
        return metrics
