# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""EncoderRunner with constraint metrics logging and barrier schedule for IPO.

Extends EncoderRunner to:
    - Update ConstraintTRPO barrier schedule each iteration
    - Log per-constraint cost returns, feasibility rates, and barrier_t
"""

from __future__ import annotations

import logging

from ..utils.logging import flush_metrics
from .encoder_runner import EncoderRunner

logger = logging.getLogger(__name__)


class ConstraintEncoderRunner(EncoderRunner):
    """EncoderRunner with IPO constraint support.

    Inherits encoder metrics and DORAEMON DR scheduling from EncoderRunner.
    Adds barrier schedule update and constraint-specific WandB/TB logging.
    """

    def _update_encoder_lr(self, _iteration: int, _total_iterations: int) -> None:
        """No-op: ConstraintTRPO updates encoder via natural gradient, not Adam."""

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        """Extended log with barrier schedule update and constraint metrics.

        Args:
            locs: Local variables from the learn() training loop.
            width: Terminal output width for formatting.
            pad: Padding for log formatting.
        """
        super().log(locs, width, pad)

        iteration = locs["it"]

        # Update barrier schedule in ConstraintTRPO
        if hasattr(self.alg, "update_barrier_schedule"):
            self.alg.update_barrier_schedule(iteration)

        # Log constraint-specific metrics
        if self.log_dir is not None and not self.disable_logs:
            self._log_constraint_metrics(locs, iteration)

    def _log_constraint_metrics(self, _locs: dict, iteration: int) -> None:
        """Log constraint metrics to TensorBoard/WandB.

        Reads monitoring metrics stored as instance attributes on ConstraintTRPO
        (populated during update()) and logs them with proper prefixes:
            - Constraint/: barrier state, cost returns, feasibility
            - Policy/: line search success
        """
        alg = self.alg
        if not hasattr(alg, "num_constraints"):
            return

        K = alg.num_constraints
        metrics: dict[str, float] = {}

        # Barrier steepness + schedule progress
        if hasattr(alg, "barrier_t"):
            metrics["Constraint/barrier_t"] = alg.barrier_t
            if alg.barrier_t_schedule_iters > 0:
                metrics["Constraint/barrier_schedule_progress"] = min(1.0, iteration / alg.barrier_t_schedule_iters)

        # Per-constraint metrics
        for k in range(K):
            if hasattr(alg, "d_k_adaptive"):
                metrics[f"Constraint/d_k_adaptive_{k}"] = alg.d_k_adaptive[k].item()
            if hasattr(alg, "_last_cost_returns"):
                metrics[f"Constraint/cost_return_{k}"] = alg._last_cost_returns[k]
            if hasattr(alg, "_last_cost_return_stds"):
                metrics[f"Constraint/cost_return_std_{k}"] = alg._last_cost_return_stds[k]
            if hasattr(alg, "_last_feasibility_rates"):
                metrics[f"Constraint/feasibility_rate_{k}"] = alg._last_feasibility_rates[k]

        # Line search (policy update metric)
        if hasattr(alg, "_last_line_search_success"):
            metrics["Policy/line_search_success"] = alg._last_line_search_success

        flush_metrics(self.writer, metrics, iteration, self.logger_type)
