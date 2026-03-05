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
        """Log constraint metrics to TensorBoard/WandB."""
        alg = self.alg
        if not hasattr(alg, "num_constraints"):
            return

        K = alg.num_constraints

        # Barrier steepness
        if hasattr(alg, "barrier_t"):
            self.writer.add_scalar("Constraint/barrier_t", alg.barrier_t, iteration)

        # Per-constraint adaptive thresholds
        if hasattr(alg, "d_k_adaptive"):
            for k in range(K):
                self.writer.add_scalar(
                    f"Constraint/d_k_adaptive_{k}",
                    alg.d_k_adaptive[k].item(),
                    iteration,
                )
