# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""EncoderRunner with C-TRPO constraint metrics logging, barrier state persistence, and auto-sync.

Extends EncoderRunner to:
    - Log per-constraint margins, recovery mode flags, and barrier penalty
    - Use constraint names from env config instead of numeric indices
    - Auto-sync num_constraints from env config to algorithm/policy config
    - Save/load barrier state (margins, recovery flags) for checkpoint persistence
"""

from __future__ import annotations

import logging

from ..utils.logging import flush_metrics
from .encoder_runner import EncoderRunner

logger = logging.getLogger(__name__)


class ConstraintEncoderRunner(EncoderRunner):
    """EncoderRunner with C-TRPO barrier constraint support.

    Inherits encoder metrics and DORAEMON DR scheduling from EncoderRunner.
    Adds barrier state persistence and constraint-specific WandB/TB logging.
    Auto-syncs num_constraints from env config.
    """

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        # Auto-sync num_constraints from env config before parent init
        constraints_cfg = getattr(env.unwrapped.cfg, "constraints", None)
        if constraints_cfg is not None:
            env_k = constraints_cfg.num_constraints
            alg_cfg = train_cfg["algorithm"]
            policy_cfg = train_cfg["policy"]

            if hasattr(alg_cfg, "num_constraints") and alg_cfg.num_constraints != env_k:
                logger.info(
                    "Auto-syncing num_constraints: alg %d -> %d",
                    alg_cfg.num_constraints,
                    env_k,
                )
                alg_cfg.num_constraints = env_k
                alg_cfg.constraint_budgets = constraints_cfg.constraint_budgets

            if hasattr(policy_cfg, "num_constraints") and policy_cfg.num_constraints != env_k:
                logger.info(
                    "Auto-syncing num_constraints: policy %d -> %d",
                    policy_cfg.num_constraints,
                    env_k,
                )
                policy_cfg.num_constraints = env_k

            # Cache constraint names for logging
            self._constraint_names = constraints_cfg.constraint_names
        else:
            self._constraint_names = ()

        super().__init__(env, train_cfg, log_dir, device)

    def _update_encoder_lr(self, _iteration: int, _total_iterations: int) -> None:
        """No-op: ConstraintTRPO manages its own encoder Adam optimizer with fixed LR."""

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        """Extended log with constraint metrics.

        Args:
            locs: Local variables from the learn() training loop.
            width: Terminal output width for formatting.
            pad: Padding for log formatting.
        """
        super().log(locs, width, pad)

        iteration = locs["it"]

        # Log constraint-specific metrics
        if self._should_log:
            self._log_constraint_metrics(locs, iteration)

    def _log_constraint_metrics(self, _locs: dict, iteration: int) -> None:
        """Log constraint metrics to TensorBoard/WandB.

        Logs per-constraint: cost_return, violation, margin, recovery flag, and d_k.
        Also logs aggregate barrier penalty and mode indicator.
        """
        alg = self.alg
        if not hasattr(alg, "num_constraints"):
            return

        K = alg.num_constraints
        metrics: dict[str, float] = {}

        # Per-constraint: cost_return, violation, margin, recovery, d_k
        for k in range(K):
            suffix = self._constraint_names[k] if k < len(self._constraint_names) else str(k)
            metrics[f"Constraint/violation_{suffix}"] = alg._last_violations[k]
            metrics[f"Constraint/cost_return_{suffix}"] = alg._last_cost_returns[k]
            metrics[f"Constraint/margin_{suffix}"] = alg._last_margins[k]
            metrics[f"Constraint/in_recovery_{suffix}"] = alg._last_in_recovery[k]
            metrics[f"Constraint/d_k_{suffix}"] = alg.d_k[k].item()

        # Aggregate metrics
        metrics["Constraint/barrier_penalty"] = alg._last_barrier_penalty
        metrics["Constraint/mode"] = float(alg._last_mode)

        # Line search (policy update metric)
        metrics["Policy/line_search_success"] = alg._last_line_search_success

        flush_metrics(self.writer, metrics, iteration, self.logger_type)

    def save(self, path, infos=None):
        """Save checkpoint with barrier state."""
        super().save(path, infos)
        self._save_aux_state(
            path,
            "barrier_state.pt",
            {
                "in_recovery": self.alg._in_recovery,
                "margins": self.alg._margins,
            },
        )

    def load(self, path, load_optimizer=True, map_location=None):
        """Load checkpoint and restore barrier state if available."""
        infos = super().load(path, load_optimizer, map_location)

        state = self._load_aux_state(path, "barrier_state.pt", self.device)
        if state is not None:
            self.alg._in_recovery = state["in_recovery"]
            self.alg._margins = state["margins"].to(self.device)
            logger.info("Restored barrier state from checkpoint")

        return infos
