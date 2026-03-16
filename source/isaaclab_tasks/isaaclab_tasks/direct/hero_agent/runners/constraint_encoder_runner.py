# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""EncoderRunner with constraint metrics logging, lambda persistence, and auto-sync.

Extends EncoderRunner to:
    - Log per-constraint cost returns, violations, and Lagrangian dual variables
    - Use constraint names from env config instead of numeric indices
    - Auto-sync num_constraints from env config to algorithm/policy config
    - Save/load lambda_k state for checkpoint persistence
"""

from __future__ import annotations

import logging
import os

import torch

from ..utils.logging import flush_metrics
from .encoder_runner import EncoderRunner

logger = logging.getLogger(__name__)


class ConstraintEncoderRunner(EncoderRunner):
    """EncoderRunner with Lagrangian constraint support.

    Inherits encoder metrics and DORAEMON DR scheduling from EncoderRunner.
    Adds Lagrangian dual variable persistence and constraint-specific WandB/TB logging.
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
        """No-op: ConstraintTRPO updates encoder via natural gradient, not Adam."""

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
        if self.log_dir is not None and not self.disable_logs:
            self._log_constraint_metrics(locs, iteration)

    def _log_constraint_metrics(self, _locs: dict, iteration: int) -> None:
        """Log constraint metrics to TensorBoard/WandB.

        Logs per-constraint: cost_return, violation, lambda, and d_k (budget).
        """
        alg = self.alg
        if not hasattr(alg, "num_constraints"):
            return

        K = alg.num_constraints
        metrics: dict[str, float] = {}

        # Per-constraint: cost_return, violation, lambda, d_k (budget)
        for k in range(K):
            suffix = self._constraint_names[k] if k < len(self._constraint_names) else str(k)
            if hasattr(alg, "_last_lambdas"):
                metrics[f"Constraint/lambda_{suffix}"] = alg._last_lambdas[k]
            if hasattr(alg, "_last_violations"):
                metrics[f"Constraint/violation_{suffix}"] = alg._last_violations[k]
            if hasattr(alg, "_last_cost_returns"):
                metrics[f"Constraint/cost_return_{suffix}"] = alg._last_cost_returns[k]
            if hasattr(alg, "d_k"):
                metrics[f"Constraint/d_k_{suffix}"] = alg.d_k[k].item()

        # Aggregate lambda stats
        if hasattr(alg, "lambda_k"):
            metrics["Constraint/lambda_mean"] = alg.lambda_k.mean().item()
            metrics["Constraint/lambda_max"] = alg.lambda_k.max().item()

        # Line search (policy update metric)
        if hasattr(alg, "_last_line_search_success"):
            metrics["Policy/line_search_success"] = alg._last_line_search_success

        flush_metrics(self.writer, metrics, iteration, self.logger_type)

    def save(self, path, infos=None):
        """Save checkpoint with lambda_k state."""
        super().save(path, infos)
        lambda_path = os.path.join(os.path.dirname(path), "lambda_state.pt")
        torch.save({"lambda_k": self.alg.lambda_k}, lambda_path)

    def load(self, path, load_optimizer=True, map_location=None):
        """Load checkpoint and restore lambda_k if available."""
        infos = super().load(path, load_optimizer, map_location)
        lambda_path = os.path.join(os.path.dirname(path), "lambda_state.pt")
        if os.path.exists(lambda_path):
            state = torch.load(lambda_path, map_location=self.device, weights_only=False)
            self.alg.lambda_k = state["lambda_k"].to(self.device)
            logger.info("Restored lambda_k from %s", lambda_path)
        return infos
