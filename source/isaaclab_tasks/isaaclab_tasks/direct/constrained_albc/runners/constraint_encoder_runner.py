# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single runner for constrained ALBC: encoder metrics + constraint state.

Flat subclass of OnPolicyRunner that combines:
    - Teacher encoder metrics logging (if encoder present)
    - Log-barrier constraint metrics (TRPO + IPO)
    - Auto-sync of num_constraints from env config
"""

from __future__ import annotations

import logging
import os

import torch
from rsl_rl.runners import OnPolicyRunner

from ..utils.logging import flush_metrics, log_encoder_metrics

logger = logging.getLogger(__name__)


class ConstraintEncoderRunner(OnPolicyRunner):
    """OnPolicyRunner with encoder metrics and log-barrier constraint support.

    Provides:
        - Encoder metrics: z latent statistics, gradient norms (when encoder present)
        - Constraint metrics: barrier margins, penalty (Modified IPO)
        - Auto-sync: num_constraints from env config to algorithm/policy config
    """

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        # Auto-sync num_constraints from env config before parent init.
        # train_cfg is a plain dict (from agent_cfg.to_dict()), so use dict
        # key access instead of hasattr/getattr which only work on objects.
        constraints_cfg = getattr(env.unwrapped.cfg, "constraints", None)
        if constraints_cfg is not None:
            env_k = constraints_cfg.num_constraints
            alg_cfg = train_cfg["algorithm"]
            policy_cfg = train_cfg["policy"]

            if "num_constraints" in alg_cfg and alg_cfg["num_constraints"] != env_k:
                logger.info(
                    "Auto-syncing num_constraints: alg %d -> %d",
                    alg_cfg["num_constraints"],
                    env_k,
                )
                alg_cfg["num_constraints"] = env_k
                alg_cfg["constraint_budgets"] = constraints_cfg.constraint_budgets

            if "num_constraints" in policy_cfg and policy_cfg["num_constraints"] != env_k:
                logger.info(
                    "Auto-syncing num_constraints: policy %d -> %d",
                    policy_cfg["num_constraints"],
                    env_k,
                )
                policy_cfg["num_constraints"] = env_k

            # Cache constraint names for logging
            self._constraint_names = constraints_cfg.constraint_names
        else:
            self._constraint_names = ()

        super().__init__(env, train_cfg, log_dir, device)

        # Detect encoder for conditional metrics logging
        self._has_encoder = hasattr(self.alg.policy, "encoder")
        if self._has_encoder:
            logger.info("[ConstraintEncoderRunner] Encoder detected. Encoder metrics logging enabled.")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def _should_log(self) -> bool:
        """Whether logging is active (log_dir set and logs not disabled)."""
        return self.log_dir is not None and not self.disable_logs

    # ------------------------------------------------------------------
    # Auxiliary state persistence helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _save_aux_state(path: str, name: str, state: dict) -> None:
        """Save auxiliary state dict alongside a model checkpoint."""
        aux_path = os.path.join(os.path.dirname(path), name)
        torch.save(state, aux_path)

    @staticmethod
    def _load_aux_state(path: str, name: str, device: str) -> dict | None:
        """Load auxiliary state dict from alongside a model checkpoint, or None."""
        aux_path = os.path.join(os.path.dirname(path), name)
        if os.path.exists(aux_path):
            return torch.load(aux_path, map_location=device, weights_only=False)
        return None

    # ------------------------------------------------------------------
    # Training loop overrides
    # ------------------------------------------------------------------

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Reset environments before training."""
        if hasattr(self.alg, "set_max_iterations"):
            self.alg.set_max_iterations(num_learning_iterations)
        self.env.reset()
        super().learn(num_learning_iterations, init_at_random_ep_len)

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        """Extended log with encoder metrics and constraint metrics."""
        super().log(locs, width, pad)

        iteration = locs["it"]

        # Encoder metrics (z latent stats, gradient norms, etc.)
        if self._has_encoder and self._should_log:
            log_encoder_metrics(
                writer=self.writer,
                policy=self.alg.policy,
                env=self.env,
                iteration=iteration,
                device=self.device,
                logger_type=self.logger_type,
                alg=self.alg,
            )

        # Constraint metrics
        if self._should_log:
            self._log_constraint_metrics(iteration)

        # DORAEMON: update DR distribution based on episode statistics
        raw_env = self.env.unwrapped
        if hasattr(raw_env, "_doraemon") and raw_env._doraemon is not None:
            metrics = raw_env._doraemon.step()
            if self._should_log:
                prefixed = {f"DORAEMON/{k}": v for k, v in metrics.items()}
                flush_metrics(self.writer, prefixed, iteration, self.logger_type)

    # ------------------------------------------------------------------
    # Checkpoint save/load
    # ------------------------------------------------------------------

    def save(self, path: str, infos: dict | None = None) -> None:
        """Save model checkpoint and DORAEMON state."""
        super().save(path, infos)

        # Save DORAEMON distribution state
        raw_env = self.env.unwrapped
        if hasattr(raw_env, "_doraemon") and raw_env._doraemon is not None:
            self._save_aux_state(path, "doraemon_state.pt", raw_env._doraemon.state_dict())

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        """Load model checkpoint and DORAEMON state if available."""
        infos = super().load(path, load_optimizer, map_location)

        # Restore DORAEMON distribution state
        raw_env = self.env.unwrapped
        if hasattr(raw_env, "_doraemon") and raw_env._doraemon is not None:
            doraemon_state = self._load_aux_state(path, "doraemon_state.pt", self.device)
            if doraemon_state is not None:
                raw_env._doraemon.load_state_dict(doraemon_state)
                logger.info("Restored DORAEMON distribution state from checkpoint")

        return infos

    # ------------------------------------------------------------------
    # Constraint metrics
    # ------------------------------------------------------------------

    def _log_constraint_metrics(self, iteration: int) -> None:
        """Log constraint metrics to TensorBoard/WandB.

        Logs per-constraint: cost_return, violation, d_k, barrier_margin.
        Also logs aggregate barrier penalty and policy diagnostics.
        """
        alg = self.alg
        if not hasattr(alg, "num_constraints"):
            return

        K = alg.num_constraints
        metrics: dict[str, float] = {}

        # Per-constraint: cost_return, violation, d_k, barrier_margin
        for k in range(K):
            suffix = self._constraint_names[k] if k < len(self._constraint_names) else str(k)
            metrics[f"Constraint/violation_{suffix}"] = alg._last_violations[k]
            metrics[f"Constraint/cost_return_{suffix}"] = alg._last_cost_returns[k]
            metrics[f"Constraint/d_k_{suffix}"] = alg.d_k[k].item()
            metrics[f"Constraint/barrier_margin_{suffix}"] = alg._last_barrier_margins[k]

        # Aggregate metrics
        metrics["Constraint/barrier_penalty"] = alg._last_barrier_penalty

        # Policy diagnostics
        metrics["Policy/line_search_success"] = alg._last_line_search_success
        metrics["Policy/entropy"] = alg._last_mean_entropy
        metrics["Policy/encoder_grad_norm"] = alg._last_encoder_grad_norm

        flush_metrics(self.writer, metrics, iteration, self.logger_type)
