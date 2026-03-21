# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single runner for constrained ALBC: encoder metrics + barrier state.

Flat subclass of OnPolicyRunner that combines:
    - HORA Phase 1 encoder metrics logging (if encoder present)
    - C-TRPO barrier state persistence and per-constraint metrics
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
    """OnPolicyRunner with encoder metrics and C-TRPO barrier support.

    Provides:
        - Encoder metrics: z latent statistics, gradient norms (when encoder present)
        - Barrier state persistence: save/load constraint margins and recovery flags
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
        """Extended log with encoder metrics and constraint metrics.

        Args:
            locs: Local variables from the learn() training loop.
            width: Terminal output width for formatting.
            pad: Padding for log formatting.
        """
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
            )

        # Constraint-specific metrics (barrier, violations, margins)
        if self._should_log:
            self._log_constraint_metrics(iteration)

    # ------------------------------------------------------------------
    # Checkpoint save/load
    # ------------------------------------------------------------------

    def save(self, path: str, infos: dict | None = None) -> None:
        """Save model checkpoint with barrier state and encoder optimizer."""
        super().save(path, infos)

        self._save_aux_state(
            path,
            "barrier_state.pt",
            {
                "margins": self.alg._margins,
                "ema_cost_returns": self.alg._ema_cost_returns,
                "ema_initialized": self.alg._ema_initialized,
                "lambda_k": self.alg._lambda_k,
            },
        )

        # Save encoder optimizer state for seamless resume (BUG-1 fix)
        if getattr(self.alg, "encoder_optimizer", None) is not None:
            self._save_aux_state(path, "encoder_optimizer.pt", self.alg.encoder_optimizer.state_dict())

        # Save EAPO state
        if getattr(self.alg, "eapo_enabled", False):
            self._save_aux_state(path, "eapo_state.pt", {"eapo_tau": self.alg.eapo_tau})

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        """Load model checkpoint, barrier state, and encoder optimizer if available."""
        infos = super().load(path, load_optimizer, map_location)

        state = self._load_aux_state(path, "barrier_state.pt", self.device)
        if state is not None:
            self.alg._margins = state["margins"].to(self.device)
            if "ema_cost_returns" in state:
                self.alg._ema_cost_returns = state["ema_cost_returns"].to(self.device)
                self.alg._ema_initialized = state["ema_initialized"]
            if "lambda_k" in state:
                self.alg._lambda_k = state["lambda_k"].to(self.device)
            logger.info("Restored constraint state from checkpoint")

        # Restore encoder optimizer state for seamless resume (BUG-1 fix)
        if load_optimizer and getattr(self.alg, "encoder_optimizer", None) is not None:
            enc_opt_state = self._load_aux_state(path, "encoder_optimizer.pt", self.device)
            if enc_opt_state is not None:
                self.alg.encoder_optimizer.load_state_dict(enc_opt_state)
                logger.info("Restored encoder optimizer state from checkpoint")

        # Restore EAPO state
        if getattr(self.alg, "eapo_enabled", False):
            eapo_state = self._load_aux_state(path, "eapo_state.pt", self.device)
            if eapo_state is not None:
                self.alg.eapo_tau = eapo_state["eapo_tau"]
                logger.info("Restored EAPO tau=%.4f from checkpoint", self.alg.eapo_tau)

        return infos

    # ------------------------------------------------------------------
    # Constraint metrics
    # ------------------------------------------------------------------

    def _log_constraint_metrics(self, iteration: int) -> None:
        """Log constraint metrics to TensorBoard/WandB.

        Logs per-constraint: cost_return, violation, margin, recovery flag, and d_k.
        Also logs aggregate barrier penalty and mode indicator.
        """
        alg = self.alg
        if not hasattr(alg, "num_constraints"):
            return

        K = alg.num_constraints
        metrics: dict[str, float] = {}

        # Per-constraint: cost_return, violation, margin, recovery, d_k, ema_cost_return
        for k in range(K):
            suffix = self._constraint_names[k] if k < len(self._constraint_names) else str(k)
            metrics[f"Constraint/violation_{suffix}"] = alg._last_violations[k]
            metrics[f"Constraint/cost_return_{suffix}"] = alg._last_cost_returns[k]
            metrics[f"Constraint/ema_cost_return_{suffix}"] = alg._ema_cost_returns[k].item()
            metrics[f"Constraint/margin_{suffix}"] = alg._last_margins[k]
            metrics[f"Constraint/in_recovery_{suffix}"] = alg._last_in_recovery[k]
            metrics[f"Constraint/d_k_{suffix}"] = alg.d_k[k].item()

        # Per-constraint Lagrangian multiplier
        for k in range(K):
            suffix = self._constraint_names[k] if k < len(self._constraint_names) else str(k)
            metrics[f"Constraint/lambda_{suffix}"] = alg._lambda_k[k].item()

        # Aggregate metrics
        metrics["Constraint/lagrangian_penalty"] = alg._last_lagrangian_penalty
        metrics["Constraint/mode"] = float(alg._last_mode)

        # Line search (policy update metric)
        metrics["Policy/line_search_success"] = alg._last_line_search_success

        # Entropy and pre-encoder KL (Fix 1 + Fix 2)
        metrics["Policy/entropy"] = alg._cached_mean_entropy
        metrics["Policy/pre_encoder_kl"] = alg._last_pre_encoder_kl

        # EAPO metrics
        if getattr(alg, "eapo_enabled", False):
            metrics["Policy/entropy_tau"] = alg._cached_entropy_tau

        flush_metrics(self.writer, metrics, iteration, self.logger_type)
