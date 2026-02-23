# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Differentiable MPC with Projected Gradient Descent for GPU-batched environments.

Solves the receding-horizon optimal control problem:

    min   sum_{k=1}^{N-1} [ (x_k - x_target)^T Q_k (x_k - x_target) ]
        + sum_{k=0}^{N-1} [ u_k^T R u_k ]
        + (x_N - x_target)^T Q_terminal (x_N - x_target)
    s.t.  x_{k+1} = dynamics(x_k, u_k)
          u_min <= u_k <= u_max

Note: x_0 state cost is excluded (current state is not a decision variable).

Phase 1 (non-differentiable):
    Projected Gradient Descent to convergence. Box constraints enforced by
    clamping after each gradient step. Uses torch.autograd.grad for efficient
    gradient computation (only w.r.t. u_seq, not dynamics parameters).

Phase 2 (differentiable, for SAC actor gradient):
    Multi-step GD refinement with create_graph=True.
    Builds computation graph from u_refined back through dynamics and cost_map
    -- enabling end-to-end SAC reparameterization gradient.
    Soft clamp via tanh(u/u_max)*u_max after each step preserves gradient
    flow while keeping u within the dynamics training range.

Key difference from AC-MPC paper: Q_k is time-varying (per horizon step),
allowing the learned Cost Map Network to specify different cost weights at
each step of the prediction horizon.

All operations are batched (num_envs, ...) for GPU parallelism.
No external QP solver dependencies -- pure PyTorch.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from isaaclab.utils import configclass

from .dynamics_mlp import DynamicsMLP


@configclass
class DifferentiableMPCCfg:
    """Configuration for the differentiable MPC solver."""

    horizon: int = 5
    """Prediction horizon (number of future steps)."""

    pgd_iters: int = 15
    """Number of projected gradient descent iterations (used for rollout)."""

    train_pgd_iters: int | None = None
    """PGD iterations for differentiable training pass. When set, the training
    MPC uses fewer PGD iterations than rollout to leave incomplete convergence,
    providing stronger gradient signal for Phase 2 refinement. None = use pgd_iters."""

    pgd_lr: float = 0.1
    """Learning rate for the projected gradient descent solver."""

    u_min: float = -3.0
    """Lower bound on control inputs (pre-tanh space). tanh(3)=0.995 covers
    nearly the full [-1,1] action range after squashing."""

    u_max: float = 3.0
    """Upper bound on control inputs (pre-tanh space). tanh(3)=0.995 covers
    nearly the full [-1,1] action range after squashing."""

    state_dim: int = 8
    """State dimension."""

    control_dim: int = 2
    """Control dimension."""

    diff_gd_lr: float = 0.05
    """Learning rate for the differentiable refinement steps."""

    diff_gd_steps: int = 1
    """Number of differentiable GD refinement steps with create_graph=True.
    More steps propagate stronger gradients to cost_map/dynamics,
    but increase GPU memory proportionally. Default 1 for backward compat."""

    refine_noise_std: float = 0.0
    """Gaussian noise std added to converged u before differentiable refinement.
    Non-zero values perturb the starting point away from the flat minimum,
    providing stronger gradient signal for upstream cost_map parameters.
    Recommended: 0.3 when enabled (pre-tanh space, tanh(0.3)~=0.29).
    Default 0.0 (disabled)."""


class DifferentiableMPC(nn.Module):
    """GPU-batched differentiable MPC with Projected Gradient Descent.

    Solves the constrained optimal control problem using:
    1. Forward rollout through learned dynamics
    2. Projected gradient descent for box-constrained cost minimization
    3. Warm-start from shifted previous solution

    Two modes:
        - differentiable=False (default): PGD solve, output detached.
          Used during rollout or inference.
        - differentiable=True: PGD converges first (detached), then multi-step
          GD refinement with create_graph=True builds the computation graph.
          Used during SAC actor loss computation for end-to-end gradient.
    """

    def __init__(self, cfg: DifferentiableMPCCfg, num_envs: int, device: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device

        # Previous solution for warm-start (shifted each call)
        self.register_buffer(
            "_u_prev",
            torch.zeros(num_envs, cfg.horizon, cfg.control_dim, device=device),
        )

        # Terminal cost from last solve (per-env), for logging
        self.last_solve_cost: torch.Tensor | None = None

        # Max |u| after differentiable refinement (for verifying soft clamp)
        self.last_u_max_after_refine: float = 0.0

        # Per-step gradient norms from differentiable refinement (for diagnostics)
        self._refine_grad_norms: list[float] = []

    def solve(
        self,
        x_current: torch.Tensor,
        dynamics: DynamicsMLP,
        Q_diag: torch.Tensor,
        Q_terminal: torch.Tensor,
        R_diag: torch.Tensor,
        x_target: torch.Tensor,
        differentiable: bool = False,
        warm_start: bool = True,
        pred_error: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Solve the MPC problem and return the first optimal control.

        Args:
            x_current: Current state. Shape: (num_envs, state_dim).
            dynamics: Learned dynamics model for forward rollout.
            Q_diag: Time-varying state cost diagonal for predicted states k=1..H-1.
                Shape: (num_envs, horizon-1, state_dim).
            Q_terminal: Terminal state cost diagonal. Shape: (num_envs, state_dim).
            R_diag: Control cost diagonal (shared across horizon).
                Shape: (num_envs, control_dim).
            x_target: Target state. Shape: (num_envs, state_dim).
            differentiable: If True, return u with computation graph attached
                (for SAC actor gradient). If False (default), return detached u.
            warm_start: If True (default), initialize PGD from the shifted
                previous solution. If False, initialize from zeros. Set to
                False for training batches where replay buffer shuffling
                makes the previous solution meaningless.
            pred_error: Previous step prediction error for ECNN-style feedback.
                Shape: (num_envs, state_dim=8). None when not using error feedback.
                Held constant across all horizon steps.

        Returns:
            Optimal first control action. Shape: (num_envs, control_dim).
        """
        # Phase 1: PGD solve (always detached, no graph needed).
        # When differentiable=True (training), use fewer PGD iters to leave
        # incomplete convergence -- provides stronger gradient for Phase 2.
        pgd_iters = (
            self.cfg.train_pgd_iters
            if differentiable and self.cfg.train_pgd_iters is not None
            else self.cfg.pgd_iters
        )
        u_converged = self._pgd_solve(
            x_current, dynamics, Q_diag, Q_terminal, R_diag, x_target,
            warm_start=warm_start, pred_error=pred_error, pgd_iters=pgd_iters,
        )

        if not differentiable:
            # Store for warm-start and return detached
            with torch.no_grad():
                self._u_prev.copy_(u_converged)
            return u_converged[:, 0, :].detach()

        # Phase 2: Differentiable 1-step refinement from converged solution
        u_first = self._differentiable_refine(
            u_converged,
            x_current,
            dynamics,
            Q_diag,
            Q_terminal,
            R_diag,
            x_target,
            pred_error=pred_error,
        )

        # Store for warm-start (detached)
        with torch.no_grad():
            self._u_prev.copy_(u_converged.detach())

        return u_first

    def _pgd_solve(
        self,
        x_current: torch.Tensor,
        dynamics: DynamicsMLP,
        Q_diag: torch.Tensor,
        Q_terminal: torch.Tensor,
        R_diag: torch.Tensor,
        x_target: torch.Tensor,
        warm_start: bool = True,
        pred_error: torch.Tensor | None = None,
        pgd_iters: int | None = None,
    ) -> torch.Tensor:
        """Projected Gradient Descent for box-constrained MPC.

        Uses autograd.grad (not backward) to compute gradient only w.r.t.
        u_seq, avoiding unnecessary gradient computation for dynamics parameters.

        Args:
            warm_start: If True, initialize from shifted previous solution.
                If False, initialize from zeros.
            pgd_iters: Override number of PGD iterations. None = use cfg.pgd_iters.

        Returns:
            Converged control sequence. Shape: (num_envs, horizon, control_dim).
        """
        cfg = self.cfg
        iters = pgd_iters if pgd_iters is not None else cfg.pgd_iters

        with torch.inference_mode(False), torch.enable_grad():
            x_c = x_current.detach().clone()
            Q_d = Q_diag.detach().clone()
            Q_t = Q_terminal.detach().clone()
            R_d = R_diag.detach().clone()
            x_t = x_target.detach().clone()
            pe_d = pred_error.detach().clone() if pred_error is not None else None

            u_seq = self._warm_start() if warm_start else torch.zeros_like(self._u_prev)
            u_seq = u_seq.detach().requires_grad_(True)

            for _ in range(iters):
                cost = self._compute_cost(u_seq, x_c, dynamics, Q_d, Q_t, R_d, x_t, pred_error=pe_d)

                # Compute gradient only w.r.t. u_seq (not dynamics parameters)
                (grad,) = torch.autograd.grad(cost, u_seq)

                with torch.no_grad():
                    u_seq = u_seq - cfg.pgd_lr * grad
                    u_seq.clamp_(cfg.u_min, cfg.u_max)
                u_seq = u_seq.detach().requires_grad_(True)

        # Cache final per-env cost for logging
        with torch.no_grad():
            self.last_solve_cost = self._compute_cost(
                u_seq, x_c, dynamics, Q_d, Q_t, R_d, x_t, reduce=False, pred_error=pe_d,
            )

        return u_seq.detach()

    def _differentiable_refine(
        self,
        u_converged: torch.Tensor,
        x_current: torch.Tensor,
        dynamics: DynamicsMLP,
        Q_diag: torch.Tensor,
        Q_terminal: torch.Tensor,
        R_diag: torch.Tensor,
        x_target: torch.Tensor,
        pred_error: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Multi-step GD refinement with graph connectivity for SAC gradient.

        Takes the converged u_seq and performs ``diff_gd_steps`` gradient steps
        with create_graph=True. The computation graph flows:
        u_refined -> dynamics rollout -> cost -> back through dynamics
        and Q_diag/Q_terminal (from cost_map).

        More steps propagate stronger gradients to upstream modules (cost_map,
        dynamics), at the cost of proportionally more GPU memory for the
        computation graph.

        Soft clamp ``u_max * tanh(u / u_max)`` is applied after each GD step
        to keep u within the dynamics training range while preserving gradient.

        Args:
            u_converged: Converged u from PGD solve. Shape: (N, H, C).
            x_current: Current state. Shape: (N, S).
            dynamics: Learned dynamics model.
            Q_diag: Running cost weights (from cost_map). Shape: (N, H-1, S).
            Q_terminal: Terminal cost weights (from cost_map). Shape: (N, S).
            R_diag: Control cost. Shape: (N, C).
            x_target: Target state. Shape: (N, S).
            pred_error: Previous step prediction error. Shape: (N, 8). None when unused.

        Returns:
            Refined first control action WITH graph. Shape: (N, C).
        """
        cfg = self.cfg

        # Start from converged solution, detached from PGD iterations
        u_seq = u_converged.detach().clone().requires_grad_(True)

        # Optional noise perturbation: push u away from the flat minimum
        # so Phase 2 refinement starts with non-trivial cost gradients.
        if cfg.refine_noise_std > 0:
            u_seq = u_seq + cfg.refine_noise_std * torch.randn_like(u_seq)
            u_seq = cfg.u_max * torch.tanh(u_seq / cfg.u_max)
            u_seq = u_seq.detach().requires_grad_(True)

        # Multi-step GD with create_graph=True: builds graph through dynamics.
        # Each step deepens the computation graph, propagating stronger
        # gradients to cost_map/dynamics upstream.
        self._refine_grad_norms.clear()
        for _ in range(cfg.diff_gd_steps):
            cost = self._compute_cost(
                u_seq, x_current, dynamics, Q_diag, Q_terminal, R_diag, x_target,
                pred_error=pred_error,
            )
            grad = torch.autograd.grad(cost, u_seq, create_graph=True)[0]
            # Track per-step gradient norm for diagnostics
            with torch.no_grad():
                self._refine_grad_norms.append(grad.norm(dim=-1).mean().item())
            u_seq = u_seq - cfg.diff_gd_lr * grad
            # Soft clamp: differentiable bound that preserves gradient flow.
            # Hard clamp has zero gradient at the boundary, severing the graph.
            # tanh(u/u_max) asymptotically approaches +-1, so output stays in
            # (-u_max, u_max). Dynamics pred_net is trained on [-u_max, u_max],
            # so extrapolation beyond this range would be unreliable.
            u_seq = cfg.u_max * torch.tanh(u_seq / cfg.u_max)

        # Track max |u| after refinement for verification logging
        with torch.no_grad():
            self.last_u_max_after_refine = u_seq.abs().max().item()

        # Return first step only (receding horizon)
        return u_seq[:, 0, :]

    def _warm_start(self) -> torch.Tensor:
        """Shift previous solution left by one step for warm-starting.

        Returns:
            Warm-started control sequence. Shape: (num_envs, horizon, control_dim).
        """
        u_shifted = torch.zeros_like(self._u_prev)
        u_shifted[:, :-1, :] = self._u_prev[:, 1:, :]
        u_shifted[:, -1, :] = self._u_prev[:, -1, :]  # repeat last
        return u_shifted

    def _compute_cost(
        self,
        u_seq: torch.Tensor,
        x_current: torch.Tensor,
        dynamics: DynamicsMLP,
        Q_diag: torch.Tensor,
        Q_terminal: torch.Tensor,
        R_diag: torch.Tensor,
        x_target: torch.Tensor,
        reduce: bool = True,
        pred_error: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute quadratic tracking cost over the prediction horizon.

        Cost = sum_{k=1}^{N-1} [(x_k - x_target)^T Q_k (x_k - x_target)]
             + sum_{k=0}^{N-1} [u_k^T R u_k]
             + (x_N - x_target)^T Q_terminal (x_N - x_target)

        Note: x_0 state cost is excluded because the current state is not
        a decision variable. In Phase 2 (create_graph=True), including it
        would create misleading gradient to cost_map for Q_0.

        Used by both PGD (Phase 1, detached) and differentiable refinement
        (Phase 2, with create_graph=True for SAC gradient).

        Args:
            reduce: If True, return scalar (summed over envs). If False,
                return per-env cost of shape (num_envs,).
            pred_error: Previous step prediction error. Shape: (N, 8). None when unused.
                Passed through to dynamics for ECNN-style error correction.

        Returns:
            Total cost (scalar if reduce=True, per-env vector otherwise).
        """
        cfg = self.cfg
        total_cost = torch.zeros(x_current.shape[0], device=x_current.device)
        x = x_current

        for k in range(cfg.horizon):
            u_k = u_seq[:, k, :]
            # State cost on predicted states only (k >= 1), skipping x_0.
            # x_0 is the current state that MPC cannot change -- its cost is
            # a constant in PGD (harmless) but creates misleading gradient
            # to cost_map's Q weights in Phase 2 differentiable refinement.
            # Index k-1 maps: k=1->Q[0], k=2->Q[1], ..., k=H-1->Q[H-2],
            # ensuring all H-1 running Q weight slices receive gradient.
            if k > 0:
                dx = x - x_target
                Q_k = Q_diag[:, k - 1, :]
                state_cost = (dx * Q_k * dx).sum(dim=-1)
                total_cost = total_cost + state_cost
            control_cost = (u_k * R_diag * u_k).sum(dim=-1)
            total_cost = total_cost + control_cost
            # pred_error held constant across all horizon steps (TDE assumption:
            # disturbance changes slowly relative to MPC horizon)
            x = dynamics(x, u_k, pred_error=pred_error)

        dx = x - x_target
        terminal_cost = (dx * Q_terminal * dx).sum(dim=-1)
        total_cost = total_cost + terminal_cost

        return total_cost.sum() if reduce else total_cost

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset warm-start buffer for specified environments."""
        self._u_prev[env_ids] = 0.0
