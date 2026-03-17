# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""C-TRPO: Barrier-based Trust Region for Constrained Policy Optimization.

Implements C-TRPO (Muller et al., ICML 2025, arXiv:2411.02957) Option C:
    1. Safe mode: barrier-augmented objective + KL-only trust region
    2. Recovery mode: cost minimization with standard TRPO trust region
    3. Option C (surrogate divergence): barrier curvature in objective gradient
       only; FVP uses pure KL Hessian (no barrier in Fisher matrix)

Key design decisions:
    - No Lagrangian dual variables: lambda completely removed. Barrier function
      naturally enforces constraints by distorting the trust region geometry --
      steps become more conservative near constraint boundaries.
    - Safe/recovery mode per constraint: each constraint independently tracks
      its margin (d_k - J_C_k). Feasible constraints use barrier penalty;
      infeasible constraints trigger recovery mode (cost minimization).
    - Hysteresis in mode switching: recovery exits only when cost drops below
      recovery_threshold_frac * budget, preventing oscillation at the boundary.
    - No detached-std cost ratio needed: in C-TRPO, cost only affects trust
      region geometry (via barrier in the objective gradient). There is no
      lambda * cost_surrogate term, so cost gradient never directly pushes
      std toward zero.
    - Cost advantage normalization maintained: per-constraint standardization
      (NORBC Sec IV-B) equalizes gradient contribution across constraints.
    - LS-gated encoder updates preserved: when line search fails, both actor
      and encoder are frozen to prevent desync.
    - Noise floor preserved: safety net against log_prob divergence.

The algorithm maintains the same interface as RSL-RL PPO (init_storage, act,
process_env_step, compute_returns, update) so it can be used as a drop-in
replacement in the OnPolicyRunner.

Reference:
    Muller et al., "Truly Constrained TRPO", ICML 2025, arXiv:2411.02957.
    Kim et al., "NORBC", IROS 2024 (cost critic, value loss structure).
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

logger = logging.getLogger(__name__)


class ConstraintTRPO:
    """C-TRPO: Barrier-based trust region for constrained policy optimization.

    Key differences from Lagrangian TRPO:
        - No lambda dual variables; barrier penalty replaces Lagrangian
        - Safe mode: reward + barrier penalty objective, KL-only trust region
        - Recovery mode: cost minimization, KL trust region + cost decrease check
        - Option C: barrier curvature in gradient only, FVP is pure KL
    """

    def __init__(
        self,
        policy: nn.Module,
        # TRPO parameters
        max_kl: float = 0.01,
        cg_iters: int = 10,
        cg_damping: float = 0.1,
        line_search_max_backtracks: int = 10,
        line_search_shrink_factor: float = 0.5,
        # Value function parameters
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        value_loss_coef: float = 1.0,
        cost_value_loss_coef: float = 1.0,
        value_lr: float = 1e-3,
        max_grad_norm: float = 1.0,
        # GAE parameters
        gamma: float = 0.99,
        lam: float = 0.95,
        # Constraint parameters
        num_constraints: int = 3,
        constraint_budgets: tuple[float, ...] = (0.15, 0.02, 0.15),
        cost_gamma: float = 0.99,
        cost_lam: float = 0.95,
        # Line search acceptance threshold
        line_search_kl_margin: float = 1.5,
        # C-TRPO barrier parameters
        beta: float = 0.01,
        recovery_threshold_frac: float = 0.8,
        # Encoder z bounds
        z_bounds_coef: float = 0.3,
        # Encoder update
        num_encoder_epochs: int = 5,
        encoder_lr: float = 1e-3,
        # Device
        device: str = "cpu",
        # Unused kwargs from RSL-RL config forwarding (including legacy Lagrangian params)
        **kwargs: Any,
    ) -> None:
        if kwargs:
            logger.debug("ConstraintTRPO ignoring unexpected kwargs: %s", list(kwargs.keys()))

        self.device = device
        self.policy = policy
        self.policy.to(self.device)

        # TRPO parameters
        self.max_kl = max_kl
        self.cg_iters = cg_iters
        self.cg_damping = cg_damping
        self.line_search_max_backtracks = line_search_max_backtracks
        self.line_search_shrink_factor = line_search_shrink_factor

        # Value function parameters
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.cost_value_loss_coef = cost_value_loss_coef
        self.max_grad_norm = max_grad_norm

        # GAE parameters
        self.gamma = gamma
        self.lam = lam

        # Constraint parameters
        self.num_constraints = num_constraints
        self.cost_gamma = cost_gamma
        self.cost_lam = cost_lam
        self.line_search_kl_margin = line_search_kl_margin
        self.z_bounds_coef = z_bounds_coef
        self.num_encoder_epochs = num_encoder_epochs

        # C-TRPO barrier parameters
        self.beta = beta
        self.recovery_threshold_frac = recovery_threshold_frac
        self._in_recovery = [False] * num_constraints
        self._margins = torch.zeros(num_constraints, device=device)

        if cost_gamma >= 1.0:
            raise ValueError(f"cost_gamma must be < 1.0, got {cost_gamma}")

        # Discounted budgets: d_k = D_k / (1 - gamma)
        self.d_k = torch.tensor(
            [b / (1.0 - cost_gamma) for b in constraint_budgets],
            device=device,
            dtype=torch.float32,
        )

        # Separate parameter groups:
        # - Actor params: TRPO natural gradient (no optimizer)
        # - Encoder params: separate Adam (indirect distribution influence)
        # - Value params (critic + cost_critic): Adam optimizer
        value_params = []
        encoder_params = []
        self._policy_params = []  # Actor-only for TRPO

        encoder_prefixes = ("encoder",)
        for name, param in self.policy.named_parameters():
            is_value = name.startswith("critic") or name.startswith("cost_critic")
            is_encoder = any(name.startswith(p) for p in encoder_prefixes)
            if is_value:
                value_params.append(param)
            elif is_encoder:
                encoder_params.append(param)
            else:
                self._policy_params.append(param)

        self._value_params = value_params
        self.value_optimizer = optim.Adam(value_params, lr=value_lr)
        self._has_encoder_params = len(encoder_params) > 0
        self.encoder_lr = encoder_lr
        if self._has_encoder_params:
            self._encoder_params = encoder_params
            self.encoder_optimizer = optim.Adam(encoder_params, lr=encoder_lr, weight_decay=1e-5)
        else:
            self._encoder_params = []
            self.encoder_optimizer = None
        logger.info(
            "ConstraintTRPO (C-TRPO): %d actor params (TRPO), %d encoder params (Adam), %d value params (Adam)",
            len(self._policy_params),
            len(encoder_params),
            len(value_params),
        )

        # Iteration counter (updated in update())
        self._iteration = 0

        # Storage
        self.storage: RolloutStorage | None = None
        self.transition = RolloutStorage.Transition()

        # Cost storage (dynamically attached to RolloutStorage)
        self._cost_storage_initialized = False

        # Learning rate (compatibility field for OnPolicyRunner logging)
        self.learning_rate = value_lr

        # Compatibility with OnPolicyRunner (expects these attributes)
        self.rnd = None  # No random network distillation
        self.optimizer = self.value_optimizer  # For checkpoint save/load

    # ==================================================================
    # Storage & Rollout Interface (matches PPO)
    # ==================================================================

    def init_storage(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],
    ) -> None:
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            self.device,
        )
        # Attach cost tensors as extra attributes
        T, N, K = num_transitions_per_env, num_envs, self.num_constraints
        self.storage.costs = torch.zeros(T, N, K, device=self.device)
        self.storage.cost_values = torch.zeros(T, N, K, device=self.device)
        self.storage.cost_returns = torch.zeros(T, N, K, device=self.device)
        self.storage.cost_advantages = torch.zeros(T, N, K, device=self.device)
        self._cost_storage_initialized = True

    def act(self, obs: TensorDict) -> torch.Tensor:
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()

        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        self.transition.observations = obs

        # Store cost values for this step
        if hasattr(self.policy, "evaluate_costs"):
            self._current_cost_values = self.policy.evaluate_costs(obs).detach()
        else:
            self._current_cost_values = torch.zeros(obs.batch_size[0], self.num_constraints, device=self.device)

        return self.transition.actions

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        self.policy.update_normalization(obs)

        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Bootstrapping on time outs
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # Store costs from environment
        step = self.storage.step
        costs = extras.get("costs", torch.zeros(self.storage.num_envs, self.num_constraints, device=self.device))

        # Bootstrap cost values on time outs (same logic as reward bootstrapping)
        if "time_outs" in extras:
            time_out_mask = extras["time_outs"].unsqueeze(1).to(self.device)  # (N, 1)
            costs = costs + self.cost_gamma * self._current_cost_values * time_out_mask
        self.storage.costs[step] = costs
        self.storage.cost_values[step] = self._current_cost_values

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        # Standard reward GAE
        last_values = self.policy.evaluate(obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

        # Cost GAE (K separate passes)
        if hasattr(self.policy, "evaluate_costs"):
            last_cost_values = self.policy.evaluate_costs(obs).detach()
        else:
            last_cost_values = torch.zeros(self.storage.num_envs, self.num_constraints, device=self.device)
        self._compute_cost_returns(last_cost_values)

    def _compute_cost_returns(self, last_cost_values: torch.Tensor) -> None:
        """Compute cost GAE returns for each constraint independently."""
        T = self.storage.num_transitions_per_env
        for k in range(self.num_constraints):
            advantage = torch.zeros(self.storage.num_envs, 1, device=self.device)
            for step in reversed(range(T)):
                next_cv = (
                    last_cost_values[:, k : k + 1]
                    if step == T - 1
                    else self.storage.cost_values[step + 1, :, k : k + 1]
                )
                not_done = 1.0 - self.storage.dones[step].float()
                delta = (
                    self.storage.costs[step, :, k : k + 1]
                    + not_done * self.cost_gamma * next_cv
                    - self.storage.cost_values[step, :, k : k + 1]
                )
                advantage = delta + not_done * self.cost_gamma * self.cost_lam * advantage
                self.storage.cost_returns[step, :, k : k + 1] = advantage + self.storage.cost_values[step, :, k : k + 1]
            self.storage.cost_advantages[:, :, k : k + 1] = (
                self.storage.cost_returns[:, :, k : k + 1] - self.storage.cost_values[:, :, k : k + 1]
            )

        # Per-constraint cost advantage standardization (NORBC Sec IV-B).
        for k in range(self.num_constraints):
            adv_k = self.storage.cost_advantages[:, :, k]
            if not torch.isfinite(adv_k).all():
                logger.warning("Non-finite cost advantages for constraint %d, zeroing.", k)
                self.storage.cost_advantages[:, :, k] = 0.0
            else:
                self.storage.cost_advantages[:, :, k] = (adv_k - adv_k.mean()) / (adv_k.std() + 1e-8)

    # ==================================================================
    # C-TRPO Barrier
    # ==================================================================

    def _compute_margins(self, mean_cost_returns: torch.Tensor) -> None:
        """Update per-constraint margins and recovery mode flags.

        Args:
            mean_cost_returns: Mean discounted cost return per constraint, shape (K,).
        """
        for k in range(self.num_constraints):
            self._margins[k] = self.d_k[k] - mean_cost_returns[k]
            if self._margins[k] <= 0:
                self._in_recovery[k] = True
            elif mean_cost_returns[k] < self.d_k[k] * self.recovery_threshold_frac:
                self._in_recovery[k] = False
            # else: hysteresis - keep current mode

    def _compute_barrier_penalty(self, cost_surrogates: list[torch.Tensor]) -> torch.Tensor:
        """Compute linearized barrier divergence for safe-mode constraints.

        For each feasible, safe-mode constraint k:
            penalty_k = beta * phi''(margin_k) * A_C_k^2
        where phi''(m) = 1/m^2 is the log-barrier second derivative.

        Args:
            cost_surrogates: List of per-constraint cost surrogate tensors.

        Returns:
            Scalar barrier penalty to add to the policy objective.
        """
        # Note: the paper's Bregman divergence has a 1/2 factor (Eq. 7):
        #   D_phi = (1/2) * phi'' * delta^2
        # We absorb this into beta for simplicity. Effective barrier strength
        # is 2x the paper's semantics when using the same beta value.
        penalty = torch.tensor(0.0, device=self.device)
        for k in range(self.num_constraints):
            if self._margins[k] > 0 and not self._in_recovery[k]:
                phi_pp = 1.0 / (self._margins[k].pow(2) + 1e-8)
                penalty = penalty + self.beta * phi_pp * cost_surrogates[k].pow(2)
        return penalty

    # ==================================================================
    # TRPO Core
    # ==================================================================

    def _get_policy_params_flat(self) -> torch.Tensor:
        """Flatten all policy parameters into a single vector."""
        return torch.cat([p.view(-1) for p in self._policy_params])

    def _set_policy_params_flat(self, flat_params: torch.Tensor) -> None:
        """Set policy parameters from a flat vector."""
        offset = 0
        for p in self._policy_params:
            numel = p.numel()
            p.data.copy_(flat_params[offset : offset + numel].view_as(p))
            offset += numel

    def _linearized_surrogate_safe(
        self,
        obs: TensorDict,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        cost_advantages: torch.Tensor,
        old_log_prob: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate safe-mode objective: reward surrogate - barrier penalty.

        The barrier penalty is computed from per-constraint cost surrogates
        using the linearized log-barrier second derivative (Option C).
        No entropy term: C-TRPO relies on KL trust region for exploration.
        """
        self.policy.act(obs)
        log_prob = self.policy.get_actions_log_prob(actions)
        ratio = torch.exp(log_prob - old_log_prob)
        reward_surr = -(advantages * ratio).mean()

        # Per-constraint cost surrogates (standard ratio, no detached std needed)
        cost_surrogates = []
        for k in range(self.num_constraints):
            cost_surr_k = (ratio * cost_advantages[:, k]).mean()
            cost_surrogates.append(cost_surr_k)

        barrier_penalty = self._compute_barrier_penalty(cost_surrogates)
        return reward_surr + barrier_penalty

    def _linearized_surrogate_recovery(
        self,
        obs: TensorDict,
        actions: torch.Tensor,
        cost_advantages: torch.Tensor,
        old_log_prob: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate recovery-mode objective: minimize cost for recovery constraints.

        Only includes recovery-mode constraints. Returns a surrogate whose
        gradient direction reduces cost.
        """
        self.policy.act(obs)
        log_prob = self.policy.get_actions_log_prob(actions)
        ratio = torch.exp(log_prob - old_log_prob)

        recovery_surr = torch.tensor(0.0, device=self.device)
        for k in range(self.num_constraints):
            if self._in_recovery[k]:
                cost_surr_k = (ratio * cost_advantages[:, k]).mean()
                recovery_surr = recovery_surr + cost_surr_k
        return recovery_surr

    def _kl_divergence(self, obs: TensorDict, old_mu: torch.Tensor, old_sigma: torch.Tensor) -> torch.Tensor:
        """Compute mean KL(pi_old || pi_new) analytically for Gaussian."""
        self.policy.act(obs)
        mu = self.policy.action_mean
        sigma = self.policy.action_std
        kl = (
            torch.log((sigma / old_sigma).clamp(min=1e-5))
            + (old_sigma.pow(2) + (old_mu - mu).pow(2)) / (2.0 * sigma.pow(2))
            - 0.5
        )
        return kl.sum(dim=-1).mean()

    def _flat_grad(self, loss: torch.Tensor, params: list[nn.Parameter], retain_graph: bool = False) -> torch.Tensor:
        """Compute flattened gradient of loss w.r.t. params."""
        grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, create_graph=False)
        return torch.cat([g.contiguous().view(-1) for g in grads])

    def _fisher_vector_product(
        self,
        obs: TensorDict,
        old_mu: torch.Tensor,
        old_sigma: torch.Tensor,
        vector: torch.Tensor,
    ) -> torch.Tensor:
        """Compute F @ v without forming F, using double backprop on KL.

        Option C core: FVP uses pure KL Hessian only. Barrier curvature
        is NOT included in the Fisher matrix -- it only affects the objective
        gradient. This keeps the CG solver stable and well-conditioned.
        """
        # Forward pass to get current distribution
        self.policy.act(obs)
        mu = self.policy.action_mean
        sigma = self.policy.action_std

        # KL divergence
        kl = (
            (
                torch.log((sigma / old_sigma).clamp(min=1e-5))
                + (old_sigma.pow(2) + (old_mu - mu).pow(2)) / (2.0 * sigma.pow(2))
                - 0.5
            )
            .sum(dim=-1)
            .mean()
        )

        # First derivative of KL
        kl_grads = torch.autograd.grad(kl, self._policy_params, create_graph=True)
        flat_kl_grad = torch.cat([g.contiguous().view(-1) for g in kl_grads])

        # Hessian-vector product: d/d_theta (flat_kl_grad . vector)
        kl_v = (flat_kl_grad * vector).sum()
        hvp_grads = torch.autograd.grad(kl_v, self._policy_params, retain_graph=False)
        fvp = torch.cat([g.contiguous().view(-1) for g in hvp_grads])

        return fvp + self.cg_damping * vector

    def _conjugate_gradient(
        self,
        obs: TensorDict,
        old_mu: torch.Tensor,
        old_sigma: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        """Solve F @ x = b using conjugate gradient.

        Returns natural gradient direction x = F^{-1} @ g.
        """
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rdotr = r.dot(r)

        for _ in range(self.cg_iters):
            fvp = self._fisher_vector_product(obs, old_mu, old_sigma, p)
            alpha = rdotr / (p.dot(fvp) + 1e-8)
            x += alpha * p
            r -= alpha * fvp
            new_rdotr = r.dot(r)
            if new_rdotr < 1e-10:
                break
            beta = new_rdotr / (rdotr + 1e-8)
            p = r + beta * p
            rdotr = new_rdotr

        return x

    def _line_search_safe(
        self,
        obs: TensorDict,
        actions: torch.Tensor,
        old_log_prob: torch.Tensor,
        advantages: torch.Tensor,
        cost_advantages: torch.Tensor,
        old_mu: torch.Tensor,
        old_sigma: torch.Tensor,
        step_dir: torch.Tensor,
        old_loss: torch.Tensor,
    ) -> bool:
        """Safe-mode line search: check KL constraint only.

        Barrier penalty is already embedded in the objective, so cost
        constraints are implicitly enforced. Only need to verify:
            1. Surrogate improvement > 0
            2. KL divergence <= max_kl * margin
        """
        old_params = self._get_policy_params_flat()
        step_size = 1.0
        kl_limit = self.max_kl * self.line_search_kl_margin

        for _ in range(self.line_search_max_backtracks):
            new_params = old_params + step_size * step_dir
            self._set_policy_params_flat(new_params)

            with torch.no_grad():
                new_loss = self._linearized_surrogate_safe(
                    obs, actions, advantages, cost_advantages, old_log_prob
                )
                kl = self._kl_divergence(obs, old_mu, old_sigma)

            improvement = old_loss - new_loss
            if improvement > 0 and kl <= kl_limit:
                return True

            step_size *= self.line_search_shrink_factor

        # Revert to old parameters if no valid step found
        self._set_policy_params_flat(old_params)
        return False

    def _line_search_recovery(
        self,
        obs: TensorDict,
        actions: torch.Tensor,
        old_log_prob: torch.Tensor,
        cost_advantages: torch.Tensor,
        old_mu: torch.Tensor,
        old_sigma: torch.Tensor,
        step_dir: torch.Tensor,
        old_loss: torch.Tensor,
    ) -> bool:
        """Recovery-mode line search: check KL + cost decrease.

        Must verify both:
            1. KL divergence <= max_kl * margin
            2. Cost surrogate decreased (recovery making progress)
        """
        old_params = self._get_policy_params_flat()
        step_size = 1.0
        kl_limit = self.max_kl * self.line_search_kl_margin

        for _ in range(self.line_search_max_backtracks):
            new_params = old_params + step_size * step_dir
            self._set_policy_params_flat(new_params)

            with torch.no_grad():
                new_loss = self._linearized_surrogate_recovery(
                    obs, actions, cost_advantages, old_log_prob
                )
                kl = self._kl_divergence(obs, old_mu, old_sigma)

            improvement = old_loss - new_loss
            if improvement > 0 and kl <= kl_limit:
                return True

            step_size *= self.line_search_shrink_factor

        self._set_policy_params_flat(old_params)
        return False

    # ==================================================================
    # Main Update
    # ==================================================================

    def update(self) -> dict[str, float]:
        """Execute one iteration of C-TRPO update.

        Update order:
            1. Compute margins + determine safe/recovery mode per constraint
            2. TRPO policy update (safe or recovery, full-batch)
            3. Encoder update (gated on line search success)
            4. Value function update (pure MSE)
        """
        self._iteration += 1

        # Flatten storage (clone to escape inference_mode)
        obs_flat = self.storage.observations.flatten(0, 1).clone()
        actions_flat = self.storage.actions.flatten(0, 1).clone()
        returns_flat = self.storage.returns.flatten(0, 1).clone()
        advantages_flat = self.storage.advantages.flatten(0, 1).clone()

        # Standardize reward advantages to O(1) scale.
        adv_raw_std = advantages_flat.std()
        if adv_raw_std > 1e-8:
            advantages_flat = (advantages_flat - advantages_flat.mean()) / adv_raw_std

        old_log_prob_flat = self.storage.actions_log_prob.flatten(0, 1).clone()
        old_mu_flat = self.storage.mu.flatten(0, 1).clone()
        old_sigma_flat = self.storage.sigma.flatten(0, 1).clone()

        # Cost storage flatten
        cost_returns_flat = self.storage.cost_returns.flatten(0, 1).clone()  # (B, K)
        cost_advantages_flat = self.storage.cost_advantages.flatten(0, 1).clone()  # (B, K)

        batch_size = obs_flat.batch_size[0]

        # Mean cost returns (computed once, needed for margins + logging)
        # Clamp to non-negative: cost value errors can make GAE return negative,
        # which would inflate barrier margin (d_k - (-X) = d_k + X).
        mean_cost_returns = cost_returns_flat.mean(dim=0).clamp(min=0.0)  # (K,)

        # ------------------------------------------------------------------
        # 1. Compute margins and determine safe/recovery mode
        # ------------------------------------------------------------------
        self._compute_margins(mean_cost_returns)

        any_recovery = any(self._in_recovery)
        all_recovery = all(self._in_recovery)

        # Compute violations for logging
        violations = []
        for k in range(self.num_constraints):
            violations.append((mean_cost_returns[k] - self.d_k[k]).item())

        # ------------------------------------------------------------------
        # 2. TRPO policy update (full-batch, single step)
        # ------------------------------------------------------------------
        # Determine mode: if ANY constraint is in recovery, do recovery step.
        # Recovery takes priority since feasibility must be restored before
        # reward optimization can proceed meaningfully.
        if any_recovery:
            mode = "recovery"
        else:
            mode = "safe"

        if mode == "safe":
            ls_success = self._trpo_step_safe(
                obs_flat, actions_flat, advantages_flat, cost_advantages_flat,
                old_log_prob_flat, old_mu_flat, old_sigma_flat,
            )
        else:
            ls_success = self._trpo_step_recovery(
                obs_flat, actions_flat, cost_advantages_flat,
                old_log_prob_flat, old_mu_flat, old_sigma_flat,
            )

        # Noise floor: numerical safety net to prevent log_prob divergence
        min_log_std = math.log(0.25)
        with torch.no_grad():
            self.policy.log_std.data.clamp_(min=min_log_std)

        # Measure KL right after TRPO step + clamp (before encoder update shifts z)
        with torch.no_grad():
            kl_after_trpo = self._kl_divergence(obs_flat, old_mu_flat, old_sigma_flat).item()

        # ------------------------------------------------------------------
        # 3. Encoder update (gated on ls_success)
        # ------------------------------------------------------------------
        mean_z_bounds_loss = 0.0
        if self.encoder_optimizer is not None and ls_success:
            mean_z_bounds_loss = self._update_encoder(obs_flat, advantages_flat, old_log_prob_flat, actions_flat)
        elif self.encoder_optimizer is not None and hasattr(self.policy, "z_bounds_loss"):
            # Still compute z_bounds_loss for logging, but don't step
            with torch.no_grad():
                self.policy.act(obs_flat)
                mean_z_bounds_loss = self.policy.z_bounds_loss().item()

        # Compute KL after full update for logging
        with torch.no_grad():
            mean_kl = self._kl_divergence(obs_flat, old_mu_flat, old_sigma_flat).item()

        # ------------------------------------------------------------------
        # 4. Value function update (pure MSE)
        # ------------------------------------------------------------------
        mean_value_loss, mean_cost_value_loss = self._update_values(
            obs_flat, returns_flat, cost_returns_flat, batch_size
        )

        # ------------------------------------------------------------------
        # Compute barrier penalty for logging
        # ------------------------------------------------------------------
        with torch.no_grad():
            self.policy.act(obs_flat)
            log_prob = self.policy.get_actions_log_prob(actions_flat)
            ratio = torch.exp(log_prob - old_log_prob_flat.squeeze(-1))
            cost_surrogates = []
            for k in range(self.num_constraints):
                cost_surr_k = (ratio * cost_advantages_flat[:, k]).mean()
                cost_surrogates.append(cost_surr_k)
            barrier_penalty_val = self._compute_barrier_penalty(cost_surrogates).item()

        # ------------------------------------------------------------------
        # Store monitoring metrics (read by ConstraintEncoderRunner)
        # ------------------------------------------------------------------
        self._last_cost_returns = [mean_cost_returns[k].item() for k in range(self.num_constraints)]
        self._last_violations = violations
        self._last_line_search_success = float(ls_success)
        self._last_margins = [self._margins[k].item() for k in range(self.num_constraints)]
        self._last_in_recovery = [float(r) for r in self._in_recovery]
        self._last_barrier_penalty = barrier_penalty_val
        self._last_mode = 2 if all_recovery else (1 if any_recovery else 0)

        # Compute entropy for logging
        with torch.no_grad():
            self.policy.act(obs_flat)
            mean_entropy = self.policy.entropy.mean().item()

        # Clear storage
        self.storage.clear()

        # ------------------------------------------------------------------
        # Return loss dict
        # ------------------------------------------------------------------
        loss_dict: dict[str, float] = {
            "value_function": mean_value_loss,
            "barrier_penalty": barrier_penalty_val,
            "entropy": mean_entropy,
            "kl": mean_kl,
            "kl_trpo": kl_after_trpo,
            "cost_value": mean_cost_value_loss,
            "mode": float(self._last_mode),
            "adv_raw_std": adv_raw_std.item(),
        }
        if hasattr(self.policy, "z_bounds_loss"):
            loss_dict["z_bounds"] = mean_z_bounds_loss

        return loss_dict

    # ==================================================================
    # Internal: TRPO steps for each mode
    # ==================================================================

    def _trpo_step_safe(
        self,
        obs_flat: TensorDict,
        actions_flat: torch.Tensor,
        advantages_flat: torch.Tensor,
        cost_advantages_flat: torch.Tensor,
        old_log_prob_flat: torch.Tensor,
        old_mu_flat: torch.Tensor,
        old_sigma_flat: torch.Tensor,
    ) -> bool:
        """Execute safe-mode TRPO step: reward + barrier penalty objective."""
        # Compute gradient of barrier-augmented objective
        self.policy.act(obs_flat)
        log_prob = self.policy.get_actions_log_prob(actions_flat)
        ratio = torch.exp(log_prob - old_log_prob_flat.squeeze(-1))
        reward_surrogate = -(advantages_flat.squeeze(-1) * ratio).mean()

        cost_surrogates = []
        for k in range(self.num_constraints):
            cost_surr_k = (ratio * cost_advantages_flat[:, k]).mean()
            cost_surrogates.append(cost_surr_k)
        barrier_penalty = self._compute_barrier_penalty(cost_surrogates)

        policy_loss = reward_surrogate + barrier_penalty

        # retain_graph=False: CG solver builds its own fresh graphs via FVP.
        # Encoder gets separate multi-step updates in _update_encoder().
        g = self._flat_grad(policy_loss, self._policy_params, retain_graph=False)

        # Natural gradient via conjugate gradient: x = F^{-1} g
        nat_grad = self._conjugate_gradient(obs_flat, old_mu_flat, old_sigma_flat, g)

        # Step size: sqrt(2 * max_kl / (g^T F^{-1} g))
        shs = 0.5 * nat_grad.dot(g)

        if shs <= 0 or not torch.isfinite(shs):
            logger.warning("TRPO safe: shs=%.6e non-positive or non-finite, skipping", shs.item())
            return False

        step_scale = torch.sqrt(self.max_kl / shs)
        step_dir = -step_scale * nat_grad

        if not torch.isfinite(step_dir).all():
            logger.warning("TRPO safe: step_dir contains NaN/Inf, skipping")
            return False

        with torch.no_grad():
            old_loss = self._linearized_surrogate_safe(
                obs_flat, actions_flat, advantages_flat.squeeze(-1),
                cost_advantages_flat, old_log_prob_flat.squeeze(-1),
            )

        return self._line_search_safe(
            obs_flat, actions_flat, old_log_prob_flat.squeeze(-1),
            advantages_flat.squeeze(-1), cost_advantages_flat,
            old_mu_flat, old_sigma_flat, step_dir, old_loss,
        )

    def _trpo_step_recovery(
        self,
        obs_flat: TensorDict,
        actions_flat: torch.Tensor,
        cost_advantages_flat: torch.Tensor,
        old_log_prob_flat: torch.Tensor,
        old_mu_flat: torch.Tensor,
        old_sigma_flat: torch.Tensor,
    ) -> bool:
        """Execute recovery-mode TRPO step: minimize cost for infeasible constraints."""
        # Compute gradient of cost minimization objective
        self.policy.act(obs_flat)
        log_prob = self.policy.get_actions_log_prob(actions_flat)
        ratio = torch.exp(log_prob - old_log_prob_flat.squeeze(-1))

        recovery_loss = torch.tensor(0.0, device=self.device)
        for k in range(self.num_constraints):
            if self._in_recovery[k]:
                cost_surr_k = (ratio * cost_advantages_flat[:, k]).mean()
                recovery_loss = recovery_loss + cost_surr_k

        # Encoder gets separate multi-step updates in _update_encoder()
        # (not gated on recovery mode -- encoder uses Adam, not TRPO trust region)

        g = self._flat_grad(recovery_loss, self._policy_params, retain_graph=False)

        nat_grad = self._conjugate_gradient(obs_flat, old_mu_flat, old_sigma_flat, g)
        shs = 0.5 * nat_grad.dot(g)

        if shs <= 0 or not torch.isfinite(shs):
            logger.warning("TRPO recovery: shs=%.6e non-positive or non-finite, skipping", shs.item())
            return False

        step_scale = torch.sqrt(self.max_kl / shs)
        step_dir = -step_scale * nat_grad

        if not torch.isfinite(step_dir).all():
            logger.warning("TRPO recovery: step_dir contains NaN/Inf, skipping")
            return False

        with torch.no_grad():
            old_loss = self._linearized_surrogate_recovery(
                obs_flat, actions_flat, cost_advantages_flat,
                old_log_prob_flat.squeeze(-1),
            )

        return self._line_search_recovery(
            obs_flat, actions_flat, old_log_prob_flat.squeeze(-1),
            cost_advantages_flat, old_mu_flat, old_sigma_flat,
            step_dir, old_loss,
        )

    def _update_encoder(
        self,
        obs_flat: TensorDict,
        advantages_flat: torch.Tensor,
        old_log_prob_flat: torch.Tensor,
        actions_flat: torch.Tensor,
    ) -> float:
        """Multi-step encoder update with fresh forward passes.

        TRPO does a single full-batch policy step, while PPO does ~20 mini-batch
        updates (5 epochs x 4 batches). The encoder in PPO gets 20 gradient steps
        per iteration; without compensation, the C-TRPO encoder would get only 1.

        This method runs num_encoder_epochs fresh forward/backward passes through
        the encoder, each time recomputing the reward surrogate and z_bounds loss.
        The actor params are frozen (only encoder_optimizer steps), so this is safe.
        """
        mean_z_bounds_loss = 0.0

        for _epoch in range(self.num_encoder_epochs):
            self.encoder_optimizer.zero_grad()
            has_grads = False

            # Fresh forward pass through encoder + actor
            self.policy.act(obs_flat)
            log_prob = self.policy.get_actions_log_prob(actions_flat)
            ratio = torch.exp(log_prob - old_log_prob_flat.squeeze(-1))
            reward_surrogate = -(advantages_flat.squeeze(-1) * ratio).mean()

            # Reward signal gradients to encoder (no cost/barrier -- see docstring
            # in _trpo_step_safe for rationale)
            enc_grads = torch.autograd.grad(
                reward_surrogate,
                self._encoder_params,
                retain_graph=True,
                allow_unused=True,
            )
            for i, p in enumerate(self._encoder_params):
                if i < len(enc_grads) and enc_grads[i] is not None:
                    p.grad = enc_grads[i]
                    has_grads = True

            # Accumulate z_bounds gradients (same forward pass, shared z tensor)
            if hasattr(self.policy, "z_bounds_loss"):
                z_b_loss = self.policy.z_bounds_loss()
                mean_z_bounds_loss = z_b_loss.item()
                if z_b_loss.requires_grad:
                    z_bounds_grads = torch.autograd.grad(
                        z_b_loss,
                        self._encoder_params,
                        allow_unused=True,
                    )
                    for i, p in enumerate(self._encoder_params):
                        if i < len(z_bounds_grads) and z_bounds_grads[i] is not None:
                            if p.grad is not None:
                                p.grad = p.grad + z_bounds_grads[i]
                            else:
                                p.grad = z_bounds_grads[i]
                            has_grads = True

            if has_grads:
                nn.utils.clip_grad_norm_(self._encoder_params, max_norm=0.2)
                self.encoder_optimizer.step()

        return mean_z_bounds_loss

    def _update_values(
        self,
        obs_flat: TensorDict,
        returns_flat: torch.Tensor,
        cost_returns_flat: torch.Tensor,
        batch_size: int,
    ) -> tuple[float, float]:
        """Update value functions (reward + cost) via MSE."""
        mean_value_loss = 0.0
        mean_cost_value_loss = 0.0
        num_value_updates = 0

        for _epoch in range(self.num_learning_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            mini_batch_size = batch_size // self.num_mini_batches

            for mb in range(self.num_mini_batches):
                start = mb * mini_batch_size
                end = (mb + 1) * mini_batch_size
                idx = indices[start:end]

                obs_mb = obs_flat[idx]
                returns_mb = returns_flat[idx]
                cost_returns_mb = cost_returns_flat[idx]

                # Reward value loss (MSE)
                value_pred = self.policy.evaluate(obs_mb)
                value_loss = (returns_mb - value_pred).pow(2).mean()

                # Cost value loss (MSE, per constraint, d_k^2-normalized)
                cost_value_loss = torch.tensor(0.0, device=self.device)
                if hasattr(self.policy, "evaluate_costs"):
                    cost_value_pred = self.policy.evaluate_costs(obs_mb)
                    target = cost_returns_mb.clamp(min=0.0)
                    per_k_mse = (target - cost_value_pred).pow(2).mean(dim=0)  # (K,)
                    cost_value_loss = (per_k_mse / self.d_k.pow(2).clamp(min=0.01)).mean()

                total_value_loss = self.value_loss_coef * value_loss + self.cost_value_loss_coef * cost_value_loss

                self.value_optimizer.zero_grad()
                total_value_loss.backward()
                nn.utils.clip_grad_norm_(self._value_params, self.max_grad_norm)
                self.value_optimizer.step()

                mean_value_loss += value_loss.item()
                mean_cost_value_loss += cost_value_loss.item()
                num_value_updates += 1

        if num_value_updates > 0:
            mean_value_loss /= num_value_updates
            mean_cost_value_loss /= num_value_updates

        return mean_value_loss, mean_cost_value_loss

    # ==================================================================
    # Compatibility
    # ==================================================================

    def set_max_iterations(self, max_iterations: int) -> None:
        """Configure iteration-based schedules.

        No lambda warmup needed in C-TRPO (no lambda), but kept for
        interface compatibility with ConstraintEncoderRunner.
        """
        logger.info("[ConstraintTRPO] C-TRPO barrier mode, beta=%.4f, max_iterations=%d", self.beta, max_iterations)
