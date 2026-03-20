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
        num_encoder_epochs: int = 1,
        encoder_lr: float = 3e-4,
        # Device
        device: str = "cpu",
        **_kwargs,
    ) -> None:
        if _kwargs:
            logger.debug("ConstraintTRPO ignoring unexpected kwargs: %s", list(_kwargs.keys()))
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

        # Initialize monitoring attributes (read by ConstraintEncoderRunner before first update)
        self._cached_barrier_penalty = 0.0
        self._last_cost_returns = [0.0] * num_constraints
        self._last_violations = [0.0] * num_constraints
        self._last_line_search_success = 0.0
        self._last_margins = [0.0] * num_constraints
        self._last_in_recovery = [0.0] * num_constraints
        self._last_barrier_penalty = 0.0
        self._last_mode = 0

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

        # RND compatibility (OnPolicyRunner.learn checks self.alg.rnd at line 84)
        self.rnd = None

        # Storage
        self.storage: RolloutStorage | None = None
        self.transition = RolloutStorage.Transition()

        # Learning rate (compatibility field for OnPolicyRunner logging)
        self.learning_rate = value_lr

        # Compatibility with OnPolicyRunner (expects this attribute for checkpoint save/load)
        self.optimizer = self.value_optimizer

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
        self._current_cost_values = self.policy.evaluate_costs(obs).detach()

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
        last_cost_values = self.policy.evaluate_costs(obs).detach()
        self._compute_cost_returns(last_cost_values)

    def _compute_cost_returns(self, last_cost_values: torch.Tensor) -> None:
        """Compute cost GAE returns for all constraints simultaneously."""
        T = self.storage.num_transitions_per_env
        N = self.storage.num_envs

        # Vectorized GAE across all K constraints in a single T-loop
        advantage = torch.zeros(N, self.num_constraints, device=self.device)
        for step in reversed(range(T)):
            next_cv = last_cost_values if step == T - 1 else self.storage.cost_values[step + 1]
            not_done = (1.0 - self.storage.dones[step].float().squeeze(-1)).unsqueeze(-1)  # (N, 1)
            delta = self.storage.costs[step] + not_done * self.cost_gamma * next_cv - self.storage.cost_values[step]
            advantage = delta + not_done * self.cost_gamma * self.cost_lam * advantage
            self.storage.cost_returns[step] = advantage + self.storage.cost_values[step]
        self.storage.cost_advantages = self.storage.cost_returns - self.storage.cost_values

        # Per-constraint cost advantage standardization (NORBC Sec IV-B).
        for k in range(self.num_constraints):
            if not torch.isfinite(self.storage.cost_advantages[:, :, k]).all():
                logger.warning("Non-finite cost advantages for constraint %d, zeroing.", k)
                self.storage.cost_advantages[:, :, k] = 0.0
        mean = self.storage.cost_advantages.mean(dim=(0, 1), keepdim=True)
        std = self.storage.cost_advantages.std(dim=(0, 1), keepdim=True)
        self.storage.cost_advantages = (self.storage.cost_advantages - mean) / (std + 1e-8)

    # ==================================================================
    # C-TRPO Barrier
    # ==================================================================

    def _compute_margins(self, mean_cost_returns: torch.Tensor) -> None:
        """Update per-constraint margins and recovery mode flags.

        Args:
            mean_cost_returns: Mean discounted cost return per constraint, shape (K,).
        """
        self._margins = self.d_k - mean_cost_returns
        # Recovery mode transitions: 3-way hysteresis per constraint
        for k in range(self.num_constraints):
            if self._margins[k] <= 0:
                self._in_recovery[k] = True
            elif mean_cost_returns[k] < self.d_k[k] * self.recovery_threshold_frac:
                self._in_recovery[k] = False
            # else: hysteresis - keep current mode

    def _compute_cost_surrogates(self, ratio: torch.Tensor, cost_advantages: torch.Tensor) -> torch.Tensor:
        """Compute per-constraint cost surrogates in a single vectorized op.

        Args:
            ratio: Importance sampling ratio pi/pi_old, shape (B,).
            cost_advantages: Per-constraint advantages, shape (B, K).

        Returns:
            Cost surrogates, shape (K,).
        """
        return (ratio.unsqueeze(-1) * cost_advantages).mean(dim=0)

    def _compute_barrier_penalty(self, cost_surrogates: torch.Tensor) -> torch.Tensor:
        """Compute linearized barrier divergence for safe-mode constraints.

        For each feasible, safe-mode constraint k:
            penalty_k = beta * phi''(margin_k) * A_C_k^2
        where phi''(m) = 1/m^2 is the log-barrier second derivative.

        Uses masked tensor ops instead of per-k loop. Only safe-mode
        constraints (positive margin and not in recovery) contribute.

        Note on re-parametrization: the paper's Bregman divergence (Eq. 7) is
            D_phi = (1/2) * (1/t) * phi''(m) * delta^2
        where t is the barrier parameter. We absorb both the 1/2 and 1/t factors
        into beta for simplicity, so: beta_code = beta_paper / (2 * t).
        When comparing to paper hyperparameters, account for this relation.
        """
        recovery = torch.tensor(self._in_recovery, device=self.device)
        safe_mask = (self._margins > 0) & ~recovery
        if not safe_mask.any():
            return torch.tensor(0.0, device=self.device)
        # Clamp margin to prevent barrier singularity when margin is small positive
        # but recovery mode hasn't triggered yet (margin in (0, ~0.01) gap).
        margin_safe = self._margins.clamp(min=0.01)
        phi_pp = 1.0 / margin_safe.pow(2)
        return self.beta * (safe_mask * phi_pp * cost_surrogates.pow(2)).sum()

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

    @staticmethod
    def _gaussian_kl(
        mu: torch.Tensor, sigma: torch.Tensor, old_mu: torch.Tensor, old_sigma: torch.Tensor
    ) -> torch.Tensor:
        """Compute mean KL(pi_old || pi_new) analytically for diagonal Gaussian."""
        kl = (
            torch.log((sigma / old_sigma).clamp(min=1e-5))
            + (old_sigma.pow(2) + (old_mu - mu).pow(2)) / (2.0 * sigma.pow(2))
            - 0.5
        )
        return kl.sum(dim=-1).mean()

    def _kl_divergence(self, obs: TensorDict, old_mu: torch.Tensor, old_sigma: torch.Tensor) -> torch.Tensor:
        """Compute mean KL(pi_old || pi_new) with a fresh forward pass."""
        self.policy.act(obs)
        return self._gaussian_kl(self.policy.action_mean, self.policy.action_std, old_mu, old_sigma)

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
        kl = self._gaussian_kl(self.policy.action_mean, self.policy.action_std, old_mu, old_sigma)

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

    def _line_search(
        self,
        obs: TensorDict,
        old_mu: torch.Tensor,
        old_sigma: torch.Tensor,
        step_dir: torch.Tensor,
        old_loss: torch.Tensor,
        surrogate_fn: object,
    ) -> bool:
        """Backtracking line search shared by safe and recovery modes.

        Accepts a step when:
            1. Surrogate improvement > 0
            2. KL divergence <= max_kl * margin

        Args:
            surrogate_fn: No-arg callable returning the surrogate loss scalar.
        """
        old_params = self._get_policy_params_flat()
        step_size = 1.0
        kl_limit = self.max_kl * self.line_search_kl_margin

        for _ in range(self.line_search_max_backtracks):
            self._set_policy_params_flat(old_params + step_size * step_dir)

            with torch.no_grad():
                new_loss = surrogate_fn()
                kl = self._kl_divergence(obs, old_mu, old_sigma)

            if (old_loss - new_loss) > 0 and kl <= kl_limit:
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
        violations = (mean_cost_returns - self.d_k).tolist()

        # ------------------------------------------------------------------
        # 2. TRPO policy update (full-batch, single step)
        # ------------------------------------------------------------------
        # Determine mode: if ANY constraint is in recovery, do recovery step.
        # Recovery takes priority since feasibility must be restored before
        # reward optimization can proceed meaningfully.
        #
        # Design note: this "any-recovery" policy is conservative. If only 1 of K
        # constraints is barely violated, ALL reward optimization halts. A per-constraint
        # blend (safe constraints use barrier, violated use cost min) would allow
        # reward progress on feasible dimensions but adds implementation complexity
        # and may interact poorly with the shared trust region. Keep as-is unless
        # training shows excessive mode oscillation.
        old_lp_sq = old_log_prob_flat.squeeze(-1)
        adv_sq = advantages_flat.squeeze(-1)

        if any_recovery:
            self._cached_barrier_penalty = 0.0
            recovery_mask = torch.tensor(self._in_recovery, dtype=torch.float32, device=self.device)

            def surrogate() -> torch.Tensor:
                self.policy.act(obs_flat)
                log_prob = self.policy.get_actions_log_prob(actions_flat)
                ratio = torch.exp(log_prob - old_lp_sq)
                cost_surrs = self._compute_cost_surrogates(ratio, cost_advantages_flat)
                return (recovery_mask * cost_surrs).sum()

            ls_success = self._trpo_step(obs_flat, old_mu_flat, old_sigma_flat, surrogate, "recovery")
        else:

            def surrogate() -> torch.Tensor:
                self.policy.act(obs_flat)
                log_prob = self.policy.get_actions_log_prob(actions_flat)
                ratio = torch.exp(log_prob - old_lp_sq)
                reward_surr = -(adv_sq * ratio).mean()
                cost_surrs = self._compute_cost_surrogates(ratio, cost_advantages_flat)
                bp = self._compute_barrier_penalty(cost_surrs)
                self._cached_barrier_penalty = bp.item()
                return reward_surr + bp

            ls_success = self._trpo_step(obs_flat, old_mu_flat, old_sigma_flat, surrogate, "safe")

        # Noise floor: numerical safety net to prevent log_prob divergence
        min_log_std = math.log(0.25)
        with torch.no_grad():
            self.policy.log_std.data.clamp_(min=min_log_std)

        # ------------------------------------------------------------------
        # 3. Encoder update (gated on ls_success)
        # ------------------------------------------------------------------
        mean_z_bounds_loss = 0.0
        if self.encoder_optimizer is not None and ls_success:
            mean_z_bounds_loss = self._update_encoder(obs_flat, advantages_flat, old_log_prob_flat, actions_flat)

        # Compute KL after full update for logging (single measurement)
        with torch.no_grad():
            mean_kl = self._kl_divergence(obs_flat, old_mu_flat, old_sigma_flat).item()

        # ------------------------------------------------------------------
        # 4. Value function update (pure MSE)
        # ------------------------------------------------------------------
        mean_value_loss, mean_cost_value_loss = self._update_values(
            obs_flat, returns_flat, cost_returns_flat, batch_size
        )

        # ------------------------------------------------------------------
        # Store monitoring metrics (read by ConstraintEncoderRunner)
        # ------------------------------------------------------------------
        barrier_penalty_val = self._cached_barrier_penalty
        self._last_cost_returns = mean_cost_returns.tolist()
        self._last_violations = violations
        self._last_line_search_success = float(ls_success)
        self._last_margins = self._margins.tolist()
        self._last_in_recovery = [float(r) for r in self._in_recovery]
        self._last_barrier_penalty = barrier_penalty_val
        self._last_mode = 2 if all_recovery else (1 if any_recovery else 0)

        # Clear storage
        self.storage.clear()

        # ------------------------------------------------------------------
        # Return loss dict
        # ------------------------------------------------------------------
        loss_dict: dict[str, float] = {
            "value_function": mean_value_loss,
            "barrier_penalty": barrier_penalty_val,
            "kl": mean_kl,
            "cost_value": mean_cost_value_loss,
            "mode": float(self._last_mode),
            "adv_raw_std": adv_raw_std.item(),
        }
        if hasattr(self.policy, "z_bounds_loss"):
            loss_dict["z_bounds"] = mean_z_bounds_loss

        return loss_dict

    # ==================================================================
    # Internal: Unified TRPO step (safe or recovery via callables)
    # ==================================================================

    def _trpo_step(
        self,
        obs_flat: TensorDict,
        old_mu_flat: torch.Tensor,
        old_sigma_flat: torch.Tensor,
        surrogate_fn: object,
        mode_name: str,
    ) -> bool:
        """Execute a single TRPO natural-gradient step.

        Args:
            surrogate_fn: No-arg callable returning the surrogate loss scalar.
                Called with grad for gradient extraction; under torch.no_grad()
                for line search evaluation.
            mode_name: "safe" or "recovery" (for warning messages only).
        """
        # 1. Compute loss + flat gradient
        loss = surrogate_fn()
        g = self._flat_grad(loss, self._policy_params, retain_graph=False)

        # 2. Natural gradient via conjugate gradient: x = F^{-1} g
        nat_grad = self._conjugate_gradient(obs_flat, old_mu_flat, old_sigma_flat, g)

        # 3. Step size: sqrt(2 * max_kl / (g^T F^{-1} g))
        shs = 0.5 * nat_grad.dot(g)

        if shs <= 0 or not torch.isfinite(shs):
            logger.warning("TRPO %s: shs=%.6e non-positive or non-finite, skipping", mode_name, shs.item())
            return False

        step_dir = -torch.sqrt(self.max_kl / shs) * nat_grad

        if not torch.isfinite(step_dir).all():
            logger.warning("TRPO %s: step_dir contains NaN/Inf, skipping", mode_name)
            return False

        # 4. Line search
        with torch.no_grad():
            old_loss = surrogate_fn()

        return self._line_search(obs_flat, old_mu_flat, old_sigma_flat, step_dir, old_loss, surrogate_fn)

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

            # Fresh forward pass through encoder + actor
            self.policy.act(obs_flat)
            log_prob = self.policy.get_actions_log_prob(actions_flat)
            ratio = torch.exp(log_prob - old_log_prob_flat.squeeze(-1))
            total_loss = -(advantages_flat.squeeze(-1) * ratio).mean()

            # Add z_bounds loss (same forward pass, shared z tensor)
            if hasattr(self.policy, "z_bounds_loss"):
                z_b_loss = self.policy.z_bounds_loss()
                mean_z_bounds_loss = z_b_loss.item()
                if z_b_loss.requires_grad:
                    total_loss = total_loss + z_b_loss

            # Guard against NaN/Inf loss propagating to encoder params
            if not torch.isfinite(total_loss):
                logger.warning("Encoder loss non-finite (%.4e), skipping epoch %d", total_loss.item(), _epoch)
                continue

            # Single backward: encoder_optimizer only steps encoder params,
            # so actor/critic grads are computed but not applied
            total_loss.backward()
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
                cost_value_pred = self.policy.evaluate_costs(obs_mb)
                target = cost_returns_mb.clamp(min=0.0)
                per_k_mse = (target - cost_value_pred).pow(2).mean(dim=0)  # (K,)
                # Guard for edge-case budgets: with default cost_gamma=0.99, min d_k=1.0
                # so d_k^2>=1.0 and the clamp never activates. Kept as defensive bound.
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
        """Interface compatibility with ConstraintEncoderRunner."""
        logger.info("[ConstraintTRPO] C-TRPO barrier mode, beta=%.4f, max_iterations=%d", self.beta, max_iterations)
