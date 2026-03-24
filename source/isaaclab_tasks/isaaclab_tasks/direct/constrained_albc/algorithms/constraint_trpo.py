# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Constrained TRPO with log-barrier constraint enforcement (Modified IPO).

TRPO policy optimization with log-barrier interior-point method for constraint
satisfaction. Based on Modified IPO (Kim et al., NORBC, IROS 2024) with
adaptive constraint thresholding for initial infeasibility.

Key design decisions:
    - Log barrier: -sum_k log(d_k^i - J_hat_C_k) / t (always-on constraint gradient)
    - Adaptive thresholding: d_k^i = max(d_k, J_C_k + alpha * d_k) for infeasible starts
    - Per-constraint cost advantage standardization (NORBC Sec IV-B): equalizes gradient scale
    - LS-gated encoder updates: when line search fails, both actor and encoder frozen
    - Noise floor (min_std): primary exploration maintenance, outside trust region
    - Multi-step encoder with KL gating: prevents encoder-induced distribution shift

Reference:
    Kim et al., "NORBC", IROS 2024 (Modified IPO, cost critic, value loss).
    Muller et al., "Truly Constrained TRPO", ICML 2025, arXiv:2411.02957.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.optim as optim
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

logger = logging.getLogger(__name__)


class ConstraintTRPO:
    """Log-barrier constrained TRPO for policy optimization (Modified IPO).

    Uses log-barrier interior-point method with adaptive thresholding for
    constraint enforcement, combined with TRPO natural gradient for policy update.
    """

    def __init__(
        self,
        policy: nn.Module,
        # TRPO parameters
        max_kl: float = 0.002,
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
        # Log barrier constraint parameters (Modified IPO)
        barrier_t: float = 50.0,
        barrier_alpha: float = 0.02,
        # Encoder update
        num_encoder_epochs: int = 5,
        encoder_lr: float = 1e-3,
        # Noise floor and sigma optimizer
        min_std: float = 0.2,
        std_lr: float = 3e-3,
        # Entropy regularization
        entropy_coef: float = 0.0,
        # Post-encoder KL gating
        max_encoder_kl: float = 0.016,
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
        self.num_encoder_epochs = num_encoder_epochs
        self.min_std = min_std
        self.std_lr = std_lr
        self._entropy_coef = entropy_coef
        self.max_encoder_kl = max_encoder_kl

        # Log barrier parameters (Modified IPO)
        self._barrier_t = barrier_t
        self._barrier_alpha = barrier_alpha

        # Monitoring attributes (read by ConstraintEncoderRunner before first update)
        self._last_cost_returns = [0.0] * num_constraints
        self._last_violations = [0.0] * num_constraints
        self._last_barrier_margins = [0.0] * num_constraints
        self._last_line_search_success = 0.0
        self._last_barrier_penalty = 0.0
        self._last_mean_entropy = 0.0
        self._last_entropy_bonus = 0.0
        self._last_surrogate_loss = 0.0
        self._last_pre_encoder_kl = 0.0

        # TRPO step quality diagnostics
        self._last_trpo_shs = 0.0
        self._last_trpo_step_norm = 0.0
        self._last_trpo_grad_norm = 0.0
        self._last_line_search_backtracks = 0
        self._last_value_grad_norm = 0.0
        self._last_encoder_grad_norm = 0.0

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
        # - Std params: separate Adam (sigma follows score-function equilibrium)
        # - Encoder params: separate Adam (indirect distribution influence)
        # - Value params (critic + cost_critic): Adam optimizer
        value_params = []
        encoder_params = []
        std_params = []
        self._policy_params = []  # Actor MLP weights only (TRPO)

        encoder_prefixes = ("encoder",)
        for name, param in self.policy.named_parameters():
            is_value = name.startswith("critic") or name.startswith("cost_critic")
            is_encoder = any(name.startswith(p) for p in encoder_prefixes)
            is_std = name == "log_std"
            if is_value:
                value_params.append(param)
            elif is_encoder:
                encoder_params.append(param)
            elif is_std:
                std_params.append(param)
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

        # Separate optimizer for log_std: decoupled from TRPO KL budget.
        # Sigma follows score-function gradient dlogpi/dsigma = ((a-mu)^2 - sigma^2)/sigma^3
        # without competing with mu for KL trust region capacity.
        self._std_params = std_params
        self.std_optimizer = optim.Adam(std_params, lr=std_lr) if std_params else None

        logger.info(
            "ConstraintTRPO: %d actor params (TRPO), %d std params (Adam lr=%.0e), "
            "%d encoder params (Adam), %d value params (Adam)",
            len(self._policy_params),
            len(std_params),
            std_lr,
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
        # Pre-allocated zero costs buffer (avoids per-step GPU allocation in process_env_step)
        self._zero_costs = torch.zeros(N, K, device=self.device)

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
        costs = extras.get("costs", self._zero_costs)

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

        # Sanitize non-finite values before standardization.
        finite_mask = torch.isfinite(self.storage.cost_advantages).all(dim=(0, 1))  # (K,)
        bad_constraints = ~finite_mask
        if bad_constraints.any():
            bad_ids = bad_constraints.nonzero(as_tuple=True)[0]
            logger.warning("Non-finite cost advantages for constraints %s, zeroing.", bad_ids.tolist())
            self.storage.cost_advantages[:, :, bad_constraints] = 0.0

    # ==================================================================
    # Log Barrier Constraint Enforcement (Modified IPO)
    # ==================================================================

    def _compute_adaptive_thresholds(self, mean_cost_returns: torch.Tensor) -> torch.Tensor:
        """Compute adaptive constraint thresholds for log barrier.

        d_k^i = max(d_k, J_C_k + alpha * d_k)

        When cost exceeds budget, the threshold is relaxed to keep the
        log barrier computable. As the policy improves, thresholds
        converge to the original budgets d_k.

        Args:
            mean_cost_returns: Current mean cost returns, shape (K,).

        Returns:
            Adaptive thresholds, shape (K,).
        """
        return torch.max(self.d_k, mean_cost_returns + self._barrier_alpha * self.d_k)

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

        FVP uses pure KL Hessian only. Constraint curvature is NOT included
        in the Fisher matrix -- it only affects the objective gradient.
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
        surrogate_fn: Callable[[], torch.Tensor],
    ) -> bool:
        """Backtracking line search.

        Accepts a step when:
            1. Surrogate improvement > 0
            2. KL divergence <= max_kl * margin
        """
        old_params = self._get_policy_params_flat()
        step_size = 1.0
        kl_limit = self.max_kl * self.line_search_kl_margin

        for i in range(self.line_search_max_backtracks):
            self._set_policy_params_flat(old_params + step_size * step_dir)

            with torch.no_grad():
                new_loss = surrogate_fn()
                kl = self._kl_divergence(obs, old_mu, old_sigma)

            if (old_loss - new_loss) > 0 and kl <= kl_limit:
                self._last_line_search_backtracks = i
                return True

            step_size *= self.line_search_shrink_factor

        self._last_line_search_backtracks = self.line_search_max_backtracks
        self._set_policy_params_flat(old_params)
        return False

    # ==================================================================
    # Main Update
    # ==================================================================

    def update(self) -> dict[str, float]:
        """Execute one iteration of constrained TRPO update.

        Update order:
            1. Compute adaptive barrier thresholds
            2. TRPO policy update (reward + log barrier, full-batch)
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

        # Per-constraint cost advantage standardization (NORBC Sec IV-B).
        # Equalizes gradient magnitude across constraints so barrier 1/margin_k
        # provides proximity-based prioritization only.
        ca_mean = cost_advantages_flat.mean(dim=0, keepdim=True)  # (1, K)
        ca_std = cost_advantages_flat.std(dim=0, keepdim=True)  # (1, K)
        cost_advantages_flat = (cost_advantages_flat - ca_mean) / (ca_std + 1e-8)

        batch_size = obs_flat.batch_size[0]

        # Mean cost returns (computed once, needed for barrier + logging)
        # Clamp to non-negative: cost value errors can make GAE return negative,
        # which would inflate the barrier margin.
        mean_cost_returns = cost_returns_flat.mean(dim=0).clamp(min=0.0)  # (K,)

        # ------------------------------------------------------------------
        # 1. Compute adaptive barrier thresholds
        # ------------------------------------------------------------------
        adaptive_d_k = self._compute_adaptive_thresholds(mean_cost_returns)

        # Compute violations and barrier margins for logging
        violations = (mean_cost_returns - self.d_k).tolist()
        with torch.no_grad():
            static_margins = (adaptive_d_k - mean_cost_returns).tolist()
        self._last_barrier_margins = static_margins

        # ------------------------------------------------------------------
        # 2. TRPO policy update (full-batch, single step)
        # ------------------------------------------------------------------
        old_lp_sq = old_log_prob_flat.squeeze(-1)
        adv_sq = advantages_flat.squeeze(-1)

        # Detach constants for the barrier (only cost_surrs depend on theta)
        barrier_base = adaptive_d_k - mean_cost_returns  # (K,) static part of margin

        def surrogate() -> torch.Tensor:
            self.policy.act(obs_flat)
            log_prob = self.policy.get_actions_log_prob(actions_flat)
            ratio = torch.exp(log_prob - old_lp_sq)
            # Reward surrogate (minimization: negative improvement)
            reward_surr = -(adv_sq * ratio).mean()
            # Log barrier: minimize -sum_k log(margin_k) / t
            cost_surrs = (ratio.unsqueeze(-1) * cost_advantages_flat).mean(dim=0)  # (K,)
            margin = barrier_base - cost_surrs  # (K,)
            barrier = -torch.log(margin.clamp(min=1e-8)).sum() / self._barrier_t
            self._last_barrier_penalty = barrier.item()
            # Entropy bonus: maximize entropy (negative in minimization objective)
            mean_entropy = self.policy.entropy.mean()
            self._last_mean_entropy = mean_entropy.item()
            entropy_bonus = -self._entropy_coef * mean_entropy
            self._last_entropy_bonus = entropy_bonus.item()
            return reward_surr + barrier + entropy_bonus

        ls_success = self._trpo_step(obs_flat, old_mu_flat, old_sigma_flat, surrogate)

        # ------------------------------------------------------------------
        # 2b. Sigma update (separate Adam, decoupled from TRPO KL budget)
        # ------------------------------------------------------------------
        # Re-snapshot post-TRPO baseline so IS ratio starts at 1.0 for sigma.
        # Gradient is the vanilla PG score function for sigma:
        #   dlogpi/dsigma = ((a - mu)^2 - sigma^2) / sigma^3
        # This is self-correcting: when advantage-weighted action spread exceeds
        # sigma^2, gradient pushes sigma up (more exploration needed); when below,
        # pushes sigma down (policy is precise enough).
        if self.std_optimizer is not None:
            with torch.no_grad():
                self.policy.act(obs_flat)
                std_baseline_lp = self.policy.get_actions_log_prob(actions_flat).squeeze(-1)

            self.policy.act(obs_flat)
            log_prob = self.policy.get_actions_log_prob(actions_flat)
            ratio = torch.exp(log_prob - std_baseline_lp)
            # Reward surrogate + log barrier (constraint feedback to sigma)
            reward_surr = -(adv_sq * ratio).mean()
            cost_surrs = (ratio.unsqueeze(-1) * cost_advantages_flat).mean(dim=0)
            margin = barrier_base - cost_surrs
            std_barrier = -torch.log(margin.clamp(min=1e-8)).sum() / self._barrier_t
            std_loss = reward_surr + std_barrier
            # Compute gradient only for std params (avoid wasteful actor/encoder grads)
            std_grads = torch.autograd.grad(std_loss, self._std_params)
            self.std_optimizer.zero_grad()
            for p, g in zip(self._std_params, std_grads):
                p.grad = g
            self.std_optimizer.step()

        # Noise floor: hard clamp after both TRPO and Adam steps.
        min_log_std = math.log(self.min_std)
        with torch.no_grad():
            self.policy.log_std.data.clamp_(min=min_log_std)

        # ------------------------------------------------------------------
        # 3. Encoder update (gated on ls_success)
        # ------------------------------------------------------------------
        # Measure pre-encoder KL for gating encoder-induced distribution shift
        with torch.no_grad():
            pre_encoder_kl = self._kl_divergence(obs_flat, old_mu_flat, old_sigma_flat).item()
        self._last_pre_encoder_kl = pre_encoder_kl

        if self.encoder_optimizer is not None and ls_success:
            self._update_encoder(
                obs_flat,
                advantages_flat,
                old_log_prob_flat,
                actions_flat,
                old_mu_flat=old_mu_flat,
                old_sigma_flat=old_sigma_flat,
                pre_encoder_kl=pre_encoder_kl,
            )

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
        self._last_cost_returns = mean_cost_returns.tolist()
        self._last_violations = violations
        self._last_line_search_success = float(ls_success)

        # Clear storage
        self.storage.clear()

        # ------------------------------------------------------------------
        # Return loss dict
        # ------------------------------------------------------------------
        return {
            "value_function": mean_value_loss,
            "kl": mean_kl,
            "cost_value": mean_cost_value_loss,
        }

    # ==================================================================
    # Internal: TRPO step
    # ==================================================================

    def _trpo_step(
        self,
        obs_flat: TensorDict,
        old_mu_flat: torch.Tensor,
        old_sigma_flat: torch.Tensor,
        surrogate_fn: Callable[[], torch.Tensor],
    ) -> bool:
        """Execute a single TRPO natural-gradient step."""
        # 1. Compute loss + flat gradient
        loss = surrogate_fn()
        self._last_surrogate_loss = loss.item()
        g = self._flat_grad(loss, self._policy_params, retain_graph=False)
        self._last_trpo_grad_norm = g.norm().item()

        # 2. Natural gradient via conjugate gradient: x = F^{-1} g
        nat_grad = self._conjugate_gradient(obs_flat, old_mu_flat, old_sigma_flat, g)

        # 3. Step size: sqrt(2 * max_kl / (g^T F^{-1} g))
        shs = 0.5 * nat_grad.dot(g)
        self._last_trpo_shs = shs.item() if torch.isfinite(shs) else 0.0

        if shs <= 0 or not torch.isfinite(shs):
            logger.warning("TRPO: shs=%.6e non-positive or non-finite, skipping", shs.item())
            self._last_trpo_step_norm = 0.0
            return False

        step_dir = -torch.sqrt(self.max_kl / shs) * nat_grad
        self._last_trpo_step_norm = step_dir.norm().item()

        if not torch.isfinite(step_dir).all():
            logger.warning("TRPO: step_dir contains NaN/Inf, skipping")
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
        old_mu_flat: torch.Tensor | None = None,
        old_sigma_flat: torch.Tensor | None = None,
        pre_encoder_kl: float = 0.0,
    ) -> None:
        """Multi-step encoder update with fresh forward passes.

        Runs num_encoder_epochs fresh forward/backward passes through the
        encoder. Actor params are frozen (only encoder_optimizer steps).

        KL gating: after each encoder step, checks if the resulting KL
        divergence exceeds pre_encoder_kl + max_encoder_kl. If so, reverts
        encoder params and stops early.
        """
        kl_gating = self.max_encoder_kl > 0 and old_mu_flat is not None and old_sigma_flat is not None

        for _epoch in range(self.num_encoder_epochs):
            # Save encoder state for potential rollback
            if kl_gating:
                saved_state = {n: p.data.clone() for n, p in self.policy.named_parameters() if n.startswith("encoder")}

            self.encoder_optimizer.zero_grad()

            # Fresh forward pass through encoder + actor
            self.policy.act(obs_flat)
            log_prob = self.policy.get_actions_log_prob(actions_flat)
            ratio = torch.exp(log_prob - old_log_prob_flat.squeeze(-1))
            total_loss = -(advantages_flat.squeeze(-1) * ratio).mean()

            # Guard against NaN/Inf loss propagating to encoder params
            if not torch.isfinite(total_loss):
                logger.warning("Encoder loss non-finite (%.4e), skipping epoch %d", total_loss.item(), _epoch)
                continue

            total_loss.backward()
            enc_grad_norm = nn.utils.clip_grad_norm_(self._encoder_params, max_norm=1.0)
            self._last_encoder_grad_norm = enc_grad_norm.item()
            self.encoder_optimizer.step()

            # KL gating: revert if encoder step caused excessive KL shift
            if kl_gating:
                with torch.no_grad():
                    post_kl = self._kl_divergence(obs_flat, old_mu_flat, old_sigma_flat).item()
                if post_kl > pre_encoder_kl + self.max_encoder_kl:
                    for n, p in self.policy.named_parameters():
                        if n in saved_state:
                            p.data.copy_(saved_state[n])
                    logger.debug(
                        "Encoder KL exceeded limit (%.4f > %.4f + %.4f), reverted epoch %d",
                        post_kl,
                        pre_encoder_kl,
                        self.max_encoder_kl,
                        _epoch,
                    )
                    break

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
        mean_value_grad_norm = 0.0
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
                cost_value_loss = (per_k_mse / self.d_k.pow(2).clamp(min=0.01)).mean()

                total_value_loss = self.value_loss_coef * value_loss + self.cost_value_loss_coef * cost_value_loss

                self.value_optimizer.zero_grad()
                total_value_loss.backward()
                val_grad_norm = nn.utils.clip_grad_norm_(self._value_params, self.max_grad_norm)
                self.value_optimizer.step()
                mean_value_grad_norm += val_grad_norm.item()

                mean_value_loss += value_loss.item()
                mean_cost_value_loss += cost_value_loss.item()
                num_value_updates += 1

        if num_value_updates > 0:
            mean_value_loss /= num_value_updates
            mean_cost_value_loss /= num_value_updates
            mean_value_grad_norm /= num_value_updates
        self._last_value_grad_norm = mean_value_grad_norm

        return mean_value_loss, mean_cost_value_loss

    # ==================================================================
    # Compatibility
    # ==================================================================

    def set_max_iterations(self, max_iterations: int) -> None:
        """Interface compatibility with ConstraintEncoderRunner."""
        logger.info(
            "[ConstraintTRPO] IPO barrier mode, barrier_t=%.1f, barrier_alpha=%.2f, max_iterations=%d",
            self._barrier_t,
            self._barrier_alpha,
            max_iterations,
        )
