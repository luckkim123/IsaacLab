# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Constraint TRPO with Lagrangian (primal-dual) constraint enforcement.

Implements constrained RL with dual variable ascent:
    1. Lagrangian cost surrogate: sum_k lambda_k * cost_adv_k
    2. TRPO natural gradient for policy update (exact KL constraint)
    3. Value function update (pure MSE)
    4. Dual variable update: lambda_k = clamp(lambda_k + lr*(J_C_k - d_k)/d_k, 0, max)

Key design decisions:
    - Detached-std cost ratio: The cost surrogate uses an importance sampling
      ratio where std is detached, so constraint gradient flows through action
      mean only. This prevents constraint pressure from collapsing variance
      (the structural root cause of entropy collapse in constrained Gaussian
      policies). Variance is controlled purely by reward-entropy balance.
    - Target entropy (SAC-style): Learned entropy coefficient alpha replaces
      fixed entropy_coef. A separate optimizer adjusts log_alpha to maintain
      entropy near a target H_target. This prevents both entropy collapse
      (alpha increases) and entropy explosion (alpha decreases). Fixed
      entropy_coef causes explosion when TRPO step size is large (low cost
      pressure during lambda warmup amplifies the entropy bonus on log_std
      via the shared step size alpha = sqrt(max_kl/shs)).
    - d_k-normalized dual update: Violation (J_C_k - d_k) is divided by d_k
      before scaling lambda, so constraints with different budget scales
      contribute equally to multiplier growth.
    - lambda_k starts at 0: no constraint pressure initially, allowing the
      random policy to explore freely.
    - Lambda LR warmup: lr_lambda linearly ramps from 0 to target over a
      warmup period (default 30% of max_iterations). This gives the policy
      a reward-dominant learning phase before constraint pressure kicks in.
    - d_k^2-normalized cost value loss: Per-constraint MSE is divided by d_k^2,
      equalizing gradient contribution across constraints with different cost
      return scales (e.g., joint_vel MSE~27000 vs effort_limit MSE~335).
    - Reward advantage normalization: Standardizes reward advantages to
      zero-mean unit-variance before gradient computation. Cost advantages
      are per-constraint standardized to O(1). Without normalizing reward
      advantages (O(0.01)), the combined gradient is cost-dominated even
      when lambda is small, causing line search failures.
    - LS-gated updates: Lambda dual update, encoder policy grads, and
      encoder z_bounds are all gated on line search success. When LS fails,
      the actor is frozen (params reverted). Updating lambda/encoder on a
      frozen actor creates a death spiral (lambda grows unchecked) and
      actor-encoder desync (encoder z drifts while actor can't adapt).

The algorithm maintains the same interface as RSL-RL PPO (init_storage, act,
process_env_step, compute_returns, update) so it can be used as a drop-in
replacement in the OnPolicyRunner.

Reference:
    Kim et al., "NORBC: Neural Online Robust Boundary Controller for
    Underwater Robots", IROS 2024.
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
    """TRPO + Lagrangian primal-dual for constrained policy optimization.

    Key differences from PPO:
        - Policy update: natural gradient via conjugate gradient (full-batch)
        - Value update: pure MSE (Lagrangian is policy-only)
        - Line search: verifies KL constraint and surrogate improvement
        - No clipped surrogate; uses exact KL constraint
        - Lagrangian dual variables: lambda_k updated via subgradient ascent
        - Cost surrogate uses mean-only gradient (std detached) to prevent
          constraint-driven entropy collapse
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
        # Constraint / Lagrangian parameters
        num_constraints: int = 3,
        constraint_budgets: tuple[float, ...] = (0.15, 0.02, 0.15),
        cost_gamma: float = 0.99,
        cost_lam: float = 0.95,
        lr_lambda: float = 0.035,
        lambda_max: float = 20.0,
        lambda_init: float = 0.0,
        # Line search acceptance threshold
        line_search_kl_margin: float = 1.5,
        # Target entropy (SAC-style automatic temperature)
        target_entropy: float = 2.0,
        alpha_entropy_lr: float = 0.01,
        alpha_entropy_init: float = 0.001,
        # Encoder z bounds
        z_bounds_coef: float = 0.3,
        # Lambda warmup
        lambda_warmup_frac: float = 0.3,
        # Device
        device: str = "cpu",
        # Unused kwargs from RSL-RL config forwarding
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

        # Target entropy: learned alpha replaces fixed entropy_coef.
        # SAC dual update: alpha adjusts to maintain entropy near target.
        self.target_entropy = target_entropy
        _safe_init = alpha_entropy_init if alpha_entropy_init > 0 else 1e-8
        self.log_alpha = torch.tensor(math.log(_safe_init), device=device, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_entropy_lr)

        # GAE parameters
        self.gamma = gamma
        self.lam = lam

        # Constraint parameters
        self.num_constraints = num_constraints
        self.cost_gamma = cost_gamma
        self.cost_lam = cost_lam
        self.lr_lambda = lr_lambda
        self.lambda_max = lambda_max
        self.line_search_kl_margin = line_search_kl_margin
        self.z_bounds_coef = z_bounds_coef

        # Lambda warmup: ramp lr_lambda from 0 to target over warmup period
        self.lambda_warmup_frac = lambda_warmup_frac
        self._lambda_warmup_end = 1  # Updated by set_max_iterations()

        if cost_gamma >= 1.0:
            raise ValueError(f"cost_gamma must be < 1.0, got {cost_gamma}")

        # Discounted budgets: d_k = D_k / (1 - gamma)
        self.d_k = torch.tensor(
            [b / (1.0 - cost_gamma) for b in constraint_budgets],
            device=device,
            dtype=torch.float32,
        )

        # Lagrangian dual variables (one per constraint)
        self.lambda_k = torch.full((num_constraints,), lambda_init, device=device, dtype=torch.float32)

        # Separate parameter groups:
        # - Actor params: TRPO natural gradient (no optimizer)
        # - Encoder params: separate Adam (indirect distribution influence)
        # - Value params (critic + cost_critic): Adam optimizer
        value_params = []
        encoder_params = []
        self._policy_params = []  # Actor-only for TRPO

        for name, param in self.policy.named_parameters():
            is_value = name.startswith("critic") or name.startswith("cost_critic")
            is_encoder = name.startswith("encoder")
            if is_value:
                value_params.append(param)
            elif is_encoder:
                encoder_params.append(param)
            else:
                self._policy_params.append(param)

        self._value_params = value_params
        self.value_optimizer = optim.Adam(value_params, lr=value_lr)
        self._has_encoder_params = len(encoder_params) > 0
        self.encoder_lr = 3e-3
        if self._has_encoder_params:
            self._encoder_params = encoder_params
            self.encoder_optimizer = optim.Adam(encoder_params, lr=self.encoder_lr, weight_decay=1e-5)
        else:
            self._encoder_params = []
            self.encoder_optimizer = None
        logger.info(
            "ConstraintTRPO: %d actor params (TRPO), %d encoder params (Adam), %d value params (Adam)",
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
        # Zero-mean ensures the policy can always find "relatively better" actions
        # even when all actions violate a constraint. Per-constraint normalization
        # equalizes gradient contribution across constraints with different physical
        # scales (e.g., rad/s vs binary 0/1), letting the dual variable alone
        # determine relative priority.
        for k in range(self.num_constraints):
            adv_k = self.storage.cost_advantages[:, :, k]
            if not torch.isfinite(adv_k).all():
                logger.warning("Non-finite cost advantages for constraint %d, zeroing.", k)
                self.storage.cost_advantages[:, :, k] = 0.0
            else:
                self.storage.cost_advantages[:, :, k] = (adv_k - adv_k.mean()) / (adv_k.std() + 1e-8)

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

    def _log_prob_mean_only(self, actions: torch.Tensor) -> torch.Tensor:
        """Log probability with std detached -- gradient flows only through mean.

        Used for cost surrogate to prevent constraint gradient from collapsing
        variance. The Gaussian log_prob formula is identical to the standard one,
        but std is detached so d(log_prob)/d(log_std) = 0.
        """
        mu = self.policy.action_mean
        std = self.policy.action_std.detach()
        return (-0.5 * (((actions - mu) / std).pow(2) + 2.0 * std.log() + math.log(2.0 * math.pi))).sum(dim=-1)

    def _linearized_surrogate(
        self,
        obs: TensorDict,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        cost_advantages: torch.Tensor,
        old_log_prob: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate linearized Lagrangian surrogate matching the gradient objective.

        Uses the SAME formula as the policy gradient computation (reward surrogate
        + Lagrangian-weighted cost surrogate - entropy). This ensures the natural
        gradient direction is guaranteed to improve the line search objective.
        """
        self.policy.act(obs)
        log_prob = self.policy.get_actions_log_prob(actions)
        ratio = torch.exp(log_prob - old_log_prob)
        reward_surr = -(advantages * ratio).mean()

        # Cost ratio with detached std: constraint gradient guides mean, not variance
        log_prob_cost = self._log_prob_mean_only(actions)
        ratio_cost = torch.exp(log_prob_cost - old_log_prob)

        cost_surr = torch.tensor(0.0, device=self.device)
        for k in range(self.num_constraints):
            cost_adv_k = (ratio_cost * cost_advantages[:, k]).mean()
            cost_surr += self.lambda_k[k] * cost_adv_k

        entropy = self.policy.entropy
        alpha = self.log_alpha.exp().detach()
        return reward_surr + cost_surr - alpha * entropy.mean()

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
        """Compute F @ v without forming F, using double backprop on KL."""
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

    def _line_search(
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
        """Backtracking line search on the linearized Lagrangian surrogate.

        Uses the same objective as the gradient computation to ensure the
        natural gradient direction is guaranteed to improve the objective.

        Checks two conditions:
            1. Linearized surrogate improvement > 0
            2. KL divergence <= max_kl * line_search_kl_margin

        Returns True if a valid step was found.
        """
        old_params = self._get_policy_params_flat()
        step_size = 1.0
        kl_limit = self.max_kl * self.line_search_kl_margin

        for i in range(self.line_search_max_backtracks):
            new_params = old_params + step_size * step_dir
            self._set_policy_params_flat(new_params)

            with torch.no_grad():
                new_loss = self._linearized_surrogate(obs, actions, advantages, cost_advantages, old_log_prob)
                kl = self._kl_divergence(obs, old_mu, old_sigma)

            improvement = old_loss - new_loss
            if improvement > 0 and kl <= kl_limit:
                return True

            step_size *= self.line_search_shrink_factor

        # Revert to old parameters if no valid step found
        self._set_policy_params_flat(old_params)
        return False

    # ==================================================================
    # Main Update
    # ==================================================================

    def update(self) -> dict[str, float]:
        """Execute one iteration of ConstraintTRPO update.

        Update order:
            1. TRPO policy update (full-batch natural gradient + line search)
            2. Value function update (pure MSE)
            3. Dual variable update (Lagrangian multiplier ascent)
        """
        self._iteration += 1

        # Flatten storage (clone to escape inference_mode)
        obs_flat = self.storage.observations.flatten(0, 1).clone()
        actions_flat = self.storage.actions.flatten(0, 1).clone()
        returns_flat = self.storage.returns.flatten(0, 1).clone()
        advantages_flat = self.storage.advantages.flatten(0, 1).clone()

        # Standardize reward advantages to O(1) scale.
        # Cost advantages are per-constraint standardized in _compute_cost_returns (std=1).
        # Without this, reward advantages O(0.01) << cost advantages O(1.0),
        # causing the combined natural gradient to be cost-dominated even when
        # lambda_k is small, which triggers line search failures.
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

        # Mean cost returns (computed once, needed by threshold + policy)
        # Clamp to non-negative: cost value errors can make GAE return negative,
        # which would inflate barrier margin (d_k - (-X) = d_k + X).
        mean_cost_returns = cost_returns_flat.mean(dim=0).clamp(min=0.0)  # (K,)

        # ------------------------------------------------------------------
        # 1. Compute constraint violations for logging
        # ------------------------------------------------------------------
        violations = []
        for k in range(self.num_constraints):
            violations.append((mean_cost_returns[k] - self.d_k[k]).item())

        # ------------------------------------------------------------------
        # 2. TRPO policy update (full-batch, single step)
        # ------------------------------------------------------------------
        # Compute policy gradient
        self.policy.act(obs_flat)
        log_prob = self.policy.get_actions_log_prob(actions_flat)
        entropy = self.policy.entropy

        ratio = torch.exp(log_prob - old_log_prob_flat.squeeze(-1))
        reward_surrogate = -(advantages_flat.squeeze(-1) * ratio).mean()

        # Lagrangian-weighted cost surrogate (detached std: gradient to mean only)
        # This prevents constraint pressure from collapsing variance -- the root
        # cause of entropy collapse in constrained Gaussian policies.
        log_prob_cost = self._log_prob_mean_only(actions_flat)
        ratio_cost = torch.exp(log_prob_cost - old_log_prob_flat.squeeze(-1))

        cost_surrogate = torch.tensor(0.0, device=self.device)
        for k in range(self.num_constraints):
            cost_adv_k = (ratio_cost * cost_advantages_flat[:, k]).mean()
            cost_surrogate += self.lambda_k[k] * cost_adv_k

        # Combined: reward surrogate + Lagrangian cost surrogate - entropy
        alpha = self.log_alpha.exp().detach()
        policy_loss = reward_surrogate + cost_surrogate - alpha * entropy.mean()

        mean_entropy = entropy.mean().item()

        # Compute encoder gradients from reward signal only (no cost surrogate).
        # Cost gradient through encoder causes z instability: as lambdas grow,
        # cost-dominated encoder updates shift z after line search KL check,
        # causing actual KL to explode (observed: kl=82 at iter 776 in run
        # 2026-03-16_13-02-33) and triggering LS failure cascades.
        # Cost critic uses asymmetric privileged obs, so encoder z is not
        # needed for cost estimation.
        _encoder_grads_cache: list[torch.Tensor | None] = []
        if self.encoder_optimizer is not None:
            encoder_loss = reward_surrogate - alpha * entropy.mean()
            encoder_grads = torch.autograd.grad(
                encoder_loss,
                self._encoder_params,
                retain_graph=True,
                allow_unused=True,
            )
            _encoder_grads_cache = list(encoder_grads)

        # Separate reward gradient norm for diagnostics (detects scale imbalance)
        g_reward = self._flat_grad(reward_surrogate, self._policy_params, retain_graph=True)
        reward_grad_norm = g_reward.norm().item()

        # Combined gradient of surrogate loss w.r.t. actor params (TRPO)
        g = self._flat_grad(policy_loss, self._policy_params, retain_graph=True)
        cost_grad_norm = (g - g_reward).norm().item()

        # Natural gradient via conjugate gradient: x = F^{-1} g
        nat_grad = self._conjugate_gradient(obs_flat, old_mu_flat, old_sigma_flat, g)

        # Step size: sqrt(2 * max_kl / (g^T F^{-1} g))
        shs = 0.5 * nat_grad.dot(g)  # 0.5 * x^T F x approximation

        if shs <= 0 or not torch.isfinite(shs):
            logger.warning("TRPO: shs=%.6e non-positive or non-finite, skipping policy step", shs.item())
            ls_success = False
        else:
            step_scale = torch.sqrt(self.max_kl / shs)
            step_dir = -step_scale * nat_grad

            if not torch.isfinite(step_dir).all():
                logger.warning("TRPO: step_dir contains NaN/Inf, skipping policy step")
                ls_success = False
            else:
                with torch.no_grad():
                    old_loss = self._linearized_surrogate(
                        obs_flat,
                        actions_flat,
                        advantages_flat.squeeze(-1),
                        cost_advantages_flat,
                        old_log_prob_flat.squeeze(-1),
                    )

                ls_success = self._line_search(
                    obs_flat,
                    actions_flat,
                    old_log_prob_flat.squeeze(-1),
                    advantages_flat.squeeze(-1),
                    cost_advantages_flat,
                    old_mu_flat,
                    old_sigma_flat,
                    step_dir,
                    old_loss,
                )

        # Noise floor: numerical safety net to prevent log_prob divergence
        # when std -> 0. No ceiling needed: target entropy auto-regulates
        # alpha (entropy coefficient) to prevent std from growing unbounded.
        min_log_std = math.log(0.2)
        with torch.no_grad():
            self.policy.log_std.data.clamp_(min=min_log_std)

        # Measure KL right after TRPO step + clamp (before encoder update shifts z)
        with torch.no_grad():
            kl_after_trpo = self._kl_divergence(obs_flat, old_mu_flat, old_sigma_flat).item()

        # Unified encoder update: policy-loss grads + z_bounds grads.
        # Both gated on ls_success to prevent actor-encoder desync: when the actor
        # is frozen (LS fail), encoder updates shift z-space causing the frozen actor
        # to produce mismatched actions. In the observed run, 33 consecutive LS failures
        # with ungated z_bounds caused roll error to spike from 7deg to 27deg.
        mean_z_bounds_loss = 0.0
        if self.encoder_optimizer is not None and ls_success:
            self.encoder_optimizer.zero_grad()
            has_grads = False

            # (1) Apply cached policy-loss gradients
            if _encoder_grads_cache:
                for i, p in enumerate(self._encoder_params):
                    if i < len(_encoder_grads_cache) and _encoder_grads_cache[i] is not None:
                        p.grad = _encoder_grads_cache[i].clone()
                        has_grads = True

            # (2) Accumulate z_bounds gradients
            if hasattr(self.policy, "z_bounds_loss"):
                self.policy.act(obs_flat)
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
                nn.utils.clip_grad_norm_(self._encoder_params, self.max_grad_norm)
                self.encoder_optimizer.step()
        elif self.encoder_optimizer is not None and hasattr(self.policy, "z_bounds_loss"):
            # Still compute z_bounds_loss for logging, but don't step
            with torch.no_grad():
                self.policy.act(obs_flat)
                mean_z_bounds_loss = self.policy.z_bounds_loss().item()

        # Compute KL after full update for logging
        with torch.no_grad():
            mean_kl = self._kl_divergence(obs_flat, old_mu_flat, old_sigma_flat).item()

        # ------------------------------------------------------------------
        # 3. Value function update (pure MSE, no barrier -- NORBC conformance)
        # ------------------------------------------------------------------
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

                # Cost value loss (MSE, per constraint, averaged).
                # Clamp targets to >=0: cost returns are theoretically non-negative
                # (J_C = E[sum gamma^t C_k] >= 0), but GAE can produce negative values
                # when the cost critic overestimates. Without clamping, softplus-bounded
                # predictions (>=0) can never match negative targets, causing systematic bias.
                cost_value_loss = torch.tensor(0.0, device=self.device)
                if hasattr(self.policy, "evaluate_costs"):
                    cost_value_pred = self.policy.evaluate_costs(obs_mb)
                    target = cost_returns_mb.clamp(min=0.0)
                    # d_k^2-normalized: equalize gradient across constraints with different scales
                    per_k_mse = (target - cost_value_pred).pow(2).mean(dim=0)  # (K,)
                    cost_value_loss = (per_k_mse / self.d_k.pow(2).clamp(min=0.01)).mean()

                # Pure MSE: no barrier in value update (barrier is policy-only in NORBC)
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

        # ------------------------------------------------------------------
        # 4. Dual variable update (Lagrangian multiplier ascent)
        # ------------------------------------------------------------------
        # Gated on ls_success: when line search fails, policy is frozen (params reverted).
        # Growing lambda on a frozen policy creates a death spiral: more cost pressure
        # -> more LS failures -> more lambda growth -> unrecoverable.
        # Lambda LR warmup: linear ramp from 0 to lr_lambda over warmup period.
        with torch.no_grad():
            warmup_progress = min(1.0, self._iteration / self._lambda_warmup_end)
            effective_lr = self.lr_lambda * warmup_progress

            if ls_success:
                for k in range(self.num_constraints):
                    self.lambda_k[k] = (
                        self.lambda_k[k] + effective_lr * (mean_cost_returns[k] - self.d_k[k]) / self.d_k[k]
                    ).clamp(min=0.0, max=self.lambda_max)

        # ------------------------------------------------------------------
        # 5. Alpha (entropy temperature) update -- SAC-style dual
        # ------------------------------------------------------------------
        # Always update (not LS-gated): entropy is an observable property of
        # the current policy regardless of whether the actor step succeeded.
        # min_alpha J(alpha) = log_alpha * (H(pi) - H_target)
        #   H > H_target -> alpha decreases -> less entropy bonus
        #   H < H_target -> alpha increases -> more entropy bonus
        alpha_loss = self.log_alpha * (mean_entropy - self.target_entropy)
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ------------------------------------------------------------------
        # Store constraint monitoring metrics
        # (read by ConstraintEncoderRunner._log_constraint_metrics)
        # ------------------------------------------------------------------
        self._last_cost_returns = [mean_cost_returns[k].item() for k in range(self.num_constraints)]
        self._last_violations = violations
        self._last_lambdas = [self.lambda_k[k].item() for k in range(self.num_constraints)]
        self._last_line_search_success = float(ls_success)

        # Clear storage
        self.storage.clear()

        # ------------------------------------------------------------------
        # Return loss dict
        # ------------------------------------------------------------------
        loss_dict: dict[str, float] = {
            "value_function": mean_value_loss,
            "surrogate": reward_surrogate.item(),
            "cost_surrogate": cost_surrogate.item(),
            "entropy": mean_entropy,
            "kl": mean_kl,
            "kl_trpo": kl_after_trpo,
            "cost_value": mean_cost_value_loss,
            "lambda_mean": self.lambda_k.mean().item(),
            "lambda_lr_eff": effective_lr,
            "grad_norm_reward": reward_grad_norm,
            "grad_norm_cost": cost_grad_norm,
            "adv_raw_std": adv_raw_std.item(),
            "alpha_entropy": self.log_alpha.exp().item(),
        }
        if hasattr(self.policy, "z_bounds_loss"):
            loss_dict["z_bounds"] = mean_z_bounds_loss

        return loss_dict

    # ==================================================================
    # Compatibility
    # ==================================================================

    def set_max_iterations(self, max_iterations: int) -> None:
        """Configure iteration-based schedules (lambda warmup)."""
        self._lambda_warmup_end = max(1, int(self.lambda_warmup_frac * max_iterations))
        logger.info(
            "[ConstraintTRPO] Lambda warmup: %d iterations (%.0f%% of %d)",
            self._lambda_warmup_end,
            self.lambda_warmup_frac * 100,
            max_iterations,
        )
