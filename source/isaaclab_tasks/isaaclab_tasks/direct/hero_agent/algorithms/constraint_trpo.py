# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Constraint TRPO with Interior-point Policy Optimization (IPO).

Implements the NORBC-style constrained RL algorithm (Algorithm 1):
    1. Adaptive threshold update (instantaneous, Eq 11)
    2. TRPO natural gradient for policy update (exact KL constraint)
    3. Value function update (pure MSE, no barrier)

The log-barrier appears only in the policy objective (Eq 10), not in
the value loss. Separate cost value heads with per-constraint GAE.

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
    """TRPO + IPO for constrained policy optimization (NORBC Algorithm 1).

    Key differences from PPO:
        - Policy update: natural gradient via conjugate gradient (full-batch)
        - Value update: pure MSE (no barrier, barrier is policy-only)
        - Line search: verifies KL constraint and surrogate improvement
        - No clipped surrogate; uses exact KL constraint
        - Update order: threshold -> policy -> value (NORBC conformance)
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
        # Constraint / IPO parameters
        num_constraints: int = 3,
        constraint_budgets: tuple[float, ...] = (0.15, 0.02, 0.15),
        cost_gamma: float = 0.99,
        cost_lam: float = 0.95,
        barrier_t: float = 10.0,
        barrier_t_final: float = 50.0,
        barrier_t_schedule_frac: float = 0.4,
        adaptive_threshold_alpha: float = 0.1,
        # Line search acceptance threshold
        line_search_kl_margin: float = 1.5,
        # Entropy
        entropy_coef: float = 0.005,
        # Encoder z bounds
        z_bounds_coef: float = 0.3,
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
        self.entropy_coef = entropy_coef

        # GAE parameters
        self.gamma = gamma
        self.lam = lam

        # Constraint parameters
        self.num_constraints = num_constraints
        self.cost_gamma = cost_gamma
        self.cost_lam = cost_lam
        self.barrier_t = barrier_t
        self.barrier_t_init = barrier_t
        self.barrier_t_final = barrier_t_final
        self.barrier_t_schedule_frac = barrier_t_schedule_frac
        self.barrier_t_schedule_iters = 0  # resolved by set_max_iterations()
        self.adaptive_threshold_scale = adaptive_threshold_alpha
        self.line_search_kl_margin = line_search_kl_margin
        self.z_bounds_coef = z_bounds_coef

        if cost_gamma >= 1.0:
            raise ValueError(f"cost_gamma must be < 1.0, got {cost_gamma}")

        # Discounted budgets: d_k = D_k / (1 - gamma)
        self.d_k = torch.tensor(
            [b / (1.0 - cost_gamma) for b in constraint_budgets],
            device=device,
            dtype=torch.float32,
        )
        # Adaptive thresholds (initialized to d_k)
        self.d_k_adaptive = self.d_k.clone()

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

        # Iteration counter for barrier schedule (updated in update())
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
        # scales (e.g., rad/s vs binary 0/1), letting the barrier margin alone
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

    def _linearized_surrogate(
        self,
        obs: TensorDict,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        cost_advantages: torch.Tensor,
        old_log_prob: torch.Tensor,
        mean_cost_returns: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate linearized IPO surrogate matching the gradient objective.

        Uses the SAME formula as the policy gradient computation (reward surrogate
        + barrier-weighted cost surrogate - entropy). This ensures the natural
        gradient direction is guaranteed to improve the line search objective.
        """
        self.policy.act(obs)
        log_prob = self.policy.get_actions_log_prob(actions)
        ratio = torch.exp(log_prob - old_log_prob)
        reward_surr = -(advantages * ratio).mean()

        cost_surr = torch.tensor(0.0, device=self.device)
        for k in range(self.num_constraints):
            margin = (self.d_k_adaptive[k] - mean_cost_returns[k]).clamp(min=0.1 * self.d_k[k].item())
            cost_adv_k = (ratio * cost_advantages[:, k]).mean()
            cost_surr += cost_adv_k / ((1.0 - self.cost_gamma) * self.barrier_t * margin)

        entropy = self.policy.entropy
        return reward_surr + cost_surr - self.entropy_coef * entropy.mean()

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
        mean_cost_returns: torch.Tensor,
    ) -> bool:
        """Backtracking line search on the linearized IPO surrogate.

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
                new_loss = self._linearized_surrogate(
                    obs, actions, advantages, cost_advantages, old_log_prob, mean_cost_returns
                )
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
        """Execute one iteration of ConstraintTRPO update (NORBC Algorithm 1).

        Update order follows NORBC:
            1. Adaptive threshold update (instantaneous, Eq 11)
            2. TRPO policy update (full-batch natural gradient + line search)
            3. Value function update (pure MSE, no barrier)
        """
        # Update barrier schedule before this iteration's update
        self.update_barrier_schedule(self._iteration)
        self._iteration += 1

        # Flatten storage (clone to escape inference_mode)
        obs_flat = self.storage.observations.flatten(0, 1).clone()
        actions_flat = self.storage.actions.flatten(0, 1).clone()
        returns_flat = self.storage.returns.flatten(0, 1).clone()
        advantages_flat = self.storage.advantages.flatten(0, 1).clone()
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
        # 1. Update adaptive thresholds (NORBC Eq 11, instantaneous)
        # ------------------------------------------------------------------
        for k in range(self.num_constraints):
            j_c_k = mean_cost_returns[k].item()
            # Instantaneous: directly set to target (no EMA smoothing)
            self.d_k_adaptive[k] = max(self.d_k[k].item(), j_c_k + self.adaptive_threshold_scale * self.d_k[k].item())

        # ------------------------------------------------------------------
        # 2. TRPO policy update (full-batch, single step)
        # ------------------------------------------------------------------
        # Compute policy gradient
        self.policy.act(obs_flat)
        log_prob = self.policy.get_actions_log_prob(actions_flat)
        entropy = self.policy.entropy

        ratio = torch.exp(log_prob - old_log_prob_flat.squeeze(-1))
        reward_surrogate = -(advantages_flat.squeeze(-1) * ratio).mean()

        # IPO: barrier-weighted cost surrogate
        cost_surrogate = torch.tensor(0.0, device=self.device)
        for k in range(self.num_constraints):
            margin = (self.d_k_adaptive[k] - mean_cost_returns[k]).clamp(min=0.1 * self.d_k[k].item())
            cost_adv_k = (ratio * cost_advantages_flat[:, k]).mean()
            cost_surrogate += cost_adv_k / ((1.0 - self.cost_gamma) * self.barrier_t * margin)

        # Combined: reward surrogate + cost barrier surrogate - entropy
        policy_loss = reward_surrogate + cost_surrogate - self.entropy_coef * entropy.mean()

        mean_entropy = entropy.mean().item()

        # Compute encoder gradients now (defer step until after TRPO line search)
        _encoder_grads_cache: list[torch.Tensor | None] = []
        if self.encoder_optimizer is not None:
            encoder_grads = torch.autograd.grad(
                policy_loss,
                self._encoder_params,
                retain_graph=True,
                allow_unused=True,
            )
            _encoder_grads_cache = list(encoder_grads)

        # Gradient of surrogate loss w.r.t. actor params (TRPO)
        g = self._flat_grad(policy_loss, self._policy_params, retain_graph=True)

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
                        mean_cost_returns,
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
                    mean_cost_returns,
                )

        # Enforce minimum noise floor BEFORE KL measurement (prevent entropy collapse).
        # std=0.25 -> 95% of samples within +-0.5 of mean for [-1,1] actions.
        # Must clamp before KL so the logged metric reflects the actual policy state.
        min_log_std = math.log(0.25)
        with torch.no_grad():
            self.policy.log_std.data.clamp_(min=min_log_std)

        # Measure KL right after TRPO step + clamp (before encoder update shifts z)
        with torch.no_grad():
            kl_after_trpo = self._kl_divergence(obs_flat, old_mu_flat, old_sigma_flat).item()

        # Unified encoder update: policy-loss grads (gated by ls_success) + z_bounds grads.
        # BUG 3 fix: only apply policy-loss grads when TRPO succeeded (prevent actor-encoder desync).
        # BUG 1 fix: z_bounds_loss() already includes coef, do NOT multiply again.
        # Merged into single optimizer step to avoid conflicting update directions.
        mean_z_bounds_loss = 0.0
        if self.encoder_optimizer is not None:
            self.encoder_optimizer.zero_grad()
            has_grads = False

            # (1) Apply cached policy-loss gradients (only if TRPO succeeded)
            if ls_success and _encoder_grads_cache:
                for i, p in enumerate(self._encoder_params):
                    if i < len(_encoder_grads_cache) and _encoder_grads_cache[i] is not None:
                        p.grad = _encoder_grads_cache[i].clone()
                        has_grads = True

            # (2) Accumulate z_bounds gradients (always applied: safe regularization)
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
                    cost_value_loss = (cost_returns_mb.clamp(min=0.0) - cost_value_pred).pow(2).mean()

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
        # Store constraint monitoring metrics
        # (read by ConstraintEncoderRunner._log_constraint_metrics)
        # ------------------------------------------------------------------
        self._last_cost_returns = [mean_cost_returns[k].item() for k in range(self.num_constraints)]
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
        }
        if hasattr(self.policy, "z_bounds_loss"):
            loss_dict["z_bounds"] = mean_z_bounds_loss

        return loss_dict

    # ==================================================================
    # Barrier Schedule
    # ==================================================================

    def set_max_iterations(self, max_iterations: int) -> None:
        """Resolve barrier schedule iterations from fraction and total iterations."""
        self.barrier_t_schedule_iters = int(max_iterations * self.barrier_t_schedule_frac)

    def update_barrier_schedule(self, iteration: int) -> None:
        """Update barrier steepness t based on training progress."""
        if self.barrier_t_schedule_iters <= 0:
            return
        progress = min(1.0, iteration / self.barrier_t_schedule_iters)
        self.barrier_t = self.barrier_t_final * progress + (1.0 - progress) * self.barrier_t_init
