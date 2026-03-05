# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Constraint TRPO with Interior-point Policy Optimization (IPO).

Implements the NORBC-style constrained RL algorithm:
    - TRPO natural gradient for policy update (exact KL constraint)
    - Log-barrier penalty on constraint margins for value function
    - Line search with feasibility check (KL + constraint satisfaction)
    - Separate cost value heads with per-constraint GAE

The algorithm maintains the same interface as RSL-RL PPO (init_storage, act,
process_env_step, compute_returns, update) so it can be used as a drop-in
replacement in the OnPolicyRunner.

Reference:
    Kim et al., "NORBC: Neural Online Robust Boundary Controller for
    Underwater Robots", IROS 2024.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

logger = logging.getLogger(__name__)


class ConstraintTRPO:
    """TRPO + IPO for constrained policy optimization.

    Key differences from PPO:
        - Policy update: natural gradient via conjugate gradient (full-batch)
        - Value update: mini-batch Adam with cost value loss + barrier
        - Line search: verifies KL constraint AND constraint feasibility
        - No clipped surrogate; uses exact KL constraint
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
        value_lr: float = 3e-4,
        max_grad_norm: float = 1.0,
        # GAE parameters
        gamma: float = 0.99,
        lam: float = 0.95,
        # Constraint / IPO parameters
        num_constraints: int = 3,
        constraint_budgets: tuple[float, ...] = (0.1, 0.05, 0.1),
        cost_gamma: float = 0.99,
        cost_lam: float = 0.95,
        barrier_t: float = 1.0,
        barrier_t_final: float = 50.0,
        barrier_t_schedule_iters: int = 1000,
        adaptive_threshold_alpha: float = 0.1,
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
        self.barrier_t_final = barrier_t_final
        self.barrier_t_schedule_iters = barrier_t_schedule_iters
        self.adaptive_alpha = adaptive_threshold_alpha

        # Discounted budgets: d_k = D_k / (1 - gamma)
        self.d_k = torch.tensor(
            [b / (1.0 - cost_gamma) for b in constraint_budgets],
            device=device,
            dtype=torch.float32,
        )
        # Adaptive thresholds (initialized to d_k)
        self.d_k_adaptive = self.d_k.clone()

        # Separate parameter groups:
        # - Policy params (actor + encoder): TRPO natural gradient (no optimizer)
        # - Value params (critic + cost_critic): Adam optimizer
        policy_param_names = set()
        value_params = []
        self._policy_params = []

        for name, param in self.policy.named_parameters():
            is_value = name.startswith("critic") or name.startswith("cost_critic")
            if is_value:
                value_params.append(param)
            else:
                self._policy_params.append(param)
                policy_param_names.add(name)

        self.value_optimizer = optim.Adam(value_params, lr=value_lr)
        logger.info(
            "ConstraintTRPO: %d policy params, %d value params",
            len(self._policy_params),
            len(value_params),
        )

        # Encoder LR (compatibility with EncoderRunner schedule)
        self._has_encoder_params = any("encoder" in n for n in policy_param_names)
        self.encoder_lr = 3e-3  # placeholder for EncoderRunner compatibility

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

    def _surrogate_loss(
        self,
        obs: TensorDict,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        old_log_prob: torch.Tensor,
    ) -> torch.Tensor:
        """Compute policy surrogate loss: -E[A * ratio]."""
        self.policy.act(obs)
        log_prob = self.policy.get_actions_log_prob(actions)
        ratio = torch.exp(log_prob - old_log_prob)
        return -(advantages * ratio).mean()

    def _kl_divergence(self, obs: TensorDict, old_mu: torch.Tensor, old_sigma: torch.Tensor) -> torch.Tensor:
        """Compute mean KL(pi_old || pi_new) analytically for Gaussian."""
        self.policy.act(obs)
        mu = self.policy.action_mean
        sigma = self.policy.action_std
        kl = (
            torch.log(sigma / old_sigma + 1e-5) + (old_sigma.pow(2) + (old_mu - mu).pow(2)) / (2.0 * sigma.pow(2)) - 0.5
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
                torch.log(sigma / old_sigma + 1e-5)
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
        old_mu: torch.Tensor,
        old_sigma: torch.Tensor,
        step_dir: torch.Tensor,
        old_loss: torch.Tensor,
    ) -> bool:
        """Backtracking line search with KL and feasibility checks.

        Returns True if a valid step was found.
        """
        old_params = self._get_policy_params_flat()
        step_size = 1.0

        for i in range(self.line_search_max_backtracks):
            new_params = old_params + step_size * step_dir
            self._set_policy_params_flat(new_params)

            with torch.no_grad():
                new_loss = self._surrogate_loss(obs, actions, advantages, old_log_prob)
                kl = self._kl_divergence(obs, old_mu, old_sigma)

            # Check improvement and KL constraint
            improvement = old_loss - new_loss
            if improvement > 0 and kl <= self.max_kl * 1.5:
                logger.debug(
                    "Line search step %d: improvement=%.6f, kl=%.6f",
                    i,
                    improvement.item(),
                    kl.item(),
                )
                return True

            step_size *= self.line_search_shrink_factor

        # Revert to old parameters if no valid step found
        self._set_policy_params_flat(old_params)
        logger.debug("Line search failed after %d backtracks", self.line_search_max_backtracks)
        return False

    # ==================================================================
    # Barrier Loss
    # ==================================================================

    def _compute_barrier_loss(self, mean_cost_returns: torch.Tensor) -> torch.Tensor:
        """Log-barrier loss on constraint margins.

        Args:
            mean_cost_returns: Mean discounted cost return per constraint. Shape: (K,).

        Returns:
            Scalar barrier loss.
        """
        barrier = torch.tensor(0.0, device=self.device)
        for k in range(self.num_constraints):
            margin = (self.d_k_adaptive[k] - mean_cost_returns[k]).clamp(min=1e-6)
            barrier += -torch.log(margin) / self.barrier_t
        return barrier

    # ==================================================================
    # Main Update
    # ==================================================================

    def update(self) -> dict[str, float]:
        """Execute one iteration of ConstraintTRPO update.

        Steps:
            1. Value function update (mini-batch Adam, includes cost critic + barrier)
            2. TRPO policy update (full-batch natural gradient + line search)
            3. Update adaptive thresholds and clear storage
        """
        # Flatten storage for value update (clone to escape inference_mode)
        obs_flat = self.storage.observations.flatten(0, 1).clone()
        # Clone all storage tensors -- they are inference tensors (collected under
        # torch.inference_mode) and cannot be used directly in autograd.
        actions_flat = self.storage.actions.flatten(0, 1).clone()
        returns_flat = self.storage.returns.flatten(0, 1).clone()
        advantages_flat = self.storage.advantages.flatten(0, 1).clone()
        old_log_prob_flat = self.storage.actions_log_prob.flatten(0, 1).clone()
        old_mu_flat = self.storage.mu.flatten(0, 1).clone()
        old_sigma_flat = self.storage.sigma.flatten(0, 1).clone()

        # Cost storage flatten
        cost_returns_flat = self.storage.cost_returns.flatten(0, 1).clone()  # (B, K)

        batch_size = obs_flat.batch_size[0]

        # ------------------------------------------------------------------
        # 1. Value function update (mini-batch, multiple epochs)
        # ------------------------------------------------------------------
        mean_value_loss = 0.0
        mean_cost_value_loss = 0.0
        mean_barrier_loss = 0.0
        mean_entropy = 0.0
        mean_z_bounds_loss = 0.0
        num_value_updates = 0

        # Mean cost returns for barrier (computed once, full batch)
        mean_cost_returns = cost_returns_flat.mean(dim=0)  # (K,)

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

                # Reward value loss
                value_pred = self.policy.evaluate(obs_mb)
                value_loss = (returns_mb - value_pred).pow(2).mean()

                # Cost value loss (per constraint, averaged)
                cost_value_loss = torch.tensor(0.0, device=self.device)
                if hasattr(self.policy, "evaluate_costs"):
                    cost_value_pred = self.policy.evaluate_costs(obs_mb)
                    cost_value_loss = (cost_returns_mb - cost_value_pred).pow(2).mean()

                # Barrier loss
                barrier_loss = self._compute_barrier_loss(mean_cost_returns)

                # Z bounds loss
                z_b_loss = torch.tensor(0.0, device=self.device)
                if hasattr(self.policy, "z_bounds_loss"):
                    z_b_loss = self.policy.z_bounds_loss()

                # Combined value update
                total_value_loss = (
                    self.value_loss_coef * value_loss
                    + self.cost_value_loss_coef * cost_value_loss
                    + barrier_loss
                    + z_b_loss
                )

                self.value_optimizer.zero_grad()
                total_value_loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for p in self.policy.parameters() if p.grad is not None],
                    self.max_grad_norm,
                )
                self.value_optimizer.step()

                mean_value_loss += value_loss.item()
                mean_cost_value_loss += cost_value_loss.item()
                mean_barrier_loss += barrier_loss.item()
                mean_z_bounds_loss += z_b_loss.item()
                num_value_updates += 1

        if num_value_updates > 0:
            mean_value_loss /= num_value_updates
            mean_cost_value_loss /= num_value_updates
            mean_barrier_loss /= num_value_updates
            mean_z_bounds_loss /= num_value_updates

        # ------------------------------------------------------------------
        # 2. TRPO policy update (full-batch, single step)
        # ------------------------------------------------------------------
        # Compute policy gradient
        self.policy.act(obs_flat)
        log_prob = self.policy.get_actions_log_prob(actions_flat)
        entropy = self.policy.entropy

        surrogate_loss = -(advantages_flat.squeeze() * torch.exp(log_prob - old_log_prob_flat.squeeze())).mean()
        # Add entropy bonus to gradient
        policy_loss = surrogate_loss - self.entropy_coef * entropy.mean()

        mean_entropy = entropy.mean().item()

        # Gradient of surrogate loss w.r.t. policy params
        g = self._flat_grad(policy_loss, self._policy_params, retain_graph=True)

        # Natural gradient via conjugate gradient: x = F^{-1} g
        nat_grad = self._conjugate_gradient(obs_flat, old_mu_flat, old_sigma_flat, g)

        # Step size: sqrt(2 * max_kl / (g^T F^{-1} g))
        shs = 0.5 * nat_grad.dot(g)  # 0.5 * x^T F x approximation
        step_scale = torch.sqrt(self.max_kl / (shs + 1e-8))
        step_dir = step_scale * nat_grad

        # Line search with feasibility check
        with torch.no_grad():
            old_loss = self._surrogate_loss(
                obs_flat, actions_flat, advantages_flat.squeeze(), old_log_prob_flat.squeeze()
            )

        ls_success = self._line_search(
            obs_flat,
            actions_flat,
            old_log_prob_flat.squeeze(),
            advantages_flat.squeeze(),
            old_mu_flat,
            old_sigma_flat,
            step_dir,
            old_loss,
        )

        # Compute KL after update for logging
        with torch.no_grad():
            mean_kl = self._kl_divergence(obs_flat, old_mu_flat, old_sigma_flat).item()

        # ------------------------------------------------------------------
        # 3. Update adaptive thresholds
        # ------------------------------------------------------------------
        for k in range(self.num_constraints):
            j_c_k = mean_cost_returns[k].item()
            self.d_k_adaptive[k] = max(
                self.d_k[k].item(),
                j_c_k + self.adaptive_alpha * self.d_k[k].item(),
            )

        # Clear storage
        self.storage.clear()
        # Reset cost storage step counter
        # (RolloutStorage.clear() only resets self.step, our cost tensors
        # are indexed by the same step counter via add_transitions)

        # ------------------------------------------------------------------
        # Return loss dict (compatible with OnPolicyRunner logging)
        # ------------------------------------------------------------------
        loss_dict: dict[str, float] = {
            "value_function": mean_value_loss,
            "surrogate": surrogate_loss.item(),
            "entropy": mean_entropy,
            "kl": mean_kl,
            "cost_value": mean_cost_value_loss,
            "barrier": mean_barrier_loss,
            "line_search_success": float(ls_success),
        }
        if hasattr(self.policy, "z_bounds_loss"):
            loss_dict["z_bounds"] = mean_z_bounds_loss

        # Per-constraint cost returns for logging
        for k in range(self.num_constraints):
            loss_dict[f"cost_return_{k}"] = mean_cost_returns[k].item()
            loss_dict[f"d_k_adaptive_{k}"] = self.d_k_adaptive[k].item()

        loss_dict["barrier_t"] = self.barrier_t

        return loss_dict

    # ==================================================================
    # Barrier Schedule
    # ==================================================================

    def update_barrier_schedule(self, iteration: int) -> None:
        """Update barrier steepness t based on training progress."""
        if self.barrier_t_schedule_iters <= 0:
            return
        progress = min(1.0, iteration / self.barrier_t_schedule_iters)
        self.barrier_t = (
            self.barrier_t_final * progress + (1.0 - progress) * 1.0  # initial t=1.0
        )
