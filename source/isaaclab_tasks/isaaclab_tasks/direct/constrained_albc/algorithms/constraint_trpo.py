# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Constrained TRPO with Lagrangian constraint enforcement.

TRPO policy optimization with adaptive Lagrangian multipliers for constraint
satisfaction. Based on C-TRPO (Muller et al., ICML 2025) but replaces the
quadratic barrier penalty with linear Lagrangian cost surrogates.

The quadratic barrier (beta * phi_pp * cost_surr^2) had two structural flaws:
    1. Vanishing gradient: dB/dtheta = 2*S*dS/dtheta -> 0 when S = E[A_cost] ~ 0
       (well-calibrated cost critic makes cost advantages zero-mean)
    2. safe_mask zeroed out violated constraints, providing no recovery gradient

The Lagrangian penalty (lambda_k * cost_surr) fixes both:
    - Linear term: gradient = lambda_k * E[A_cost * dlog_pi/dtheta], nonzero
      per-sample even when batch mean E[A_cost] = 0
    - No masking: lambda_k grows via dual ascent for violated constraints,
      providing continuous pressure proportional to violation magnitude
    - lambda_max caps constraint gradient to protect reward priority

Key design decisions:
    - Adaptive lambda via dual ascent: lambda_k += lr * (J_C_k - d_k)
    - lambda_max = 0.5: constraint gradient <= 50% of reward gradient O(1)
    - Cost advantage standardization preserved (NORBC Sec IV-B)
    - LS-gated encoder updates: when line search fails, both actor and encoder frozen
    - Noise floor (min_std): primary exploration maintenance, outside trust region

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
        - Blend mode: reward always active + cost minimization for violated
          constraints + barrier for safe constraints (RC3 per-constraint blend)
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
        # Lagrangian constraint parameters
        beta: float = 0.01,  # deprecated (barrier removed), kept for compat
        recovery_threshold_frac: float = 0.8,  # deprecated, kept for compat
        lambda_lr: float = 0.035,
        lambda_max: float = 0.5,
        ema_cost_alpha: float = 0.3,
        # Encoder z bounds
        z_bounds_coef: float = 0.0,
        # Encoder update
        num_encoder_epochs: int = 1,
        encoder_lr: float = 3e-4,
        # Noise floor (exploration maintenance)
        min_std: float = 0.2,
        # Entropy bonus
        entropy_coef: float = 0.0,
        # EAPO parameters
        eapo_enabled: bool = False,
        eapo_tau_init: float = 0.01,
        eapo_target_entropy: float = 0.5,
        eapo_tau_lr: float = 0.001,
        eapo_tau_min: float = 0.001,
        eapo_tau_max: float = 0.5,
        # Post-encoder KL gating (Fix 2: prevents encoder-induced KL violation)
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
        self.z_bounds_coef = z_bounds_coef
        self.num_encoder_epochs = num_encoder_epochs
        self.min_std = min_std
        self.entropy_coef = entropy_coef
        self.max_encoder_kl = max_encoder_kl

        # EAPO state
        self.eapo_enabled = eapo_enabled
        self.eapo_tau = eapo_tau_init
        self.eapo_target_entropy = eapo_target_entropy
        self.eapo_tau_lr = eapo_tau_lr
        self.eapo_tau_min = eapo_tau_min
        self.eapo_tau_max = eapo_tau_max
        self._cached_entropy_tau = eapo_tau_init

        # Lagrangian multipliers (adaptive dual variables)
        self._lambda_k = torch.zeros(num_constraints, device=device)
        self._lambda_lr = lambda_lr
        self._lambda_max = lambda_max
        self._margins = torch.zeros(num_constraints, device=device)

        # B1: EMA smoothing on mean_cost_returns to prevent phantom mode switches.
        # When cost critic updates without actor change, raw mean_cost_returns can
        # jump, causing spurious mode transitions. EMA damps this noise.
        self._ema_cost_returns = torch.zeros(num_constraints, device=device)
        self._ema_alpha = ema_cost_alpha
        self._ema_initialized = False

        # Initialize monitoring attributes (read by ConstraintEncoderRunner before first update)
        self._cached_lagrangian_penalty = 0.0
        self._last_cost_returns = [0.0] * num_constraints
        self._last_violations = [0.0] * num_constraints
        self._last_line_search_success = 0.0
        self._last_margins = [0.0] * num_constraints
        self._last_in_recovery = [0.0] * num_constraints
        self._last_lagrangian_penalty = 0.0
        self._last_mode = 0
        self._cached_mean_entropy = 0.0
        self._cached_surrogate_loss = 0.0
        self._last_pre_encoder_kl = 0.0

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
        self._base_value_lr = value_lr  # B2: saved for LR gating
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
        self.storage.cost_advantages_raw = torch.zeros(T, N, K, device=self.device)

        if self.eapo_enabled:
            self.storage.entropy_advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

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

        if self.eapo_enabled:
            self._compute_entropy_advantages()

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

        # Store raw (unstandardized) cost advantages for barrier penalty.
        # Standardized advantages have E[A_cost] = 0, making barrier gradient
        # structurally zero (cost_surr = mean(ratio * A) = 0 at ratio=1).
        # Barrier needs the actual cost change signal to repel from boundaries.
        self.storage.cost_advantages_raw = self.storage.cost_advantages.clone()

        mean = self.storage.cost_advantages.mean(dim=(0, 1), keepdim=True)
        std = self.storage.cost_advantages.std(dim=(0, 1), keepdim=True)
        self.storage.cost_advantages = (self.storage.cost_advantages - mean) / (std + 1e-8)

    def _compute_entropy_advantages(self) -> None:
        """Compute per-sample entropy advantages for EAPO.

        Uses -log_prob as entropy signal. With state-independent log_std,
        no entropy critic needed -- batch normalization as baseline.
        """
        entropy_signal = -self.storage.actions_log_prob  # (T, N, 1)
        ea_mean = entropy_signal.mean()
        ea_std = entropy_signal.std()
        if ea_std > 1e-8:
            self.storage.entropy_advantages = (entropy_signal - ea_mean) / ea_std
        else:
            self.storage.entropy_advantages = torch.zeros_like(entropy_signal)

    # ==================================================================
    # C-TRPO Barrier
    # ==================================================================

    def _compute_margins(self, mean_cost_returns: torch.Tensor) -> None:
        """Update per-constraint margins.

        Args:
            mean_cost_returns: Mean discounted cost return per constraint, shape (K,).
        """
        self._margins = self.d_k - mean_cost_returns

    def _compute_cost_surrogates(self, ratio: torch.Tensor, cost_advantages: torch.Tensor) -> torch.Tensor:
        """Compute per-constraint cost surrogates in a single vectorized op.

        Args:
            ratio: Importance sampling ratio pi/pi_old, shape (B,).
            cost_advantages: Per-constraint advantages, shape (B, K).

        Returns:
            Cost surrogates, shape (K,).
        """
        return (ratio.unsqueeze(-1) * cost_advantages).mean(dim=0)

    def _compute_lagrangian_penalty(self, cost_surrogates_std: torch.Tensor) -> torch.Tensor:
        """Linear Lagrangian penalty: sum_k lambda_k * E[ratio * A_cost_k_std].

        Gradient = lambda_k * E[A_cost_std * d_log_pi/d_theta] -- nonzero per-sample
        even when batch mean E[A_cost] = 0 (fixes quadratic barrier vanishing gradient).

        lambda_k adapted via dual ascent in _update_lambda():
        - Starts at 0 (reward-first learning)
        - Grows proportionally to constraint violation
        - Capped at lambda_max to protect reward gradient priority
        """
        return (self._lambda_k * cost_surrogates_std).sum()

    def _update_lambda(self, mean_cost_returns: torch.Tensor) -> None:
        """Dual gradient ascent on Lagrangian multipliers.

        lambda_k += lr * (J_C_k - d_k). Positive violation increases lambda.
        Clamped to [0, lambda_max] to keep reward gradient dominant.
        """
        with torch.no_grad():
            violation = mean_cost_returns - self.d_k
            self._lambda_k = (self._lambda_k + self._lambda_lr * violation).clamp(0.0, self._lambda_max)

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
        """Backtracking line search.

        Accepts a step when:
            1. Surrogate improvement > 0
            2. KL divergence <= max_kl * margin
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
        cost_advantages_std_flat = self.storage.cost_advantages.flatten(0, 1).clone()  # (B, K) standardized

        batch_size = obs_flat.batch_size[0]

        # Mean cost returns (computed once, needed for margins + logging)
        # Clamp to non-negative: cost value errors can make GAE return negative,
        # which would inflate barrier margin (d_k - (-X) = d_k + X).
        mean_cost_returns = cost_returns_flat.mean(dim=0).clamp(min=0.0)  # (K,)

        # ------------------------------------------------------------------
        # 1. Compute margins and determine safe/recovery mode
        # ------------------------------------------------------------------
        # B1: EMA smoothing to prevent phantom mode switches from cost critic drift.
        if not self._ema_initialized:
            self._ema_cost_returns = mean_cost_returns.clone()
            self._ema_initialized = True
        else:
            self._ema_cost_returns = (
                1 - self._ema_alpha
            ) * self._ema_cost_returns + self._ema_alpha * mean_cost_returns
        self._compute_margins(self._ema_cost_returns)
        self._update_lambda(self._ema_cost_returns)

        # Compute violations for logging
        violations = (mean_cost_returns - self.d_k).tolist()

        # ------------------------------------------------------------------
        # 2. TRPO policy update (full-batch, single step)
        # ------------------------------------------------------------------
        # Pure barrier-based: reward + barrier penalty for all constraints.
        # No recovery mode switching — barrier provides continuous gradient
        # pressure that strengthens as costs approach budget.
        old_lp_sq = old_log_prob_flat.squeeze(-1)
        adv_sq = advantages_flat.squeeze(-1)

        # EAPO: soft advantage = task + tau * entropy advantage
        if self.eapo_enabled:
            entropy_adv_flat = self.storage.entropy_advantages.flatten(0, 1).clone().squeeze(-1)
            soft_adv = adv_sq + self.eapo_tau * entropy_adv_flat
        else:
            soft_adv = adv_sq

        def surrogate() -> torch.Tensor:
            self.policy.act(obs_flat)
            log_prob = self.policy.get_actions_log_prob(actions_flat)
            ratio = torch.exp(log_prob - old_lp_sq)
            # Reward + EAPO entropy advantage
            reward_surr = -(soft_adv * ratio).mean()
            # Lagrangian constraint penalty (linear in cost surrogate)
            cost_surrs_std = self._compute_cost_surrogates(ratio, cost_advantages_std_flat)
            lp = self._compute_lagrangian_penalty(cost_surrs_std)
            self._cached_lagrangian_penalty = lp.item()
            # Entropy bonus
            entropy_bonus = -self.entropy_coef * self.policy.entropy.mean()
            self._cached_mean_entropy = self.policy.entropy.mean().item()
            return reward_surr + lp + entropy_bonus

        ls_success = self._trpo_step(obs_flat, old_mu_flat, old_sigma_flat, surrogate)

        # Noise floor: applied after TRPO step (outside trust region optimization).
        # Does not consume KL budget or distort natural gradient direction.
        # Prevents exploration collapse that entropy_coef failed to stabilize
        # (entropy in surrogate competes with reward for single KL budget step).
        min_log_std = math.log(self.min_std)
        with torch.no_grad():
            self.policy.log_std.data.clamp_(min=min_log_std)

        # EAPO: Adaptive tau (SAC v2 dual gradient, outside trust region)
        if self.eapo_enabled:
            with torch.no_grad():
                self.policy.act(obs_flat)
                current_entropy = self.policy.entropy.mean().item()
                # Dual gradient: tau increases when H < target
                self.eapo_tau -= self.eapo_tau_lr * (current_entropy - self.eapo_target_entropy)
                self.eapo_tau = max(self.eapo_tau_min, min(self.eapo_tau_max, self.eapo_tau))
            self._cached_entropy_tau = self.eapo_tau
            self._cached_mean_entropy = current_entropy

        # ------------------------------------------------------------------
        # 3. Encoder update (gated on ls_success)
        # ------------------------------------------------------------------
        # Fix 2: Measure pre-encoder KL for gating encoder-induced distribution shift
        with torch.no_grad():
            pre_encoder_kl = self._kl_divergence(obs_flat, old_mu_flat, old_sigma_flat).item()
        self._last_pre_encoder_kl = pre_encoder_kl

        mean_z_bounds_loss = 0.0
        if self.encoder_optimizer is not None and ls_success:
            mean_z_bounds_loss = self._update_encoder(
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
            obs_flat, returns_flat, cost_returns_flat, batch_size, actor_updated=ls_success
        )

        # ------------------------------------------------------------------
        # Store monitoring metrics (read by ConstraintEncoderRunner)
        # ------------------------------------------------------------------
        lagrangian_val = self._cached_lagrangian_penalty
        self._last_cost_returns = mean_cost_returns.tolist()
        self._last_violations = violations
        self._last_line_search_success = float(ls_success)
        self._last_margins = self._margins.tolist()
        self._last_in_recovery = [0.0] * self.num_constraints
        self._last_lagrangian_penalty = lagrangian_val
        self._last_mode = 0

        # Clear storage
        self.storage.clear()

        # ------------------------------------------------------------------
        # Return loss dict
        # ------------------------------------------------------------------
        loss_dict: dict[str, float] = {
            "value_function": mean_value_loss,
            "lagrangian_penalty": lagrangian_val,
            "kl": mean_kl,
            "cost_value": mean_cost_value_loss,
            "mode": float(self._last_mode),
            "adv_raw_std": adv_raw_std.item(),
            "surrogate": self._cached_surrogate_loss,
        }
        if hasattr(self.policy, "z_bounds_loss"):
            loss_dict["z_bounds"] = mean_z_bounds_loss
        if self.eapo_enabled:
            loss_dict["entropy_tau"] = self._cached_entropy_tau

        return loss_dict

    # ==================================================================
    # Internal: Unified TRPO step (safe or blend via callables)
    # ==================================================================

    def _trpo_step(
        self,
        obs_flat: TensorDict,
        old_mu_flat: torch.Tensor,
        old_sigma_flat: torch.Tensor,
        surrogate_fn: object,
    ) -> bool:
        """Execute a single TRPO natural-gradient step."""
        # 1. Compute loss + flat gradient
        loss = surrogate_fn()
        self._cached_surrogate_loss = loss.item()
        g = self._flat_grad(loss, self._policy_params, retain_graph=False)

        # 2. Natural gradient via conjugate gradient: x = F^{-1} g
        nat_grad = self._conjugate_gradient(obs_flat, old_mu_flat, old_sigma_flat, g)

        # 3. Step size: sqrt(2 * max_kl / (g^T F^{-1} g))
        shs = 0.5 * nat_grad.dot(g)

        if shs <= 0 or not torch.isfinite(shs):
            logger.warning("TRPO: shs=%.6e non-positive or non-finite, skipping", shs.item())
            return False

        step_dir = -torch.sqrt(self.max_kl / shs) * nat_grad

        if not torch.isfinite(step_dir).all():
            logger.warning("TRPO: step_dir contains NaN/Inf, skipping")
            return False

        # 4. Line search
        with torch.no_grad():
            old_loss = surrogate_fn()

        return self._line_search(
            obs_flat, old_mu_flat, old_sigma_flat, step_dir, old_loss, surrogate_fn
        )

    def _update_encoder(
        self,
        obs_flat: TensorDict,
        advantages_flat: torch.Tensor,
        old_log_prob_flat: torch.Tensor,
        actions_flat: torch.Tensor,
        old_mu_flat: torch.Tensor | None = None,
        old_sigma_flat: torch.Tensor | None = None,
        pre_encoder_kl: float = 0.0,
    ) -> float:
        """Multi-step encoder update with fresh forward passes.

        TRPO does a single full-batch policy step, while PPO does ~20 mini-batch
        updates (5 epochs x 4 batches). The encoder in PPO gets 20 gradient steps
        per iteration; without compensation, the C-TRPO encoder would get only 1.

        This method runs num_encoder_epochs fresh forward/backward passes through
        the encoder, each time recomputing the reward surrogate and z_bounds loss.
        The actor params are frozen (only encoder_optimizer steps), so this is safe.

        Fix 2: After each encoder step, checks if the resulting KL divergence
        exceeds pre_encoder_kl + max_encoder_kl. If so, reverts encoder params
        and stops early to prevent encoder-induced distribution shift.
        """
        mean_z_bounds_loss = 0.0
        kl_gating = self.max_encoder_kl > 0 and old_mu_flat is not None and old_sigma_flat is not None

        for _epoch in range(self.num_encoder_epochs):
            # Save encoder state for potential rollback (Fix 2)
            if kl_gating:
                saved_state = {n: p.data.clone() for n, p in self.policy.named_parameters() if n.startswith("encoder")}

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
            nn.utils.clip_grad_norm_(self._encoder_params, max_norm=0.5)
            self.encoder_optimizer.step()

            # Fix 2: KL gating -- revert if encoder step caused excessive KL shift
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

        return mean_z_bounds_loss

    def _update_values(
        self,
        obs_flat: TensorDict,
        returns_flat: torch.Tensor,
        cost_returns_flat: torch.Tensor,
        batch_size: int,
        actor_updated: bool = True,
    ) -> tuple[float, float]:
        """Update value functions (reward + cost) via MSE.

        B2: When actor is frozen (actor_updated=False), cost critic LR is reduced
        10x to prevent phantom mode switches from cost value drift.
        """
        mean_value_loss = 0.0
        mean_cost_value_loss = 0.0
        num_value_updates = 0

        # B2: Reduce value LR when actor is frozen to slow cost critic drift
        if not actor_updated:
            for pg in self.value_optimizer.param_groups:
                pg["lr"] = self._base_value_lr * 0.1

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

        # B2: Restore value LR after update
        if not actor_updated:
            for pg in self.value_optimizer.param_groups:
                pg["lr"] = self._base_value_lr

        return mean_value_loss, mean_cost_value_loss

    # ==================================================================
    # Compatibility
    # ==================================================================

    def set_max_iterations(self, max_iterations: int) -> None:
        """Interface compatibility with ConstraintEncoderRunner."""
        logger.info("[ConstraintTRPO] Lagrangian mode, lambda_lr=%.4f, lambda_max=%.2f, max_iterations=%d", self._lambda_lr, self._lambda_max, max_iterations)
