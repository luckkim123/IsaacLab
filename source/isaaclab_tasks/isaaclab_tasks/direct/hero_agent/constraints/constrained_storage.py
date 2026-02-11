# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Constrained rollout storage with cost buffers and cost GAE.

Extends RSL-RL's RolloutStorage to maintain per-constraint cost signals alongside
the standard reward/value/advantage buffers. Each constraint k has its own:
    - costs[t, n, k]: per-step cost
    - cost_values[t, n, k]: cost value function V_Ck(s)
    - cost_returns[t, n, k]: discounted cost return (from cost GAE)
    - cost_advantages[t, n, k]: cost advantage (for barrier loss)
"""

from __future__ import annotations

from collections.abc import Generator

import torch
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict


class ConstrainedRolloutStorage(RolloutStorage):
    """RolloutStorage extended with K constraint cost buffers.

    Adds cost-specific tensors of shape (T, N, K) and overrides compute_returns()
    to additionally compute cost GAE, and mini_batch_generator() to yield cost
    data alongside standard PPO data.
    """

    def __init__(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],
        device: str = "cpu",
        num_constraints: int = 0,
    ) -> None:
        super().__init__(training_type, num_envs, num_transitions_per_env, obs, actions_shape, device)

        self.num_constraints = num_constraints

        if num_constraints > 0 and training_type == "rl":
            T, N, K = num_transitions_per_env, num_envs, num_constraints
            self.costs = torch.zeros(T, N, K, device=device)
            self.cost_values = torch.zeros(T, N, K, device=device)
            self.cost_returns = torch.zeros(T, N, K, device=device)
            self.cost_advantages = torch.zeros(T, N, K, device=device)

    def add_transitions(self, transition: RolloutStorage.Transition) -> None:
        """Store transition including optional cost data.

        Cost data is attached to the transition as extra attributes:
            transition.costs: (num_envs, K) tensor
            transition.cost_values: (num_envs, K) tensor

        Args:
            transition: Standard transition (extended with cost attributes).
        """
        step = self.step  # capture before parent increments

        # Store cost data before parent call (parent increments self.step)
        has_costs = hasattr(transition, "costs") and transition.costs is not None
        if has_costs and self.num_constraints > 0:
            self.costs[step].copy_(transition.costs)
            self.cost_values[step].copy_(transition.cost_values)

        # Call parent to store standard data and increment step
        super().add_transitions(transition)

    def compute_returns(
        self,
        last_values: torch.Tensor,
        gamma: float,
        lam: float,
        normalize_advantage: bool = True,
        last_cost_values: torch.Tensor | None = None,
        cost_gamma: float = 0.99,
    ) -> None:
        """Compute reward GAE + cost GAE for each constraint.

        Args:
            last_values: Bootstrap value V(s_T), shape (N, 1).
            gamma: Reward discount factor.
            lam: GAE lambda.
            normalize_advantage: Whether to normalize reward advantages.
            last_cost_values: Bootstrap cost values V_C(s_T), shape (N, K).
            cost_gamma: Discount factor for cost returns.
        """
        # Standard reward GAE
        super().compute_returns(last_values, gamma, lam, normalize_advantage)

        # Cost GAE (no lambda smoothing -- use lambda=1.0 for unbiased cost return)
        if self.num_constraints > 0 and last_cost_values is not None:
            cost_advantage = torch.zeros(self.num_envs, self.num_constraints, device=self.device)
            for step in reversed(range(self.num_transitions_per_env)):
                next_cv = last_cost_values if step == self.num_transitions_per_env - 1 else self.cost_values[step + 1]
                not_done = 1.0 - self.dones[step].float()  # (N, 1)
                # TD error for costs: c_t + gamma * V_C(s_{t+1}) - V_C(s_t)
                delta = self.costs[step] + not_done * cost_gamma * next_cv - self.cost_values[step]
                # GAE with lambda=1.0 (Monte Carlo-like for unbiased cost return estimate)
                cost_advantage = delta + not_done * cost_gamma * cost_advantage
                self.cost_returns[step] = cost_advantage + self.cost_values[step]

            self.cost_advantages = self.cost_returns - self.cost_values

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8) -> Generator:
        """Yield standard PPO mini-batches extended with cost data.

        Each yielded tuple is the standard 10-tuple followed by:
            - cost_advantages_batch: (batch, K) or None
            - cost_returns_batch: (batch, K) or None

        Yields:
            12-tuple: (obs, actions, values, advantages, returns,
                       old_log_prob, old_mu, old_sigma,
                       hidden_states, masks,
                       cost_advantages, cost_returns)
        """
        if self.num_constraints == 0:
            # No constraints: delegate to parent and pad with None
            for batch in super().mini_batch_generator(num_mini_batches, num_epochs):
                yield (*batch, None, None)
            return

        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        # Flatten standard buffers
        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        # Flatten cost buffers
        cost_advantages_flat = self.cost_advantages.flatten(0, 1)  # (T*N, K)
        cost_returns_flat = self.cost_returns.flatten(0, 1)  # (T*N, K)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size
                batch_idx = indices[start:stop]

                yield (
                    observations[batch_idx],
                    actions[batch_idx],
                    values[batch_idx],
                    advantages[batch_idx],
                    returns[batch_idx],
                    old_actions_log_prob[batch_idx],
                    old_mu[batch_idx],
                    old_sigma[batch_idx],
                    (None, None),  # hidden_states (not recurrent)
                    None,  # masks
                    cost_advantages_flat[batch_idx],
                    cost_returns_flat[batch_idx],
                )
