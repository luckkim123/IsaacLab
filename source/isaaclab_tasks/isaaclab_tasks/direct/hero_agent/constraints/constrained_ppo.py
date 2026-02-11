# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Constrained PPO with log-barrier loss and multi-head cost critics.

Extends RSL-RL's PPO to add:
    1. Cost value computation in act() via multi-head cost critic
    2. Cost storage in process_env_step()
    3. Barrier loss + cost value loss in update()
    4. Adaptive threshold updates after each update cycle

The loss becomes:
    L = L_surrogate + value_coef * L_value - entropy_coef * H
        - barrier_loss + cost_value_coef * L_cost_value
"""

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

from .barrier import LogBarrierManager
from .config import ConstrainedTrainingCfg, ConstraintCfg
from .constrained_storage import ConstrainedRolloutStorage


class ConstrainedPPO(PPO):
    """PPO with Interior-point Policy Optimization (IPO) for constrained RL.

    Overrides:
        - init_storage(): creates ConstrainedRolloutStorage
        - act(): additionally computes cost values via policy.evaluate_costs()
        - process_env_step(): stores costs in transition
        - compute_returns(): additionally computes cost returns
        - update(): adds barrier loss and cost value loss
    """

    def __init__(
        self,
        policy,
        constraint_cfgs: list[ConstraintCfg],
        constrained_cfg: ConstrainedTrainingCfg,
        **ppo_kwargs,
    ) -> None:
        """Initialize constrained PPO.

        Args:
            policy: ActorCriticConstrained with evaluate_costs() method.
            constraint_cfgs: List of active constraint configs.
            constrained_cfg: IPO hyperparameters.
            **ppo_kwargs: Standard PPO constructor arguments.
        """
        super().__init__(policy, **ppo_kwargs)

        self.constraint_cfgs = constraint_cfgs
        self.constrained_cfg = constrained_cfg
        self.num_constraints = len(constraint_cfgs)
        self.cost_value_coef = constrained_cfg.cost_value_coef

        # Log-barrier manager
        self.barrier = LogBarrierManager(
            constraints=constraint_cfgs,
            t=constrained_cfg.barrier_t,
            alpha=constrained_cfg.adaptive_alpha,
            gamma=constrained_cfg.cost_gamma,
            device=self.device,
        )

    def init_storage(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],
    ) -> None:
        """Create constrained rollout storage with cost buffers."""
        self.storage = ConstrainedRolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            self.device,
            num_constraints=self.num_constraints,
        )

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample action and compute cost values.

        Extends parent act() to additionally compute V_Ck(s) for each constraint.
        """
        actions = super().act(obs)

        # Compute cost values if policy supports it
        if self.num_constraints > 0 and hasattr(self.policy, "evaluate_costs"):
            self.transition.cost_values = self.policy.evaluate_costs(obs).detach()
        else:
            self.transition.cost_values = None

        return actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict
    ) -> None:
        """Process environment step, additionally storing costs.

        Costs are extracted from extras["costs"] (set by the constrained env).
        """
        # Extract costs from extras before parent call
        costs = extras.get("costs", None)
        if costs is not None and self.num_constraints > 0:
            self.transition.costs = costs.clone()
        else:
            self.transition.costs = None

        super().process_env_step(obs, rewards, dones, extras)

    def compute_returns(self, obs: TensorDict) -> None:
        """Compute reward returns + cost returns.

        Extends parent to compute cost GAE using separate cost discount factor.
        """
        # Standard reward returns
        last_values = self.policy.evaluate(obs).detach()

        # Cost returns
        last_cost_values = None
        if self.num_constraints > 0 and hasattr(self.policy, "evaluate_costs"):
            last_cost_values = self.policy.evaluate_costs(obs).detach()

        self.storage.compute_returns(
            last_values,
            self.gamma,
            self.lam,
            normalize_advantage=not self.normalize_advantage_per_mini_batch,
            last_cost_values=last_cost_values,
            cost_gamma=self.constrained_cfg.cost_gamma,
        )

    def update(self) -> dict[str, float]:
        """PPO update with barrier loss and cost value loss.

        Returns:
            Loss dictionary with standard PPO losses + constraint-specific losses.
        """
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_barrier_loss = 0.0
        mean_cost_value_loss = 0.0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hidden_states_batch,
            masks_batch,
            cost_advantages_batch,
            cost_returns_batch,
        ) in generator:
            original_batch_size = obs_batch.batch_size[0]

            # Normalize advantages per mini-batch if configured
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (
                        advantages_batch.std() + 1e-8
                    )

            # Forward pass
            self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(
                obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1]
            )
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # Adaptive LR (same as parent PPO)
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # Standard PPO loss
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Barrier loss and cost value loss
            barrier_loss = torch.tensor(0.0, device=self.device)
            cost_value_loss = torch.tensor(0.0, device=self.device)

            if self.num_constraints > 0 and cost_returns_batch is not None:
                # Barrier loss from log-barrier manager
                barrier_loss = self.barrier.compute_barrier_loss(cost_returns_batch)
                loss = loss + barrier_loss

                # Cost value loss (MSE for each constraint head)
                if hasattr(self.policy, "evaluate_costs"):
                    cost_value_pred = self.policy.evaluate_costs(obs_batch)
                    cost_value_loss = (cost_value_pred - cost_returns_batch).pow(2).mean()
                    loss = loss + self.cost_value_coef * cost_value_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Accumulate losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_barrier_loss += barrier_loss.item()
            mean_cost_value_loss += cost_value_loss.item()

        # Average over updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_barrier_loss /= num_updates
        mean_cost_value_loss /= num_updates

        # Update adaptive thresholds after full update cycle
        if self.num_constraints > 0:
            cost_returns_all = self.storage.cost_returns.flatten(0, 1)  # (T*N, K)
            self.barrier.update_thresholds(cost_returns_all)

        # Clear storage
        self.storage.clear()

        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "barrier": mean_barrier_loss,
            "cost_value": mean_cost_value_loss,
        }
        return loss_dict
