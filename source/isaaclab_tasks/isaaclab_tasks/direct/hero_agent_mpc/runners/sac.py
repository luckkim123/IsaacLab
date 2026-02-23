# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SAC (Soft Actor-Critic) algorithm for AC-MPC training.

Custom SAC implementation without external RL library dependency.
Designed for GPU-batched Isaac Lab environments with TensorDict observations.

Components:
    - ReplayBuffer: GPU circular buffer storing TensorDict observations
    - SAC: actor/critic/target_critic optimization with entropy tuning

Key design choices:
    - Critic: configurable obs keys for flexible asymmetric/symmetric design
    - Actor uses reparameterization through MPC (end-to-end dynamics gradient)
    - Auto-tuning alpha (temperature) with target entropy = -action_dim
    - Twin Q-networks with soft target update
    - Separate dynamics optimizer: pred_net trained by prediction loss
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =============================================================================
# Replay Buffer
# =============================================================================


@dataclass
class Transition:
    """Single-step transition for replay buffer storage."""

    obs: dict[str, torch.Tensor]  # {key: (num_envs, dim)}
    action: torch.Tensor  # (num_envs, action_dim)
    reward: torch.Tensor  # (num_envs,)
    next_obs: dict[str, torch.Tensor]  # {key: (num_envs, dim)}
    done: torch.Tensor  # (num_envs,) bool -- TRUE terminations only (for SAC bootstrap)
    pred_error: torch.Tensor | None = None  # (num_envs, 8) prediction error at this step
    episode_boundary: torch.Tensor | None = None  # (num_envs,) bool -- episode end including timeouts


@dataclass
class SequenceBatch:
    """Multi-step consecutive transitions for dynamics prediction loss.

    Used by the multi-step dynamics auxiliary loss to reduce compounding error
    over the MPC prediction horizon. States include seq_len+1 entries
    (initial state + seq_len successor states from ground truth simulation).
    """

    states: torch.Tensor  # (batch, seq_len+1, state_dim)
    actions: torch.Tensor  # (batch, seq_len, action_dim)
    pred_error: torch.Tensor | None = None  # (batch, 8) from t=0, None when no error feedback


class ReplayBuffer:
    """GPU circular replay buffer for multi-key observations.

    Stores transitions as flat tensors on GPU. Each add() call stores
    num_envs transitions simultaneously (one per environment).

    Memory layout: (capacity, dim) for each tensor. When capacity is reached,
    old entries are overwritten cyclically.
    """

    def __init__(
        self,
        capacity: int,
        num_envs: int,
        obs_shapes: dict[str, int],
        action_dim: int,
        device: str,
        pred_error_dim: int = 0,
    ) -> None:
        self.capacity = capacity
        self.num_envs = num_envs
        self.device = device
        self._size = 0
        self._ptr = 0
        self._add_count = 0  # monotonic counter for sequence validity checking

        # Pre-allocate flat storage (capacity, dim) for each field
        self._obs = {k: torch.zeros(capacity, d, device=device) for k, d in obs_shapes.items()}
        self._next_obs = {k: torch.zeros(capacity, d, device=device) for k, d in obs_shapes.items()}
        self._actions = torch.zeros(capacity, action_dim, device=device)
        self._rewards = torch.zeros(capacity, device=device)
        self._dones = torch.zeros(capacity, dtype=torch.bool, device=device)
        # Episode boundaries: True for ANY episode end (termination OR timeout).
        # Used by sample_sequences to detect episode discontinuities.
        # Separate from _dones which stores only TRUE terminations (for SAC bootstrap).
        self._boundaries = torch.zeros(capacity, dtype=torch.bool, device=device)
        # Per-slot step counter for verifying temporal consecutiveness in sequences.
        self._step_ids = torch.full((capacity,), -1, dtype=torch.long, device=device)

        # Prediction error storage for ECNN-style error feedback
        self._pred_error_dim = pred_error_dim
        if pred_error_dim > 0:
            self._pred_errors = torch.zeros(capacity, pred_error_dim, device=device)
        else:
            self._pred_errors = None

    def add(self, transition: Transition) -> None:
        """Add num_envs transitions to the buffer."""
        n = self.num_envs
        indices = torch.arange(n, device=self.device) + self._ptr
        indices = indices % self.capacity

        for k in self._obs:
            self._obs[k][indices] = transition.obs[k]
            self._next_obs[k][indices] = transition.next_obs[k]
        self._actions[indices] = transition.action
        self._rewards[indices] = transition.reward
        self._dones[indices] = transition.done
        if transition.episode_boundary is not None:
            self._boundaries[indices] = transition.episode_boundary
        else:
            self._boundaries[indices] = transition.done
        self._step_ids[indices] = self._add_count
        if self._pred_errors is not None and transition.pred_error is not None:
            self._pred_errors[indices] = transition.pred_error

        self._ptr = (self._ptr + n) % self.capacity
        self._size = min(self._size + n, self.capacity)
        self._add_count += 1

    def sample(self, batch_size: int) -> Transition:
        """Sample a random batch of transitions."""
        indices = torch.randint(0, self._size, (batch_size,), device=self.device)
        return Transition(
            obs={k: v[indices] for k, v in self._obs.items()},
            action=self._actions[indices],
            reward=self._rewards[indices],
            next_obs={k: v[indices] for k, v in self._next_obs.items()},
            done=self._dones[indices],
            pred_error=self._pred_errors[indices] if self._pred_errors is not None else None,
        )

    def sample_sequences(
        self,
        batch_size: int,
        seq_len: int,
        state_key: str = "mpc_state",
    ) -> SequenceBatch | None:
        """Sample valid sequences of consecutive transitions for multi-step prediction.

        Exploits parallel env structure: env j's consecutive transitions are
        spaced num_envs apart in the buffer (j, j+num_envs, j+2*num_envs, ...).
        """
        if state_key not in self._obs or self._size < self.num_envs * (seq_len + 1):
            return None

        n = self.num_envs
        max_attempts = batch_size * 4  # oversample for rejection

        start_indices = torch.randint(0, self._size, (max_attempts,), device=self.device)
        offsets = torch.arange(seq_len, device=self.device) * n
        seq_indices = (start_indices.unsqueeze(1) + offsets.unsqueeze(0)) % self.capacity

        # Check temporal consecutiveness
        step_ids = self._step_ids[seq_indices]
        expected = step_ids[:, 0:1] + torch.arange(seq_len, device=self.device)
        steps_consecutive = (step_ids == expected).all(dim=1)

        # No episode boundaries (termination OR timeout) in any sequence slot.
        # Uses _boundaries (includes timeouts) rather than _dones (terminations only)
        # to prevent sequences from crossing episode discontinuities.
        boundaries = self._boundaries[seq_indices]
        no_mid_dones = ~boundaries.any(dim=1)

        valid_mask = steps_consecutive & no_mid_dones
        valid_indices = valid_mask.nonzero(as_tuple=False).view(-1)

        if valid_indices.shape[0] < batch_size:
            return None

        sel = valid_indices[:batch_size]
        sel_seq = seq_indices[sel]

        states_list = [self._obs[state_key][sel_seq[:, k]] for k in range(seq_len)]
        states_list.append(self._next_obs[state_key][sel_seq[:, -1]])
        states = torch.stack(states_list, dim=1)

        actions = self._actions[sel_seq]

        # Prediction error from t=0 (held constant across sequence, matching MPC rollout)
        pe = self._pred_errors[sel_seq[:, 0]] if self._pred_errors is not None else None

        return SequenceBatch(states=states, actions=actions, pred_error=pe)

    def __len__(self) -> int:
        return self._size


# =============================================================================
# SAC Algorithm
# =============================================================================


class SAC:
    """Soft Actor-Critic for AC-MPC training.

    Manages three optimization targets:
        1. Critic (Twin Q): minimizes TD error with target network
        2. Actor: maximizes Q - alpha * log_pi (via reparameterization)
        3. Alpha: auto-tunes entropy coefficient

    Plus a separate dynamics optimizer:
        4. Dynamics (pred_net): supervised prediction loss
    """

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        target_critic: nn.Module,
        *,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        dynamics_lr: float = 1e-3,
        dynamics_loss_weight: float = 1.0,
        multistep_horizon: int = 1,
        multistep_decay: float = 0.5,
        gamma: float = 0.99,
        tau: float = 0.005,
        init_alpha: float = 0.2,
        target_entropy: float | None = None,
        alpha_min: float = 0.0,
        actor_delay: int = 1,
        action_dim: int = 2,
        max_q: float = 0.0,
        critic_grad_clip: float = 1.0,
        policy_obs_key: str = "policy",
        critic_obs_keys: list[str] | None = None,
        dynamics_diag_interval: int = 50,
        multistep_eval_steps: list[int] | None = None,
        dynamics_dim_weights: list[float] | None = None,
        multistep_weight: float = 0.5,
        q_aggregation: str = "min",
        adaptive_dynamics_weights: bool = False,
        adaptive_weights_warmup: int = 2000,
        adaptive_weights_ema: float = 0.99,
    ) -> None:
        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.gamma = gamma
        self.tau = tau
        self.policy_obs_key = policy_obs_key
        self.critic_obs_keys = critic_obs_keys
        self.dynamics_loss_weight = dynamics_loss_weight
        self._multistep_horizon = multistep_horizon
        self._multistep_decay = multistep_decay
        self._multistep_weight = multistep_weight
        self._alpha_min = alpha_min
        # Eval steps: sparse checkpoints where dynamics loss is computed.
        # None/empty -> every step (backward-compatible original behavior).
        if multistep_eval_steps:
            invalid = [s for s in multistep_eval_steps if s < 1 or s > multistep_horizon]
            if invalid:
                raise ValueError(f"multistep_eval_steps {invalid} out of range [1, {multistep_horizon}]")
            self._eval_steps: set[int] = set(multistep_eval_steps)
        else:
            self._eval_steps: set[int] = set(range(1, multistep_horizon + 1))
        self._actor_delay = max(1, actor_delay)
        self._max_q = max_q
        self._critic_grad_clip = critic_grad_clip

        # Q-value aggregation strategy.
        # Ref: FastSAC (Seo et al., arXiv:2512.01996, 2025).
        if q_aggregation not in ("min", "avg"):
            raise ValueError(f"q_aggregation must be 'min' or 'avg', got '{q_aggregation}'")
        self._q_aggregation = q_aggregation

        # Auto-tuning alpha
        self.log_alpha = torch.tensor(
            math.log(init_alpha),
            requires_grad=True,
            device=next(actor.parameters()).device,
        )
        self.target_entropy = target_entropy if target_entropy is not None else -float(action_dim)

        # Actor optimizer: cost_map + log_std only.
        # dynamics params are excluded (trained by separate dynamics_optimizer).
        dynamics_param_ids = {id(p) for p in actor.dynamics.parameters()}
        actor_params = [p for p in actor.parameters() if id(p) not in dynamics_param_ids]
        self._actor_only_params = actor_params
        self.actor_optimizer = torch.optim.Adam(actor_params, lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        # Dynamics optimizer: pred_net (entire DynamicsMLP).
        self.dynamics_optimizer = torch.optim.Adam(actor.dynamics.parameters(), lr=dynamics_lr)

        # Per-dimension dynamics loss weights (None = uniform)
        if dynamics_dim_weights is not None:
            self._dim_weights = torch.tensor(
                dynamics_dim_weights,
                dtype=torch.float32,
                device=next(actor.parameters()).device,
            )
        else:
            self._dim_weights = None

        # VaGraM adaptive per-dim dynamics loss weighting.
        # Ref: Voelcker et al. (2022), Lambert et al. (L4DC 2020).
        self._adaptive_dynamics_weights = adaptive_dynamics_weights
        self._adaptive_weights_warmup = adaptive_weights_warmup
        self._adaptive_weights_ema = adaptive_weights_ema
        self._adaptive_dim_weights: torch.Tensor | None = None

        # pred_error dimension for critic input injection.
        # When "pred_error" is in critic_obs_keys, _build_critic_obs() appends
        # pred_error (or zeros) at the corresponding position.
        self._pred_error_dim = actor.dynamics.state_dim  # 8

        # Update counter
        self._update_count = 0
        self._diag_interval = max(1, dynamics_diag_interval)

        # Copy initial weights to target
        self._hard_update_target()

        logger.info(
            "SAC: actor_lr=%.1e, critic_lr=%.1e, dynamics_lr=%.1e, dyn_loss_w=%.2f, "
            "multistep=%d(decay=%.2f, eval_steps=%s, weight=%.2f), gamma=%.3f, tau=%.4f, "
            "target_entropy=%.2f, init_alpha=%.3f, alpha_min=%.3f, actor_delay=%d, "
            "max_q=%.1f, critic_grad_clip=%.1f, q_agg=%s, diag_interval=%d, "
            "dim_weights=%s, critic_obs_keys=%s",
            actor_lr,
            critic_lr,
            dynamics_lr,
            dynamics_loss_weight,
            multistep_horizon,
            multistep_decay,
            sorted(self._eval_steps),
            multistep_weight,
            gamma,
            tau,
            self.target_entropy,
            init_alpha,
            alpha_min,
            actor_delay,
            max_q,
            critic_grad_clip,
            q_aggregation,
            dynamics_diag_interval,
            dynamics_dim_weights,
            critic_obs_keys,
        )

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @property
    def multistep_horizon(self) -> int:
        """Multi-step prediction horizon for dynamics auxiliary loss."""
        return self._multistep_horizon

    def _aggregate_q(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Aggregate twin Q-values for critic target and actor loss.

        Ref: FastSAC (Seo et al., arXiv:2512.01996, 2025) showed 'avg'
        outperforms 'min' in sim-to-real with strong domain randomization.
        """
        if self._q_aggregation == "avg":
            return ((q1 + q2) * 0.5).squeeze(-1)
        return torch.min(q1, q2).squeeze(-1)

    def update(
        self,
        batch: Transition,
        seq_batch: SequenceBatch | None = None,
    ) -> dict[str, float]:
        """Perform one SAC update step on a batch of transitions.

        Args:
            batch: Sampled transitions from replay buffer.
            seq_batch: Optional multi-step sequences for dynamics loss.

        Returns:
            Dictionary of scalar metrics for logging.
        """
        metrics = {}

        # ---- Fix off-policy error feedback distribution shift ----
        # Replay buffer stores pred_error computed by OLD dynamics model.
        # Recompute with CURRENT model so all downstream consumers see fresh errors.
        fresh_pred_error = self._recompute_fresh_pred_error(batch)
        if fresh_pred_error is not None:
            if batch.pred_error is not None:
                metrics["ErrorFeedback/stale_vs_fresh_l2"] = (
                    (batch.pred_error - fresh_pred_error).norm(dim=-1).mean().item()
                )
            batch = Transition(
                obs=batch.obs,
                action=batch.action,
                reward=batch.reward,
                next_obs=batch.next_obs,
                done=batch.done,
                pred_error=fresh_pred_error,
            )

        if seq_batch is not None:
            fresh_seq_pe = self._recompute_fresh_seq_pred_error(seq_batch)
            if fresh_seq_pe is not None:
                seq_batch = SequenceBatch(
                    states=seq_batch.states,
                    actions=seq_batch.actions,
                    pred_error=fresh_seq_pe,
                )

        # ---- Critic update (every step) ----
        metrics.update(self._update_critic(batch))

        # ---- Actor update (delayed: every actor_delay steps) ----
        do_actor = self._update_count % self._actor_delay == 0
        if do_actor:
            actor_metrics, log_prob = self._update_actor(batch)
            metrics.update(actor_metrics)

            # ---- Alpha update (reuses log_prob from actor) ----
            metrics.update(self._update_alpha(log_prob))

        # ---- Dynamics update (every step -- independent of actor) ----
        if self.dynamics_loss_weight > 0.0:
            metrics.update(self._update_dynamics(batch, seq_batch))

        # ---- Target network soft update ----
        self._soft_update_target()

        # ---- Periodic dynamics diagnostics ----
        if self._update_count % self._diag_interval == 0:
            metrics.update(self._compute_dynamics_diagnostics(batch))

        self._update_count += 1
        return metrics

    def _build_critic_obs(
        self,
        obs_dict: dict[str, torch.Tensor],
        pred_error: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Concatenate obs keys for critic input, injecting pred_error if configured.

        When "pred_error" appears in critic_obs_keys, the corresponding tensor
        is taken from the ``pred_error`` argument (not from obs_dict, since env
        observations do not include it). If pred_error is None, zeros are used.

        Args:
            obs_dict: Observation dictionary from a Transition batch.
            pred_error: Dynamics prediction error. Shape: (batch, 8). None -> zeros.

        Returns:
            Concatenated critic input tensor. Shape: (batch, critic_obs_dim).
        """
        if self.critic_obs_keys is not None:
            parts = []
            for k in self.critic_obs_keys:
                if k == "pred_error":
                    if pred_error is not None:
                        parts.append(pred_error)
                    else:
                        # Infer batch size from first available obs tensor
                        ref = next(iter(obs_dict.values()))
                        parts.append(torch.zeros(ref.shape[0], self._pred_error_dim, device=ref.device))
                else:
                    parts.append(obs_dict[k])
            return torch.cat(parts, dim=-1)
        # Fallback: policy_obs only
        return obs_dict[self.policy_obs_key]

    def _update_critic(self, batch: Transition) -> dict[str, float]:
        """Update twin Q-networks with TD target."""
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample_no_diff(
                batch.next_obs,
                pred_error=batch.pred_error,
            )

            next_q1, next_q2 = self.target_critic(
                self._build_critic_obs(batch.next_obs, pred_error=batch.pred_error),
                None,
                next_action,
            )
            next_q = self._aggregate_q(next_q1, next_q2)
            target_q = batch.reward + self.gamma * (~batch.done).float() * (next_q - self.alpha * next_log_prob)

            if self._max_q > 0:
                target_q = target_q.clamp(-self._max_q, self._max_q)

        q1, q2 = self.critic(
            self._build_critic_obs(batch.obs, pred_error=batch.pred_error),
            None,
            batch.action,
        )
        q1 = q1.squeeze(-1)
        q2 = q2.squeeze(-1)

        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self._critic_grad_clip)
        self.critic_optimizer.step()

        return {
            "Loss/critic": critic_loss.item(),
            "SAC/q1_mean": q1.mean().item(),
            "SAC/q2_mean": q2.mean().item(),
            "SAC/target_q_mean": target_q.mean().item(),
        }

    def _update_actor(self, batch: Transition) -> tuple[dict[str, float], torch.Tensor]:
        """Update actor via reparameterization gradient through MPC."""
        action, log_prob = self.actor.sample(batch.obs, pred_error=batch.pred_error)

        q1, q2 = self.critic(
            self._build_critic_obs(batch.obs, pred_error=batch.pred_error),
            None,
            action,
        )
        q_agg = self._aggregate_q(q1, q2)

        actor_loss = (self.alpha.detach() * log_prob - q_agg).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(self._actor_only_params, max_norm=1.0)
        self.actor_optimizer.step()

        metrics = {
            "Loss/actor": actor_loss.item(),
            "SAC/log_prob_mean": log_prob.mean().item(),
            "SAC/actor_grad_norm": actor_grad_norm.item(),
        }
        return metrics, log_prob.detach()

    def _update_alpha(self, log_prob: torch.Tensor) -> dict[str, float]:
        """Update temperature parameter alpha."""
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        if self._alpha_min > 0:
            with torch.no_grad():
                self.log_alpha.clamp_(min=math.log(self._alpha_min))

        return {
            "SAC/alpha": self.alpha.item(),
            "Loss/alpha": alpha_loss.item(),
        }

    def _update_dynamics(
        self,
        batch: Transition,
        seq_batch: SequenceBatch | None = None,
    ) -> dict[str, float]:
        """Auxiliary supervised loss for dynamics MLP prediction accuracy.

        Always computes single-step loss (1-step accuracy for MPC rollouts).
        When multi-step sequences are available, also computes multi-step loss
        to reduce compounding error over long prediction horizons.
        The two losses are combined: single + multistep_weight * multi.
        """
        metrics: dict[str, float] = {}

        # --- VaGraM adaptive weights (override static dim_weights when active) ---
        adaptive_w = self._compute_adaptive_dim_weights(batch)
        if adaptive_w is not None:
            self._dim_weights = adaptive_w

        # --- Single-step loss (ALWAYS computed) ---
        single_loss, single_metrics = self._compute_single_step_loss(batch)
        if single_loss is None:
            return {}
        metrics.update(single_metrics)
        metrics["Loss/dynamics_single"] = single_loss.item()

        # --- Multi-step loss (when sequences available) ---
        device = single_loss.device
        multi_loss = torch.tensor(0.0, device=device)
        if seq_batch is not None and self._multistep_horizon > 1:
            multi_loss_val, step_errors = self._compute_multistep_loss(
                seq_batch.states,
                seq_batch.actions,
                self._multistep_horizon,
                pred_error=seq_batch.pred_error,
            )
            multi_loss = multi_loss_val

            eval_steps_sorted = sorted(self._eval_steps)
            for i, err in enumerate(step_errors):
                step_idx = eval_steps_sorted[i] if i < len(eval_steps_sorted) else i + 1
                metrics[f"Dynamics/pred_err_step{step_idx}"] = err
            if step_errors:
                metrics["Dynamics/multistep_err_mean"] = sum(step_errors) / len(step_errors)
            metrics["Loss/dynamics_multistep"] = multi_loss.item()

        # --- Combined loss ---
        total_loss = single_loss + self._multistep_weight * multi_loss

        self.dynamics_optimizer.zero_grad()
        total_loss.backward()
        dyn_grad_norm = nn.utils.clip_grad_norm_(self.actor.dynamics.parameters(), max_norm=1.0)
        self.dynamics_optimizer.step()

        if self._update_count < 100 and dyn_grad_norm < 1e-8:
            logger.warning(
                "Dynamics grad norm near zero (%.2e) at update %d -- check auxiliary loss graph connectivity.",
                dyn_grad_norm,
                self._update_count,
            )

        metrics["Loss/dynamics"] = total_loss.item()
        metrics["Dynamics/grad_norm"] = dyn_grad_norm.item()

        return metrics

    def _compute_single_step_loss(
        self,
        batch: Transition,
    ) -> tuple[torch.Tensor | None, dict[str, float]]:
        """Compute single-step dynamics prediction loss.

        Returns (loss_tensor, metrics_dict). Loss is None when no valid data.
        Does NOT call optimizer -- caller handles backward/step.
        """
        mpc_state = batch.obs.get(self.actor.mpc_state_key)
        next_mpc_state = batch.next_obs.get(self.actor.mpc_state_key)
        if mpc_state is None or next_mpc_state is None:
            return None, {}

        valid = ~batch.done
        if valid.sum() < 2:
            return None, {}
        mpc_state = mpc_state[valid]
        next_mpc_state = next_mpc_state[valid]
        action_valid = batch.action[valid]

        # Prediction error from replay buffer (for ECNN-style error feedback)
        pe_valid: torch.Tensor | None = None
        if batch.pred_error is not None:
            pe_valid = batch.pred_error[valid]

        _EPS = 1e-6
        action_pretanh = torch.atanh(action_valid.clamp(-1 + _EPS, 1 - _EPS)).clamp(-3.0, 3.0)

        # Per-member loss for ensemble diversity; single-member fast path otherwise.
        # Ref: PETS (Chua et al., NeurIPS 2018) trains each member independently.
        ensemble_size = self.actor.dynamics._ensemble_size
        if ensemble_size == 1:
            pred_next = self.actor.dynamics(mpc_state, action_pretanh, pred_error=pe_valid)
            if self._dim_weights is not None:
                loss = ((pred_next - next_mpc_state) ** 2 * self._dim_weights).mean() * self.dynamics_loss_weight
            else:
                loss = F.mse_loss(pred_next, next_mpc_state) * self.dynamics_loss_weight
        else:
            total_loss = torch.tensor(0.0, device=mpc_state.device)
            pred_next = None
            for m in range(ensemble_size):
                pred_m = self.actor.dynamics.forward_member(
                    m,
                    mpc_state,
                    action_pretanh,
                    pred_error=pe_valid,
                )
                if self._dim_weights is not None:
                    total_loss = total_loss + ((pred_m - next_mpc_state) ** 2 * self._dim_weights).mean()
                else:
                    total_loss = total_loss + F.mse_loss(pred_m, next_mpc_state)
                if pred_next is None:
                    pred_next = pred_m  # use first member for pred_err diagnostic
            loss = (total_loss / ensemble_size) * self.dynamics_loss_weight

        with torch.no_grad():
            pred_err_mean = (pred_next - next_mpc_state).abs().mean().item()

        return loss, {"Dynamics/pred_err_mean": pred_err_mean}

    def _compute_multistep_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        seq_len: int,
        pred_error: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[float]]:
        """Compute multi-step dynamics prediction loss at sparse eval checkpoints.

        Unrolls dynamics for seq_len micro-steps from the first ground-truth
        state using PREDICTED states (not ground truth). Loss is computed only
        at micro-step indices in self._eval_steps, with exponential decay
        applied per eval step (not per micro-step).
        """
        _EPS = 1e-6
        actions_pretanh = torch.atanh(actions.clamp(-1 + _EPS, 1 - _EPS)).clamp(-3.0, 3.0)

        x_pred = states[:, 0, :]  # start from ground truth x_0
        total_loss = torch.tensor(0.0, device=states.device)
        step_errors: list[float] = []
        eval_idx = 0

        dynamics = self.actor.dynamics
        for k in range(seq_len):
            # pred_error from t=0 held constant across all unrolled steps
            # (matches MPC rollout behavior: no future actuals available)
            x_pred = dynamics(x_pred, actions_pretanh[:, k, :], pred_error=pred_error)

            if (k + 1) in self._eval_steps:
                target = states[:, k + 1, :]
                if self._dim_weights is not None:
                    step_loss = ((x_pred - target) ** 2 * self._dim_weights).mean()
                else:
                    step_loss = F.mse_loss(x_pred, target)
                total_loss = total_loss + (self._multistep_decay**eval_idx) * step_loss
                with torch.no_grad():
                    step_errors.append((x_pred - target).abs().mean().item())
                eval_idx += 1

        return total_loss * self.dynamics_loss_weight, step_errors

    def _recompute_fresh_pred_error(self, batch: Transition) -> torch.Tensor | None:
        """Recompute pred_error using CURRENT dynamics to fix off-policy distribution shift.

        Stored pred_error from replay buffer was computed by an older dynamics model.
        Fresh recomputation: dynamics(x, u, pred_error=zeros) - x_next
        uses the current model's base prediction (zero error input = no prior correction).
        """
        if not getattr(self.actor.dynamics, "use_error_feedback", False):
            return None

        mpc_state = batch.obs.get(self.actor.mpc_state_key)
        next_mpc_state = batch.next_obs.get(self.actor.mpc_state_key)
        if mpc_state is None or next_mpc_state is None:
            return None

        state_dim = self.actor.dynamics.state_dim  # 8

        with torch.no_grad():
            _EPS = 1e-6
            action_pretanh = torch.atanh(batch.action.clamp(-1 + _EPS, 1 - _EPS)).clamp(-3.0, 3.0)

            # Zero error = base prediction (no prior correction)
            zero_error = torch.zeros(mpc_state.shape[0], state_dim, device=mpc_state.device)
            pred_next = self.actor.dynamics(
                mpc_state,
                action_pretanh,
                pred_error=zero_error,
            )
            fresh_error = (pred_next[:, :state_dim] - next_mpc_state[:, :state_dim]).clamp(-10.0, 10.0)

        return fresh_error

    def _recompute_fresh_seq_pred_error(self, seq_batch: SequenceBatch) -> torch.Tensor | None:
        """Recompute pred_error at t=0 of multi-step sequences using current dynamics."""
        if not getattr(self.actor.dynamics, "use_error_feedback", False):
            return None
        if seq_batch is None:
            return None

        state_dim = self.actor.dynamics.state_dim
        x0 = seq_batch.states[:, 0, :]
        x1 = seq_batch.states[:, 1, :]
        a0 = seq_batch.actions[:, 0, :]

        with torch.no_grad():
            _EPS = 1e-6
            a0_pretanh = torch.atanh(a0.clamp(-1 + _EPS, 1 - _EPS)).clamp(-3.0, 3.0)
            zero_error = torch.zeros(x0.shape[0], state_dim, device=x0.device)
            pred_next = self.actor.dynamics(x0, a0_pretanh, pred_error=zero_error)
            fresh_error = (pred_next[:, :state_dim] - x1[:, :state_dim]).clamp(-10.0, 10.0)

        return fresh_error

    def _compute_adaptive_dim_weights(self, batch: Transition) -> torch.Tensor | None:
        """Compute VaGraM-style adaptive per-dim dynamics loss weights.

        Weights are proportional to |dQ/ds| per state dimension, normalized to
        mean=1.0 (keeps total loss scale stable). EMA-smoothed across updates.

        Ref: Voelcker et al., "Value Gradient weighted Model-Based RL" (2022).

        Returns None during warmup or when disabled.
        """
        if not self._adaptive_dynamics_weights:
            return None
        if self._update_count < self._adaptive_weights_warmup:
            return None

        mpc_state_key = self.actor.mpc_state_key
        mpc_state = batch.obs.get(mpc_state_key)
        if mpc_state is None:
            return None

        # Compute dQ/ds using the critic's Q-values.
        # Use a small random subset for efficiency (64 samples).
        n = min(64, mpc_state.shape[0])
        state_sub = mpc_state[:n].detach().requires_grad_(True)

        # Reconstruct critic obs with state_sub spliced in at the mpc_state
        # position so autograd can trace dQ/ds through the critic forward pass.
        obs_sub = {k: v[:n].detach() for k, v in batch.obs.items() if isinstance(v, torch.Tensor)}
        obs_sub[mpc_state_key] = state_sub
        pe_sub = batch.pred_error[:n].detach() if batch.pred_error is not None else None
        critic_obs_sub = self._build_critic_obs(obs_sub, pred_error=pe_sub)
        action_sub = batch.action[:n].detach()

        q1, q2 = self.critic(critic_obs_sub, None, action_sub)
        q_mean = ((q1 + q2) * 0.5).squeeze(-1).sum()

        grad = torch.autograd.grad(q_mean, state_sub, create_graph=False, allow_unused=True)[0]
        if grad is None:
            # mpc_state not in critic_obs_keys -- gradient cannot be computed.
            return None
        # Per-dim magnitude averaged across batch: (full_state_dim,)
        grad_mag = grad.abs().mean(dim=0)

        # Only weight physical dims (first 8); q_target dims (8:10) are always exact.
        phys_grad = grad_mag[:8]

        # Normalize to mean=1.0 to keep loss scale stable
        grad_mean = phys_grad.mean()
        if grad_mean < 1e-12:
            return None
        weights_phys = phys_grad / grad_mean

        # Pad to full state dim (10D) with 1.0 for q_target dims
        weights = torch.ones(mpc_state.shape[-1], device=mpc_state.device)
        weights[:8] = weights_phys

        # EMA update
        if self._adaptive_dim_weights is None:
            self._adaptive_dim_weights = weights
        else:
            ema = self._adaptive_weights_ema
            self._adaptive_dim_weights = ema * self._adaptive_dim_weights + (1 - ema) * weights

        return self._adaptive_dim_weights.detach()

    def _compute_dynamics_diagnostics(
        self,
        batch: Transition,
    ) -> dict[str, float]:
        """Compute diagnostic metrics that expose whether dynamics MLP learns beyond identity.

        The residual formulation (x'=x+f*dt) gives "free" accuracy from the identity
        prediction. These diagnostics compare learned model vs identity baseline to
        reveal genuine learning.

        All computation is in torch.no_grad() for zero training overhead.
        Called every _diag_interval updates.
        """
        metrics: dict[str, float] = {}

        mpc_state = batch.obs.get(self.actor.mpc_state_key)
        next_mpc_state = batch.next_obs.get(self.actor.mpc_state_key)
        if mpc_state is None or next_mpc_state is None:
            return metrics

        valid = ~batch.done
        if valid.sum() < 2:
            return metrics

        with torch.no_grad():
            state = mpc_state[valid]
            next_state = next_mpc_state[valid]
            action = batch.action[valid]

            # Prediction error from replay buffer
            pe_valid: torch.Tensor | None = None
            if batch.pred_error is not None:
                pe_valid = batch.pred_error[valid]

            # Pre-tanh action (same transform used in dynamics training)
            _EPS = 1e-6
            action_pretanh = torch.atanh(action.clamp(-1 + _EPS, 1 - _EPS)).clamp(-3.0, 3.0)

            # ---- 1. Identity baseline vs model prediction ----
            baseline_err = next_state - state
            baseline_mse = (baseline_err**2).mean().item()

            pred_next = self.actor.dynamics(state, action_pretanh, pred_error=pe_valid)
            model_err = pred_next - next_state
            model_mse = (model_err**2).mean().item()

            metrics["Dynamics/diag_model_mse"] = model_mse

            if baseline_mse > 1e-12:
                metrics["Dynamics/diag_improvement"] = 1.0 - model_mse / baseline_mse
            else:
                metrics["Dynamics/diag_improvement"] = 0.0

            # ---- 2. Residual magnitude ----
            mlp_parts = [state, action_pretanh]
            if pe_valid is not None:
                mlp_parts.append(pe_valid)
            mlp_input = torch.cat(mlp_parts, dim=-1)
            mlp_input = mlp_input / self.actor.dynamics._input_scale
            delta = self.actor.dynamics.pred_net(mlp_input)
            residual_mag = (delta * self.actor.dynamics.dt).abs().mean().item()
            metrics["Dynamics/diag_residual_mag"] = residual_mag

            # ---- 3. Aggregate error breakdown ----
            model_mae_per_dim = model_err.abs().mean(dim=0)
            # attitude = mean(phi, theta), joint = mean(q1, q2)
            if model_mae_per_dim.shape[0] >= 6:
                metrics["Dynamics/diag_err_attitude"] = ((model_mae_per_dim[0] + model_mae_per_dim[1]) / 2).item()
                metrics["Dynamics/diag_err_joint"] = ((model_mae_per_dim[4] + model_mae_per_dim[5]) / 2).item()

            # ---- 3b. Prediction error norm (error feedback only) ----
            if pe_valid is not None:
                metrics["ErrorFeedback/pred_error_norm"] = pe_valid.norm(dim=-1).mean().item()

            # ---- 4. Base dynamics quality (zero error feedback) ----
            # Measures prediction error when error feedback is withheld,
            # revealing how much base dynamics relies on the error crutch.
            if pe_valid is not None:
                zero_pe = torch.zeros_like(pe_valid)
                pred_next_base = self.actor.dynamics(state, action_pretanh, pred_error=zero_pe)
                base_err = (pred_next_base - next_state).abs().mean().item()
                metrics["Dynamics/diag_base_pred_err"] = base_err

        return metrics

    def _hard_update_target(self) -> None:
        """Copy critic parameters to target critic."""
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _soft_update_target(self) -> None:
        """Polyak averaging: target = tau * critic + (1-tau) * target."""
        with torch.no_grad():
            for tp, cp in zip(self.target_critic.parameters(), self.critic.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * cp.data)

    def state_dict(self) -> dict:
        """Save all SAC state for checkpointing."""
        d = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "log_alpha": self.log_alpha.detach().clone(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "dynamics_optimizer": self.dynamics_optimizer.state_dict(),
            "update_count": self._update_count,
        }
        return d

    def load_state_dict(self, state: dict) -> None:
        """Load SAC state from checkpoint."""
        # strict=False: MPC solver buffers (_u_prev) are saved but lazily initialized.
        # Also handles checkpoint incompatibility when cost_map dimensions change
        # (e.g., output dim 62D -> 52D after Q_diag fix). Mismatched layers will
        # be randomly re-initialized, which is safe (cost_map retrains quickly).
        missing, unexpected = self.actor.load_state_dict(state["actor"], strict=False)
        if missing or unexpected:
            logger.warning(
                "Actor checkpoint mismatch (strict=False): %d missing, %d unexpected keys. "
                "Mismatched layers will use random init.",
                len(missing),
                len(unexpected),
            )
        self.critic.load_state_dict(state["critic"])
        self.target_critic.load_state_dict(state["target_critic"])
        self.log_alpha.data.copy_(state["log_alpha"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
        if "dynamics_optimizer" in state:
            self.dynamics_optimizer.load_state_dict(state["dynamics_optimizer"])
        self._update_count = state.get("update_count", 0)
