# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AC-MPC policy: MPC solver inside the actor for end-to-end SAC gradient.

Architecture:
    Cost Map Network: policy_obs -> Q_diag per horizon step
    DynamicsMLP (hybrid): phys(8D) learned residual + q_target(2D) deterministic
    MPC Solver: Q_diag + dynamics -> u* (2D joint velocity, differentiable)

SAC reparameterization:
    sample(obs):
        Q = cost_map(policy_obs)
        u* = MPC.solve(mpc_state, dynamics, Q, diff=True)
        action = tanh(u* + sigma * noise)
        log_prob = gaussian_log_prob - log(1 - tanh^2)

Reference:
    AC-MPC (Romero et al., 2024): Actor outputs cost parameters, MPC computes
    optimal control. We extend with SAC for off-policy + end-to-end dynamics.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from rsl_rl.networks import MLP

from ..controllers.dynamics_mlp import DynamicsMLP, DynamicsMLPCfg
from ..controllers.mpc import DifferentiableMPC, DifferentiableMPCCfg

if TYPE_CHECKING:
    from tensordict import TensorDict

logger = logging.getLogger(__name__)

# Numerical constant for log_prob computation
_LOG_STD_MIN = -5.0
_LOG_STD_MAX = 2.0
_LOG2PI = math.log(2 * math.pi)


class ActorCriticMPC(nn.Module):
    """AC-MPC policy with MPC solver inside the actor.

    For SAC compatibility, this module provides:
        - sample(obs) -> (action, log_prob): reparameterized sampling
        - act(obs) -> action: calls sample(), discards log_prob
        - act_inference(obs) -> action: deterministic MPC solve
        - get_dynamics() -> DynamicsMLP
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        # Policy dimensions
        policy_obs_dim: int = 13,
        # MPC parameters
        mpc_cfg: DifferentiableMPCCfg | None = None,
        mpc_horizon: int = 5,
        mpc_state_dim: int = 10,
        q_min: float = 0.1,
        q_max: float = 1000.0,
        r_min: float = 0.01,
        r_max: float = 100.0,
        # Dynamics parameters
        dynamics_hidden_dims: list[int] | None = None,
        dynamics_activation: str = "elu",
        dynamics_dt: float = 0.02,
        dynamics_output_scale: float = 0.01,
        # Cost map network dims
        cost_map_hidden_dims: list[int] | None = None,
        # Cost map Q bias initialization (per state dim, used to compute Q_base)
        cost_map_q_bias_init: list[float] | None = None,
        # Residual cost learning scales
        q_residual_scale: float = 50.0,
        r_residual_scale: float = 5.0,
        # MPC solver hyperparameters
        pgd_iters: int = 15,
        diff_gd_lr: float = 0.05,
        diff_gd_steps: int = 1,
        train_pgd_iters: int | None = None,
        refine_noise_std: float = 0.0,
        # Noise parameters for SAC exploration
        init_log_std: float = -1.0,
        # Error feedback
        use_error_feedback: bool = False,
        error_feedback_dropout: float = 0.0,
        # Ensemble dynamics
        dynamics_ensemble_size: int = 1,
        # Unused kwargs from config (absorb parent-style parameters)
        **kwargs: Any,
    ) -> None:
        # num_actions=2 for actual MPC output
        assert num_actions == 2, f"AC-MPC requires num_actions=2, got {num_actions}"
        super().__init__()

        if kwargs:
            # Absorb unused encoder/actor/critic config fields from parent-style configs
            logger.debug(
                "ActorCriticMPC: ignoring unused kwargs: %s",
                list(kwargs.keys()),
            )

        self.policy_obs_dim = policy_obs_dim
        self.mpc_horizon = mpc_horizon
        self.mpc_state_dim = mpc_state_dim

        # Extract obs key names from obs_groups
        policy_groups = obs_groups["policy"]
        self._policy_obs_key = policy_groups[0]
        self._mpc_state_key = "mpc_state"
        self._mpc_target_key = "mpc_target"

        # Sigmoid scaling bounds for Q and R
        self.register_buffer("_q_min", torch.tensor(q_min))
        self.register_buffer("_q_max", torch.tensor(q_max))
        self.register_buffer("_r_min", torch.tensor(r_min))
        self.register_buffer("_r_max", torch.tensor(r_max))

        # Cost Map Network
        # Input: policy_obs(13D)
        # Output: horizon*state_dim + control_dim = Q_running(H-1) + Q_terminal(1) + R
        # Note: H-1 running Q weights (for predicted states k=1..H-1) plus 1 terminal
        # Q weight (for predicted state k=H) = H groups total.
        cost_map_output = mpc_horizon * mpc_state_dim + num_actions
        cm_hidden = cost_map_hidden_dims if cost_map_hidden_dims is not None else [256, 128, 64]
        self.cost_map = MLP(policy_obs_dim, cost_map_output, cm_hidden, "elu")

        # Residual cost learning: Q = Q_base + tanh(raw) * residual_scale.
        # Q_base is a frozen buffer computed from bias init values using the old
        # sigmoid scaling formula. The network learns residual adjustments
        # around this physically motivated base cost, preventing drift to
        # meaningless cost structures during training.
        self._q_residual_scale = q_residual_scale
        self._r_residual_scale = r_residual_scale

        per_dim_q_bias = (
            cost_map_q_bias_init
            if cost_map_q_bias_init is not None
            else [
                2.0,   # phi (roll)
                2.0,   # theta (pitch)
                0.0,   # p (roll rate)
                0.0,   # q (pitch rate)
                0.0,   # q1 (joint pos)
                0.0,   # q2 (joint pos)
                -4.0,  # q1_dot (joint vel)
                -4.0,  # q2_dot (joint vel)
                -6.0,  # q1_target (near-zero cost)
                -6.0,  # q2_target (near-zero cost)
            ]
        )
        assert len(per_dim_q_bias) == mpc_state_dim, (
            f"cost_map_q_bias_init length ({len(per_dim_q_bias)}) != mpc_state_dim ({mpc_state_dim})"
        )

        # Frozen base Q: sigmoid(bias_init) scaled to [q_min, q_max] per dim.
        q_base_per_dim = q_min + torch.sigmoid(torch.tensor(per_dim_q_bias, dtype=torch.float32)) * (q_max - q_min)
        self.register_buffer("_q_base", q_base_per_dim)

        # Frozen base R: midpoint of [r_min, r_max] (sigmoid(0) = 0.5).
        r_base_val = r_min + 0.5 * (r_max - r_min)
        self.register_buffer("_r_base", torch.full((num_actions,), r_base_val))

        # Output layer bias = 0: tanh(0) = 0, so initial Q = Q_base exactly.
        cm_output_layer = self.cost_map[-1]
        assert isinstance(cm_output_layer, nn.Linear)
        cm_output_layer.bias.data.zero_()

        # Dynamics MLP
        # state_dim=8 (pred_net output: physical dims only)
        # full_state_dim=mpc_state_dim (10D: phys + q_target, for input)
        dyn_hidden = dynamics_hidden_dims if dynamics_hidden_dims is not None else [256, 128, 64]
        dyn_cfg = DynamicsMLPCfg(
            state_dim=8,
            full_state_dim=mpc_state_dim,
            control_dim=2,
            hidden_dims=dyn_hidden,
            activation=dynamics_activation,
            dt=dynamics_dt,
            output_scale=dynamics_output_scale,
            use_error_feedback=use_error_feedback,
            error_feedback_dropout=error_feedback_dropout,
            ensemble_size=dynamics_ensemble_size,
        )
        self.dynamics = DynamicsMLP(dyn_cfg)

        # Error feedback buffers (lazily initialized on first _run_mpc call)
        self._pred_error_buf: torch.Tensor | None = None
        self._last_predicted_next: torch.Tensor | None = None

        # MPC solver (created lazily on first solve)
        mpc_cfg_obj = (
            mpc_cfg
            if mpc_cfg is not None
            else DifferentiableMPCCfg(
                horizon=mpc_horizon,
                state_dim=mpc_state_dim,
                pgd_iters=pgd_iters,
                train_pgd_iters=train_pgd_iters,
                diff_gd_lr=diff_gd_lr,
                diff_gd_steps=diff_gd_steps,
                refine_noise_std=refine_noise_std,
            )
        )
        self._mpc_cfg = mpc_cfg_obj
        self._mpc_rollout: DifferentiableMPC | None = None
        self._mpc_train: DifferentiableMPC | None = None

        # Learnable log_std for SAC exploration noise
        self.log_std = nn.Parameter(torch.full((num_actions,), init_log_std))

        logger.info(
            "ActorCriticMPC: cost_map %dD->%dD, dynamics %s, "
            "Q [%.1f, %.1f] (residual +-%.1f), R [%.2f, %.2f] (residual +-%.1f), "
            "MPC pgd_iters=%d, diff_gd_lr=%.3f, diff_gd_steps=%d, error_feedback=%s, "
            "ensemble_size=%d",
            policy_obs_dim,
            cost_map_output,
            dyn_cfg.hidden_dims,
            q_min,
            q_max,
            q_residual_scale,
            r_min,
            r_max,
            r_residual_scale,
            mpc_cfg_obj.pgd_iters,
            mpc_cfg_obj.diff_gd_lr,
            mpc_cfg_obj.diff_gd_steps,
            dyn_cfg.use_error_feedback,
            dynamics_ensemble_size,
        )

    @property
    def mpc_state_key(self) -> str:
        """Obs dict key for MPC state vector."""
        return self._mpc_state_key

    @property
    def mpc_target_key(self) -> str:
        """Obs dict key for MPC target vector."""
        return self._mpc_target_key

    @property
    def policy_obs_key(self) -> str:
        """Obs dict key for policy observations."""
        return self._policy_obs_key

    def get_last_solve_cost(self) -> torch.Tensor | None:
        """Return per-env cost from the last rollout MPC solve, or None."""
        if self._mpc_rollout is not None and self._mpc_rollout.last_solve_cost is not None:
            return self._mpc_rollout.last_solve_cost
        return None

    def _ensure_mpc(self, num_envs: int, device: str, for_rollout: bool = False) -> DifferentiableMPC:
        """Lazily create MPC solver, maintaining separate rollout/train instances."""
        if for_rollout:
            if self._mpc_rollout is None or self._mpc_rollout.num_envs != num_envs:
                self._mpc_rollout = DifferentiableMPC(self._mpc_cfg, num_envs, device)
            return self._mpc_rollout
        else:
            if self._mpc_train is None or self._mpc_train.num_envs != num_envs:
                self._mpc_train = DifferentiableMPC(self._mpc_cfg, num_envs, device)
            return self._mpc_train

    def decode_cost_params(
        self,
        raw_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode raw cost map output into Q_diag, Q_terminal, and R_diag.

        Uses residual cost learning: Q = Q_base + tanh(raw) * scale, clamped
        to [q_min, q_max]. Q_base is a frozen physically-motivated starting
        point; the network learns bounded residual adjustments around it.

        Args:
            raw_logits: Cost map output. Shape: (N, horizon*state_dim + control_dim).

        Returns:
            Q_diag: Running cost weights. (N, horizon-1, state_dim).
            Q_terminal: Terminal cost weights. (N, state_dim).
            R_diag: Learned control cost weights. (N, control_dim).
        """
        H = self.mpc_horizon
        S = self.mpc_state_dim
        batch = raw_logits.shape[0]

        # Split: first H*S for Q (H-1 running + 1 terminal), remainder for R
        q_size = H * S
        raw_q = raw_logits[:, :q_size].view(batch, H, S)
        raw_r = raw_logits[:, q_size:]  # (batch, C)

        # Residual cost: base + bounded adjustment.
        # _q_base shape (S,) broadcasts to (N, H, S).
        q_residual = torch.tanh(raw_q) * self._q_residual_scale
        Q_all = (self._q_base + q_residual).clamp(self._q_min, self._q_max)
        Q_diag = Q_all[:, : H - 1, :]  # H-1 running cost weights: one per predicted state k=1..H-1
        Q_terminal = Q_all[:, H - 1, :]  # 1 terminal cost weight: for predicted state k=H

        r_residual = torch.tanh(raw_r) * self._r_residual_scale
        R_diag = (self._r_base + r_residual).clamp(self._r_min, self._r_max)
        return Q_diag, Q_terminal, R_diag

    def _run_mpc(
        self,
        obs: TensorDict,
        differentiable: bool = False,
        for_rollout: bool = False,
        pred_error_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run full AC-MPC pipeline: cost_map -> MPC.solve.

        Args:
            obs: Observation TensorDict with policy, mpc_state, mpc_target.
            differentiable: If True, return u with graph for SAC gradient.
            for_rollout: If True, use rollout MPC instance (warm-start preserved).
            pred_error_override: External prediction error to use instead of the
                live buffer. Required for off-policy training where batch size
                differs from num_envs. Shape: (batch, 8). None uses live buffer.

        Returns:
            MPC optimal control u*. Shape: (N, 2).
        """
        policy_obs = obs[self._policy_obs_key]

        # Cost map input: policy_obs only
        raw_logits = self.cost_map(policy_obs)
        Q_diag, Q_terminal, R_diag = self.decode_cost_params(raw_logits)

        # MPC state and target from env
        mpc_state = obs[self._mpc_state_key]
        mpc_target = obs[self._mpc_target_key]

        # Ensure MPC solver exists
        mpc = self._ensure_mpc(policy_obs.shape[0], policy_obs.device, for_rollout=for_rollout)

        # Prediction error: use override (off-policy training) or live buffer (rollout)
        pred_error = pred_error_override if pred_error_override is not None else self.get_pred_error()

        # Solve MPC
        u_star = mpc.solve(
            x_current=mpc_state,
            dynamics=self.dynamics,
            Q_diag=Q_diag,
            Q_terminal=Q_terminal,
            R_diag=R_diag,
            x_target=mpc_target,
            differentiable=differentiable,
            warm_start=for_rollout and not differentiable,
            pred_error=pred_error,
        )

        return u_star

    def _sample_from_u_star(
        self,
        u_star: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reparameterized sample + tanh squash + log_prob from pre-computed u*.

        Args:
            u_star: MPC output (pre-tanh mean). Shape: (N, 2).

        Returns:
            action: Squashed action in [-1, 1]. Shape: (N, 2).
            log_prob: Log probability. Shape: (N,).
        """
        log_std = self.log_std.clamp(_LOG_STD_MIN, _LOG_STD_MAX)
        std = log_std.exp()

        # Reparameterized sample: u = u* + std * eps
        noise = torch.randn_like(u_star)
        u_sample = u_star + std * noise  # pre-tanh value

        # Tanh squashing
        action = torch.tanh(u_sample)

        # Log probability: log N(noise; 0, 1) - sum log(std)
        log_prob = -0.5 * (_LOG2PI + 2 * log_std + noise**2)
        log_prob = log_prob.sum(dim=-1)
        # Tanh correction (numerically stable): -sum log(1 - tanh^2(u))
        log_prob -= (2 * (math.log(2) - u_sample - torch.nn.functional.softplus(-2 * u_sample))).sum(dim=-1)

        return action, log_prob

    def sample(
        self,
        obs: TensorDict,
        pred_error: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """SAC reparameterized sampling with differentiable MPC.

        Full graph: actor_loss -> action -> MPC.solve -> dynamics + cost_map.
        Use for SAC actor update (training batch, not rollout).

        Args:
            obs: Observation TensorDict.
            pred_error: Prediction error from replay buffer for off-policy training.
                None during rollout (uses live buffer).

        Returns:
            action: Squashed action in [-1, 1]. Shape: (N, 2).
            log_prob: Log probability. Shape: (N,).
        """
        u_star = self._run_mpc(obs, differentiable=True, for_rollout=False, pred_error_override=pred_error)
        return self._sample_from_u_star(u_star)

    def sample_no_diff(
        self,
        obs: TensorDict,
        for_rollout: bool = False,
        pred_error: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample without differentiable MPC (for critic target or rollout).

        Non-differentiable MPC solve + stochastic sample with log_prob.
        Avoids the expensive differentiable refinement step.

        Args:
            for_rollout: If True, use rollout MPC instance (warm-start preserved).
            pred_error: Prediction error from replay buffer for off-policy training.
                None during rollout (uses live buffer).

        Returns:
            action: Squashed action in [-1, 1]. Shape: (N, 2).
            log_prob: Log probability. Shape: (N,).
        """
        u_star = self._run_mpc(obs, differentiable=False, for_rollout=for_rollout, pred_error_override=pred_error)
        return self._sample_from_u_star(u_star)

    def act(self, obs: TensorDict, **_kwargs: Any) -> torch.Tensor:
        """Sample action (for rollout collection). Discards log_prob."""
        action, _ = self.sample_no_diff(obs, for_rollout=True)
        return action

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        """Deterministic action: MPC solve without noise."""
        u_star = self._run_mpc(obs, differentiable=False, for_rollout=True)
        return torch.tanh(u_star)

    def get_dynamics(self) -> DynamicsMLP:
        """Return the dynamics MLP sub-module."""
        return self.dynamics

    def reset_mpc(self, env_ids: torch.Tensor) -> None:
        """Reset MPC warm-start buffer for specified environments (rollout only)."""
        if self._mpc_rollout is not None:
            self._mpc_rollout.reset(env_ids)
        if self._pred_error_buf is not None:
            self._pred_error_buf[env_ids] = 0.0
        # NOTE: In the current runner call order (update_pred_error -> reset_mpc),
        # _last_predicted_next is always None here. Per-env zeroing is defensive
        # code that protects correctness if call order is ever reorganized.
        if self._last_predicted_next is not None:
            self._last_predicted_next[env_ids] = 0.0

    # -- Error feedback --

    def _ensure_pred_error_buf(self, num_envs: int, device: torch.device | str) -> None:
        """Lazily initialize prediction error buffer on first use."""
        if self._pred_error_buf is None or self._pred_error_buf.shape[0] != num_envs:
            self._pred_error_buf = torch.zeros(num_envs, self.dynamics.state_dim, device=device)
            self._last_predicted_next = None

    def store_prediction(self, mpc_state: torch.Tensor, action: torch.Tensor) -> None:
        """Store predicted next state for later error computation.

        Called AFTER act() produces an action, BEFORE env.step().
        Uses dynamics model to predict what next state should be, given
        the current state and the action about to be applied.
        """
        if not self.dynamics.use_error_feedback:
            return
        with torch.no_grad():
            self._ensure_pred_error_buf(mpc_state.shape[0], mpc_state.device)
            _EPS = 1e-6
            action_pretanh = torch.atanh(action.clamp(-1 + _EPS, 1 - _EPS)).clamp(-3.0, 3.0)
            self._last_predicted_next = self.dynamics(
                mpc_state, action_pretanh, pred_error=self._pred_error_buf,
            )

    def update_pred_error(self, actual_next_mpc_state: torch.Tensor) -> None:
        """Compute prediction error from stored prediction vs actual next state.

        Called AFTER env.step() returns next_obs. Updates the error buffer
        that will be used by the next MPC solve call.

        Error = predicted - actual (signed, 8D physical dims only).
        Uses in-place copy to preserve buffer identity (so reset_mpc
        always targets the same tensor regardless of call order).
        """
        if not self.dynamics.use_error_feedback or self._last_predicted_next is None:
            return
        with torch.no_grad():
            self._ensure_pred_error_buf(actual_next_mpc_state.shape[0], actual_next_mpc_state.device)
            self._pred_error_buf.copy_(
                self._last_predicted_next[:, :self.dynamics.state_dim]
                - actual_next_mpc_state[:, :self.dynamics.state_dim]
            )
        self._last_predicted_next = None

    def get_pred_error(self) -> torch.Tensor | None:
        """Return current prediction error buffer for MPC, or None if unused."""
        if not self.dynamics.use_error_feedback or self._pred_error_buf is None:
            return None
        return self._pred_error_buf
