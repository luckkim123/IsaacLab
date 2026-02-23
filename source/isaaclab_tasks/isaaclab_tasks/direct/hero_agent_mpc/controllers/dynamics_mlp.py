# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Learned residual dynamics model for differentiable MPC.

Hybrid dynamics model with learned physics + deterministic actuator update:

    Physical state (8D): x_{t+1} = x_t + pred_net(x_full, u) * dt
    Actuator state (2D): q_target_{t+1} = clamp(q_target_t + dt * max_vel * tanh(u))

where:
    x_full: [phys(8D), q_target(2D)] = 10D full state
    x_phys: [phi, theta, phi_dot, theta_dot, q1, q2, q1_dot, q2_dot] (8D)
    u: control [q1_dot_ref, q2_dot_ref] (2D)

The q_target update is deterministic (exactly mirrors env integration logic),
so it is hard-coded rather than learned. This avoids unnecessary learning burden
and eliminates multi-step compounding error for the trivial actuator dynamics.

The residual formulation with small-scale output initialization ensures the
model starts near-identity (x' ~ x), providing stable behavior before learning.

Reference:
    AC-MPC (Romero et al., 2024) uses analytic dynamics. We replace it with
    a learned MLP, enabling the dynamics to adapt to unknown parameters via
    error feedback.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn

from isaaclab.utils import configclass

logger = logging.getLogger(__name__)


@configclass
class DynamicsMLPCfg:
    """Configuration for the learned residual dynamics model.

    The model operates on a full state (full_state_dim=10) that includes
    both physical state (state_dim=8) and actuator state (q_target, 2D).
    The pred_net outputs only the physical state delta (8D), while the
    q_target update is deterministic (hard-coded in forward()).
    """

    state_dim: int = 8
    """Physical state dimension (pred_net output): [phi, theta, phi_dot, theta_dot,
    q1, q2, q1_dot, q2_dot]."""

    full_state_dim: int = 10
    """Full state dimension (MPC state): physical(8D) + q_target(2D).
    Used for pred_net input and input_scale construction."""

    control_dim: int = 2
    """Control dimension: [q1_dot_ref, q2_dot_ref]."""

    hidden_dims: list[int] = [128, 128]
    """Hidden layer dimensions for the prediction MLP."""

    activation: str = "elu"
    """Activation function for hidden layers."""

    dt: float = 0.02
    """Control timestep (50Hz = 0.02s). Scales the residual output."""

    output_scale: float = 0.01
    """Scale factor for output layer initialization. Small value ensures
    near-zero initial residuals (near-identity dynamics)."""

    state_scale: list[float] = [3.14, 3.14, 10.0, 10.0, 1.5, 1.5, 5.0, 5.0, 6.28, 6.28]
    """Per-dimension scale for full state input normalization (10D).
    Order: [phi(pi), theta(pi), p(10), q(10), q1(1.5), q2(1.5),
            q1d(5), q2d(5), q1_target(2pi), q2_target(2pi)]."""

    control_scale: list[float] = [3.0, 3.0]
    """Per-dimension scale for control input normalization.
    Order: [u1(3), u2(3)] -- pre-tanh bounds."""

    max_joint_velocity: float = math.pi
    """Maximum joint velocity (rad/s) for q_target integration.
    Mirrors env config: position_delta = dt * max_vel * tanh(u)."""

    joint_limit_lower: float = -6.28
    """Lower joint position limit (rad) for q_target clamping (~-2*pi)."""

    joint_limit_upper: float = 6.28
    """Upper joint position limit (rad) for q_target clamping (~+2*pi)."""

    use_error_feedback: bool = False
    """When True, pred_net input includes previous prediction error (8D).
    Enables ECNN-style error correction.
    References:
        - ECNN: Zimmermann et al., "Modeling Dynamical Systems by Error Correction
          Neural Networks"
        - CaDM: Lee et al., "Context-aware Dynamics Model for Generalization in
          Model-Based Reinforcement Learning" (ICML 2020)
        - TDE: Hsia & Gao, "Robot Manipulator Control Using Decentralized Linear
          Controller and Nonlinear Disturbance Observer" (1990)"""

    error_feedback_dropout: float = 0.0
    """Probability of zeroing out the entire pred_error input during training.
    Applied per-env (batch-level masking), only when self.training=True.
    Prevents base dynamics from over-relying on error feedback, improving
    multi-step prediction quality when error feedback becomes stale.
    Recommended: 0.3. Ref: Denoising autoencoder principle."""

    ensemble_size: int = 1
    """Number of independent pred_net members for ensemble dynamics.
    Ref: PETS (Chua et al., NeurIPS 2018), MBPO (Janner et al., NeurIPS 2019).
    1 = single network (default, no overhead). 3 for ensemble averaging.
    Ensemble reduces overfitting, provides uncertainty via disagreement."""


_ACTIVATIONS = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "selu": nn.SELU,
}


class DynamicsMLP(nn.Module):
    """Hybrid dynamics: learned physics residual + deterministic actuator update.

    Two components:
        pred_net:    cat(full_state, action[, pred_error]) -> phys delta (8D)
        q_target:    deterministic update (hard-coded, not learned)

    Physical (learned):  x_phys_{t+1} = x_phys_t + pred_net(x_full, u) * dt
    Actuator (exact):    q_target_{t+1} = clamp(q_target_t + dt * max_vel * tanh(u))

    All operations are batched for GPU-parallel environments: (num_envs, dim).
    """

    def __init__(self, cfg: DynamicsMLPCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.state_dim = cfg.state_dim
        self.full_state_dim = cfg.full_state_dim
        self.control_dim = cfg.control_dim
        self.dt = cfg.dt
        self._max_vel = cfg.max_joint_velocity
        self._joint_limit_lower = cfg.joint_limit_lower
        self._joint_limit_upper = cfg.joint_limit_upper
        self.use_error_feedback = cfg.use_error_feedback
        self._ef_dropout = cfg.error_feedback_dropout

        activation_cls = _ACTIVATIONS.get(cfg.activation, nn.ELU)

        # Prediction network(s): cat(full_state, action[, pred_error]) -> phys delta
        # Input: full_state(10D) + action(2D) [+ pred_error(8D)] = 12D+
        # Output: physical state delta (8D only, q_target is hard-coded)
        # Ensemble: multiple independent pred_nets (PETS/MBPO).
        error_feedback_dim = cfg.state_dim if cfg.use_error_feedback else 0
        input_dim = cfg.full_state_dim + cfg.control_dim + error_feedback_dim
        self._ensemble_size = cfg.ensemble_size
        self.pred_nets = nn.ModuleList()
        for _ in range(cfg.ensemble_size):
            layers: list[nn.Module] = []
            prev_dim = input_dim
            for hidden_dim in cfg.hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(activation_cls())
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, cfg.state_dim))  # 8D output
            net = nn.Sequential(*layers)
            # Small-scale output initialization for near-identity initial dynamics.
            # With output_scale=0.01, initial |delta| ~ 0.01 * normal ~ negligible.
            nn.init.uniform_(net[-1].weight, -cfg.output_scale, cfg.output_scale)
            nn.init.zeros_(net[-1].bias)
            self.pred_nets.append(net)

        # Per-dimension input scaling to normalize heterogeneous feature ranges.
        # Full state (10D) + control (2D) [+ pred_error(8D)]
        scale_parts = [torch.tensor(cfg.state_scale), torch.tensor(cfg.control_scale)]
        if cfg.use_error_feedback:
            # Reuse physical state scales for error normalization (same units)
            scale_parts.append(torch.tensor(cfg.state_scale[: cfg.state_dim]))
        input_scale = torch.cat(scale_parts)
        self.register_buffer("_input_scale", input_scale)

    @property
    def pred_net(self) -> nn.Sequential:
        """First ensemble member, for diagnostic code that accesses pred_net directly."""
        return self.pred_nets[0]

    def forward(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        pred_error: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict next state via hybrid dynamics.

        Physical state (8D): learned residual via pred_net.
        Actuator state (2D): deterministic q_target integration (hard-coded).

        Args:
            x: Current full state. Shape: (batch, full_state_dim=10).
                [phys(8D), q_target(2D)]
            u: Control input (pre-tanh). Shape: (batch, control_dim=2).
            pred_error: Previous step prediction error (predicted - actual).
                Shape: (batch, state_dim=8). None when use_error_feedback=False.
                Held constant during MPC horizon rollout (no future actuals).

        Returns:
            Predicted next full state x_{t+1}. Shape: (batch, full_state_dim=10).
        """
        # Error feedback dropout: zero-mask entire pred_error per-env during training.
        # Forces base dynamics to maintain prediction quality without error crutch.
        if pred_error is not None and self.training and self._ef_dropout > 0:
            mask = (torch.rand(pred_error.shape[0], 1, device=pred_error.device) > self._ef_dropout).float()
            pred_error = pred_error * mask

        # MLP input: full state(10D) + action(2D) [+ pred_error(8D)]
        parts = [x, u]
        if pred_error is not None:
            parts.append(pred_error)
        mlp_input = torch.cat(parts, dim=-1) / self._input_scale

        # Ensemble forward: single member fast path, ensemble mean for size > 1.
        if self._ensemble_size == 1:
            delta = self.pred_nets[0](mlp_input)  # (batch, 8)
        else:
            deltas = torch.stack([net(mlp_input) for net in self.pred_nets], dim=0)
            delta = deltas.mean(dim=0)  # (batch, 8)

        # Physical state update (learned residual)
        x_phys_next = x[:, :8] + delta * self.dt

        # Actuator state update (deterministic, mirrors env integration exactly)
        q_target_next = x[:, 8:10] + self.dt * self._max_vel * torch.tanh(u)
        q_target_next = q_target_next.clamp(self._joint_limit_lower, self._joint_limit_upper)

        return torch.cat([x_phys_next, q_target_next], dim=-1)

    def forward_member(
        self,
        member_idx: int,
        x: torch.Tensor,
        u: torch.Tensor,
        pred_error: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict next state using a single ensemble member (for per-member training loss).

        Same interface as forward() but uses only pred_nets[member_idx].
        """
        if pred_error is not None and self.training and self._ef_dropout > 0:
            mask = (torch.rand(pred_error.shape[0], 1, device=pred_error.device) > self._ef_dropout).float()
            pred_error = pred_error * mask

        parts = [x, u]
        if pred_error is not None:
            parts.append(pred_error)
        mlp_input = torch.cat(parts, dim=-1) / self._input_scale
        delta = self.pred_nets[member_idx](mlp_input)

        x_phys_next = x[:, :8] + delta * self.dt
        q_target_next = x[:, 8:10] + self.dt * self._max_vel * torch.tanh(u)
        q_target_next = q_target_next.clamp(self._joint_limit_lower, self._joint_limit_upper)
        return torch.cat([x_phys_next, q_target_next], dim=-1)

    def forward_with_disagreement(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        pred_error: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict next state with ensemble disagreement (uncertainty estimate).

        Returns:
            mean_next: Ensemble mean prediction. Shape: (batch, full_state_dim).
            disagreement: Per-dim std across ensemble members. Shape: (batch, state_dim=8).
                Only for physical state dims (q_target is deterministic).
        """
        parts = [x, u]
        if pred_error is not None:
            parts.append(pred_error)
        mlp_input = torch.cat(parts, dim=-1) / self._input_scale

        if self._ensemble_size == 1:
            delta = self.pred_nets[0](mlp_input)
            x_phys_next = x[:, :8] + delta * self.dt
            q_target_next = x[:, 8:10] + self.dt * self._max_vel * torch.tanh(u)
            q_target_next = q_target_next.clamp(self._joint_limit_lower, self._joint_limit_upper)
            mean_next = torch.cat([x_phys_next, q_target_next], dim=-1)
            disagreement = torch.zeros(x.shape[0], self.state_dim, device=x.device)
            return mean_next, disagreement

        deltas = torch.stack([net(mlp_input) for net in self.pred_nets], dim=0)  # (E, batch, 8)
        delta_mean = deltas.mean(dim=0)
        delta_std = deltas.std(dim=0)  # per-dim disagreement

        x_phys_next = x[:, :8] + delta_mean * self.dt
        q_target_next = x[:, 8:10] + self.dt * self._max_vel * torch.tanh(u)
        q_target_next = q_target_next.clamp(self._joint_limit_lower, self._joint_limit_upper)
        mean_next = torch.cat([x_phys_next, q_target_next], dim=-1)

        return mean_next, delta_std

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load state dict with backward-compatible migration.

        Handles two legacy formats:
        1. Old single ``pred_net`` (nn.Sequential) -> ``pred_nets.0.*``
        2. Old Teacher checkpoints with ``domain_head.*`` keys -> stripped with warning
        """
        # Strip domain_head keys from old Teacher checkpoints
        domain_keys = [k for k in state_dict if k.startswith("domain_head.")]
        if domain_keys:
            logger.warning(
                "Stripping %d domain_head.* keys from old Teacher checkpoint (domain_head removed in simplification).",
                len(domain_keys),
            )
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("domain_head.")}

        # Migrate single pred_net -> pred_nets.0
        has_old_keys = any(k.startswith("pred_net.") for k in state_dict)
        if has_old_keys and self._ensemble_size > 1:
            logger.warning(
                "Loading single-member checkpoint into ensemble_size=%d. "
                "Only member 0 will be loaded; others retain random init.",
                self._ensemble_size,
            )
        migrated = {}
        for key, val in state_dict.items():
            if key.startswith("pred_net."):
                migrated[key.replace("pred_net.", "pred_nets.0.", 1)] = val
            else:
                migrated[key] = val
        super().load_state_dict(migrated, strict=strict, assign=assign)
