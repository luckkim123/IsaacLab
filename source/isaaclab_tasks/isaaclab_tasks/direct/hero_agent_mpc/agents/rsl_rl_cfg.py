# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL agent configuration for Hero Agent SAC-MPC environment.

Provides:
    - ActorCriticMPCCfg: Policy config (cost map + MPC + dynamics)
    - HeroAgentSACMPCRunnerCfg: SAC-MPC runner config (off-policy, MPC inside policy)
"""

import rsl_rl.runners.on_policy_runner as _runner_module

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from ..encoder.actor_critic_mpc import ActorCriticMPC
from ..runners.sac_mpc_runner import SACMPCRunner

# Register MPC classes in RSL-RL runner module namespace for dynamic resolution.
_runner_module.ActorCriticMPC = ActorCriticMPC
_runner_module.SACMPCRunner = SACMPCRunner


# =============================================================================
# Policy Configuration
# =============================================================================


@configclass
class ActorCriticMPCCfg(RslRlPpoActorCriticCfg):
    """Policy configuration for AC-MPC (SAC, not PPO).

    Uses ActorCriticMPC which:
        - Cost Map Network: policy_obs -> Q_diag
        - DynamicsMLP: x' = x + pred_net(x, u) * dt
        - TwinQNetwork (separate): critic with configurable obs keys
    """

    class_name: str = "ActorCriticMPC"

    policy_obs_dim: int = 13

    # MPC cost map parameters
    mpc_horizon: int = 5
    mpc_state_dim: int = 10
    q_min: float = 0.1
    q_max: float = 100.0
    r_min: float = 0.01
    r_max: float = 10.0

    # MPC solver parameters (forwarded to DifferentiableMPCCfg)
    pgd_iters: int = 8
    """PGD iterations for rollout (data collection). Full convergence for
    high-quality actions in the replay buffer."""
    train_pgd_iters: int | None = None
    """PGD iterations for training MPC (differentiable pass). None = use pgd_iters
    (no separation). Set to 4 after confirming training stability."""
    diff_gd_lr: float = 0.05
    """Differentiable refinement step size. Lowered from 0.08 to 0.05 to
    compensate for increased diff_gd_steps (2). Total displacement per solve:
    2*0.05=0.10 (vs previous 1*0.08=0.08). Dynamics pred_err stable at 0.019,
    so gradient amplification risk is low."""
    diff_gd_steps: int = 2
    """Number of differentiable GD refinement steps. Restored to 2 (was 1)
    to strengthen actor gradient signal through MPC Phase 2 chain.
    With 1 step, actor_grad_norm collapsed to 0.03 (cost_map barely learning).
    2 steps doubles the gradient path length through the cost landscape."""
    refine_noise_std: float = 0.0
    """Gaussian noise std added to converged u before differentiable refinement.
    REVERTED to 0.0: noise caused critic explosion (out-of-distribution actions
    from perturbed MPC solve triggered extrapolation error in Q-network)."""

    # Cost map network
    cost_map_hidden_dims: list[int] = [256, 128, 64]
    cost_map_q_bias_init: list[float] = [2.0, 2.0, 0.0, 0.0, 0.0, 0.0, -4.0, -4.0, -6.0, -6.0]
    """Per-state-dim bias init for cost map Q output layer (sigmoid domain).
    High bias -> high initial Q (strong tracking). Order matches mpc_state_dim=10:
    [phi, theta, p, q, q1, q2, q1_dot, q2_dot, q1_target, q2_target].
    q_target dims: -6.0 -> sigmoid(-6)~=0.0025 -> near-zero cost weight."""

    # Residual cost learning: Q = Q_base + tanh(raw) * scale.
    q_residual_scale: float = 50.0
    """Max residual adjustment for Q state costs."""
    r_residual_scale: float = 5.0
    """Max residual adjustment for R control costs."""

    # Dynamics MLP parameters (3-layer for complex nonlinear physics with wide DR)
    dynamics_hidden_dims: list[int] = [256, 128, 64]
    dynamics_activation: str = "elu"
    dynamics_dt: float = 0.02  # fallback; overridden by env.step_dt in SACMPCRunner
    dynamics_output_scale: float = 0.01
    dynamics_ensemble_size: int = 3
    """Number of ensemble members for dynamics pred_net.
    Ref: PETS (Chua et al., NeurIPS 2018), MBPO (Janner et al., NeurIPS 2019).
    3 for ensemble averaging + disagreement."""

    use_error_feedback: bool = True
    """Enable ECNN-style prediction error feedback for dynamics MLP.
    Feeds previous step's prediction error (predicted - actual, 8D) back
    into pred_net input, providing implicit domain adaptation signal."""

    error_feedback_dropout: float = 0.3
    """Probability of zeroing out pred_error input during training (per-env).
    Prevents base dynamics from over-relying on error feedback, improving
    multi-step prediction quality when error becomes stale in MPC rollout."""


# =============================================================================
# Runner Configuration
# =============================================================================


@configclass
class HeroAgentSACMPCRunnerCfg(RslRlOnPolicyRunnerCfg):
    """SAC-MPC runner configuration.

    Trains AC-MPC policy with SAC (off-policy):
        - Actor (AC-MPC): cost_map + MPC.solve(diff=True) + dynamics
        - Critic: Twin Q-networks with configurable obs keys

    Note: Inherits RslRlOnPolicyRunnerCfg despite being off-policy because
    Isaac Lab's gym.register and train.py resolve runner class_name through
    this base config's __init_subclass__ mechanism. The ``algorithm`` and
    ``num_steps_per_env`` fields are required by the base class but unused
    by SACMPCRunner.
    """

    class_name: str = "SACMPCRunner"

    seed = 42
    max_iterations = 10000
    save_interval = 500
    experiment_name = "hero_agent_sac_mpc"
    empirical_normalization = False

    # Required by RslRlOnPolicyRunnerCfg base class, unused by SACMPCRunner.
    num_steps_per_env = 1

    # Symmetric critic: actor and critic see identical observations.
    obs_groups = {
        "policy": ["policy", "mpc_state", "mpc_target"],
        "critic": ["policy", "mpc_state", "mpc_target"],
    }

    policy = ActorCriticMPCCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    # Required by RslRlOnPolicyRunnerCfg base class, unused by SACMPCRunner.
    algorithm = RslRlPpoAlgorithmCfg()

    # -- SAC-specific parameters --
    buffer_capacity: int = 1_000_000
    warmup_iters: int = 500
    updates_per_step: int = 16
    batch_size: int = 512
    log_interval: int = 10

    # SAC hyperparameters
    actor_lr: float = 3e-4
    """Same as critic_lr. Higher values (1e-3) cause gradient explosion through
    the differentiable MPC chain (Phase 2 refinement amplifies gradients)."""
    critic_lr: float = 3e-4
    alpha_lr: float = 1e-4
    """Slow alpha tuning to prevent premature entropy collapse."""
    dynamics_lr: float = 1e-3
    dynamics_loss_weight: float = 1.0
    multistep_horizon: int = 10
    """Number of micro-steps for multi-step dynamics loss unroll."""
    multistep_decay: float = 0.9
    multistep_eval_steps: list[int] | None = [5, 10]
    multistep_weight: float = 0.5
    gamma: float = 0.99
    tau: float = 0.002
    """Target network EMA rate. Kept at 0.002 (not SAC default 0.005).
    With 16 updates/step, effective per-step tau: 1-(1-0.002)^16 = 0.032.
    tau=0.005 caused critic loss 20x increase without improving tracking."""
    init_alpha: float = 0.2
    alpha_min: float = 0.05
    """Minimum alpha floor. OLD run (alpha_min=0.10) stabilized log_prob at
    -1.33. alpha_min=0.01 caused entropy collapse (log_prob -1.2 -> -0.6).
    0.05 prevents collapse while allowing more exploitation than 0.10."""
    target_entropy: float = -1.5
    """Slightly higher than -2.0 to maintain moderate exploration."""
    actor_delay: int = 2
    """Actor updates every 2 critic steps for stability. delay=1 caused
    premature actor updates before critic stabilizes."""
    max_q: float = 40.0
    """Bidirectional bound for TD target Q-values: clamp(-max_q, max_q).
    10.0 was too tight and caused asymmetric Q-value divergence (Q -> -22)."""
    critic_grad_clip: float = 2.0
    """Critic gradient norm clipping."""

    q_aggregation: str = "min"
    """Q-value aggregation for critic target and actor loss.
    Ref: FastSAC (Seo et al., arXiv:2512.01996, 2025).
    'min': standard SAC pessimistic estimate.
    'avg': (Q1 + Q2) / 2, less pessimistic, better exploration."""

    dynamics_diag_interval: int = 50

    dynamics_dim_weights: list[float] | None = None

    adaptive_dynamics_weights: bool = False
    """VaGraM-style adaptive per-dim dynamics loss weighting.
    Ref: Voelcker et al. (2022); Lambert et al. (L4DC 2020)."""

    adaptive_weights_warmup: int = 2000
    adaptive_weights_ema: float = 0.99
