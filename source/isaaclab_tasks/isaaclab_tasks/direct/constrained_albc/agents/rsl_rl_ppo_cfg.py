# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL agent configurations for constrained ALBC environments.

Provides runner configurations for TRPO + IPO constrained encoder training:
    - ConstrainedALBCEncoderRunnerCfg: TRPO + IPO with teacher encoder
"""

import rsl_rl.runners.on_policy_runner as _runner_module

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

# Register custom classes in RSL-RL runner module namespace.
from ..algorithms import ConstraintTRPO
from ..encoder import (
    ActorCriticConstrained,
    ActorCriticEncoder,
    ActorCriticEncoderConstrained,
)
from ..runners import ConstraintEncoderRunner

# Use ALBC-prefixed names to avoid collision with hero_agent registrations.
_runner_module.ALBCActorCriticConstrained = ActorCriticConstrained
_runner_module.ALBCActorCriticEncoder = ActorCriticEncoder
_runner_module.ALBCActorCriticEncoderConstrained = ActorCriticEncoderConstrained
_runner_module.ALBCConstraintEncoderRunner = ConstraintEncoderRunner
_runner_module.ALBCConstraintTRPO = ConstraintTRPO


# =============================================================================
# Policy Configurations
# =============================================================================


@configclass
class _EncoderPolicyCfg(RslRlPpoActorCriticCfg):
    """Shared network architecture for ALBC encoder policies (teacher)."""

    # Actor
    init_noise_std: float = 1.0
    noise_std_type: str = "log"
    actor_obs_normalization: bool = True
    critic_obs_normalization: bool = True
    actor_hidden_dims: list[int] = [256, 128, 64]
    critic_hidden_dims: list[int] = [512, 256, 128]
    activation: str = "elu"
    # Encoder (privileged only -> z)
    encoder_hidden_dims: list[int] = [256, 128, 64]
    encoder_latent_dim: int = 13
    encoder_activation: str = "elu"
    encoder_obs_normalization: bool = True
    # Observation dimensions
    policy_obs_dim: int = 14
    privileged_dim: int = 23


@configclass
class RslRlPpoActorCriticEncoderConstrainedCfg(_EncoderPolicyCfg):
    """Policy config for ActorCriticEncoderConstrained (teacher with cost critic)."""

    class_name: str = "ALBCActorCriticEncoderConstrained"
    num_constraints: int = 0  # Auto-synced from env config by ConstraintEncoderRunner
    cost_critic_hidden_dims: list[int] = [512, 256, 128, 64]


# =============================================================================
# Algorithm Configuration
# =============================================================================


@configclass
class RslRlConstraintTRPOAlgorithmCfg:
    """Algorithm configuration for TRPO + IPO (Interior-Point Optimization).

    Uses log-barrier interior-point method with adaptive thresholding for
    constraint enforcement, combined with TRPO natural gradient for policy update.
    """

    class_name: str = "ALBCConstraintTRPO"

    # TRPO parameters
    max_kl: float = 0.005
    cg_iters: int = 10
    cg_damping: float = 0.1
    line_search_max_backtracks: int = 10
    line_search_shrink_factor: float = 0.5

    # Value function
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    value_loss_coef: float = 1.0
    cost_value_loss_coef: float = 1.0
    value_lr: float = 1e-3
    max_grad_norm: float = 1.0

    # GAE
    gamma: float = 0.99
    lam: float = 0.95

    # Constraint
    num_constraints: int = 0  # Auto-synced from env config
    constraint_budgets: tuple[float, ...] = ()  # Auto-synced from env config
    cost_gamma: float = 0.99
    cost_lam: float = 0.95
    line_search_kl_margin: float = 1.5

    # Log barrier (IPO)
    barrier_t: float = 100.0
    barrier_alpha: float = 0.05

    # Noise floor
    min_std: float = 0.2


# =============================================================================
# Runner Configuration
# =============================================================================


@configclass
class ConstrainedALBCEncoderRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Runner configuration for TRPO + IPO constrained encoder training."""

    class_name: str = "ALBCConstraintEncoderRunner"
    seed = 30
    num_steps_per_env = 64
    max_iterations = 2500
    save_interval = 50
    experiment_name = "constrained_albc_encoder"
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy", "privileged"],
        "critic": ["policy", "privileged"],
    }

    algorithm = RslRlConstraintTRPOAlgorithmCfg()
    policy = RslRlPpoActorCriticEncoderConstrainedCfg()


# =============================================================================
# Debug Configuration (pure PPO, no encoder, no constraints)
# =============================================================================


@configclass
class _DebugPolicyCfg(RslRlPpoActorCriticCfg):
    """Standard actor-critic for debug (no encoder)."""

    class_name: str = "ActorCritic"
    init_noise_std: float = 1.0
    noise_std_type: str = "log"
    actor_obs_normalization: bool = True
    critic_obs_normalization: bool = True
    actor_hidden_dims: list[int] = [256, 128, 64]
    critic_hidden_dims: list[int] = [256, 128, 64]
    activation: str = "elu"


@configclass
class _DebugAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Standard PPO for debug."""

    class_name: str = "PPO"
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 3e-4
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    entropy_coef: float = 0.01
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2


@configclass
class ALBCDebugRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Debug runner: standard PPO, no encoder, no constraints."""

    seed = 30
    num_steps_per_env = 64
    max_iterations = 500
    save_interval = 50
    experiment_name = "constrained_albc_debug"

    algorithm = _DebugAlgorithmCfg()
    policy = _DebugPolicyCfg()


@configclass
class ALBCDebugDRRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Step 1: PPO + DR (no encoder, no constraints)."""

    seed = 30
    num_steps_per_env = 64
    max_iterations = 500
    save_interval = 50
    experiment_name = "constrained_albc_debug_dr"

    algorithm = _DebugAlgorithmCfg()
    policy = _DebugPolicyCfg()


@configclass
class ALBCDebugTRPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Step 2: TRPO + DR (no encoder, no barriers). Tests TRPO vs PPO."""

    class_name: str = "ALBCConstraintEncoderRunner"
    seed = 30
    num_steps_per_env = 64
    max_iterations = 500
    save_interval = 50
    experiment_name = "constrained_albc_debug_trpo"

    algorithm = RslRlConstraintTRPOAlgorithmCfg()
    policy = _DebugPolicyCfg()


@configclass
class _BarrierPolicyCfg(RslRlPpoActorCriticCfg):
    """ActorCritic + cost critic (no encoder) for barrier ablation."""

    class_name: str = "ALBCActorCriticConstrained"
    init_noise_std: float = 1.0
    noise_std_type: str = "log"
    actor_obs_normalization: bool = True
    critic_obs_normalization: bool = True
    actor_hidden_dims: list[int] = [256, 128, 64]
    critic_hidden_dims: list[int] = [256, 128, 64]
    activation: str = "elu"
    num_constraints: int = 0  # Auto-synced by runner


@configclass
class ALBCDebugBarrierRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Step 3: TRPO + IPO barrier + DR (no encoder). Tests barrier impact."""

    class_name: str = "ALBCConstraintEncoderRunner"
    seed = 30
    num_steps_per_env = 64
    max_iterations = 500
    save_interval = 50
    experiment_name = "constrained_albc_debug_barrier"

    algorithm = RslRlConstraintTRPOAlgorithmCfg()
    policy = _BarrierPolicyCfg()


@configclass
class ALBCDebugEncoderRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Step 4: TRPO + Encoder + DR (no constraints). Tests encoder impact."""

    class_name: str = "ALBCConstraintEncoderRunner"
    seed = 30
    num_steps_per_env = 64
    max_iterations = 500
    save_interval = 50
    experiment_name = "constrained_albc_debug_encoder"
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy", "privileged"],
        "critic": ["policy", "privileged"],
    }

    algorithm = RslRlConstraintTRPOAlgorithmCfg()
    policy = RslRlPpoActorCriticEncoderConstrainedCfg()


@configclass
class _PPOEncoderPolicyCfg(_EncoderPolicyCfg):
    """Encoder policy for PPO (no cost critic)."""

    class_name: str = "ALBCActorCriticEncoder"


@configclass
class ALBCDebugPPOEncoderRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Step 4b: PPO + Encoder + DR (no constraints). Tests PPO vs TRPO with encoder."""

    class_name: str = "ALBCConstraintEncoderRunner"
    seed = 30
    num_steps_per_env = 64
    max_iterations = 500
    save_interval = 50
    experiment_name = "constrained_albc_debug_ppo_encoder"
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy", "privileged"],
        "critic": ["policy", "privileged"],
    }

    algorithm = _DebugAlgorithmCfg()
    policy = _PPOEncoderPolicyCfg()


@configclass
class _PPOEncoderHistPolicyCfg(_EncoderPolicyCfg):
    """Encoder policy with proprio history for PPO.

    Actor input: cat([o_t(14D), hist_flat(240D), z(13D)]) = 267D.
    z/input ratio: 13/267 = 4.9% (was 48% without history).
    """

    class_name: str = "ALBCActorCriticEncoder"
    proprio_hist_dim: int = 240  # 30 steps * 8 features


@configclass
class ALBCDebugPPOEncoderHistRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Step 4c: PPO + Encoder + History + DR (no constraints).

    Adds proprio history to actor input to reduce z/input ratio from 48% to 5%.
    Matches NORBC paper architecture: Actor = cat([o_t(+history), z]).
    """

    class_name: str = "ALBCConstraintEncoderRunner"
    seed = 30
    num_steps_per_env = 64
    max_iterations = 500
    save_interval = 50
    experiment_name = "constrained_albc_debug_ppo_encoder_hist"
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy", "privileged", "proprio_hist"],
        "critic": ["policy", "privileged"],
    }

    algorithm = _DebugAlgorithmCfg()
    policy = _PPOEncoderHistPolicyCfg()
