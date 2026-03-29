# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL agent configurations for constrained ALBC environments.

Provides runner configurations for:
    - ConstrainedALBCEncoderRunnerCfg: TRPO + IPO with teacher encoder (production)
    - ALBCHardDRHistOnlyRunnerCfg: History-only PPO baseline (hard DR)
    - ALBCHardDRFrozenEncoderRunnerCfg: Frozen encoder fine-tuning (hard DR)
"""

import rsl_rl.runners.on_policy_runner as _runner_module

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

# Register custom classes in RSL-RL runner module namespace.
from ..algorithms import ConstraintTRPO
from ..encoder import (
    ActorCriticEncoder,
    ActorCriticEncoderConstrained,
)
from ..runners import ConstraintEncoderRunner

# Use ALBC-prefixed names to avoid collision with hero_agent registrations.
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
# Runner Configuration (Production: TRPO + IPO + Encoder)
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
# Hard DR: History-Only Baseline (PPO)
# =============================================================================


@configclass
class _HistOnlyPolicyCfg(RslRlPpoActorCriticCfg):
    """Standard actor-critic for history-only baseline (no encoder)."""

    class_name: str = "ActorCritic"
    init_noise_std: float = 1.0
    noise_std_type: str = "log"
    actor_obs_normalization: bool = True
    critic_obs_normalization: bool = True
    actor_hidden_dims: list[int] = [256, 128, 64]
    critic_hidden_dims: list[int] = [256, 128, 64]
    activation: str = "elu"


@configclass
class _PPOHistOnlyAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO for history-only training."""

    class_name: str = "PPO"
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 3e-4
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    entropy_coef: float = 0.0
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2


@configclass
class ALBCHardDRHistOnlyRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Hard DR baseline: history-only (no encoder).

    Same structure as Step 4d but with aggressive DR + ocean current.
    Target: ~10 deg attitude error. Establishes the performance gap
    that offline encoder should close.
    """

    seed = 30
    num_steps_per_env = 64
    max_iterations = 500
    save_interval = 50
    experiment_name = "constrained_albc_hard_dr_hist_only"
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy", "proprio_hist"],
        "critic": ["policy", "proprio_hist"],
    }

    algorithm = _PPOHistOnlyAlgorithmCfg()
    policy = _HistOnlyPolicyCfg()


# =============================================================================
# Hard DR: Frozen Encoder Fine-tuning
# =============================================================================

# 23D privileged obs bounds derived from DomainRandomizationCfg + HydrodynamicsCfg.
# Each pair is (lower, upper) with ~10% margin beyond actual DR range.
# Layout: hydro(6) + inertia(4) + damping(4) + body(2) + payload(4) + actuator(2) + env(1)
_PRIV_OBS_LOWER: list[float] = [
    # Hydrodynamics (6D): main [volume, CoG_z, CoB_z], buoy [volume, CoG_z, CoB_z]
    0.007,  # [0] main volume: nom=0.009, DR*0.9=0.0081
    -0.08,  # [1] main CoG_z: nom=-0.05, offset=(-0.02,0.02)
    -0.03,  # [2] main CoB_z: nom=0.0, offset=(-0.02,0.02)
    0.002,  # [3] buoy volume: nom=0.00268, DR*0.9=0.00241
    0.03,  # [4] buoy CoG_z: nom=0.059, offset=(-0.02,0.02)
    0.03,  # [5] buoy CoB_z: nom=0.059, offset=(-0.02,0.02)
    # Inertia (4D): main [Ixx, Iyy], buoy [Ixx, Iyy]
    0.06,  # [6] main Ixx: nom=0.0994, DR*0.75=0.0746
    0.06,  # [7] main Iyy: same
    0.0015,  # [8] buoy Ixx: nom=0.00278, DR*0.75=0.00209
    0.0015,  # [9] buoy Iyy: same
    # Damping (4D): linear [roll, pitch], quadratic [roll, pitch]
    0.10,  # [10] lin_damp roll: nom=0.3, DR*0.5=0.15
    0.10,  # [11] lin_damp pitch: same
    0.3,  # [12] quad_damp roll: nom=1.0, DR*0.5=0.5
    0.3,  # [13] quad_damp pitch: same
    # Body properties (2D): mass, added_mass_surge
    7.0,  # [14] body_mass: nom=9.18, DR*0.9=8.26
    5.0,  # [15] added_mass_surge: nom=8.0, DR*0.85=6.8
    # Payload (4D): mass, cog_offset [x, y, z]
    -0.1,  # [16] payload_mass: DR range (0.0, 1.0)
    -0.12,  # [17] payload_cog_x: disk r=0.10
    -0.12,  # [18] payload_cog_y: disk r=0.10
    -0.04,  # [19] payload_cog_z: DR range (-0.03, 0.0)
    # Actuator (2D): stiffness, damping
    30.0,  # [20] joint_stiffness: DR range (40, 120)
    0.3,  # [21] joint_damping: DR range (0.5, 5.0)
    # Environment (1D): water_density
    990.0,  # [22] water_density: DR range (995, 1025)
]

_PRIV_OBS_UPPER: list[float] = [
    # Hydrodynamics (6D)
    0.011,  # [0] main volume: DR*1.1=0.0099
    -0.02,  # [1] main CoG_z: nom=-0.05+0.02=-0.03
    0.03,  # [2] main CoB_z: nom=0.0+0.02=0.02
    0.0035,  # [3] buoy volume: DR*1.1=0.00295
    0.09,  # [4] buoy CoG_z: nom=0.059+0.02=0.079
    0.09,  # [5] buoy CoB_z: nom=0.059+0.02=0.079
    # Inertia (4D)
    0.15,  # [6] main Ixx: DR*1.3=0.1292
    0.15,  # [7] main Iyy: same
    0.004,  # [8] buoy Ixx: DR*1.3=0.00361
    0.004,  # [9] buoy Iyy: same
    # Damping (4D)
    0.50,  # [10] lin_damp roll: DR*1.5=0.45
    0.50,  # [11] lin_damp pitch: same
    1.8,  # [12] quad_damp roll: DR*1.5=1.5
    1.8,  # [13] quad_damp pitch: same
    # Body properties (2D)
    12.0,  # [14] body_mass: DR*1.1=10.10
    11.0,  # [15] added_mass_surge: DR*1.15=9.2
    # Payload (4D)
    1.2,  # [16] payload_mass: DR max=1.0
    0.12,  # [17] payload_cog_x: disk r=0.10
    0.12,  # [18] payload_cog_y: disk r=0.10
    0.01,  # [19] payload_cog_z: DR max=0.0
    # Actuator (2D)
    130.0,  # [20] joint_stiffness: DR max=120
    6.0,  # [21] joint_damping: DR max=5.0
    # Environment (1D)
    1030.0,  # [22] water_density: DR max=1025
]


@configclass
class _FrozenEncoderAlgorithmCfg(_PPOHistOnlyAlgorithmCfg):
    """PPO for frozen encoder -- uses standard update path (not encoder update).

    Encoder is frozen so _update_encoder_ppo() is unnecessary and harmful:
    it uses per-epoch LR adaptation (4x slower reaction) and weight_decay=0
    optimizer, causing noise_std explosion identical to online encoder.
    """

    use_encoder_update: bool = False


@configclass
class _FrozenEncoderPolicyCfg(_EncoderPolicyCfg):
    """Encoder policy with pre-trained frozen encoder.

    Actor input: cat([o_t_norm(14D), hist(240D), z_frozen(13D)]) = 267D.
    Encoder: frozen, loaded from offline checkpoint via pretrained_encoder_path.
    z-related actor weights initialized to near-zero for smooth integration.
    History: 30 steps stride 1, matching history-only baseline for warm-start.
    """

    class_name: str = "ALBCActorCriticFrozenEncoder"
    shared_backbone: bool = False
    proprio_hist_dim: int = 240  # 30 steps * 8 features (matches history-only)
    encoder_obs_normalization: bool = False  # static norm loaded from checkpoint
    pretrained_encoder_path: str = "logs/offline_encoder/encoder.pt"
    z_init_scale: float = 1.0


@configclass
class ALBCHardDRFrozenEncoderRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Hard DR + Frozen encoder fine-tuning with actor warm-start.

    Offline encoder pipeline Step 3:
    1. Encoder frozen, loaded from offline-trained checkpoint
    2. Actor warm-started from history-only checkpoint
    3. z-related weights near-zero -> gradual z integration
    4. Standard PPO training (actor already competent -> stable sigma)

    Set pretrained_encoder_path in policy config before training:
        cfg.policy.pretrained_encoder_path = "path/to/encoder.pt"
    """

    class_name: str = "ALBCConstraintEncoderRunner"
    seed = 30
    num_steps_per_env = 64
    max_iterations = 500
    save_interval = 50
    experiment_name = "constrained_albc_hard_dr_frozen_encoder"
    normalize_value: bool = True
    hist_only_checkpoint: str = ""
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy", "privileged", "proprio_hist"],
        "critic": ["policy", "privileged", "proprio_hist"],
    }

    algorithm = _FrozenEncoderAlgorithmCfg()
    policy = _FrozenEncoderPolicyCfg()
