# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO agent configurations for Hero Agent ALBC environments.

Provides three runner configurations:
    - HeroAgentPPORunnerCfg: Standard PPO for joint-based attitude control
    - HeroAgentEncoderPPORunnerCfg: HORA Phase 1 with extrinsics encoder
    - HeroAgentEncoderTDCPPORunnerCfg: TDC-integrated training with gains output

For evaluation, use CLI overrides instead of separate config classes:
    --max_iterations 100 --save_interval 25
"""

import rsl_rl.runners.on_policy_runner as _runner_module

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

# Register ActorCriticEncoder and ActorCriticEncoderTDC in RSL-RL runner module namespace.
# The runner resolves policy class_name dynamically (on_policy_runner.py:416).
# This injection makes custom network classes resolvable in that scope.
from ..encoder import ActorCriticEncoder, ActorCriticEncoderTDC

_runner_module.ActorCriticEncoder = ActorCriticEncoder
_runner_module.ActorCriticEncoderTDC = ActorCriticEncoderTDC


# =============================================================================
# Policy Configurations
# =============================================================================


@configclass
class RslRlPpoActorCriticEncoderCfg(RslRlPpoActorCriticCfg):
    """PPO actor-critic configuration with extrinsics encoder for HORA Phase 1.

    The encoder compresses privileged hydrodynamic parameters into a latent z
    that can later be replaced by a history-based adaptation module (Phase 2).
    """

    class_name: str = "ActorCriticEncoder"

    # Encoder architecture
    encoder_hidden_dims: list[int] = [64, 32]
    encoder_latent_dim: int = 6  # 6-DOF diagonal inertia matrix
    encoder_activation: str = "elu"

    # Softplus minimum: z = softplus(x) + z_min guarantees z > z_min
    z_min: float = 0.1

    # Observation dimensions (must match environment)
    policy_obs_dim: int = 13
    privileged_dim: int = 22  # Main(10D) + Buoy(10D) + Payload(2D) = 22D


@configclass
class RslRlPpoActorCriticEncoderTDCCfg(RslRlPpoActorCriticCfg):
    """PPO actor-critic configuration with TDC gains output.

    Similar to RslRlPpoActorCriticEncoderCfg but outputs 4D gains instead of
    2D joint velocities:
        - Actor output: [K_p_roll, K_d_roll, K_p_pitch, K_d_pitch]
        - Encoder latent z exposed as M_hat for TDC controller
    """

    class_name: str = "ActorCriticEncoderTDC"

    # Encoder architecture (same as base encoder)
    encoder_hidden_dims: list[int] = [64, 32]
    encoder_latent_dim: int = 6  # 6-DOF diagonal inertia matrix
    encoder_activation: str = "elu"

    # Softplus minimum: z = softplus(x) + z_min guarantees z > z_min
    z_min: float = 0.1

    # Observation dimensions (must match TDC environment: 15D = 11 base + 4 prev gains)
    policy_obs_dim: int = 15
    privileged_dim: int = 22  # Main(10D) + Buoy(10D) + Payload(2D) = 22D


# =============================================================================
# Runner Configurations
# =============================================================================


@configclass
class HeroAgentPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO configuration for Hero Agent ALBC (Active Linear Buoyancy Controller).

    Optimized for 2-DOF joint control with potential-based rewards.
    Uses standard ActorCritic without encoder.
    """

    seed = 42
    num_steps_per_env = 32
    max_iterations = 600
    save_interval = 50
    experiment_name = "hero_agent_albc"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[64, 64],
        critic_hidden_dims=[64, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,  # Disabled for dynamic systems (prevents std explosion)
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class HeroAgentEncoderPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO configuration for Hero Agent encoder training (HORA Phase 1).

    Uses ActorCriticEncoder that compresses privileged info into latent z:
        - Encoder: privileged (22D) -> softplus -> z (6D)
        - Actor: cat([policy_obs, z]) = 19D -> actions
        - Critic: cat([policy_obs, z]) = 19D -> value (symmetric, forces encoder learning)
    """

    seed = 42
    num_steps_per_env = 32
    max_iterations = 600
    save_interval = 50
    experiment_name = "hero_agent_albc_encoder"
    empirical_normalization = False

    obs_groups = {
        "policy": ["policy", "privileged"],
        "critic": ["policy", "privileged"],
    }

    policy = RslRlPpoActorCriticEncoderCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[64, 64],
        critic_hidden_dims=[64, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,  # Disabled for dynamic systems (prevents std explosion)
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class HeroAgentEncoderTDCPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO configuration for Hero Agent TDC training.

    Uses ActorCriticEncoderTDC that outputs PD gains for TDC controller:
        - Encoder: privileged (22D) -> softplus -> z (6D) -> M_hat
        - Actor: cat([policy_obs, z]) = 21D -> gains (4D)  (15D obs + 6D z)
        - TDC: gains + M_hat -> p_EE_desired -> IK -> joint targets
    """

    seed = 42
    num_steps_per_env = 32
    max_iterations = 600
    save_interval = 50
    experiment_name = "hero_agent_albc_tdc"
    empirical_normalization = False

    obs_groups = {
        "policy": ["policy", "privileged"],
        "critic": ["policy", "privileged"],
    }

    policy = RslRlPpoActorCriticEncoderTDCCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[64, 64],
        critic_hidden_dims=[64, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,  # Disabled for dynamic systems (prevents std explosion)
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class HeroAgentBaseTDCPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO configuration for Hero Agent TDC gain-only training (no encoder).

    Uses standard ActorCritic (no encoder, no privileged obs):
        - Actor: 15D obs -> 4D gains [K_p_roll, K_d_roll, K_p_pitch, K_d_pitch]
        - Critic: 15D obs -> value
        - M_hat fixed at default_m_hat=(0.14, 0.15)
    """

    seed = 42
    num_steps_per_env = 32
    max_iterations = 600
    save_interval = 50
    experiment_name = "hero_agent_base_tdc"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[64, 64],
        critic_hidden_dims=[64, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,  # Prevents entropy collapse / local minimum trapping
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
