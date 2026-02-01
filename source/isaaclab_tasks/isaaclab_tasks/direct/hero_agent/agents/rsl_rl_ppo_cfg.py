# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO agent configurations for Hero Agent ALBC environments.

Hyperparameters are tuned for joint-based attitude control (no thrusters).
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

# Register ActorCriticEncoder in RSL-RL runner module namespace.
# The runner resolves policy class_name via eval() (on_policy_runner.py:416).
# This injection makes "ActorCriticEncoder" resolvable in that scope.
from isaaclab_tasks.models.actor_critic_encoder import ActorCriticEncoder

import rsl_rl.runners.on_policy_runner as _runner_module

_runner_module.ActorCriticEncoder = ActorCriticEncoder


@configclass
class HeroAgentPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO configuration for Hero Agent ALBC (Active Linear Buoyancy Controller).

    Optimized for 2-DOF joint control with potential-based rewards.
    Smaller network architecture for simpler action space.
    """

    num_steps_per_env = 32
    max_iterations = 500
    save_interval = 50
    experiment_name = "hero_agent_albc"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[64, 64],
        critic_hidden_dims=[64, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
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
class HeroAgentTrainPPORunnerCfg(HeroAgentPPORunnerCfg):
    """PPO configuration for Hero Agent ALBC training with domain randomization."""

    seed = -1  # Random seed for each run (ensures different domain randomization)
    max_iterations = 600
    # @configclass does not support partial override of nested configclass fields,
    # so the entire algorithm block must be redefined. The only difference from
    # the base HeroAgentPPORunnerCfg is entropy_coef: 0.008 vs 0.005.
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,  # Higher exploration for robustness
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
class HeroAgentEvalPPORunnerCfg(HeroAgentPPORunnerCfg):
    """PPO configuration for Hero Agent ALBC evaluation."""

    max_iterations = 100
    save_interval = 25
    experiment_name = "hero_agent_albc_eval"


@configclass
class RslRlPpoActorCriticEncoderCfg(RslRlPpoActorCriticCfg):
    """PPO actor-critic configuration with extrinsics encoder for HORA Phase 1."""

    class_name: str = "ActorCriticEncoder"

    # Encoder architecture
    encoder_hidden_dims: list[int] = [64, 32]
    encoder_latent_dim: int = 2
    encoder_activation: str = "elu"

    # Scaled sigmoid bounds for latent z.
    # z is used as TDC's M_hat = diag(z), so must be positive.
    # Range [0.1, 5.0] covers the domain randomization scale factors
    # (e.g., added_mass 0.8-1.2x, damping 0.8-1.2x, mass 0.9-1.1x).
    z_min: float = 0.1
    z_max: float = 5.0

    # Observation dimensions (must match environment)
    policy_obs_dim: int = 13
    privileged_dim: int = 64


@configclass
class HeroAgentEncoderTrainPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner configuration for Hero Agent encoder training (HORA Phase 1).

    Both actor and critic receive ["policy", "privileged"] as obs_groups.
    The ActorCriticEncoder internally routes them differently:
    - Actor: cat([policy_obs, z]) = 15D (compressed via encoder)
    - Critic: cat([policy_obs, z, privileged]) = 79D (full information)
    """

    seed = -1  # Random seed for each run (ensures different domain randomization)
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
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[64, 64],
        critic_hidden_dims=[64, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
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
class HeroAgentEncoderEvalPPORunnerCfg(HeroAgentEncoderTrainPPORunnerCfg):
    """PPO runner configuration for Hero Agent encoder evaluation.

    Inherits encoder architecture from training config but with reduced
    iterations for evaluation purposes.
    """

    max_iterations = 100
    save_interval = 25
    experiment_name = "hero_agent_albc_encoder"
