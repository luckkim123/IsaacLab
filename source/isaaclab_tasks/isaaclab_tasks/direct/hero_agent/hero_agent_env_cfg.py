# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration classes for Hero Agent ALBC environments.

Hero Agent uses ALBC (Active Linear Buoyancy Controller) for attitude control
via 2 revolute joints (joint1, joint2) that position a buoyancy element.
No thrusters are used.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.uuv import (
    HERO_AGENT_ALBC_JOINT_NAMES,
    HERO_AGENT_CFG,
    HeroAgentBuoyHydrodynamicsCfg,
    HeroAgentHydrodynamicsCfg,
    HydrodynamicsCfg,
    OceanCurrentCfg,
)

from .mdp import ALBCRewardCfg, TDCRewardCfg


@configclass
class DomainRandomizationCfg:
    """Configuration for domain randomization in Hero Agent ALBC environments.

    Note:
        Mass randomization has been removed because weight is now handled
        by PhysX (disable_gravity=False). To randomize mass, modify PhysX
        rigid body properties directly via the physics API.
    """

    enable: bool = False

    # -- Initial Position (meters) --
    position_x_range: tuple[float, float] = (-0.5, 0.5)
    position_y_range: tuple[float, float] = (-0.5, 0.5)
    position_z_range: tuple[float, float] = (4.0, 5.0)

    # -- Initial Orientation (radians) --
    roll_range: tuple[float, float] = (-0.785, 0.785)
    pitch_range: tuple[float, float] = (-0.785, 0.785)
    yaw_range: tuple[float, float] = (-math.pi, math.pi)

    # -- Hydrodynamic Parameter Scales --
    added_mass_scale: tuple[float, float] = (0.8, 1.2)
    linear_damping_scale: tuple[float, float] = (0.8, 1.2)
    quadratic_damping_scale: tuple[float, float] = (0.8, 1.2)
    volume_scale: tuple[float, float] = (0.95, 1.05)

    # -- Center of Buoyancy Offset (meters) --
    cob_offset_x: tuple[float, float] = (-0.01, 0.01)
    cob_offset_y: tuple[float, float] = (-0.01, 0.01)
    cob_offset_z: tuple[float, float] = (-0.02, 0.02)

    # -- Center of Gravity Offset (meters) --
    cog_offset_x: tuple[float, float] = (-0.01, 0.01)
    cog_offset_y: tuple[float, float] = (-0.01, 0.01)
    cog_offset_z: tuple[float, float] = (-0.02, 0.02)

    # -- Inertia --
    inertia_scale: tuple[float, float] = (0.9, 1.1)

    # ==========================================================================
    # Payload Randomization (only used when enable_payload=True)
    # Simple weight-based payload: mass + attachment offset
    # ==========================================================================
    payload_mass_range: tuple[float, float] = (0.3, 0.7)  # kg
    payload_attachment_x_range: tuple[float, float] = (-0.05, 0.05)  # m, offset from base
    payload_attachment_y_range: tuple[float, float] = (-0.05, 0.05)  # m, offset from base
    payload_attachment_z_range: tuple[float, float] = (-0.25, -0.15)  # m, absolute value


@configclass
class HeroAgentEnvCfg(DirectRLEnvCfg):
    """Configuration for Hero Agent ALBC environment.

    The vehicle uses 2 revolute joints (joint1, joint2) to position a buoyancy
    element for attitude stabilization. No thrusters are used.
    """

    # ==========================================================================
    # Environment Settings
    # ==========================================================================
    episode_length_s: float = 15.0
    decimation: int = 1
    action_space: int = 2
    observation_space: int = 13
    state_space: int = 0
    debug_vis: bool = True

    # Top-down camera view (looking down at robot from above)
    viewer: ViewerCfg = ViewerCfg(
        eye=(0.0, 0.0, 12.0),
        lookat=(0.0, 0.0, 4.5),
    )

    # ==========================================================================
    # Simulation
    # ==========================================================================
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=2,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.0,
        ),
        physx=PhysxCfg(
            enable_external_forces_every_iteration=True,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=2.0,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    # Terrain disabled for underwater environment (no ground collision needed)
    terrain: TerrainImporterCfg | None = None

    # ==========================================================================
    # Robot and Hydrodynamics
    # ==========================================================================
    robot = HERO_AGENT_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    hydrodynamics: HydrodynamicsCfg = HeroAgentHydrodynamicsCfg()
    buoy_hydrodynamics: HydrodynamicsCfg = HeroAgentBuoyHydrodynamicsCfg()
    ocean_current: OceanCurrentCfg = OceanCurrentCfg(
        max_velocity=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        noise_scale=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )

    # ==========================================================================
    # ALBC Joint Control
    # ==========================================================================
    albc_joint_names: list[str] = HERO_AGENT_ALBC_JOINT_NAMES
    max_joint_velocity: float = 2 * math.pi
    control_decimation: int = 1
    initial_joint_pos_range: tuple[float, float] = (-0.3, 0.3)

    # ==========================================================================
    # Attitude Task and Rewards
    # ==========================================================================
    # Target attitude [roll, pitch, yaw] in radians (default: upright)
    # Note: yaw is included for observation but EXCLUDED from reward calculation
    # because buoyancy control cannot generate Z-axis torque
    target_attitude: tuple[float, float, float] = (0.0, 0.0, 0.0)
    randomize_target_attitude: bool = False
    target_attitude_range: tuple[float, float, float] = (0.3, 0.3, 0.0)

    reward: ALBCRewardCfg = ALBCRewardCfg()

    # ==========================================================================
    # Initialization and Termination
    # ==========================================================================
    initial_height: float = 4.5
    min_height: float = -10.0
    max_height: float = 10.0
    max_distance_from_origin: float = 10.0

    # ==========================================================================
    # Domain Randomization
    # ==========================================================================
    randomization: DomainRandomizationCfg = DomainRandomizationCfg()

    # ==========================================================================
    # Virtual Payload Configuration (simple weight model)
    # ==========================================================================
    enable_payload: bool = False
    payload_mass: float = 0.5  # kg
    payload_attachment_offset: tuple[float, float, float] = (0.0, 0.0, -0.2)  # m, body frame


@configclass
class HeroAgentTrainEnvCfg(HeroAgentEnvCfg):
    """Hero Agent ALBC training environment with domain randomization."""

    randomization = DomainRandomizationCfg(enable=True)
    ocean_current = OceanCurrentCfg(
        max_velocity=(0.2, 0.2, 0.1, 0.0, 0.0, 0.0),
        noise_scale=(0.05, 0.05, 0.02, 0.0, 0.0, 0.0),
    )
    enable_payload: bool = True


@configclass
class HeroAgentEncoderTrainEnvCfg(HeroAgentTrainEnvCfg):
    """Hero Agent encoder training with privileged hydrodynamic info.

    state_space=22 returns compact privileged information for HORA/RMA Phase 1 training.
    Uses get_privileged_info_compact(): Main body (10D) + Buoy (10D) + Payload (2D) = 22D.
    Payload privileged info: mass (1D) + attachment_offset_z (1D).

    Network Input Dimensions (ActorCriticEncoder):
        - observation_space (13): Used for gym.spaces.Box definition only
        - state_space (22): Privileged info, returned as observations["privileged"]
        - Encoder: privileged(22D) -> latent z(6D)
        - Actual Actor/Critic input: policy_obs(13) + z(6) = 19D
    """

    state_space: int = 22
    enable_payload: bool = True


# =============================================================================
# TDC-Specific Configurations
# =============================================================================


@configclass
class HeroAgentEncoderTDCEnvCfg(HeroAgentEncoderTrainEnvCfg):
    """Hero Agent TDC environment configuration.

    Uses TDC controller for attitude control with learned PD gains.
    Actor outputs 4D gains instead of 2D joint velocities.

    Control Flow:
        1. Actor outputs gains: [K_p_roll, K_d_roll, K_p_pitch, K_d_pitch]
        2. Encoder outputs z (6D) -> M_hat = diag(z[:2])
        3. TDC controller: gains + M_hat + attitude_error -> p_EE_desired
        4. IK: p_EE_desired -> delta_joint_angles
        5. Joint position control: integrate delta to joint targets

    Network Input Dimensions (ActorCriticEncoderTDC):
        - observation_space (15): Used for gym.spaces.Box definition (11 base + 4 prev gains)
        - state_space (22): Privileged info for encoder
        - Encoder: privileged(22D) -> z(6D) -> M_hat(2D for roll/pitch)
        - Actor input: policy_obs(15) + z(6) = 21D -> gains(4D)
    """

    # Override action and observation space for TDC gains (4D instead of 2D)
    action_space: int = 4
    observation_space: int = 15  # 11 base + 4 previous gains (vs 13 = 11 + 2 in base env)

    # TDC-specific reward configuration
    reward: TDCRewardCfg = TDCRewardCfg()

    # ==========================================================================
    # TDC Controller Configuration
    # ==========================================================================
    # Gain bounds (RL actor output is scaled to these ranges)
    tdc_k_p_min: float = 1.0
    """Minimum proportional gain for TDC."""

    tdc_k_p_max: float = 50.0
    """Maximum proportional gain for TDC."""

    tdc_k_d_min: float = 0.1
    """Minimum derivative gain for TDC."""

    tdc_k_d_max: float = 10.0
    """Maximum derivative gain for TDC."""

    # TDE (Time Delay Estimation) parameters
    tdc_tde_delay_steps: int = 1
    """Number of steps for time delay estimation."""

    # Default inertia estimate
    tdc_default_m_hat: tuple[float, float] = (1.0, 1.0)
    """Default diagonal inertia estimate for roll and pitch."""

    # Note: ALBC arm geometry (link lengths, height offset, workspace) is sourced
    # from robot config constants (HERO_AGENT_ALBC_*) in isaaclab_assets.robots.uuv.
    # This ensures consistency with the URDF and eliminates duplicate hardcoded values.
