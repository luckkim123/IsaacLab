# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration classes for Hero Agent ALBC environments.

Hero Agent uses ALBC (Active Linear Buoyancy Controller) for attitude control
via 2 revolute joints (joint1, joint2) that position a buoyancy element.
No thrusters are used.

This module consolidates all environment configurations:
- DomainRandomizationCfg: DR parameter ranges
- HeroAgentEnvCfg: Base environment config (debug, no DR)
- HeroAgentTrainEnvCfg: Training config (DR + ocean current + payload)
- HeroAgentEncoderTrainEnvCfg: Encoder training with privileged info
- HeroAgentTDCEnvCfg: Classical TDC control (no RL)
- HeroAgentEncoderTDCEnvCfg: Encoder-TDC integration (RL adaptive gains + M_hat)
- HeroAgentAdaptTDCEnvCfg: Phase 2 adaptation (proprio history -> z_hat)
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg, UniformNoiseCfg

from isaaclab_assets.robots.uuv import (
    HERO_AGENT_ALBC_JOINT_NAMES,
    HERO_AGENT_CFG,
    HeroAgentBuoyHydrodynamicsCfg,
    HeroAgentHydrodynamicsCfg,
    HydrodynamicsCfg,
    OceanCurrentCfg,
)

from .constraints.config import ConstrainedTrainingCfg, ConstraintCfg
from .controllers import TDCControllerCfg
from .mdp import ALBCRewardCfg, ConstrainedEncoderTDCRewardCfg, EncoderTDCRewardCfg


@configclass
class DomainRandomizationCfg:
    """Configuration for domain randomization in Hero Agent ALBC environments.

    Note:
        Body mass is randomized via PhysX ``set_masses()`` API (see ``body_mass_scale``).
        Weight (gravity) is handled by PhysX natively (disable_gravity=False).
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
    added_mass_scale: tuple[float, float] = (0.7, 1.3)
    linear_damping_scale: tuple[float, float] = (0.7, 1.3)
    quadratic_damping_scale: tuple[float, float] = (0.6, 1.4)
    volume_scale: tuple[float, float] = (0.9, 1.1)

    # -- Center of Buoyancy Offset (meters) --
    cob_offset_x: tuple[float, float] = (-0.01, 0.01)
    cob_offset_y: tuple[float, float] = (-0.01, 0.01)
    cob_offset_z: tuple[float, float] = (-0.04, 0.04)

    # -- Center of Gravity Offset (meters) --
    cog_offset_x: tuple[float, float] = (-0.01, 0.01)
    cog_offset_y: tuple[float, float] = (-0.01, 0.01)
    cog_offset_z: tuple[float, float] = (-0.04, 0.04)

    # -- Inertia --
    inertia_scale: tuple[float, float] = (0.8, 1.2)

    # -- Body Mass Scale (applied uniformly to all bodies) --
    body_mass_scale: tuple[float, float] = (0.9, 1.1)

    # -- Water Density (kg/m^3) --
    water_density_range: tuple[float, float] = (995.0, 1025.0)

    # -- Joint Actuator Gains (absolute values) --
    # Asset defaults: stiffness=100.0, damping=3.0 (ImplicitActuatorCfg)
    joint_stiffness_range: tuple[float, float] = (80.0, 120.0)
    joint_damping_range: tuple[float, float] = (2.4, 3.6)

    # -- Joint Friction --
    joint_static_friction_range: tuple[float, float] = (0.0, 0.05)
    joint_viscous_friction_range: tuple[float, float] = (0.0, 0.3)

    def disable_all(
        self,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
        position: tuple[float, float, float] = (0.0, 0.0, 4.5),
    ) -> None:
        """Fix all randomization to exact values for controlled experiments."""
        self.enable = True
        self.roll_range = (roll, roll)
        self.pitch_range = (pitch, pitch)
        self.yaw_range = (yaw, yaw)
        self.position_x_range = (position[0], position[0])
        self.position_y_range = (position[1], position[1])
        self.position_z_range = (position[2], position[2])
        # Fix all parameter scales to nominal
        for attr in (
            "added_mass_scale",
            "linear_damping_scale",
            "quadratic_damping_scale",
            "volume_scale",
            "inertia_scale",
            "body_mass_scale",
        ):
            setattr(self, attr, (1.0, 1.0))
        self.water_density_range = (998.0, 998.0)
        # Fix joint gains to asset defaults
        self.joint_stiffness_range = (100.0, 100.0)
        self.joint_damping_range = (3.0, 3.0)
        self.joint_static_friction_range = (0.0, 0.0)
        self.joint_viscous_friction_range = (0.0, 0.0)
        for attr in (
            "cob_offset_x",
            "cob_offset_y",
            "cob_offset_z",
            "cog_offset_x",
            "cog_offset_y",
            "cog_offset_z",
            "payload_cog_offset_x",
            "payload_cog_offset_y",
            "payload_cog_offset_z",
        ):
            setattr(self, attr, (0.0, 0.0))
        # Fix payload mass to nominal (matches HeroAgentEnvCfg.payload_mass default)
        self.payload_mass_range = (0.5, 0.5)

    @classmethod
    def fixed_pose(
        cls,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
        position: tuple[float, float, float] = (0.0, 0.0, 4.5),
    ) -> DomainRandomizationCfg:
        """Create a DR config with all randomization disabled except a fixed pose.

        Use this classmethod for declarative config construction (no mutable
        ``disable_all()`` needed).
        """
        return cls(
            enable=True,
            roll_range=(roll, roll),
            pitch_range=(pitch, pitch),
            yaw_range=(yaw, yaw),
            position_x_range=(position[0], position[0]),
            position_y_range=(position[1], position[1]),
            position_z_range=(position[2], position[2]),
            added_mass_scale=(1.0, 1.0),
            linear_damping_scale=(1.0, 1.0),
            quadratic_damping_scale=(1.0, 1.0),
            volume_scale=(1.0, 1.0),
            inertia_scale=(1.0, 1.0),
            body_mass_scale=(1.0, 1.0),
            water_density_range=(998.0, 998.0),
            joint_stiffness_range=(100.0, 100.0),
            joint_damping_range=(3.0, 3.0),
            joint_static_friction_range=(0.0, 0.0),
            joint_viscous_friction_range=(0.0, 0.0),
            cob_offset_x=(0.0, 0.0),
            cob_offset_y=(0.0, 0.0),
            cob_offset_z=(0.0, 0.0),
            cog_offset_x=(0.0, 0.0),
            cog_offset_y=(0.0, 0.0),
            cog_offset_z=(0.0, 0.0),
            payload_cog_offset_x=(0.0, 0.0),
            payload_cog_offset_y=(0.0, 0.0),
            payload_cog_offset_z=(0.0, 0.0),
            payload_mass_range=(0.5, 0.5),
        )

    # ==========================================================================
    # Payload Randomization (only used when enable_payload=True)
    # Payload is attached to the gripper body (fixed to base via base_to_gripper joint).
    # Offsets are in gripper body frame.
    # ==========================================================================
    payload_mass_range: tuple[float, float] = (0.0, 1.0)  # kg

    # -- Payload CoG Offset (meters, relative to attachment point) --
    payload_cog_offset_x: tuple[float, float] = (-0.50, 0.50)
    payload_cog_offset_y: tuple[float, float] = (-0.50, 0.50)
    payload_cog_offset_z: tuple[float, float] = (-0.20, 0.0)


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
    decimation: int = 1  # 0.005 * 1 = 0.005s step; 50Hz control via control_decimation=4
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
        dt=0.005,  # 200Hz sim (physics)
        render_interval=4,
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
    control_decimation: int = 4  # target updates every 4th step = 0.02s (50Hz control)
    initial_joint_pos_range: tuple[float, float] = (-math.pi, math.pi)

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
    max_angular_velocity: float = 3.14159  # rad/s (~180 deg/s); terminate if roll/pitch rate exceeds this

    # ==========================================================================
    # Domain Randomization
    # ==========================================================================
    randomization: DomainRandomizationCfg = DomainRandomizationCfg()

    # ==========================================================================
    # Virtual Payload Configuration (simple weight model)
    # Payload is applied to the gripper body (fixed to base). Offsets in gripper frame.
    # ==========================================================================
    enable_payload: bool = False
    payload_mass: float = 0.5  # kg
    payload_attachment_offset: tuple[float, float, float] = (0.0, 0.0, -0.05)  # m, gripper frame


@configclass
class HeroAgentTrainEnvCfg(HeroAgentEnvCfg):
    """Hero Agent ALBC training environment with domain randomization."""

    randomization = DomainRandomizationCfg(enable=True)
    ocean_current = OceanCurrentCfg(
        max_velocity=(0.2, 0.2, 0.1, 0.0, 0.0, 0.0),
        noise_scale=(0.05, 0.05, 0.02, 0.0, 0.0, 0.0),
    )
    enable_payload: bool = True
    randomize_target_attitude: bool = True

    # IMU sensor noise: bias (per-episode drift) + white noise (per-step)
    # Dims 0-2: euler angles (rad), 3-5: angular velocity (rad/s),
    # 6-8: attitude errors (same IMU source as euler -> same noise), 9-12: no noise
    # Values stored as tuples for OmegaConf/Hydra compatibility; converted to tensors at env init.
    observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(
            mean=0.0,
            std=(0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0),
        ),
        bias_noise_cfg=UniformNoiseCfg(
            n_min=(-0.005, -0.005, -0.005, -0.01, -0.01, -0.01, -0.005, -0.005, -0.005, 0, 0, 0, 0),
            n_max=(0.005, 0.005, 0.005, 0.01, 0.01, 0.01, 0.005, 0.005, 0.005, 0, 0, 0, 0),
        ),
    )


@configclass
class HeroAgentEncoderTrainEnvCfg(HeroAgentTrainEnvCfg):
    """Hero Agent encoder training with privileged hydrodynamic info.

    state_space=24 returns compact privileged information for HORA/RMA Phase 1 training.
    Main body (10D) + Buoy (10D) + Payload (4D) = 24D.
    Payload privileged info: mass (1D) + cog_offset (3D).

    Network Input Dimensions (ActorCriticEncoder):
        - observation_space (13): Used for gym.spaces.Box definition only
        - state_space (24): Privileged info, returned as observations["privileged"]
        - Encoder: privileged(24D) -> latent z(6D)
        - Actual Actor/Critic input: policy_obs(13) + z(6) = 19D
    """

    state_space: int = 24


# =============================================================================
# TDC (Time Delay Control) Configurations
# =============================================================================


@configclass
class HeroAgentTDCEnvCfg(HeroAgentTrainEnvCfg):
    """Hero Agent TDC (Time Delay Control) environment configuration.

    Uses classical TDC controller instead of RL for attitude stabilization.
    Inherits DR, ocean current, and payload from HeroAgentTrainEnvCfg.

    Control timing (matching C++ reference):
        - decimation=1: step_dt = physics_dt = 0.005s (200Hz step)
        - control_decimation=4: TDC runs every 4th step = 0.02s (50Hz)

    Joint gains are centered at TDC-optimal values (Kp=200, Kd=10) with
    +/-20% randomization, unlike the base RL config (Kp=100, Kd=3).
    """

    tdc: TDCControllerCfg = TDCControllerCfg()

    # TDC runs at 50Hz (every 4th physics step, matching C++ reference)
    control_decimation: int = 4  # TDC dt = 0.005 * 4 = 0.02s

    # No privileged obs for pure TDC (classical control, no encoder)
    state_space: int = 0

    # Override joint gain ranges for TDC (centered at Kp=200, Kd=10)
    randomization: DomainRandomizationCfg = DomainRandomizationCfg(
        enable=True,
        joint_stiffness_range=(160.0, 240.0),
        joint_damping_range=(8.0, 12.0),
    )


@configclass
class HeroAgentEncoderTDCEnvCfg(HeroAgentTrainEnvCfg):
    """Encoder-TDC integration: RL learns adaptive gains + M_hat for TDC.

    The RL actor outputs 4D actions [Kp_roll, Kp_pitch, Kd_roll, Kd_pitch],
    which are converted to TDC gains via sigmoid scaling. The encoder latent
    z[3:5] provides adaptive M_hat for the TDC controller.

    Inherits DR, ocean current, and payload from HeroAgentTrainEnvCfg.
    Joint gains centered at TDC-optimal values (Kp=200, Kd=10).
    """

    tdc: TDCControllerCfg = TDCControllerCfg(log_interval=0)

    # TDC timing: control_decimation=4 (50Hz TDC)
    control_decimation: int = 4

    state_space: int = 24  # privileged obs for encoder
    action_space: int = 4  # Kp_roll, Kp_pitch, Kd_roll, Kd_pitch
    observation_space: int = 13  # same policy obs

    # Gain bounds (after sigmoid scaling in env)
    kp_range: tuple[float, float] = (10.0, 100.0)
    kd_range: tuple[float, float] = (2.0, 30.0)

    # Encoder-TDC specific reward config (adjusted weights + TDE residual)
    reward: EncoderTDCRewardCfg = EncoderTDCRewardCfg()

    # Override joint gain ranges for TDC (centered at Kp=200, Kd=10)
    randomization: DomainRandomizationCfg = DomainRandomizationCfg(
        enable=True,
        joint_stiffness_range=(160.0, 240.0),
        joint_damping_range=(8.0, 12.0),
    )


@configclass
class HeroAgentConstrainedEncoderTDCEnvCfg(HeroAgentEncoderTDCEnvCfg):
    """Encoder-TDC with NORBC constrained RL training.

    Extends HeroAgentEncoderTDCEnvCfg with IPO constraint definitions.
    The ConstrainedTrainingCfg contains 5 constraints:
        1. workspace: EE position within reachable radius (probabilistic, D=0.001)
        2. control_smoothness: EE position change (average, D=0.002)
        3. inertia_rate: Encoder z change rate (average, D=0.001)
        4. angular_velocity: Body angular velocity limit (probabilistic, D=0.01)
        5. joint_velocity: Joint velocity sim-to-real limit (probabilistic, D=0.5)

    D_k values are chosen so that discounted limits d_k = D_k/(1-gamma)
    are comparable to observed cost returns, ensuring log-barrier gradient
    is strong enough to influence the policy.
    """

    # Safety penalties disabled -- handled by IPO constraints instead
    reward: ConstrainedEncoderTDCRewardCfg = ConstrainedEncoderTDCRewardCfg()

    constrained_training: ConstrainedTrainingCfg = ConstrainedTrainingCfg(
        barrier_t=10.0,
        adaptive_alpha=0.5,
        cost_gamma=0.99,
        cost_value_coef=0.5,
        constraints=[
            ConstraintCfg(
                name="workspace",
                constraint_type="probabilistic",
                limit_D=0.001,
                params={"r_max": 0.466},
            ),
            ConstraintCfg(
                name="control_smoothness",
                constraint_type="average",
                limit_D=0.002,
            ),
            ConstraintCfg(
                name="inertia_rate",
                constraint_type="average",
                limit_D=0.001,
            ),
            ConstraintCfg(
                name="angular_velocity",
                constraint_type="probabilistic",
                limit_D=0.01,
                params={"omega_max": 1.5},
            ),
            ConstraintCfg(
                name="joint_velocity",
                constraint_type="probabilistic",
                limit_D=0.5,
                params={"vel_max": 1.0},
            ),
        ],
    )


@configclass
class HeroAgentAdaptTDCEnvCfg(HeroAgentEncoderTDCEnvCfg):
    """Phase 2 adaptation training config.

    Adds proprioception history buffer for the adaptation module.
    Per-timestep feature (12D):
        [roll(1), pitch(1), p(1), q(1), joint_pos_norm(2), joint_vel(2), actions(4)]
    """

    proprio_history_len: int = 30
    proprio_feature_dim: int = 12  # body(4) + joint_pos(2) + joint_vel(2) + actions(4)
