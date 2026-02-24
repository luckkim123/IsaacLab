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
- HeroAgentAdaptBaseEnvCfg: Phase 2 adaptation (proprio history -> z_hat, base RL)

MPC configurations are in the separate hero_agent_mpc package.
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

from .controllers import TDCControllerCfg
from .mdp import ALBCRewardCfg


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
    added_mass_scale: tuple[float, float] = (0.8, 1.2)
    linear_damping_scale: tuple[float, float] = (0.7, 1.3)
    quadratic_damping_scale: tuple[float, float] = (0.6, 1.4)
    volume_scale: tuple[float, float] = (0.85, 1.15)

    # -- Center of Buoyancy Offset (meters) --
    cob_offset_x: tuple[float, float] = (-0.01, 0.01)
    cob_offset_y: tuple[float, float] = (-0.01, 0.01)
    cob_offset_z: tuple[float, float] = (-0.02, 0.02)

    # -- Center of Gravity Offset (meters) --
    cog_offset_x: tuple[float, float] = (-0.01, 0.01)
    cog_offset_y: tuple[float, float] = (-0.01, 0.01)
    cog_offset_z: tuple[float, float] = (-0.02, 0.02)

    # -- Inertia (moderate: max ratio M_true/M_hat ~2.1, near TDE stability boundary) --
    inertia_scale: tuple[float, float] = (0.7, 1.5)

    # -- Body Mass Scale (applied uniformly to all bodies) --
    body_mass_scale: tuple[float, float] = (0.85, 1.15)

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
        """Fix all randomization to exact values for controlled experiments.

        Delegates to ``fixed_pose()`` and copies all fields to self.
        """
        fixed = type(self).fixed_pose(roll, pitch, yaw, position)
        for field_name in self.__dataclass_fields__:
            setattr(self, field_name, getattr(fixed, field_name))

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
            enable_perturbation=False,
            action_latency_range=(0, 0),
            payload_cog_offset_xy_radius=0.0,
            payload_cog_offset_z=(0.0, 0.0),
            payload_mass_range=(0.5, 0.5),
        )

    # ==========================================================================
    # Random Perturbation (per-step external disturbance, Tan et al. 2018)
    # Periodically applies random wrench (force + torque) to the base body.
    # Models: tether tension variation, sudden current changes, contact forces.
    # ==========================================================================
    enable_perturbation: bool = True
    perturbation_force_range: tuple[float, float] = (0.0, 5.0)  # N (Hero Agent ~10kg -> 0.5 m/s^2 max)
    perturbation_torque_range: tuple[float, float] = (0.0, 0.75)  # Nm (5N x 0.15m half-body)
    perturbation_interval: int = 100  # physics steps between events (~0.5s at 200Hz)
    perturbation_duration: int = 20  # physics steps active (~0.1s)

    # ==========================================================================
    # Action Latency (delays RL action application by random physics steps)
    # Models: communication delay, computation latency in real hardware.
    # Sampled per-env at reset time, held constant during episode.
    # ==========================================================================
    action_latency_range: tuple[int, int] = (0, 4)  # physics steps (0-20ms at 200Hz)

    # ==========================================================================
    # Payload Randomization (only used when enable_payload=True)
    # Payload is attached to the gripper body (fixed to base via base_to_gripper joint).
    # Offsets are in gripper body frame.
    # ==========================================================================
    payload_mass_range: tuple[float, float] = (0.0, 1.5)  # kg (up to ~15% body weight)

    # -- Payload CoG Offset (meters, relative to attachment point) --
    # XY sampled uniformly in disk of radius payload_cog_offset_xy_radius.
    payload_cog_offset_xy_radius: float = 0.10
    payload_cog_offset_z: tuple[float, float] = (-0.03, 0.0)

    # Physical constant: CoG-to-ABPC vertical offset for buoy moment calculation.
    # Used to limit payload CoG offset so max payload moment <= buoy restoring moment.
    buoy_moment_arm: float = 0.180  # m (matches TDCControllerCfg.h)


@configclass
class DRCurriculumCfg:
    """Linearly ramp DR ranges from mild (start) to full (DomainRandomizationCfg values).

    The "full" values are whatever DomainRandomizationCfg has. Start values define the
    initial easy regime. Linear interpolation over end_iter training iterations.
    """

    enable: bool = False
    end_iter: int = 2000

    # Per-step perturbation start ranges (baseline-equivalent, ramp to full)
    perturbation_force_range_start: tuple[float, float] = (0.0, 1.5)
    perturbation_torque_range_start: tuple[float, float] = (0.0, 0.2)

    # Episode-level DR start ranges (baseline-equivalent)
    inertia_scale_start: tuple[float, float] = (0.85, 1.15)
    body_mass_scale_start: tuple[float, float] = (0.92, 1.08)
    volume_scale_start: tuple[float, float] = (0.92, 1.08)
    added_mass_scale_start: tuple[float, float] = (0.92, 1.08)
    payload_mass_range_start: tuple[float, float] = (0.0, 0.2)

    # High-impact stability parameters (start near nominal, ramp to full range)
    cog_offset_z_start: tuple[float, float] = (-0.008, 0.008)
    cob_offset_z_start: tuple[float, float] = (-0.004, 0.004)
    action_latency_range_start: tuple[int, int] = (0, 0)


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
    max_joint_velocity: float = math.pi
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
    target_attitude_range: tuple[float, float, float] = (0.5, 0.5, 0.0)

    reward: ALBCRewardCfg = ALBCRewardCfg()

    # ==========================================================================
    # Initialization and Termination
    # ==========================================================================
    initial_height: float = 4.5
    max_angular_velocity: float = 3.14159  # rad/s (~180 deg/s); terminate if roll/pitch rate exceeds this
    max_attitude_angle: float = 1.5708  # rad (~90 deg), prevents Lambda sign reversal

    # ==========================================================================
    # Domain Randomization
    # ==========================================================================
    randomization: DomainRandomizationCfg = DomainRandomizationCfg()
    dr_curriculum: DRCurriculumCfg = DRCurriculumCfg()

    # ==========================================================================
    # Virtual Payload Configuration (simple weight model)
    # Payload is applied to the gripper body (fixed to base). Offsets in gripper frame.
    # ==========================================================================
    enable_payload: bool = False
    payload_mass: float = 0.5  # kg
    payload_attachment_offset: tuple[float, float, float] = (0.0, 0.0, -0.05)  # m, gripper frame

    # ==========================================================================
    # TDE Observation (optional dynamics mismatch signal)
    # Computes H_hat = Lambda*p_EE + T_b - M_bar*nu_dot (2D roll/pitch).
    # Encodes all unmodeled dynamics (inertia error, coupling, damping, etc.)
    # without requiring a TDC controller or learned dynamics model.
    # When enabled, observation_space increases by 2 (appended to policy obs).
    # ==========================================================================
    enable_tde_obs: bool = False
    tde_m_hat: tuple[float, float] = (0.15, 0.16)  # nominal design inertia [roll, pitch]
    tde_nu_dot_ema_alpha: float = 0.05  # EMA filter for angular acceleration
    tde_h: float = 0.180  # CoG-to-ABPC vertical offset (m), same as TDCControllerCfg


@configclass
class HeroAgentTrainEnvCfg(HeroAgentEnvCfg):
    """Hero Agent ALBC training environment with domain randomization."""

    randomization = DomainRandomizationCfg(enable=True)

    # DR curriculum: baseline-equivalent start -> full DR over end_iter.
    # Single source of truth: reward sigma/weight curricula also follow end_iter.
    # Shared by all training envs (Base, Encoder, TDC, Adapt).
    dr_curriculum: DRCurriculumCfg = DRCurriculumCfg(enable=True)
    ocean_current = OceanCurrentCfg(
        max_velocity=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        noise_scale=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    enable_payload: bool = True
    randomize_target_attitude: bool = True

    # IMU sensor noise: bias (per-episode drift) + white noise (per-step)
    # Dims 0-2: euler angles (rad), 3-5: angular velocity (rad/s),
    # 6-8: attitude errors (same IMU source as euler -> same noise), 9-12: no noise
    # Values calibrated for general MEMS IMU (BIR Survey / Tan et al. reference).
    # Values stored as tuples for OmegaConf/Hydra compatibility; converted to tensors at env init.
    observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(
            mean=0.0,
            std=(0.02, 0.02, 0.02, 0.04, 0.04, 0.04, 0.02, 0.02, 0.02, 0.0, 0.0, 0.0, 0.0),
        ),
        bias_noise_cfg=UniformNoiseCfg(
            n_min=(-0.02, -0.02, -0.02, -0.03, -0.03, -0.03, -0.02, -0.02, -0.02, 0, 0, 0, 0),
            n_max=(0.02, 0.02, 0.02, 0.03, 0.03, 0.03, 0.02, 0.02, 0.02, 0, 0, 0, 0),
        ),
    )


@configclass
class HeroAgentEncoderTrainEnvCfg(HeroAgentTrainEnvCfg):
    """Hero Agent encoder training with privileged hydrodynamic info.

    state_space=26 returns privileged information for HORA/RMA Phase 1 training.
    Main hydro (7D) + Buoy hydro (7D) + Main dynamics (4D: Ixx,Iyy,Izz,body_mass)
    + Buoy dynamics (4D) + Payload (4D: mass, cog_offset_xyz) = 26D.
    Includes inertia and body mass for both bodies in privileged obs.

    Network Input Dimensions (ActorCriticEncoder):
        - observation_space (13): Used for gym.spaces.Box definition only
        - state_space (26): Privileged info, returned as observations["privileged"]
        - Encoder: privileged(26D) -> latent z(13D)
        - Actual Actor/Critic input: policy_obs(13) + z(13) = 26D

    DR curriculum is inherited from HeroAgentTrainEnvCfg (shared across all
    training envs). Sigmoid encoder activation prevents z collapse even with
    mild early DR.
    """

    state_space: int = 26


@configclass
class HeroAgentTDEBaseEnvCfg(HeroAgentTrainEnvCfg):
    """Hero Agent TDE-Base: base RL with TDE dynamics mismatch observation.

    Adds 2D H_hat observation (roll/pitch dynamics mismatch) to the standard
    13D policy obs, giving the policy direct access to unmodeled dynamics
    without requiring an encoder or adaptation module.

    H_hat = Lambda*p_EE + T_b - M_bar*nu_dot encodes: inertia error, coupling,
    Coriolis, damping, and external disturbances. Computable from onboard
    sensors (IMU + joint encoders) in both sim and real.

    observation_space = 15 (13D policy + 2D TDE, auto-adjusted by __init__).
    No privileged info (state_space=0): encoder-free pipeline.
    """

    # observation_space stays at 13 (inherited); __init__ auto-increments by 2
    # when enable_tde_obs=True, and also pads the noise config accordingly.
    enable_tde_obs: bool = True


# =============================================================================
# TDC (Time Delay Control) Configurations
# =============================================================================


def _tdc_randomization() -> DomainRandomizationCfg:
    """Create DomainRandomizationCfg with TDC-specific overrides.

    TDC envs need higher joint gains (centered at Kp=200, Kd=10) and no action
    latency (TDC overrides _pre_physics_step entirely).
    """
    return DomainRandomizationCfg(
        enable=True,
        joint_stiffness_range=(160.0, 240.0),
        joint_damping_range=(8.0, 12.0),
        action_latency_range=(0, 0),
    )


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

    # TDC-specific DR: higher joint gains (Kp=200, Kd=10), no action latency
    randomization: DomainRandomizationCfg = _tdc_randomization()


@configclass
class HeroAgentAdaptBaseEnvCfg(HeroAgentEncoderTrainEnvCfg):
    """Phase 2 adaptation training config (base RL pipeline).

    Inherits from HeroAgentEncoderTrainEnvCfg (state_space=26 for privileged z_gt).
    Adds proprioception history buffer for the adaptation module.

    Per-timestep feature (8D):
        [roll(1), pitch(1), p(1), q(1), joint_pos_norm(2), prev_actions(2)]

    The temporal conv learns dynamics from the command-response relationship:
    same action with different physical parameters produces different angular velocity.
    """

    proprio_history_len: int = 30
    proprio_feature_dim: int = 8  # body_state(2) + ang_vel(2) + joint_pos(2) + prev_actions(2)
