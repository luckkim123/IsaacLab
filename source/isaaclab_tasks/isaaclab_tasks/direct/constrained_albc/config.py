# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration classes for ALBC environments.

ALBC (Active Linear Buoyancy Controller) uses 2 revolute joints (joint1, joint2)
to position a buoyancy element for attitude stabilization. No thrusters are used.

This module consolidates all environment configurations:
- DomainRandomizationCfg: DR parameter ranges
- ALBCEnvCfg: Base environment config (debug, no DR)
- ALBCTrainEnvCfg: Training config (DR + ocean current + payload)
- ALBCEncoderTrainEnvCfg: Encoder training with privileged info
- ConstrainedALBCEncoderEnvCfg: Constrained encoder training with IPO
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

from .doraemon import DoraemonCfg
from .mdp import (
    ALBCConstraintCfg,
    ALBCRewardCfg,
    ConstraintTermCfg,
    accumulated_rotation_cost,
    attitude_absolute_cost,
    joint_torque_cost,
    joint_velocity_limit_cost,
    overshoot_cost,
    yaw_velocity_cost,
)


@configclass
class DomainRandomizationCfg:
    """Configuration for domain randomization in ALBC environments.

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
    roll_range: tuple[float, float] = (-0.524, 0.524)  # +-30 deg
    pitch_range: tuple[float, float] = (-0.524, 0.524)  # +-30 deg
    yaw_range: tuple[float, float] = (-math.pi, math.pi)

    # -- Hydrodynamic Parameter Scales --
    added_mass_scale: tuple[float, float] = (0.85, 1.15)
    linear_damping_scale: tuple[float, float] = (0.5, 1.5)
    quadratic_damping_scale: tuple[float, float] = (0.5, 1.5)
    volume_scale: tuple[float, float] = (0.9, 1.1)

    # -- Center of Buoyancy Offset (meters) --
    cob_offset_x: tuple[float, float] = (-0.01, 0.01)
    cob_offset_y: tuple[float, float] = (-0.01, 0.01)
    cob_offset_z: tuple[float, float] = (-0.02, 0.02)

    # -- Center of Gravity Offset (meters) --
    cog_offset_x: tuple[float, float] = (-0.01, 0.01)
    cog_offset_y: tuple[float, float] = (-0.01, 0.01)
    cog_offset_z: tuple[float, float] = (-0.02, 0.02)

    # -- Inertia (max ratio M_true/M_hat ~1.7, reduced from ~2.1) --
    inertia_scale: tuple[float, float] = (0.75, 1.3)

    # -- Body Mass Scale (applied uniformly to all bodies) --
    body_mass_scale: tuple[float, float] = (0.9, 1.1)

    # -- Water Density (kg/m^3) --
    water_density_range: tuple[float, float] = (995.0, 1025.0)

    # -- Joint Actuator Gains (absolute values, Nm/rad and Nm*s/rad) --
    # Dynamixel XW540-T260-R: stall torque 9.5Nm, no-load 40rpm (4.19 rad/s)
    # Lower bound accounts for payload/seal friction reducing effective stiffness.
    # Upper bound = asset default (100.0 Kp saturates at ~5.4 deg error).
    # Unified across all envs (same physical motor regardless of control algorithm).
    joint_stiffness_range: tuple[float, float] = (40.0, 120.0)
    joint_damping_range: tuple[float, float] = (0.5, 5.0)

    # -- Yaw-specific quadratic damping scale (independent of general quad_damping) --
    yaw_damping_scale: tuple[float, float] = (0.5, 1.5)

    # -- Joint Effort Limit (scale applied to asset default 9.5 Nm) --
    # Upper bound 1.0: stall torque is the physical maximum, cannot exceed.
    # Lower bound 0.7: payload/friction/thermal derating reduces usable torque.
    joint_effort_limit_range: tuple[float, float] = (0.7, 1.0)

    # -- Joint Friction --
    joint_static_friction_range: tuple[float, float] = (0.0, 0.03)
    joint_viscous_friction_range: tuple[float, float] = (0.0, 0.2)

    @classmethod
    def fixed_pose(
        cls,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
        position: tuple[float, float, float] = (0.0, 0.0, 4.5),
    ) -> DomainRandomizationCfg:
        """Create a DR config with all randomization disabled except a fixed pose."""
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
            yaw_damping_scale=(1.0, 1.0),
            joint_stiffness_range=(100.0, 100.0),
            joint_damping_range=(3.0, 3.0),
            joint_effort_limit_range=(1.0, 1.0),
            joint_static_friction_range=(0.0, 0.0),
            joint_viscous_friction_range=(0.0, 0.0),
            cob_offset_x=(0.0, 0.0),
            cob_offset_y=(0.0, 0.0),
            cob_offset_z=(0.0, 0.0),
            cog_offset_x=(0.0, 0.0),
            cog_offset_y=(0.0, 0.0),
            cog_offset_z=(0.0, 0.0),
            enable_perturbation=False,
            enable_buoy_perturbation=False,
            action_latency_range=(0, 0),
            payload_cog_offset_xy_radius=0.0,
            payload_cog_offset_z=(0.0, 0.0),
            payload_mass_range=(0.5, 0.5),
        )

    @classmethod
    def half_strength(cls) -> DomainRandomizationCfg:
        """Create DR config at ~50% of full range for diagnostics.

        Narrows all scale/offset ranges to midpoint between 1.0 and full range.
        No DORAEMON -- constant DR from start. Use to isolate whether encoder/reward
        design can handle mild DR before enabling full strength.
        """
        return cls(
            enable=True,
            added_mass_scale=(0.9, 1.1),
            linear_damping_scale=(0.85, 1.15),
            quadratic_damping_scale=(0.8, 1.2),
            volume_scale=(0.92, 1.08),
            inertia_scale=(0.85, 1.25),
            body_mass_scale=(0.92, 1.08),
            cob_offset_z=(-0.01, 0.01),
            cog_offset_z=(-0.01, 0.01),
            joint_stiffness_range=(70.0, 110.0),
            joint_damping_range=(1.75, 4.0),
            joint_static_friction_range=(0.0, 0.025),
            joint_viscous_friction_range=(0.0, 0.15),
            water_density_range=(997.0, 1003.0),
            yaw_damping_scale=(0.8, 1.2),
            joint_effort_limit_range=(0.85, 1.0),
            payload_mass_range=(0.0, 0.5),
            payload_cog_offset_xy_radius=0.05,
            payload_cog_offset_z=(-0.015, 0.0),
            perturbation_force_range=(0.0, 2.5),
            perturbation_torque_range=(0.0, 0.2),
            action_latency_range=(0, 2),
        )

    # ==========================================================================
    # Random Perturbation (per-step external disturbance, Tan et al. 2018)
    # Periodically applies random wrench (force + torque) to the base body.
    # Models: tether tension variation, sudden current changes, contact forces.
    # ==========================================================================
    enable_perturbation: bool = True
    perturbation_force_range: tuple[float, float] = (0.0, 5.0)  # N (Hero Agent ~10kg -> 0.5 m/s^2 max)
    perturbation_torque_range: tuple[float, float] = (0.0, 0.4)  # Nm (~38% of restoring torque at 10deg)
    perturbation_interval: int = 100  # physics steps between events (~0.5s at 200Hz)
    perturbation_duration: int = 20  # physics steps active (~0.1s)

    # -- Buoy Perturbation (independent from main body perturbation) --
    # Buoy (~0.93kg) at arm tip is exposed to different turbulence than main body (~9.18kg).
    # Mass-proportional scaling: same acceleration (0.54 m/s^2) -> force = 0.93/9.18 * 5.0 ~ 0.5N.
    # Torque: 0.5N * 0.085m (buoy radius) ~ 0.05 Nm.
    # Shares perturbation_interval/duration timing parameters but uses independent phase timer.
    enable_buoy_perturbation: bool = False
    buoy_perturbation_force_range: tuple[float, float] = (0.0, 0.5)  # N
    buoy_perturbation_torque_range: tuple[float, float] = (0.0, 0.05)  # Nm

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
    payload_mass_range: tuple[float, float] = (0.0, 1.0)  # kg (up to ~10% body weight)

    # -- Payload CoG Offset (meters, relative to attachment point) --
    # XY sampled uniformly in disk of radius payload_cog_offset_xy_radius.
    payload_cog_offset_xy_radius: float = 0.10
    payload_cog_offset_z: tuple[float, float] = (-0.03, 0.0)

    # Physical constant: CoG-to-ABPC vertical offset for buoy moment calculation.
    # Used to limit payload CoG offset so max payload moment <= buoy restoring moment.
    buoy_moment_arm: float = 0.180  # m (matches TDCControllerCfg.h)


@configclass
class ALBCEnvCfg(DirectRLEnvCfg):
    """Configuration for ALBC environment.

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
    max_joint_velocity: float = 4.0 * math.pi / 3.0  # 40 RPM at 12V = 4*pi/3 rad/s
    control_decimation: int = 4  # target updates every 4th step = 0.02s (50Hz control)
    initial_joint_pos_range: tuple[float, float] = (-math.pi, math.pi)
    joint_init_mode: str = "random"  # "equilibrium" or "random"
    equilibrium_joint_noise: tuple[float, float] = (-0.3, 0.3)  # rad, noise around equilibrium

    # ==========================================================================
    # Attitude Task and Rewards
    # ==========================================================================
    # Target attitude [roll, pitch, yaw] in radians (default: upright)
    # Note: yaw is included for observation but EXCLUDED from reward calculation
    # because buoyancy control cannot generate Z-axis torque
    target_attitude: tuple[float, float, float] = (0.0, 0.0, 0.0)
    randomize_target_attitude: bool = False
    target_attitude_range: tuple[float, float, float] = (0.349, 0.349, 0.0)

    reward: ALBCRewardCfg = ALBCRewardCfg()

    # ==========================================================================
    # Initialization and Termination
    # ==========================================================================
    initial_height: float = 4.5
    max_angular_velocity: float = math.pi  # rad/s (~180 deg/s); terminate if roll/pitch rate exceeds this
    max_attitude_angle: float = math.pi / 2.0  # rad (~90 deg), prevents Lambda sign reversal

    # ==========================================================================
    # Domain Randomization
    # ==========================================================================
    randomization: DomainRandomizationCfg = DomainRandomizationCfg()
    doraemon: DoraemonCfg = DoraemonCfg(enable=False)

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

    # ==========================================================================
    # Proprioception History (for history-augmented encoder)
    # 0 = disabled (default), 30 = standard for history encoder.
    # When > 0, base_env creates a ring buffer and adds "proprio_hist" to observations.
    # ==========================================================================
    proprio_history_len: int = 0
    proprio_feature_dim: int = 8  # [roll, pitch, p, q, joint_pos_norm(2), prev_actions(2)]


@configclass
class ALBCTrainEnvCfg(ALBCEnvCfg):
    """ALBC training environment with domain randomization."""

    randomization = DomainRandomizationCfg(enable=True)

    # DORAEMON: adaptive DR distribution (replaces linear DR curriculum).
    doraemon: DoraemonCfg = DoraemonCfg()
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
            std=tuple([0.02] * 3 + [0.04] * 3 + [0.02] * 3 + [0.0] * 4),
        ),
        bias_noise_cfg=UniformNoiseCfg(
            n_min=tuple([-0.02] * 3 + [-0.03] * 3 + [-0.02] * 3 + [0] * 4),
            n_max=tuple([0.02] * 3 + [0.03] * 3 + [0.02] * 3 + [0] * 4),
        ),
    )


@configclass
class ALBCEncoderTrainEnvCfg(ALBCTrainEnvCfg):
    """ALBC encoder training with privileged hydrodynamic info.

    state_space=19 returns privileged information for HORA/RMA Phase 1 training.
    Main hydro (5D: volume, CoG_xyz, CoB_z) + Buoy hydro (5D)
    + Main inertia (2D: Ixx, Iyy) + Buoy inertia (2D: Ixx, Iyy)
    + Payload (4D: mass, cog_offset_xyz) + Main added mass surge (1D) = 19D.

    Removed from 20D: buoy added mass surge (1D, zero encoder sensitivity).

    Network Input Dimensions (ActorCriticEncoder):
        - observation_space (13): Used for gym.spaces.Box definition only
        - state_space (19): Privileged info, returned as observations["privileged"]
        - Encoder: privileged(19D) -> tanh -> latent z(13D) in [-1, 1]
        - Actual Actor/Critic input: policy_obs(13) + z(13) = 26D

    DORAEMON disabled: fixed uniform DR covers the full DomainRandomizationCfg range.
    """

    state_space: int = 19
    proprio_history_len: int = 30  # Enable history for all encoder configs
    doraemon: DoraemonCfg = DoraemonCfg(enable=False)


@configclass
class ConstrainedALBCEncoderEnvCfg(ALBCEncoderTrainEnvCfg):
    """Constrained encoder training with IPO (Interior-point Policy Optimization).

    Inherits from ALBCEncoderTrainEnvCfg (state_space=19 for privileged obs).
    Adds constraint configuration for the NORBC-style constrained RL pipeline.

    Constraint terms (joint_velocity, joint_oscillation) that were previously
    soft reward penalties are moved to explicit constraints. Their reward weights
    are zeroed to avoid double-counting.
    """

    constraints: ALBCConstraintCfg = ALBCConstraintCfg(
        terms=[
            # --- Binary constraints (4 terms) ---
            ConstraintTermCfg(
                func=accumulated_rotation_cost,
                params={"max_rotations": 2.0},
                budget=0.02,
                name="accum_rot",
            ),
            ConstraintTermCfg(
                func=attitude_absolute_cost,
                params={"limit": 1.396},
                budget=0.01,
                name="attitude_abs",
            ),
            # singularity: disabled -- DLS IK handles singularity smoothly
            # attitude_err: disabled -- quadratic command reward covers tracking
            ConstraintTermCfg(
                func=joint_torque_cost,
                budget=0.20,
                name="joint_torque",
            ),
            ConstraintTermCfg(
                func=joint_velocity_limit_cost,
                params={"limit_rad_per_s": 4.189},
                budget=0.05,
                name="joint_vel_limit",
            ),
            ConstraintTermCfg(
                func=overshoot_cost,
                params={"threshold": 0.035},
                budget=0.10,
                name="overshoot",
            ),
            # --- Continuous constraints (1 term) ---
            ConstraintTermCfg(
                func=yaw_velocity_cost,
                budget=0.35,
                cost_type="average",
                name="yaw_vel",
            ),
        ],
    )

    # Restore unified DR default (0.7-1.0x effort limit)
    randomization: DomainRandomizationCfg = DomainRandomizationCfg(
        enable=True,
    )

    # settling replaced by attitude_err constraint (adaptive lambda vs fixed weight).
    # smoothness replaces joint_osc constraint (fixed weight avoids lambda competition).
    # PBRS progress: strengthens reward signal against growing lambda pressure.
    reward: ALBCRewardCfg = ALBCRewardCfg(
        command_type="quadratic",
        settling_weight=0.0,
        energy_weight=0.0,
        smoothness_weight=-0.1,
        progress_weight=2.0,
    )
