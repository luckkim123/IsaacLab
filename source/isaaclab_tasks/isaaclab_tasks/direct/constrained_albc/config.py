# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Constrained ALBC environment.

ALBC (Active Linear Buoyancy Controller) uses 2 revolute joints (joint1, joint2)
to position a buoyancy element for attitude stabilization. No thrusters are used.

Single registered task: Isaac-Constrained-ALBC-Encoder-v0
    C-TRPO + encoder constrained RL with 6 constraint terms.
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
    effort_limit_cost,
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
    joint_stiffness_range: tuple[float, float] = (40.0, 120.0)
    joint_damping_range: tuple[float, float] = (0.5, 5.0)

    # -- Yaw-specific quadratic damping scale (independent of general quad_damping) --
    yaw_damping_scale: tuple[float, float] = (0.5, 1.5)

    # -- Joint Effort Limit (scale applied to asset default 9.5 Nm) --
    joint_effort_limit_range: tuple[float, float] = (0.7, 1.0)

    # -- Joint Friction --
    joint_static_friction_range: tuple[float, float] = (0.0, 0.03)
    joint_viscous_friction_range: tuple[float, float] = (0.0, 0.2)

    # ==========================================================================
    # Random Perturbation (per-step external disturbance, Tan et al. 2018)
    # Periodically applies random wrench (force + torque) to the base body.
    # ==========================================================================
    enable_perturbation: bool = True
    perturbation_force_range: tuple[float, float] = (0.0, 5.0)  # N
    perturbation_torque_range: tuple[float, float] = (0.0, 0.4)  # Nm
    perturbation_interval: int = 100  # physics steps between events (~0.5s at 200Hz)
    perturbation_duration: int = 20  # physics steps active (~0.1s)

    # ==========================================================================
    # Action Latency (delays RL action application by random physics steps)
    # ==========================================================================
    action_latency_range: tuple[int, int] = (0, 4)  # physics steps (0-20ms at 200Hz)

    # ==========================================================================
    # Payload Randomization
    # Payload is attached to the gripper body (fixed to base via base_to_gripper joint).
    # ==========================================================================
    payload_mass_range: tuple[float, float] = (0.0, 1.0)  # kg (up to ~10% body weight)

    # -- Payload CoG Offset (meters, relative to attachment point) --
    payload_cog_offset_xy_radius: float = 0.10
    payload_cog_offset_z: tuple[float, float] = (-0.03, 0.0)

    # Physical constant: CoG-to-ABPC vertical offset for buoy moment calculation.
    buoy_moment_arm: float = 0.180  # m (matches TDCControllerCfg.h)


@configclass
class ALBCEnvCfg(DirectRLEnvCfg):
    """Base configuration for ALBC environment.

    Used directly by ALBCEnv. Inheritable for specialized configs.

    The vehicle uses 2 revolute joints (joint1, joint2) to position a buoyancy
    element for attitude stabilization. No thrusters are used.

    Network Input Dimensions (ActorCriticEncoder):
        - observation_space (13): Used for gym.spaces.Box definition only
        - state_space (27): Privileged info, returned as observations["privileged"]
        - Encoder: cat([policy_obs(13), hist(240), privileged(27)]) = 280D -> softsign -> z(13D)
        - Actor input: policy_obs(13) + hist(240) + z(13) = 266D
        - Critic input: policy_obs(13) + hist(240) + privileged(27) = 280D
    """

    # ==========================================================================
    # Environment Settings
    # ==========================================================================
    episode_length_s: float = 15.0
    decimation: int = 1  # 0.005 * 1 = 0.005s step; 50Hz control via control_decimation=4
    action_space: int = 2
    observation_space: int = 13
    state_space: int = 27  # 27D privileged obs for encoder (+4: latency, friction x2, density)
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
    max_joint_velocity: float = 2.0 * math.pi  # rad/s, matches PhysX velocity_limit_sim=6.28
    control_decimation: int = 4  # target updates every 4th step = 0.02s (50Hz control)
    initial_joint_pos_range: tuple[float, float] = (-math.pi, math.pi)
    joint_init_mode: str = "random"  # "equilibrium" or "random"
    equilibrium_joint_noise: tuple[float, float] = (-0.3, 0.3)  # rad, noise around equilibrium

    # Action mode: "joint_velocity" (legacy) or "ee_position" (IK-based)
    action_mode: str = "ee_position"
    """Action interpretation:
    "joint_velocity": actions are joint velocity commands, integrated to position targets.
    "ee_position": actions are desired EE position (x, y) in body frame, converted to
        joint angles via analytical 2-link IK. Provides a more natural action space
        since buoy (x, y) maps directly to (pitch, roll) torque."""

    workspace_radius: float = 0.40
    """Maximum EE reach for action scaling (meters). Actions in [-1, 1] are scaled to
    [-workspace_radius, workspace_radius]. Set below the kinematic max (0.466m) to avoid
    IK singularity at full extension."""

    # EMA alpha for constraint system joint velocity filtering
    ema_joint_vel_alpha: float = 0.2

    # ==========================================================================
    # Attitude Task and Rewards
    # ==========================================================================
    target_attitude: tuple[float, float, float] = (0.0, 0.0, 0.0)
    randomize_target_attitude: bool = True
    target_attitude_range: tuple[float, float, float] = (0.349, 0.349, 0.0)

    # smoothness replaces joint_osc constraint (fixed weight avoids lambda competition).
    reward: ALBCRewardCfg = ALBCRewardCfg(
        command_type="exponential",
        command_coeff_roll=5.0,
        command_coeff_pitch=7.5,
        smoothness_weight=-0.5,
        torque_weight=-0.0001,
    )

    # ==========================================================================
    # Initialization and Termination
    # ==========================================================================
    initial_height: float = 4.5
    max_angular_velocity: float = math.pi  # rad/s (~180 deg/s)
    max_attitude_angle: float = math.pi / 2.0  # rad (~90 deg)

    # ==========================================================================
    # Domain Randomization (enabled by default for training)
    # ==========================================================================
    randomization: DomainRandomizationCfg = DomainRandomizationCfg(enable=True)

    # ==========================================================================
    # DORAEMON Adaptive DR Curriculum (environment-like parameters)
    # ==========================================================================
    doraemon: DoraemonCfg = DoraemonCfg(enable=True)

    # ==========================================================================
    # Virtual Payload Configuration
    # Payload is applied to the gripper body (fixed to base). Offsets in gripper frame.
    # ==========================================================================
    payload_mass: float = 0.5  # kg
    payload_attachment_offset: tuple[float, float, float] = (0.0, 0.0, -0.05)  # m, gripper frame

    # ==========================================================================
    # Proprioception History (for history-augmented encoder)
    # ==========================================================================
    proprio_history_len: int = 30
    proprio_feature_dim: int = 8  # [roll, pitch, p, q, joint_pos_norm(2), prev_actions(2)]

    # ==========================================================================
    # IMU Sensor Noise
    # ==========================================================================
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

    # ==========================================================================
    # Constraints (C-TRPO)
    # ==========================================================================
    # Budget values are per-step raw budgets D_k. The algorithm transforms them to
    # discounted budgets d_k = D_k / (1 - cost_gamma). With cost_gamma=0.99 (default),
    # d_k = D_k * 100. Barrier penalty activates as mean cost return approaches d_k.
    constraints: ALBCConstraintCfg = ALBCConstraintCfg(
        terms=[
            # --- Binary constraints (5 terms) ---
            ConstraintTermCfg(
                func=accumulated_rotation_cost,
                params={"max_rotations": 2.0},
                budget=0.02,
                name="accum_rot",
            ),
            ConstraintTermCfg(
                func=attitude_absolute_cost,
                params={"limit": 1.396},  # ~80 deg
                budget=0.01,
                name="attitude_abs",
            ),
            ConstraintTermCfg(
                func=effort_limit_cost,
                params={"limit_nm": 9.5},  # Dynamixel XW540 stall torque @ 12V
                budget=0.20,
                name="joint_torque",
            ),
            ConstraintTermCfg(
                func=joint_velocity_limit_cost,
                params={"limit_rad_per_s": 4.189},  # 40 RPM (Dynamixel XW540 no-load)
                budget=0.10,
                name="joint_vel_limit",
            ),
            ConstraintTermCfg(
                func=overshoot_cost,
                params={"threshold": 0.087},
                budget=0.20,
                name="overshoot",
            ),
            # --- Continuous constraints (1 term) ---
            ConstraintTermCfg(
                func=yaw_velocity_cost,
                budget=0.785,
                name="yaw_vel",
            ),
        ],
    )
