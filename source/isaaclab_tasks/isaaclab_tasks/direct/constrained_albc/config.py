# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Constrained ALBC environment.

ALBC (Active Linear Buoyancy Controller) uses 2 revolute joints (joint1, joint2)
to position a buoyancy element for attitude stabilization. No thrusters are used.

Single registered task: Isaac-Constrained-ALBC-Encoder-v0
    TRPO + IPO encoder constrained RL with 4 constraint terms.
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
from .mdp.constraints import (
    ALBCConstraintCfg,
    ConstraintTermCfg,
    attitude_limit_cost,
    torque_limit_cost,
    velocity_limit_cost,
    yaw_velocity_cost,
)
from .mdp.rewards import ALBCRewardCfg


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

    The vehicle uses 2 revolute joints (joint1, joint2) to position a buoyancy
    element for attitude stabilization. No thrusters are used.

    Network Input Dimensions (Teacher Policy):
        - observation_space (14): o_t policy obs for gym.spaces.Box
        - state_space (23): p_t privileged info, returned as observations["privileged"]
        - Encoder: p_t(23D) -> MLP[256,128,64] -> softsign -> z(13D)
        - Actor:   cat([o_t(14D), z(13D)]) = 27D -> MLP[256,128,64] -> a_t(2D)
        - Action:  q_des = q_nominal + action_scale * a_t (Joint PD targets)
        - Critic:  cat([o_t(14D), p_t(23D)]) = 37D -> Shared Backbone[512,256,128]->64D
                   -> Reward Head(1D) + Cost Head(K=4)

    Control Frequency (1:40 ratio):
        - Physics: dt=0.0005s (2000Hz PD)
        - Env step: decimation=40 -> 0.02s (50Hz)
        - Policy: control_decimation=1 -> 50Hz
    """

    # ==========================================================================
    # Environment Settings
    # ==========================================================================
    episode_length_s: float = 15.0
    decimation: int = 40  # 40 physics substeps per env step: 0.0005 * 40 = 0.02s (50Hz)
    action_space: int = 2
    observation_space: int = 14  # o_t: euler(3) + ang_vel(3) + att_err(2) + jpos(2) + jvel(2) + prev_act(2)
    state_space: int = 23  # p_t: hydro(6)+inertia(4)+damping(4)+mass(2)+payload(4)+joint(2)+density(1)
    debug_vis: bool = False

    # Top-down camera view (looking down at robot from above)
    viewer: ViewerCfg = ViewerCfg(
        eye=(0.0, 0.0, 12.0),
        lookat=(0.0, 0.0, 4.5),
    )

    # ==========================================================================
    # Simulation
    # ==========================================================================
    sim: SimulationCfg = SimulationCfg(
        dt=0.0005,  # 2000Hz physics (PD control), 1:40 ratio with 50Hz policy
        render_interval=40,  # render every 40 physics steps = 50Hz
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
    control_decimation: int = 1  # policy updates every env step (50Hz = 2000Hz / 40)
    initial_joint_pos_range: tuple[float, float] = (-math.pi, math.pi)

    nominal_joint_pos: tuple[float, float] = (0.0, math.pi)
    """Nominal joint configuration (g1, g2). At (0, pi), EE is at body center (0, 0).
    Actions offset from this: q_des = q_nominal + action_scale * a_t."""

    action_scale: float = math.pi
    """Action scaling factor (sigma_a). Maps action [-1, 1] to joint offset [-pi, pi].
    q_des = q_nominal + action_scale * a_t. PD controller tracks q_des at 2000Hz."""

    # ==========================================================================
    # Attitude Task and Rewards
    # ==========================================================================
    target_attitude: tuple[float, float, float] = (0.0, 0.0, 0.0)
    randomize_target_attitude: bool = True
    target_attitude_range: tuple[float, float, float] = (0.349, 0.349, 0.0)

    reward: ALBCRewardCfg = ALBCRewardCfg(
        k_c=-1.0,
        k_tau=-0.001,
        k_s=-0.05,
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
    # IMU Sensor Noise (14D: euler(3), ang_vel(3), att_err(2), jpos(2), jvel(2), prev_act(2))
    # ==========================================================================
    observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(
            mean=0.0,
            std=tuple(
                [0.02] * 3  # euler
                + [0.04] * 3  # ang_vel
                + [0.02] * 2  # att_err
                + [0.02] * 2  # joint_pos
                + [0.04] * 2  # joint_vel
                + [0.0] * 2  # prev_actions (no noise)
            ),
        ),
        bias_noise_cfg=UniformNoiseCfg(
            n_min=tuple([-0.02] * 3 + [-0.03] * 3 + [-0.02] * 2 + [-0.02] * 2 + [-0.03] * 2 + [0] * 2),
            n_max=tuple([0.02] * 3 + [0.03] * 3 + [0.02] * 2 + [0.02] * 2 + [0.03] * 2 + [0] * 2),
        ),
    )

    # ==========================================================================
    # Constraints (IPO): 3 probabilistic + 1 average = 4 terms
    # Budget D_k is per-step; algorithm converts to d_k = D_k / (1 - gamma).
    # ==========================================================================
    constraints: ALBCConstraintCfg = ALBCConstraintCfg(
        terms=[
            # --- Probabilistic (binary indicator) ---
            ConstraintTermCfg(
                func=attitude_limit_cost,
                params={"limit": 1.396},
                budget=0.01,
                name="attitude",
            ),
            ConstraintTermCfg(
                func=torque_limit_cost,
                params={"limit_nm": 9.5},
                budget=0.20,
                name="torque",
            ),
            ConstraintTermCfg(
                func=velocity_limit_cost,
                params={"limit_rad_per_s": 4.189},
                budget=0.10,
                name="velocity",
            ),
            # --- Average (continuous) ---
            ConstraintTermCfg(
                func=yaw_velocity_cost,
                budget=0.785,
                name="yaw_vel",
            ),
        ],
    )
