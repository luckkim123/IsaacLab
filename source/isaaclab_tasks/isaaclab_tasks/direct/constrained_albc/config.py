# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Constrained ALBC environment.

ALBC (Active Linear Buoyancy Controller) uses 2 revolute joints (joint1, joint2)
to position a buoyancy element for attitude stabilization. No thrusters are used.

Registered tasks:
    Isaac-Constrained-ALBC-Encoder-v0:          TRPO + IPO + Encoder (production)
    Isaac-Constrained-ALBC-HardDR-HistOnly-v0:  History-only baseline (hard DR)
    Isaac-Constrained-ALBC-HardDR-FrozenEncoder-v0: Frozen encoder fine-tuning (hard DR)
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
        - Action:  q_des += delta_scale * a_t (Delta joint PD targets)
        - Critic:  cat([o_t(14D), p_t(23D)]) = 37D -> Shared Backbone[512,256,128]->64D
                   -> Reward Head(1D) + Cost Head(K=4)

    Control Frequency (1:4 ratio):
        - Physics: dt=0.005s (200Hz)
        - Env step: decimation=4 -> 0.02s (50Hz)
        - Policy: control_decimation=1 -> 50Hz
    """

    # ==========================================================================
    # Environment Settings
    # ==========================================================================
    episode_length_s: float = 15.0
    decimation: int = 4  # 4 physics substeps per env step: 0.005 * 4 = 0.02s (50Hz)
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
        dt=0.005,  # 200Hz physics, 1:4 ratio with 50Hz policy
        render_interval=4,  # render every 4 physics steps = 50Hz
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
    control_decimation: int = 1  # policy updates every env step (50Hz = 200Hz / 4)
    initial_joint_pos_range: tuple[float, float] = (-math.pi, math.pi)

    nominal_joint_pos: tuple[float, float] = (0.0, math.pi / 2.0)
    """Nominal joint configuration (g1, g2). At (0, pi/2), EE is at (L1, L2) = (0.233, 0.233).
    This provides full-rank Jacobian (maximum manipulability) with balanced authority
    in both x (pitch) and y (roll) directions. The previous (0, pi) was a kinematic
    singularity with zero pitch authority. Small bias torque (~0.06 Nm) is negligible
    compared to buoy restoring force (~17N). Used as initial target on episode reset."""

    delta_scale: float = 0.08
    """Delta action scaling (rad/step). q_des += delta_scale * a_t each control step.
    At 50Hz, max joint velocity = delta_scale * 50 = 4.0 rad/s (within 4.189 limit).
    Any absolute position reachable via accumulation over multiple steps."""

    # ==========================================================================
    # Proprioceptive History (for actor input, reduces z/input ratio)
    # ==========================================================================
    proprio_history_len: int = 0
    """Number of timesteps in proprioception history buffer. 0 = disabled.
    When > 0, actor receives cat([o_t, hist_flat, z]) instead of cat([o_t, z]).
    Features per step (8D): roll, pitch, p, q, joint_pos_norm(2), prev_actions(2).
    Flattened: history_len * 8 dimensions added to actor input."""

    proprio_feature_dim: int = 8
    """Dimension of proprioceptive features per timestep."""

    proprio_history_stride: int = 1
    """Stride for proprioceptive history recording. 1 = every control step (default).
    N = record every N-th control step. Effective temporal span = history_len * stride * step_dt.
    Example: stride=5 at 50Hz -> 10Hz sampling, 15 entries = 1.5s window."""

    # ==========================================================================
    # Attitude Task and Rewards
    # ==========================================================================
    target_attitude: tuple[float, float, float] = (0.0, 0.0, 0.0)
    randomize_target_attitude: bool = True
    target_attitude_range: tuple[float, float, float] = (0.349, 0.349, 0.0)

    reward: ALBCRewardCfg = ALBCRewardCfg(
        k_c=-8.0,
        k_tau=-0.005,  # Reduced: constraint handles hard torque limit
        k_s=-0.1,  # Reduced: constraint handles velocity limit
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
                budget=0.08,
                name="torque",
            ),
            ConstraintTermCfg(
                func=velocity_limit_cost,
                params={"limit_rad_per_s": 4.189},
                budget=0.02,
                name="velocity",
            ),
            # --- Average (continuous) ---
            ConstraintTermCfg(
                func=yaw_velocity_cost,
                budget=0.40,
                name="yaw_vel",
            ),
        ],
    )


@configclass
class HardDomainRandomizationCfg(DomainRandomizationCfg):
    """Aggressive DR for offline encoder experiments.

    Widened ranges so that history-only policy degrades to ~10 deg attitude error,
    creating a gap that encoder should fill. Key changes from base DR:
    - added_mass: +-15% -> +-40% (real robot added mass is estimated, not measured)
    - body_mass: +-10% -> +-25% (manufacturing tolerance + biofouling)
    - volume: +-10% -> +-25% (affects buoyancy directly)
    - CoG/CoB offsets: doubled (assembly tolerance)
    - inertia: (0.75, 1.3) -> (0.5, 1.8) (uncertain geometry)
    - payload: 0-1kg -> 0-2kg (20% body weight)
    - ocean current: enabled (0.5 m/s xy, 0.25 m/s z)
    """

    enable: bool = True

    # Hydrodynamic parameters -- significantly widened
    added_mass_scale: tuple[float, float] = (0.6, 1.4)
    volume_scale: tuple[float, float] = (0.75, 1.25)

    # CoG/CoB offsets -- doubled
    cob_offset_x: tuple[float, float] = (-0.02, 0.02)
    cob_offset_y: tuple[float, float] = (-0.02, 0.02)
    cob_offset_z: tuple[float, float] = (-0.04, 0.04)
    cog_offset_x: tuple[float, float] = (-0.02, 0.02)
    cog_offset_y: tuple[float, float] = (-0.02, 0.02)
    cog_offset_z: tuple[float, float] = (-0.04, 0.04)

    # Inertia -- wider range
    inertia_scale: tuple[float, float] = (0.5, 1.8)

    # Body mass -- wider
    body_mass_scale: tuple[float, float] = (0.75, 1.25)

    # Payload -- doubled
    payload_mass_range: tuple[float, float] = (0.0, 2.0)
    payload_cog_offset_xy_radius: float = 0.15
    payload_cog_offset_z: tuple[float, float] = (-0.05, 0.0)


@configclass
class ALBCHardDRHistOnlyEnvCfg(ALBCEnvCfg):
    """Hard DR + History-only (no encoder). Baseline for offline encoder experiments.

    Target: ~10 deg attitude error (vs 3 deg with standard DR).
    If error is < 5 deg, DR is not hard enough; if > 20 deg, DR is too aggressive.
    """

    state_space: int = 0
    proprio_history_len: int = 30

    randomization: HardDomainRandomizationCfg = HardDomainRandomizationCfg()
    ocean_current: OceanCurrentCfg = OceanCurrentCfg(
        max_velocity=(0.5, 0.5, 0.25, 0.0, 0.0, 0.0),
        noise_scale=(0.1, 0.1, 0.05, 0.0, 0.0, 0.0),
    )
    doraemon: DoraemonCfg = DoraemonCfg(enable=False)
    constraints: ALBCConstraintCfg = ALBCConstraintCfg(terms=[])


@configclass
class ALBCHardDRFrozenEncoderEnvCfg(ALBCEnvCfg):
    """Hard DR + Frozen Encoder fine-tuning environment.

    History: 30 steps, stride 1 (matches history-only baseline for warm-start).
    state_space=23 for privileged obs (encoder input).
    """

    proprio_history_len: int = 30
    proprio_history_stride: int = 1

    randomization: HardDomainRandomizationCfg = HardDomainRandomizationCfg()
    ocean_current: OceanCurrentCfg = OceanCurrentCfg(
        max_velocity=(0.5, 0.5, 0.25, 0.0, 0.0, 0.0),
        noise_scale=(0.1, 0.1, 0.05, 0.0, 0.0, 0.0),
    )
    doraemon: DoraemonCfg = DoraemonCfg(enable=False)
    constraints: ALBCConstraintCfg = ALBCConstraintCfg(terms=[])
