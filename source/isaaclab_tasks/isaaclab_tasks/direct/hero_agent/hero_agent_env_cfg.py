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

from .mdp import ALBCRewardCfg


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
        ):
            setattr(self, attr, (1.0, 1.0))
        for attr in (
            "cob_offset_x",
            "cob_offset_y",
            "cob_offset_z",
            "cog_offset_x",
            "cog_offset_y",
            "cog_offset_z",
        ):
            setattr(self, attr, (0.0, 0.0))

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
    decimation: int = 2  # 0.005 * 2 = 0.01s = 100Hz policy (RL action)
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
    control_decimation: int = 4  # target updates every 4th RL step (25Hz servo command)
    initial_joint_pos_range: tuple[float, float] = (-0.3, 0.3)

    albc_joint_stiffness: float | None = None
    """ALBC joint stiffness (PhysX PD gain). None = use actuator default (500.0).
    Override for runtime tuning. Target damping ratio ~0.7 with Kd=10."""

    albc_joint_damping: float | None = None
    """ALBC joint damping (PhysX PD gain). None = use actuator default (10.0).
    Override for runtime tuning. Target damping ratio ~0.7 with Kp=500."""

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
class HeroAgentTDCEnvCfg(HeroAgentEnvCfg):
    """Hero Agent TDC (Time Delay Control) environment configuration.

    Uses classical TDC controller instead of RL for attitude stabilization.
    All domain randomization and ocean currents are disabled for controlled testing.
    Initial pose is tilted 15 degrees in roll and pitch.
    """

    # TDC controller parameters
    tdc_m_hat: tuple[float, float] = (0.15, 0.15)  # kg*m^2 (roll, pitch)
    tdc_kp: float = 40.0  # omega_n ~= 6.3 rad/s (aggressive for TDE dominance)
    tdc_kd: float = 12.0  # zeta ~= 0.95 (near-critically damped)
    tdc_dls_damping: float = 0.01  # DLS regularization for singularity
    tdc_h: float = 0.230  # buoyancy height offset (m)
    tdc_workspace_radius: float = 0.45  # EE clamp radius (< l1+l2=0.466)
    tdc_nu_dot_ema_alpha: float = 0.3  # EMA filter for angular accel (lower=smoother)
    tdc_tde_gain: float = 1.0  # TDE contribution scale (0.0=pure PD, 1.0=full TDC)
    tdc_h_hat_filter_alpha: float = 1.0  # U_hat EMA filter (1.0=no filter, <1=smoother)
    tdc_log_interval: int = 200  # Console log every N steps (0 = disabled)

    def __post_init__(self):
        """Set up TDC-specific defaults."""
        super().__post_init__()

        # TDC runs at 200Hz (every physics step) for accurate TDE
        self.decimation = 1  # step_dt = physics_dt = 0.005s
        self.control_decimation = 1

        # Fixed initial pose: 15 degrees tilt in roll and pitch
        self.randomization.disable_all(roll=0.2618, pitch=0.2618)

        # Disable ocean current for pure control testing
        self.ocean_current.max_velocity = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.ocean_current.noise_scale = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # Disable payload
        self.enable_payload = False
        self.state_space = 0


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
