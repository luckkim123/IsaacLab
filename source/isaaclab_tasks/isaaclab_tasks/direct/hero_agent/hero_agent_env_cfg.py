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
from isaaclab.envs import DirectRLEnvCfg
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

from .rewards import ALBCRewardCfg
from .tasks import ALBCAttitudeTaskCfg


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
    debug_vis: bool = False

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

    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

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
    # Task and Rewards
    # ==========================================================================
    task: ALBCAttitudeTaskCfg = ALBCAttitudeTaskCfg(
        target_attitude=(0.0, 0.0, 0.0),
        randomize_target=False,
    )
    reward: ALBCRewardCfg = ALBCRewardCfg()

    # ==========================================================================
    # Initialization and Termination
    # ==========================================================================
    initial_height: float = 4.5
    min_height: float = 0.0
    max_height: float = 10.0
    max_distance_from_origin: float = 10.0

    # ==========================================================================
    # Domain Randomization
    # ==========================================================================
    randomization: DomainRandomizationCfg = DomainRandomizationCfg()


@configclass
class HeroAgentTrainEnvCfg(HeroAgentEnvCfg):
    """Hero Agent ALBC training environment with domain randomization."""

    randomization = DomainRandomizationCfg(enable=True)
    ocean_current = OceanCurrentCfg(
        max_velocity=(0.2, 0.2, 0.1, 0.0, 0.0, 0.0),
        noise_scale=(0.05, 0.05, 0.02, 0.0, 0.0, 0.0),
    )


@configclass
class HeroAgentEvalEnvCfg(HeroAgentEnvCfg):
    """Hero Agent ALBC evaluation environment with moderate perturbations."""

    randomization = DomainRandomizationCfg(
        enable=True,
        position_x_range=(-0.3, 0.3),
        position_y_range=(-0.3, 0.3),
        position_z_range=(4.2, 4.8),
        roll_range=(-0.5, 0.5),
        pitch_range=(-0.5, 0.5),
        yaw_range=(-math.pi / 2, math.pi / 2),
        added_mass_scale=(0.9, 1.1),
        linear_damping_scale=(0.9, 1.1),
        quadratic_damping_scale=(0.9, 1.1),
        volume_scale=(0.98, 1.02),
        cob_offset_x=(-0.005, 0.005),
        cob_offset_y=(-0.005, 0.005),
        cob_offset_z=(-0.01, 0.01),
        cog_offset_x=(-0.005, 0.005),
        cog_offset_y=(-0.005, 0.005),
        cog_offset_z=(-0.01, 0.01),
        inertia_scale=(0.95, 1.05),
    )

    ocean_current = OceanCurrentCfg(
        max_velocity=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        noise_scale=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )


@configclass
class HeroAgentEncoderTrainEnvCfg(HeroAgentTrainEnvCfg):
    """Hero Agent encoder training with privileged hydrodynamic info.

    state_space=20 signals the environment to return compact privileged information
    alongside policy observations for HORA/RMA Phase 1 teacher training.
    Uses get_privileged_info_compact(): Main body (10D) + Buoy (10D) = 20D.
    Contains: volume, r_cg, r_cb, inertia (core hydrostatic params).
    Excludes: damping, added_mass, ocean_current (velocity-dependent).

    Network Input Dimensions (ActorCriticEncoder):
        - observation_space (13): Used for gym.spaces.Box definition only
        - state_space (20): Privileged info, returned as observations["privileged"]
        - Encoder: privileged(20D) -> latent z(6D)
        - Actual Actor/Critic input: policy_obs(13) + z(6) = 19D
    """

    state_space: int = 20


@configclass
class HeroAgentEncoderEvalEnvCfg(HeroAgentEvalEnvCfg):
    """Hero Agent encoder evaluation with privileged hydrodynamic info.

    Uses moderate domain randomization (same as HeroAgentEvalEnvCfg) while
    providing privileged information for encoder-based policy evaluation.

    Network Input Dimensions (ActorCriticEncoder):
        - observation_space (13): Used for gym.spaces.Box definition only
        - state_space (20): Privileged info, returned as observations["privileged"]
        - Encoder: privileged(20D) -> latent z(6D)
        - Actual Actor/Critic input: policy_obs(13) + z(6) = 19D
    """

    state_space: int = 20
