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
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

# Import configuration classes from isaaclab_assets
from isaaclab_assets.robots.uuv import (
    HERO_AGENT_CFG,
    HeroAgentBuoyHydrodynamicsCfg,
    HeroAgentHydrodynamicsCfg,
    HydrodynamicsCfg,
    OceanCurrentCfg,
)

from .tasks import ALBCAttitudeTaskCfg


@configclass
class DomainRandomizationCfg:
    """Configuration for domain randomization in Hero Agent ALBC environments."""

    enable: bool = False

    # Initial position randomization range (meters)
    position_x_range: tuple[float, float] = (-0.5, 0.5)
    position_y_range: tuple[float, float] = (-0.5, 0.5)
    position_z_range: tuple[float, float] = (4.0, 5.0)

    # Initial orientation randomization range (radians)
    roll_range: tuple[float, float] = (-0.785, 0.785)   # +/-45 degrees
    pitch_range: tuple[float, float] = (-0.785, 0.785)
    yaw_range: tuple[float, float] = (-3.14159, 3.14159)

    # Hydrodynamic parameter scale ranges
    added_mass_scale: tuple[float, float] = (0.8, 1.2)
    linear_damping_scale: tuple[float, float] = (0.8, 1.2)
    quadratic_damping_scale: tuple[float, float] = (0.8, 1.2)
    volume_scale: tuple[float, float] = (0.95, 1.05)
    mass_scale: tuple[float, float] = (0.9, 1.1)

    # Center of Buoyancy offset scale
    cob_offset_scale: tuple[float, float] = (0.8, 1.2)

    # Rigid body inertia scale
    inertia_scale: tuple[float, float] = (0.9, 1.1)

    # Payload randomization
    payload_mass_ratio: tuple[float, float] = (0.0, 0.1)
    payload_cog_offset_z: tuple[float, float] = (-0.02, 0.02)


@configclass
class HeroAgentEnvCfg(DirectRLEnvCfg):
    """Configuration for Hero Agent ALBC environment.

    The vehicle uses 2 revolute joints (joint1, joint2) to position a buoyancy
    element for attitude stabilization. No thrusters are used.
    """

    # Environment settings
    episode_length_s: float = 15.0
    decimation: int = 2
    action_space: int = 2  # 2 joint velocity commands
    observation_space: int = 13  # euler(3) + ang_vel(3) + errors(3) + joint_pos(2) + prev_act(2)
    state_space: int = 0
    debug_vis: bool = False  # ALBC doesn't have goal position visualization

    # Simulation configuration
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

    # Scene configuration
    # Note: clone_in_fabric=False to ensure proper visual mesh cloning
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=2.0,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    # Terrain (underwater floor)
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

    # Robot configuration
    robot = HERO_AGENT_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # Body link names
    body_link_name: str = "base"
    buoy_link_name: str = "link3"

    # Hydrodynamics for main body
    hydrodynamics: HydrodynamicsCfg = HeroAgentHydrodynamicsCfg()

    # Buoy hydrodynamics (link3)
    buoy_hydrodynamics: HydrodynamicsCfg = HeroAgentBuoyHydrodynamicsCfg()

    # Child body simple buoyancy parameters (force only, no torque)
    # Format: {"body_name": {"mass": kg, "volume": m^3}}
    # Note: heroagent2.py does not apply separate buoyancy to these bodies,
    # so we set them to near-neutral buoyancy (minimal effect)
    child_body_buoyancy: dict[str, dict[str, float]] = {
        "gripper": {"mass": 0.3, "volume": 0.0003},    # near-neutral
        "link1": {"mass": 0.1, "volume": 0.0001},      # near-neutral
        "link2": {"mass": 0.1, "volume": 0.0001},      # near-neutral
    }

    # Water density for child body buoyancy calculation (kg/m^3)
    # Matched to heroagent2.py
    water_density: float = 998.0

    # Ocean current configuration
    ocean_current: OceanCurrentCfg = OceanCurrentCfg(
        max_velocity=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        noise_scale=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )

    # ALBC joint names
    albc_joint_names: list[str] = ["joint1", "joint2"]

    # Action speed scale: pi rad/s
    action_speed_scale: float = math.pi

    # Control decimation (100 Hz sim / 4 = 25 Hz control)
    control_decimation: int = 4

    # Initial joint position randomization range (radians)
    initial_joint_pos_range: tuple[float, float] = (-6.0, 6.0)

    # ALBC task configuration
    task: ALBCAttitudeTaskCfg = ALBCAttitudeTaskCfg(
        target_attitude=(0.0, 0.0, 0.0),
        randomize_target=False,
    )

    # Legacy parameters for compatibility
    goal_pos_range: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_height: float = 4.5

    # Termination conditions
    min_height: float = 0.0
    max_height: float = 10.0
    max_distance_from_origin: float = 10.0

    # ALBC reward weight
    albc_action_cost_weight: float = -0.1

    # Domain randomization configuration
    randomization: DomainRandomizationCfg = DomainRandomizationCfg()


@configclass
class HeroAgentTrainEnvCfg(HeroAgentEnvCfg):
    """Hero Agent ALBC training environment with domain randomization."""

    randomization = DomainRandomizationCfg(
        enable=True,
        position_x_range=(-0.5, 0.5),
        position_y_range=(-0.5, 0.5),
        position_z_range=(4.0, 5.0),
        roll_range=(-0.785, 0.785),
        pitch_range=(-0.785, 0.785),
        yaw_range=(-3.14159, 3.14159),
        added_mass_scale=(0.8, 1.2),
        linear_damping_scale=(0.8, 1.2),
        quadratic_damping_scale=(0.8, 1.2),
        volume_scale=(0.95, 1.05),
        mass_scale=(0.9, 1.1),
        cob_offset_scale=(0.8, 1.2),
        inertia_scale=(0.9, 1.1),
        payload_mass_ratio=(0.0, 0.1),
        payload_cog_offset_z=(-0.02, 0.02),
    )

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
        yaw_range=(-1.57, 1.57),
        added_mass_scale=(0.9, 1.1),
        linear_damping_scale=(0.9, 1.1),
        quadratic_damping_scale=(0.9, 1.1),
        volume_scale=(0.98, 1.02),
        mass_scale=(0.95, 1.05),
        cob_offset_scale=(0.9, 1.1),
        inertia_scale=(0.95, 1.05),
        payload_mass_ratio=(0.0, 0.05),
        payload_cog_offset_z=(-0.01, 0.01),
    )

    ocean_current = OceanCurrentCfg(
        max_velocity=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        noise_scale=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
