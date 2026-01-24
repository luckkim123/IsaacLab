# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration classes for UUV (Underwater Vehicle) environments."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from .hydrodynamics_model import HydrodynamicsCfg, OceanCurrentCfg


@configclass
class ThrusterCfg:
    """Configuration for underwater vehicle thrusters.

    This configuration supports simplified thruster dynamics based on the
    Blue Robotics T200 thruster model.

    BlueROV2 Heavy thruster layout (6 thrusters):
        - Thrusters 0-3: Horizontal vectored at 45 degrees (X-Y-Yaw control)
        - Thrusters 4-5: Vertical (Z-Roll-Pitch control)

    The allocation matrix maps thruster commands to body wrench:
        wrench = allocation_matrix @ thrusts
    """

    # Number of thrusters
    num_thrusters: int = 6

    # Maximum thrust per thruster (N) - T200 max thrust
    max_thrust: float = 50.0

    # Thruster time constant for first-order dynamics (s)
    # From MarineGym T200 model tau_up/tau_down (throttle dynamics)
    # Note: BlueROV.yaml time_constants=0.01 is for RPM filter, not throttle
    time_constant_up: float = 0.43
    time_constant_down: float = 0.43

    # Thrust coefficient (N per normalized throttle [-1, 1])
    thrust_coefficient: float = 50.0

    # Thruster arm lengths (m) - distance from center of mass
    arm_length_x: float = 0.12  # X distance to vertical thrusters
    arm_length_y: float = 0.12  # Y distance to vertical thrusters
    arm_length_xy: float = 0.15  # Distance to horizontal thrusters

    # Thruster allocation matrix (6 DOF x num_thrusters)
    # Maps thruster forces to body wrench [Fx, Fy, Fz, Mx, My, Mz]
    # BlueROV2 Heavy configuration with 45-degree vectored horizontal thrusters
    # Row order: Fx, Fy, Fz, Mx, My, Mz
    # Column order: T0, T1, T2, T3, T4, T5
    # Horizontal thrusters (0-3) at 45 degrees, Vertical thrusters (4-5)
    allocation_matrix: tuple[tuple[float, ...], ...] = (
        # Fx: forward thrust from horizontal thrusters
        (0.707, 0.707, -0.707, -0.707, 0.0, 0.0),
        # Fy: lateral thrust from horizontal thrusters
        (0.707, -0.707, 0.707, -0.707, 0.0, 0.0),
        # Fz: vertical thrust from vertical thrusters
        (0.0, 0.0, 0.0, 0.0, 1.0, 1.0),
        # Mx: roll moment from vertical thrusters
        (0.0, 0.0, 0.0, 0.0, 0.12, -0.12),
        # My: pitch moment from vertical thrusters
        (0.0, 0.0, 0.0, 0.0, -0.12, -0.12),
        # Mz: yaw moment from horizontal thrusters
        (-0.15, 0.15, 0.15, -0.15, 0.0, 0.0),
    )


@configclass
class DomainRandomizationCfg:
    """Configuration for domain randomization in UUV environments.

    This configuration supports train/evaluate mode switching and follows
    MarineGym's scale-based parameter randomization pattern.

    Randomization is applied on every environment reset when enabled.
    """

    # Enable/disable randomization (set False for deterministic training/evaluation)
    enable: bool = False

    # Initial position randomization range (meters)
    # Following MarineGym defaults: XY [-2.5, 2.5], Z [1.5, 2.5]
    position_x_range: tuple[float, float] = (-2.5, 2.5)
    position_y_range: tuple[float, float] = (-2.5, 2.5)
    position_z_range: tuple[float, float] = (1.5, 2.5)

    # Initial orientation randomization range (radians)
    # Following MarineGym defaults: Roll/Pitch ±36°, Yaw 0-360°
    roll_range: tuple[float, float] = (-0.628, 0.628)    # ±36 degrees
    pitch_range: tuple[float, float] = (-0.628, 0.628)   # ±36 degrees
    yaw_range: tuple[float, float] = (0.0, 6.283)        # 0-360 degrees

    # Hydrodynamic parameter scale ranges (multipliers applied to base values)
    # Following MarineGym defaults
    added_mass_scale: tuple[float, float] = (0.5, 1.0)
    linear_damping_scale: tuple[float, float] = (0.5, 1.0)
    quadratic_damping_scale: tuple[float, float] = (0.5, 1.0)
    volume_scale: tuple[float, float] = (0.9, 1.1)

    # Robot mass scale (applied to all bodies)
    mass_scale: tuple[float, float] = (0.8, 1.2)

    # Thruster parameter scales
    thrust_coefficient_scale: tuple[float, float] = (0.8, 1.2)
    time_constant_scale: tuple[float, float] = (0.8, 1.2)


@configclass
class UUVEnvCfg(DirectRLEnvCfg):
    """Configuration for the UUV hover/tracking environment.

    This environment trains an underwater vehicle to hover at a target position
    or track a trajectory while experiencing hydrodynamic forces and optional
    ocean current disturbances.
    """

    # Environment settings
    episode_length_s: float = 15.0
    decimation: int = 2
    action_space: int = 6  # 6 thruster commands
    observation_space: int = 18  # pos(3) + quat(4) + lin_vel(3) + ang_vel(3) + goal_pos_b(3) + up(2)
    state_space: int = 0
    debug_vis: bool = True

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
        # Enable external forces every iteration for accurate hydrodynamic force integration
        # This is important for underwater vehicles with continuous external forces
        physx=PhysxCfg(
            enable_external_forces_every_iteration=True,
        ),
    )

    # Scene configuration
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=5.0,
        replicate_physics=True,
        clone_in_fabric=True,
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

    # Robot configuration (will be set by specific vehicle configs)
    robot: ArticulationCfg = None  # type: ignore

    # Name of the body link to apply hydrodynamic forces to
    body_link_name: str = "base_link"

    # Hydrodynamics configuration (should be overridden by specific robot configs)
    hydrodynamics: HydrodynamicsCfg = HydrodynamicsCfg()

    # Ocean current configuration
    ocean_current: OceanCurrentCfg = OceanCurrentCfg()

    # Thruster configuration
    thrusters: ThrusterCfg = ThrusterCfg()

    # Task parameters
    # Goal position randomization range (relative to spawn)
    goal_pos_range: tuple[float, float, float] = (2.0, 2.0, 1.0)

    # Initial position height (underwater depth)
    initial_height: float = 2.0

    # Termination conditions
    max_height: float = 5.0  # Max distance from ground
    min_height: float = 0.2  # Min distance from ground
    max_distance_from_origin: float = 10.0

    # Reward scales
    # Position tracking
    position_reward_scale: float = 10.0
    position_reward_exp_scale: float = 0.5  # For exponential reward shaping

    # Orientation tracking (upright)
    orientation_reward_scale: float = 2.0

    # Velocity penalties
    linear_velocity_penalty_scale: float = -0.05
    angular_velocity_penalty_scale: float = -0.01

    # Action penalties
    action_rate_penalty_scale: float = -0.01
    action_magnitude_penalty_scale: float = -0.001

    # Alive bonus
    alive_reward_scale: float = 0.5

    # Domain randomization configuration
    randomization: DomainRandomizationCfg = DomainRandomizationCfg()
