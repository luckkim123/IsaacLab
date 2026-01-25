# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration classes for UUV (Underwater Vehicle) environments."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

# Import configuration classes from isaaclab_assets
from isaaclab_assets.robots.uuv import HydrodynamicsCfg, OceanCurrentCfg, ThrusterCfg
from .mdp import events as uuv_events
from .tasks import TaskBaseCfg, HoverTaskCfg


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
class EventCfg:
    """Isaac Lab EventManager-based domain randomization configuration.

    NOTE: Currently NOT USED. UUVEnv inherits DirectRLEnv which does not have
    EventManager. Randomization is handled manually in _reset_idx() using
    DomainRandomizationCfg above.

    This is kept for future migration to ManagerBasedRLEnv, which would
    automatically process these EventTerms via EventManager.

    Usage (when migrating to ManagerBasedRLEnv):
        class UUVEnvCfg(ManagerBasedRLEnvCfg):
            events: EventCfg = EventCfg()
    """

    randomize_hydrodynamics = EventTerm(
        func=uuv_events.randomize_hydrodynamics,
        mode="reset",
        params={
            "added_mass_scale": (0.5, 1.0),
            "linear_damping_scale": (0.5, 1.0),
            "quadratic_damping_scale": (0.5, 1.0),
            "volume_scale": (0.9, 1.1),
            "mass_scale": (0.8, 1.2),
        },
    )

    randomize_thrusters = EventTerm(
        func=uuv_events.randomize_thruster_params,
        mode="reset",
        params={
            "thrust_coeff_scale": (0.8, 1.2),
            "time_constant_scale": (0.8, 1.2),
        },
    )

    randomize_current = EventTerm(
        func=uuv_events.randomize_ocean_current,
        mode="reset",
        params={
            "max_velocity": (0.5, 0.5, 0.1),
        },
    )

    randomize_pose = EventTerm(
        func=uuv_events.randomize_robot_pose,
        mode="reset",
        params={
            "position_x_range": (-2.5, 2.5),
            "position_y_range": (-2.5, 2.5),
            "position_z_range": (1.5, 2.5),
            "roll_range": (-0.628, 0.628),
            "pitch_range": (-0.628, 0.628),
            "yaw_range": (0.0, 6.283),
        },
    )


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

    # Task configuration (will be overridden by specific robot/task configs)
    # Default is HoverTask with standard parameters
    task: TaskBaseCfg = HoverTaskCfg()

    # Legacy task parameters (deprecated - use task config instead)
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
    # exp(-distance/exp_scale): at 3m distance with scale=2.0, reward=0.22 (vs 0.0025 with scale=0.5)
    position_reward_scale: float = 15.0
    position_reward_exp_scale: float = 2.0  # Larger scale = denser reward at distance

    # Orientation tracking (upright) - uses exponential reward
    # exp(-tilt_angle/exp_scale): at 30deg with scale=0.5, reward=0.35
    orientation_reward_scale: float = 5.0  # Increased for stricter upright control
    orientation_exp_scale: float = 0.5  # 0.5 rad ~ 29 degrees for exp(-1) = 0.37

    # Velocity penalties
    linear_velocity_penalty_scale: float = -0.01
    angular_velocity_penalty_scale: float = -0.005

    # Action penalties
    action_rate_penalty_scale: float = -0.005
    action_magnitude_penalty_scale: float = -0.0005

    # Alive bonus
    alive_reward_scale: float = 0.1

    # Domain randomization configuration (used by DirectRLEnv)
    randomization: DomainRandomizationCfg = DomainRandomizationCfg()

    # Reserved for future ManagerBasedRLEnv migration (currently unused)
    events: EventCfg | None = None
