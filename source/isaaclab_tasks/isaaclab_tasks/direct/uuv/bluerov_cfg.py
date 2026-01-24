# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""BlueROV2 robot configuration for Isaac Lab.

The BlueROV2 is a popular open-source underwater ROV manufactured by Blue Robotics.
This configuration uses the USD model with hydrodynamic parameters
identified through experiments (originally from MarineGym, IROS 2025).
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from .hydrodynamics_model import HydrodynamicsCfg, OceanCurrentCfg
from .uuv_env_cfg import UUVEnvCfg, ThrusterCfg, DomainRandomizationCfg


@configclass
class BlueROVHydrodynamicsCfg(HydrodynamicsCfg):
    """Hydrodynamic parameters for BlueROV2.

    Parameters are from experimental identification (MarineGym, IROS 2025).
    All values are taken from BlueROV.yaml in the MarineGym repository.
    """

    # Added mass coefficients [surge, sway, heave, roll, pitch, yaw]
    added_mass: tuple[float, ...] = (5.5, 12.7, 14.57, 0.12, 0.12, 0.12)

    # Linear damping coefficients
    linear_damping: tuple[float, ...] = (4.03, 6.22, 5.18, 0.07, 0.07, 0.07)

    # Quadratic damping coefficients
    quadratic_damping: tuple[float, ...] = (18.18, 21.66, 36.99, 1.55, 1.55, 1.55)

    # Vehicle volume (m^3) - from BlueROV2 specifications
    volume: float = 0.0113459

    # Center of buoyancy offset (m) - positive means CoB above CoM
    center_of_buoyancy_offset: float = 0.01

    # Water density (kg/m^3) - freshwater
    water_density: float = 997.0

# Path to BlueROV USD file (relative to this module)
_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
BLUEROV_USD_PATH = os.path.join(_ASSETS_DIR, "BlueROV", "BlueROV.usd")


BLUEROV_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=BLUEROV_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,  # Gravity is balanced by buoyancy
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 2.0),  # Start 2m above ground (underwater)
        rot=(1.0, 0.0, 0.0, 0.0),  # Upright orientation
        joint_pos={
            "rotor_.*": 0.0,
        },
        joint_vel={
            "rotor_.*": 0.0,
        },
    ),
    actuators={
        "thrusters": ImplicitActuatorCfg(
            joint_names_expr=["rotor_.*"],
            stiffness=0.0,
            damping=0.0,
        ),
    },
)


@configclass
class BlueROVEnvCfg(UUVEnvCfg):
    """Environment configuration for BlueROV2 hover task.

    This configuration sets up a BlueROV2 with 6 thrusters to hover
    at a target position while experiencing hydrodynamic forces.
    """

    # Robot
    robot: ArticulationCfg = BLUEROV_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # Hydrodynamics (BlueROV-specific parameters)
    hydrodynamics: BlueROVHydrodynamicsCfg = BlueROVHydrodynamicsCfg()

    # Thrusters (BlueROV has 6 thrusters)
    thrusters: ThrusterCfg = ThrusterCfg(
        num_thrusters=6,
        max_thrust=50.0,
        thrust_coefficient=40.0,  # Adjusted for BlueROV
    )

    # Action space matches number of thrusters
    action_space: int = 6

    # Ocean current (disabled by default, enable for domain randomization)
    ocean_current: OceanCurrentCfg = OceanCurrentCfg(
        max_velocity=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        noise_scale=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )


@configclass
class BlueROVCurrentEnvCfg(BlueROVEnvCfg):
    """BlueROV environment with ocean current disturbances.

    This configuration adds random ocean currents for domain randomization
    and robustness training.
    """

    ocean_current: OceanCurrentCfg = OceanCurrentCfg(
        max_velocity=(0.3, 0.3, 0.1, 0.0, 0.0, 0.0),  # m/s
        noise_scale=(0.1, 0.1, 0.05, 0.0, 0.0, 0.0),
    )

    # Slightly harder task with currents
    position_reward_scale: float = 12.0
    linear_velocity_penalty_scale: float = -0.03


@configclass
class BlueROVTrainEnvCfg(BlueROVEnvCfg):
    """BlueROV training environment with minimal domain randomization.

    Training mode: Randomization typically disabled for stable learning.
    Use this configuration for initial policy training.
    """

    # Disable randomization for deterministic training
    randomization: DomainRandomizationCfg = DomainRandomizationCfg(enable=False)


@configclass
class BlueROVEvalEnvCfg(BlueROVEnvCfg):
    """BlueROV evaluation environment with full domain randomization.

    Evaluation mode: Full randomization enabled for robustness testing.
    Use this configuration to evaluate policy generalization.

    Randomization ranges follow MarineGym defaults:
    - Position: XY ±2.5m, Z 1.5-2.5m
    - Orientation: Roll/Pitch ±36°, Yaw 0-360°
    - Hydrodynamics: Added mass/damping 0.5-1.0x, Volume 0.9-1.1x
    - Mass: 0.8-1.2x
    - Thrusters: 0.8-1.2x
    """

    # Enable full domain randomization for robustness evaluation
    randomization: DomainRandomizationCfg = DomainRandomizationCfg(
        enable=True,
        # Initial pose randomization (MarineGym defaults)
        position_x_range=(-2.5, 2.5),
        position_y_range=(-2.5, 2.5),
        position_z_range=(1.5, 2.5),
        roll_range=(-0.628, 0.628),    # ±36 degrees
        pitch_range=(-0.628, 0.628),   # ±36 degrees
        yaw_range=(0.0, 6.283),        # 0-360 degrees
        # Hydrodynamic parameter randomization
        added_mass_scale=(0.5, 1.0),
        linear_damping_scale=(0.5, 1.0),
        quadratic_damping_scale=(0.5, 1.0),
        volume_scale=(0.9, 1.1),
        mass_scale=(0.8, 1.2),
        # Thruster randomization
        thrust_coefficient_scale=(0.8, 1.2),
        time_constant_scale=(0.8, 1.2),
    )

    # Enable ocean currents for evaluation
    ocean_current: OceanCurrentCfg = OceanCurrentCfg(
        max_velocity=(0.3, 0.3, 0.1, 0.0, 0.0, 0.0),  # m/s
        noise_scale=(0.1, 0.1, 0.05, 0.0, 0.0, 0.0),
    )
