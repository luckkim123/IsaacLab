# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hero Agent underwater vehicle configuration for Isaac Lab.

The Hero Agent is a custom deep-sea intervention ROV with:
    - Main body with integrated sensors
    - 2-DOF manipulator arm
    - Gripper system

Design Notes:
    Unlike BlueROV which has explicit rotor joints, Hero Agent uses virtual thrusters.
    Thruster forces are computed via ThrusterModel and applied directly to the base_link
    through the allocation matrix. This approach is valid for RL training since
    the policy learns thruster commands, not individual rotor velocities.

    Hydrodynamic parameters are estimated and should be updated
    with experimentally identified values for accurate simulation.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from ..uuv_cfg import HydrodynamicsCfg, ThrusterCfg

# Path to Hero Agent USD file
_HERO_AGENT_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Agent")
HERO_AGENT_USD_PATH = os.path.join(_HERO_AGENT_ASSETS_DIR, "Agent.usd")


@configclass
class HeroAgentHydrodynamicsCfg(HydrodynamicsCfg):
    """Hydrodynamic parameters for Hero Agent.

    Note:
        These are estimated values based on vehicle dimensions.
        Should be replaced with experimentally identified parameters.
    """

    # Estimated added mass coefficients [surge, sway, heave, roll, pitch, yaw]
    # Larger vehicle = larger added mass
    added_mass: tuple[float, ...] = (8.0, 15.0, 18.0, 0.2, 0.2, 0.2)

    # Estimated linear damping coefficients
    linear_damping: tuple[float, ...] = (6.0, 8.0, 7.0, 0.1, 0.1, 0.1)

    # Estimated quadratic damping coefficients
    quadratic_damping: tuple[float, ...] = (25.0, 30.0, 45.0, 2.0, 2.0, 2.0)

    # Vehicle mass (kg) - None for neutral buoyancy assumption
    vehicle_mass: float | None = None

    # Estimated volume (m^3) - larger than BlueROV
    volume: float = 0.015

    # Center of buoyancy position in body frame (m)
    center_of_buoyancy: tuple[float, float, float] = (0.0, 0.0, 0.02)

    # Center of gravity at body frame origin
    center_of_gravity: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Freshwater density
    water_density: float = 997.0

    # Enable full Coriolis matrix
    use_full_coriolis: bool = True


@configclass
class HeroAgentThrusterCfg(ThrusterCfg):
    """Thruster configuration for Hero Agent.

    Note:
        Thruster layout and allocation matrix should be updated
        based on actual vehicle design.
    """

    num_thrusters: int = 6
    max_thrust: float = 60.0  # Larger thrusters for bigger vehicle
    thrust_coefficient: float = 50.0
    time_constant_up: float = 0.1
    time_constant_down: float = 0.05

    # Estimated allocation matrix (similar to BlueROV layout)
    # Should be updated based on actual thruster positions
    allocation_matrix: tuple[tuple[float, ...], ...] = (
        # Fx: forward surge force
        (0.707, 0.707, -0.707, -0.707, 0.0, 0.0),
        # Fy: lateral sway force
        (-0.707, 0.707, 0.707, -0.707, 0.0, 0.0),
        # Fz: vertical heave force
        (0.0, 0.0, 0.0, 0.0, 1.0, 1.0),
        # Mx: roll torque
        (0.0, 0.0, 0.0, 0.0, 0.12, -0.12),
        # My: pitch torque
        (0.0, 0.0, 0.0, 0.0, 0.15, 0.15),
        # Mz: yaw torque
        (0.22, -0.22, 0.22, -0.22, 0.0, 0.0),
    )


HERO_AGENT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=HERO_AGENT_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,  # Gravity handled by hydrodynamics model
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
        joint_pos={},
        joint_vel={},
    ),
    actuators={
        # Arm actuators (if any joints exist)
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint.*"],
            stiffness=100.0,
            damping=10.0,
        ),
    },
)
"""Configuration of Hero Agent underwater vehicle."""
