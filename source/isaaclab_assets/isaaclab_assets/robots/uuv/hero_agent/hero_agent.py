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
_HERO_AGENT_MESHES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meshes")
HERO_AGENT_USD_PATH = os.path.join(_HERO_AGENT_MESHES_DIR, "Agent.usd")


@configclass
class HeroAgentHydrodynamicsCfg(HydrodynamicsCfg):
    """Hydrodynamic parameters for Hero Agent main body.

    Based on heroagent2.py (IsaacGym) parameters:
        Main body: cylinder R=0.0825m, L=0.27m, m=9.18kg
        rho = 998 kg/m^3

    Buoyancy balance (from heroagent2.py):
        Base buoyancy: 7.88 * 9.81 = 77.3 N (volume ~0.00789 m^3)
        Base weight:   9.18 * 9.81 = 90.1 N
        Base net: -12.8 N (sinks)

    Drag coefficients from heroagent2.py:
        D_x = D_y = 1.17, D_z = 1.0
        A_x = A_y = 0.0825 * 2 * 0.27 = 0.04455 m^2
        A_z = pi * 0.0825^2 = 0.0214 m^2
        Quad drag X/Y: 0.5 * 998 * 1.17 * 0.04455 = 26.0 Ns^2/m^2
        Quad drag Z:   0.5 * 998 * 1.0 * 0.0214 = 10.7 Ns^2/m^2

    Reference:
        heroagent2.py (IsaacGym implementation)
    """

    # Added mass coefficients [surge, sway, heave, roll, pitch, yaw]
    # Based on R=0.0825m, L=0.27m geometry
    # Lateral: M_a = rho * pi * r^2 * L = 998 * pi * 0.0825^2 * 0.27 = 5.76 kg
    added_mass: tuple[float, ...] = (0.6, 5.76, 5.76, 0.04, 0.05, 0.05)

    # Linear damping coefficients (skin friction, Ns/m and Nms/rad)
    linear_damping: tuple[float, ...] = (2.0, 4.0, 4.0, 0.1, 0.1, 0.1)

    # Quadratic damping coefficients (form drag, Ns^2/m^2 and Nms^2/rad^2)
    # From heroagent2.py: D = 0.5 * rho * Cd * A
    #   X/Y: 0.5 * 998 * 1.17 * 0.04455 = 26.0
    #   Z:   0.5 * 998 * 1.0 * 0.0214 = 10.7
    # Rotational damping from heroagent2.py empirical values
    quadratic_damping: tuple[float, ...] = (26.0, 26.0, 10.7, 1.5, 1.5, 0.01)

    # Vehicle mass from heroagent2.py (kg)
    vehicle_mass: float | None = 9.18

    # Volume for buoyancy = 7.88 kg equivalent (from heroagent2.py)
    # V = buoyancy_force / (rho * g) = 77.3 / (998 * 9.81) = 0.00789 m^3
    volume: float = 0.00789

    # Center of buoyancy at geometric center of cylinder (body frame)
    center_of_buoyancy: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Center of gravity below CoB for passive stability
    center_of_gravity: tuple[float, float, float] = (0.0, 0.0, -0.10)

    # Freshwater density from heroagent2.py (kg/m^3)
    water_density: float = 998.0

    # Enable full Coriolis matrix (C_RB + C_A per Fossen model)
    use_full_coriolis: bool = True


@configclass
class HeroAgentBuoyHydrodynamicsCfg(HydrodynamicsCfg):
    """Hydrodynamic parameters for Hero Agent buoy (link3 / ABPC).

    Based on heroagent2.py (IsaacGym) parameters:
        ABPC buoyancy: 1.55 * 9.81 = 15.2 N (slightly increased for positive buoyancy)
        ABPC mass: 0.18 kg (adjusted for slight positive buoyancy)
        rho = 998 kg/m^3

    Buoyancy balance:
        ABPC buoyancy: 15.2 N
        ABPC weight:   0.18 * 9.81 = 1.8 N
        ABPC net: +13.4 N (floats)

    System total (with main body):
        Main body net: -12.8 N
        ABPC net:      +13.4 N
        System net:    +0.6 N (slightly positive buoyancy)

    Drag from heroagent2.py:
        A_x_abpc = 0.1 * 2 * 0.065 = 0.013 m^2
        Uses 0.3 coefficient factor (reduced drag for ABPC)
        Quad drag: 0.3 * 998 * 1.17 * 0.013 = 4.6 Ns^2/m^2
    """

    # Added mass coefficients [surge, sway, heave, roll, pitch, yaw]
    added_mass: tuple[float, ...] = (0.15, 1.5, 1.5, 0.01, 0.01, 0.01)

    # Linear damping coefficients (skin friction, Ns/m)
    linear_damping: tuple[float, ...] = (0.5, 0.5, 0.5, 0.01, 0.01, 0.01)

    # Quadratic damping coefficients (form drag, Ns^2/m^2)
    # From heroagent2.py: 0.3 * rho * Cd * A = 0.3 * 998 * 1.17 * 0.013 = 4.6
    quadratic_damping: tuple[float, ...] = (4.6, 4.6, 4.6, 0.1, 0.1, 0.1)

    # ABPC mass (kg) - adjusted for slight positive buoyancy
    vehicle_mass: float | None = 0.18

    # Volume for buoyancy = 1.55 kg equivalent (slightly > heroagent2.py's 1.51)
    # V = 15.2 / (998 * 9.81) = 0.00155 m^3
    volume: float = 0.00155

    # Center of buoyancy at geometric center
    center_of_buoyancy: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Center of gravity at body frame origin
    center_of_gravity: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Freshwater density from heroagent2.py (kg/m^3)
    water_density: float = 998.0

    # Disable full Coriolis for small body (negligible effect)
    use_full_coriolis: bool = False


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
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=8,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 2.0),  # Start 2m above ground (underwater)
        rot=(1.0, 0.0, 0.0, 0.0),  # Upright orientation
        joint_pos={},
        joint_vel={},
    ),
    actuators={
        # Arm actuators - increased stiffness/damping for stability under hydrodynamic forces
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint.*"],
            stiffness=400.0,  # Fixed: was 100.0, increased for external force resistance
            damping=40.0,     # Fixed: was 10.0, increased for oscillation damping
        ),
    },
)
"""Configuration of Hero Agent underwater vehicle."""
