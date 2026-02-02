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

    Physics Model:
        - PhysX gravity is ENABLED (disable_gravity=False)
        - Weight is handled by PhysX naturally for all bodies
        - This model applies ONLY buoyancy as external force
        - Buoyancy-weight difference determines net vertical force

    Buoyancy calculation (from heroagent2.py):
        Volume ~0.00789 m^3 -> Buoyancy = 998 * 9.81 * 0.00789 = 77.3 N

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

    # Volume for buoyancy calculation (m^3)
    # From URDF: Cylinder R=0.09m, L=0.325m -> V = pi * 0.09^2 * 0.325 = 0.00827 m^3
    # Buoyancy = 998 * 0.00827 * 9.81 = 80.9 N
    volume: float = 0.00827

    # Body name (for reference, not used when volume is explicitly set)
    body_name: str = "base"

    # Center of buoyancy at geometric center of cylinder (body frame)
    center_of_buoyancy: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Center of gravity below CoB for passive stability
    # Note: This is for restoring moment calculation. Actual CoG is from PhysX.
    center_of_gravity: tuple[float, float, float] = (0.0, 0.0, -0.10)

    # Freshwater density from heroagent2.py (kg/m^3)
    water_density: float = 998.0

    # Rigid body inertia [I_xx, I_yy, I_zz] (kg*m^2)
    # Solid cylinder approximation: R=0.0825m, L=0.27m, m=9.18kg
    #   I_xx = I_yy = m*(3*R^2 + L^2)/12 = 9.18*(0.0204 + 0.0729)/12 = 0.071
    #   I_zz = m*R^2/2 = 9.18*0.006806/2 = 0.031
    rigid_body_inertia: tuple[float, float, float] | None = (0.071, 0.071, 0.031)

    # Enable full Coriolis matrix (C_RB + C_A per Fossen model)
    use_full_coriolis: bool = True


@configclass
class HeroAgentBuoyHydrodynamicsCfg(HydrodynamicsCfg):
    """Hydrodynamic parameters for Hero Agent buoy (link3 / ABPC).

    Based on heroagent2.py (IsaacGym) parameters:
        ABPC buoyancy: 1.55 * 9.81 = 15.2 N (slightly increased for positive buoyancy)
        ABPC mass: 0.18 kg (adjusted for slight positive buoyancy)
        rho = 998 kg/m^3

    Physics Model:
        - PhysX gravity is ENABLED (disable_gravity=False)
        - Weight is handled by PhysX naturally
        - This model applies ONLY buoyancy as external force

    Buoyancy calculation:
        Volume ~0.00155 m^3 -> Buoyancy = 998 * 9.81 * 0.00155 = 15.2 N

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

    # Volume for buoyancy calculation (m^3)
    # From URDF: Cylinder R=0.085m, H=0.09m -> V = pi * 0.085^2 * 0.09 = 0.00204 m^3
    # Buoyancy = 998 * 0.00204 * 9.81 = 20.0 N (link3 mass=0.93kg -> weight=9.1N -> net=+10.9N)
    volume: float = 0.00204

    # Body name (for reference, not used when volume is explicitly set)
    body_name: str = "link3"

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


# ALBC (Active Linear Buoyancy Controller) joint names
HERO_AGENT_ALBC_JOINT_NAMES: list[str] = ["joint1", "joint2"]


HERO_AGENT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=HERO_AGENT_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,  # PhysX handles gravity; HydrodynamicsModel applies buoyancy only
            max_depenetration_velocity=10.0,  # Limits separation speed when bodies overlap
            enable_gyroscopic_forces=True,  # Enables gyroscopic torque from rotational inertia
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
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint.*"],
            stiffness=100.0,
            damping=40.0,
        ),
    },
)
"""Configuration of Hero Agent underwater vehicle."""
