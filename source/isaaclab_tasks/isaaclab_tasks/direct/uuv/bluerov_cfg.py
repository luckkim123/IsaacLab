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
from .uuv_env_cfg import DomainRandomizationCfg, ThrusterCfg, UUVEnvCfg


@configclass
class BlueROVHydrodynamicsCfg(HydrodynamicsCfg):
    """Hydrodynamic parameters for BlueROV2.

    Parameters are from experimental identification (MarineGym, IROS 2025).
    All values are taken from BlueROV.yaml in the MarineGym repository.

    BlueROV2 Heavy specifications:
        - Air weight: ~11.5 kg
        - Displacement volume: 11.35 L (0.0113459 m^3)
        - Designed for neutral buoyancy in freshwater
    """

    # Added mass coefficients [surge, sway, heave, roll, pitch, yaw]
    added_mass: tuple[float, ...] = (5.5, 12.7, 14.57, 0.12, 0.12, 0.12)

    # Linear damping coefficients
    linear_damping: tuple[float, ...] = (4.03, 6.22, 5.18, 0.07, 0.07, 0.07)

    # Quadratic damping coefficients
    quadratic_damping: tuple[float, ...] = (18.18, 21.66, 36.99, 1.55, 1.55, 1.55)

    # Vehicle mass (kg) - BlueROV2 Heavy air weight
    # Set to None for neutral buoyancy assumption (mass = volume * water_density)
    vehicle_mass: float | None = None

    # Vehicle volume (m^3) - from BlueROV2 specifications (11.35 L displacement)
    volume: float = 0.0113459

    # Center of buoyancy position in body frame (m, [x, y, z])
    # Positive z means CoB above CoG (provides passive stability)
    center_of_buoyancy: tuple[float, float, float] = (0.0, 0.0, 0.01)

    # Center of gravity position in body frame (m, [x, y, z])
    # Assumed at body frame origin
    center_of_gravity: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Water density (kg/m^3) - freshwater
    water_density: float = 997.0

    # Enable full Coriolis matrix computation C(v) = C_RB(v) + C_A(v)
    use_full_coriolis: bool = True

    # Rigid body inertia for full Coriolis (kg*m^2, [I_xx, I_yy, I_zz])
    # Estimated from added mass rotational terms * 0.5 (default if None)
    rigid_body_inertia: tuple[float, float, float] | None = None


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
    """BlueROV environment with ocean current disturbances and domain randomization.

    This configuration adds random ocean currents and full domain randomization
    for robust policy training that can handle real-world variations.
    """

    ocean_current: OceanCurrentCfg = OceanCurrentCfg(
        max_velocity=(0.3, 0.3, 0.1, 0.0, 0.0, 0.0),  # m/s
        noise_scale=(0.1, 0.1, 0.05, 0.0, 0.0, 0.0),
    )

    # Domain randomization for robust training
    randomization: DomainRandomizationCfg = DomainRandomizationCfg(
        enable=True,
        # Initial pose randomization
        position_x_range=(-2.5, 2.5),
        position_y_range=(-2.5, 2.5),
        position_z_range=(1.5, 2.5),
        roll_range=(-0.628, 0.628),
        pitch_range=(-0.628, 0.628),
        yaw_range=(0.0, 6.283),
        # Hydrodynamic parameter randomization
        added_mass_scale=(0.8, 1.2),
        linear_damping_scale=(0.8, 1.2),
        quadratic_damping_scale=(0.8, 1.2),
        volume_scale=(0.95, 1.05),
        mass_scale=(0.9, 1.1),
        # Thruster randomization
        thrust_coefficient_scale=(0.9, 1.1),
        time_constant_scale=(0.9, 1.1),
    )

    # Slightly harder task with currents - need stronger position signal
    position_reward_scale: float = 20.0
    position_reward_exp_scale: float = 2.5  # Even larger scale for current disturbance
    orientation_reward_scale: float = 8.0  # Strong orientation control against currents
    orientation_exp_scale: float = 0.4  # Stricter than base (0.5)
    linear_velocity_penalty_scale: float = -0.01


@configclass
class BlueROVTrainEnvCfg(BlueROVEnvCfg):
    """BlueROV training environment with domain randomization.

    Training mode: Randomization enabled for robust policy learning.
    Domain randomization during training helps sim-to-real transfer.
    """

    # Enable domain randomization for robust training
    randomization: DomainRandomizationCfg = DomainRandomizationCfg(
        enable=True,
        # Initial pose randomization
        position_x_range=(-2.5, 2.5),
        position_y_range=(-2.5, 2.5),
        position_z_range=(1.5, 2.5),
        roll_range=(-0.628, 0.628),  # ±36 degrees
        pitch_range=(-0.628, 0.628),  # ±36 degrees
        yaw_range=(0.0, 6.283),  # 0-360 degrees
        # Hydrodynamic parameter randomization
        added_mass_scale=(0.8, 1.2),
        linear_damping_scale=(0.8, 1.2),
        quadratic_damping_scale=(0.8, 1.2),
        volume_scale=(0.95, 1.05),
        mass_scale=(0.9, 1.1),
        # Thruster randomization
        thrust_coefficient_scale=(0.9, 1.1),
        time_constant_scale=(0.9, 1.1),
    )


@configclass
class BlueROVEvalEnvCfg(BlueROVEnvCfg):
    """BlueROV evaluation environment with extreme domain randomization.

    Evaluation mode: More aggressive randomization than training to stress-test
    the learned policy's robustness and generalization capability.

    Uses wider randomization ranges than training:
    - Hydrodynamics: 0.5-1.5x (vs 0.8-1.2x in training)
    - Thrusters: 0.7-1.3x (vs 0.9-1.1x in training)
    - Stronger ocean currents
    """

    # Aggressive domain randomization for stress testing
    randomization: DomainRandomizationCfg = DomainRandomizationCfg(
        enable=True,
        # Initial pose randomization
        position_x_range=(-3.0, 3.0),
        position_y_range=(-3.0, 3.0),
        position_z_range=(1.0, 3.0),
        roll_range=(-0.785, 0.785),  # ±45 degrees (wider than training)
        pitch_range=(-0.785, 0.785),  # ±45 degrees
        yaw_range=(0.0, 6.283),  # 0-360 degrees
        # Wider hydrodynamic parameter randomization
        added_mass_scale=(0.5, 1.5),
        linear_damping_scale=(0.5, 1.5),
        quadratic_damping_scale=(0.5, 1.5),
        volume_scale=(0.85, 1.15),
        mass_scale=(0.7, 1.3),
        # Wider thruster randomization
        thrust_coefficient_scale=(0.7, 1.3),
        time_constant_scale=(0.7, 1.3),
    )

    # Stronger ocean currents for evaluation
    ocean_current: OceanCurrentCfg = OceanCurrentCfg(
        max_velocity=(0.5, 0.5, 0.2, 0.0, 0.0, 0.0),  # Stronger than training
        noise_scale=(0.15, 0.15, 0.1, 0.0, 0.0, 0.0),
    )
