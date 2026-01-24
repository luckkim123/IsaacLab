# Adding a New Underwater Vehicle to Isaac Lab UUV

This guide explains how to add a new underwater vehicle (e.g., Neo11, Firefly, custom AUV)
to the Isaac Lab UUV framework.

## Prerequisites

1. USD model of your vehicle (`.usd` file)
2. Hydrodynamic parameters (from CFD, experiments, or literature)
3. Thruster configuration (positions, orientations, allocation matrix)

## Step-by-Step Guide

### Step 1: Create Asset Directory

Create a directory for your vehicle under `assets/`:

```
uuv/
└── assets/
    └── YourVehicle/
        ├── YourVehicle.usd      # USD model
        ├── YourVehicle.yaml     # Hydrodynamic parameters
        └── config.yaml          # USD conversion config (optional)
```

### Step 2: Create YAML Configuration

Create `YourVehicle.yaml` with hydrodynamic coefficients:

```yaml
name: YourVehicle

drag_coef: 0.3  # Drag coefficient
volume: 0.01    # Displacement volume (m^3)
coBM: 0.01      # Center of buoyancy offset (m, z-direction)

hydro_coef:
  added_mass:  # [surge, sway, heave, roll, pitch, yaw] (kg or kg*m^2)
  - 5.0
  - 10.0
  - 12.0
  - 0.1
  - 0.1
  - 0.1
  linear_damping:  # Ns/m or Ns*m/rad
  - 4.0
  - 6.0
  - 5.0
  - 0.05
  - 0.05
  - 0.05
  quadratic_damping:  # Ns^2/m^2 or Ns^2*m^2/rad^2
  - 15.0
  - 20.0
  - 30.0
  - 1.0
  - 1.0
  - 1.0

rotor_configuration:
  num_rotors: 6
  directions: [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
  time_constants: [0.43, 0.43, 0.43, 0.43, 0.43, 0.43]  # Thruster response time (s)
  force_constants: [4.4e-07, 4.4e-07, 4.4e-07, 4.4e-07, 4.4e-07, 4.4e-07]  # N/(RPM)^2
  max_rotation_velocities: [3900, 3900, 3900, 3900, 3900, 3900]  # Max RPM
  moment_constants: [1.37e-09, 1.37e-09, 1.37e-09, 1.37e-09, 1.37e-09, 1.37e-09]  # Nm/(RPM)^2
```

### Step 3: Create Configuration File

Create `yourvehicle_cfg.py` (use `bluerov_cfg.py` as reference):

```python
from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from .hydrodynamics_model import HydrodynamicsCfg, OceanCurrentCfg
from .uuv_env_cfg import DomainRandomizationCfg, ThrusterCfg, UUVEnvCfg


@configclass
class YourVehicleHydrodynamicsCfg(HydrodynamicsCfg):
    """Hydrodynamic parameters for YourVehicle.

    Provide source of parameters (CFD, experiments, literature).
    """
    added_mass: tuple[float, ...] = (5.0, 10.0, 12.0, 0.1, 0.1, 0.1)
    linear_damping: tuple[float, ...] = (4.0, 6.0, 5.0, 0.05, 0.05, 0.05)
    quadratic_damping: tuple[float, ...] = (15.0, 20.0, 30.0, 1.0, 1.0, 1.0)
    volume: float = 0.01  # m^3
    center_of_buoyancy: tuple[float, float, float] = (0.0, 0.0, 0.01)


# USD path
_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
YOURVEHICLE_USD_PATH = os.path.join(_ASSETS_DIR, "YourVehicle", "YourVehicle.usd")


YOURVEHICLE_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=YOURVEHICLE_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
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
        pos=(0.0, 0.0, 2.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={"rotor_.*": 0.0},
        joint_vel={"rotor_.*": 0.0},
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
class YourVehicleEnvCfg(UUVEnvCfg):
    """Environment configuration for YourVehicle."""

    robot: ArticulationCfg = YOURVEHICLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    hydrodynamics: YourVehicleHydrodynamicsCfg = YourVehicleHydrodynamicsCfg()

    # Customize thruster configuration
    thrusters: ThrusterCfg = ThrusterCfg(
        num_thrusters=6,  # Adjust for your vehicle
        max_thrust=50.0,
        allocation_matrix=(
            # Define your thruster allocation matrix
            (0.707, 0.707, -0.707, -0.707, 0.0, 0.0),  # Fx
            (0.707, -0.707, 0.707, -0.707, 0.0, 0.0),  # Fy
            (0.0, 0.0, 0.0, 0.0, 1.0, 1.0),            # Fz
            (0.0, 0.0, 0.0, 0.0, 0.12, -0.12),         # Mx
            (0.0, 0.0, 0.0, 0.0, -0.12, -0.12),        # My
            (-0.15, 0.15, 0.15, -0.15, 0.0, 0.0),      # Mz
        ),
    )

    action_space: int = 6  # Match num_thrusters
```

### Step 4: Register in `__init__.py`

Add to `uuv/__init__.py`:

```python
# Import your configuration
from .yourvehicle_cfg import (
    YOURVEHICLE_CFG,
    YourVehicleEnvCfg,
    YourVehicleHydrodynamicsCfg,
)

# Register environment
gym.register(
    id="Isaac-UUV-YourVehicle-v0",
    entry_point="isaaclab_tasks.direct.uuv:UUVEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.uuv:YourVehicleEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BlueROVPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)
```

### Step 5: (Optional) Create Agent Configuration

If your vehicle has significantly different dynamics, create custom PPO config:

```python
# In agents/rsl_rl_ppo_cfg.py
@configclass
class YourVehiclePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # Customize hyperparameters for your vehicle
    pass
```

## Thruster Allocation Matrix

The allocation matrix maps thruster forces to body wrench:

```
[Fx]   [a11 a12 ... a1n] [T1]
[Fy]   [a21 a22 ... a2n] [T2]
[Fz] = [a31 a32 ... a3n] [T3]
[Mx]   [a41 a42 ... a4n] [.]
[My]   [a51 a52 ... a5n] [.]
[Mz]   [a61 a62 ... a6n] [Tn]
```

Where:
- `Tx, Ty, Tz`: Force components in body frame
- `Mx, My, Mz`: Moment components in body frame
- `T1...Tn`: Individual thruster forces

For vectored thrusters at angle theta:
- Force contribution: `cos(theta)` in one axis, `sin(theta)` in another
- Moment arm: distance from CoM to thruster

## Hydrodynamic Parameter Sources

1. **Experimental identification**: Free-decay tests, tow-tank experiments
2. **CFD simulation**: Use Capytaine, OpenFOAM, or similar tools
3. **Literature**: Similar vehicles from research papers
4. **Estimation**: Empirical formulas (e.g., Eidsvik method, DNV standards)

### References

- Fossen (2011): Handbook of Marine Craft Hydrodynamics and Motion Control
- MarineGym (IROS 2025): GPU-accelerated underwater vehicle simulation
- Blue Robotics: T200 thruster specifications
