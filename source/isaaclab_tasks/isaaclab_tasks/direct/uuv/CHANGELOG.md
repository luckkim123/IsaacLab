# Changelog

All notable changes to the UUV (Underwater Vehicle) environment module.

## [0.7.0] - 2026-01-25

### Added
- **Full Fossen model compliance**: Improved hydrodynamics to match Fossen textbook formulation
  - Weight-buoyancy difference (W-B) for non-neutral buoyancy vehicles
  - 3D center of buoyancy and center of gravity vectors
  - Full Coriolis matrix C(v) = C_RB(v) + C_A(v) (rigid body + added mass)
  - Configurable rigid body inertia for accurate C_RB computation

- **Quaternion-based buoyancy calculation**: Eliminates gimbal lock issues
  - `_compute_buoyancy_quat()`: Uses quaternion rotation instead of Euler angles
  - Proper separation of weight force (at CoG) and buoyancy force (at CoB)
  - Moment arms computed from force application points

- **Vehicle mass support**: Configurable mass for W != B scenarios
  - `vehicle_mass` config parameter (defaults to neutral buoyancy if None)
  - `robot_mass` parameter in `__init__` for physics engine integration
  - Mass randomization in domain randomization (`mass_scale` parameter)

### Changed
- **HydrodynamicsCfg**: New fields for complete Fossen model
  - `vehicle_mass: float | None` - explicit mass (None = neutral buoyancy)
  - `center_of_buoyancy: tuple[float, float, float]` - 3D CoB position
  - `center_of_gravity: tuple[float, float, float]` - 3D CoG position
  - `use_full_coriolis: bool` - enable C_RB + C_A (default: True)
  - `rigid_body_inertia: tuple[float, float, float] | None` - I_xx, I_yy, I_zz
  - `center_of_buoyancy_offset` deprecated (use `center_of_buoyancy` instead)

- **HydrodynamicsModel**: Enhanced initialization and computation
  - Accepts `robot_mass` from physics engine
  - Priority: cfg.vehicle_mass > robot_mass > neutral buoyancy
  - `randomize_parameters()`: Added `mass_scale` for mass randomization

- **BlueROVHydrodynamicsCfg**: Updated with new config fields
  - Explicit 3D center_of_buoyancy and center_of_gravity
  - Enabled full Coriolis by default
  - Added BlueROV2 Heavy specifications in docstring

- **UUVEnv**: Passes robot mass to HydrodynamicsModel
  - `_robot_mass` extracted before hydro model init
  - Domain randomization now includes mass_scale

### Technical Notes
- Full Coriolis: C_RB uses skew-symmetric matrices for angular momentum coupling
- Buoyancy: Gravity direction rotated to body frame via quat_apply_inverse
- Backward compatible: center_of_buoyancy_offset still works (deprecated)

## [0.6.0] - 2026-01-25

### Changed
- **Domain randomization strategy**: Fixed training/eval randomization logic
  - `BlueROVTrainEnvCfg`: Now ENABLED (was disabled) - robust policy learning
  - `BlueROVCurrentEnvCfg`: Now ENABLED with currents - full training setup
  - `BlueROVEvalEnvCfg`: More aggressive ranges for stress testing
- **Eval environment**: Wider randomization ranges than training for robustness testing
  - Hydrodynamics: 0.5-1.5x (vs 0.8-1.2x in training)
  - Thrusters: 0.7-1.3x (vs 0.9-1.1x in training)
  - Orientation: ±45° (vs ±36° in training)
  - Stronger ocean currents: 0.5 m/s (vs 0.3 m/s)

## [0.5.3] - 2026-01-24

### Added
- **Ocean current visualization**: Bright yellow arrow showing current direction and magnitude
  - Arrow positioned 1.0m above robot for visibility
  - Arrow length proportional to current magnitude (1.0m - 4.0m)
  - Thick arrow (0.4 scale) with emissive yellow color
  - `_direction_to_quat()` helper for direction vector to quaternion conversion
  - `get_ocean_current()` and `get_ocean_current_info()` API methods

### Changed
- `_set_debug_vis_impl()`: Now creates both goal position (red cube) and current (yellow arrow) markers
- `_debug_vis_callback()`: Calls `_visualize_ocean_current()` in addition to goal visualization

## [0.5.2] - 2026-01-24

### Fixed
- **experiment_name consistency**: Unified `experiment_name` to `"bluerov_direct"` across all RSL-RL configs
  - **Root cause**: Train/Eval/Current variants had different experiment_names, causing checkpoint path mismatch
  - **Problem**: Training saved to `logs/rsl_rl/bluerov_train/`, evaluation looked in `logs/rsl_rl/bluerov_hover/`
  - **Solution**: All variants now inherit `experiment_name = "bluerov_direct"` from base config
  - Follows Isaac Lab convention (e.g., `cartpole_direct`, `quadcopter_direct`)

### Changed
- **RSL-RL config structure**: Removed `experiment_name` overrides from derived classes
  - `BlueROVTrainPPORunnerCfg`, `BlueROVEvalPPORunnerCfg`, `BlueROVCurrentPPORunnerCfg` now inherit from base
  - Ensures checkpoint compatibility across all environment variants

## [0.5.1] - 2026-01-24

### Fixed
- **Deprecated function replacement**: Replaced `quat_rotate_inverse` with `quat_apply_inverse`
  - Affects `uuv_env.py` and `hydrodynamics_model.py`
  - Fixes deprecation warnings during simulation

### Changed
- **PhysX external forces**: Enabled `enable_external_forces_every_iteration=True`
  - Important for accurate hydrodynamic force integration in underwater vehicles
  - Removes the warning about noisy velocities

## [0.5.0] - 2026-01-24

### Fixed
- **Damping coupling implementation**: Fixed to match MarineGym's approach
  - Now uses velocity values in off-diagonal positions (not damping coefficients)
  - `maintained_body_vels` matrix follows MarineGym pattern exactly
- **Thruster time constant**: Corrected from 0.01s to 0.43s
  - 0.01s was RPM filter time constant, not throttle dynamics
  - 0.43s matches T200 model's `tau_up`/`tau_down` in MarineGym

### Changed
- **Code structure improvements**:
  - Moved `BlueROVHydrodynamicsCfg` from `uuv_env_cfg.py` to `bluerov_cfg.py`
  - Generic `UUVEnvCfg` now uses base `HydrodynamicsCfg` as default
  - Better separation between generic UUV code and BlueROV-specific code
- **Configurable body link name**: Added `body_link_name` to `UUVEnvCfg`
  - Previously hardcoded as "base_link"
  - Now configurable for different robot models
- **Removed unused import**: Removed `MISSING` from `hydrodynamics_model.py`

### Verified (No changes needed)
- **Coriolis force calculation**: Confirmed identical to MarineGym implementation
  - Both use `-(M_A * v_lin) x omega` formulation
  - This is a valid simplification for diagonal added mass matrices

## [0.4.0] - 2026-01-24

### Fixed
- **Thruster allocation**: Replaced hardcoded allocation coefficients (0.707, 0.1) with
  configurable allocation matrix in `ThrusterCfg`
- **Thruster time constant**: Fixed from 0.15s (initially set incorrectly)
- **Unused config parameters**: Now all config parameters are actually used in code
  - `max_thrust`: Applied as clamp in `_apply_action()`
  - `time_constant_scale`: Applied in domain randomization
  - `action_magnitude_penalty_scale`: Added to reward calculation
- **Hardcoded USD path**: Replaced absolute path with relative path using `__file__`

### Added
- **BlueROV USD assets**: Copied to `assets/BlueROV/` directory (no external dependency)
- **Thruster allocation matrix** in `ThrusterCfg`:
  - Configurable 6x6 allocation matrix mapping thruster commands to body wrench
  - Default values for BlueROV2 Heavy with 45-degree vectored horizontal thrusters
  - Thruster arm length parameters (`arm_length_x`, `arm_length_y`, `arm_length_xy`)
- **Time constant randomization**: `_randomized_time_constant_up/down` buffers
- **Action magnitude penalty**: Added to reward function and episode logging

### Changed
- `bluerov_cfg.py`: USD path now uses `os.path.dirname(__file__)` for portability
- `_apply_action()`: Now uses matrix multiplication with allocation matrix
- `_pre_physics_step()`: Supports per-environment time constant randomization
- `_get_rewards()`: Includes `action_magnitude_penalty` term
- `_reset_idx()`: Randomizes time constants when domain randomization enabled

## [0.3.0] - 2026-01-24

### Added
- **RL Agent Configurations** for training underwater vehicle control policies
  - `agents/rsl_rl_ppo_cfg.py`: RSL-RL PPO configurations
    - `BlueROVPPORunnerCfg`: Base configuration with [128,128,64] networks
    - `BlueROVTrainPPORunnerCfg`: Training mode (300 iterations, no randomization)
    - `BlueROVEvalPPORunnerCfg`: Evaluation mode (800 iterations, higher entropy)
    - `BlueROVCurrentPPORunnerCfg`: Ocean current disturbances (600 iterations)
  - `agents/rl_games_ppo_cfg.yaml`: RL-Games PPO configuration
  - `agents/skrl_ppo_cfg.yaml`: SKRL PPO configuration
- Registered RL configurations with Gymnasium environments

### Changed
- Updated `__init__.py` to include `rl_games_cfg_entry_point`, `rsl_rl_cfg_entry_point`, and `skrl_cfg_entry_point` for all environments

## [0.2.0] - 2026-01-24

### Added
- **Domain Randomization** support following MarineGym patterns
  - `DomainRandomizationCfg` class in `uuv_env_cfg.py`
    - Initial position randomization: XY +/-2.5m, Z 1.5-2.5m
    - Initial orientation randomization: Roll/Pitch +/-36 deg, Yaw 0-360 deg
    - Hydrodynamic parameter scales: added_mass/damping 0.5-1.0x, volume 0.9-1.1x
    - Thruster coefficient scale: 0.8-1.2x
  - `randomize_parameters()` method in `HydrodynamicsModel`
  - Per-environment randomization buffers for thrust coefficients
- **New Environment Configurations**
  - `BlueROVTrainEnvCfg`: Training mode with randomization disabled
  - `BlueROVEvalEnvCfg`: Evaluation mode with full randomization + ocean currents
- **New Gymnasium Environments**
  - `Isaac-UUV-BlueROV-Train-v0`: Deterministic training environment
  - `Isaac-UUV-BlueROV-Eval-v0`: Randomized evaluation environment

### Changed
- Modified `_reset_idx()` in `UUVEnv` to apply domain randomization on reset
- Modified `_apply_action()` to use per-environment thrust coefficients

## [0.1.0] - 2026-01-24

### Added
- **Initial UUV Environment Implementation**
  - `UUVEnv` class extending `DirectRLEnv` for underwater vehicle control
  - `UUVEnvCfg` configuration class with all environment parameters

- **Fossen Model Hydrodynamics** (`hydrodynamics_model.py`)
  - `HydrodynamicsModel` class computing 6-DOF hydrodynamic forces
  - Added mass effects (diagonal 6x6 matrix)
  - Linear and quadratic damping
  - Coriolis and centripetal forces
  - Buoyancy with restoring moments
  - `HydrodynamicsCfg` and `OceanCurrentCfg` configuration classes

- **BlueROV2 Robot Configuration** (`bluerov_cfg.py`)
  - `BLUEROV_CFG`: Articulation configuration for BlueROV2 USD model
  - `BlueROVEnvCfg`: Base environment configuration
  - `BlueROVCurrentEnvCfg`: Environment with ocean current disturbances
  - `BlueROVHydrodynamicsCfg`: Experimentally-identified parameters from MarineGym

- **Thruster Model** (`uuv_env_cfg.py`)
  - `ThrusterCfg` with T200 thruster parameters
  - First-order dynamics with configurable time constants
  - Thrust allocation matrix support

- **Observation Space** (18 dimensions)
  - Position (3): Robot position in world frame
  - Orientation (4): Quaternion
  - Linear velocity (3): Body frame
  - Angular velocity (3): Body frame
  - Goal position (3): Relative to body frame
  - Up vector (2): Projected gravity direction

- **Action Space** (6 dimensions)
  - 6 thruster commands normalized to [-1, 1]

- **Reward Function**
  - Position tracking reward (exponential)
  - Orientation reward (upright bonus)
  - Velocity penalties (linear and angular)
  - Action penalties (rate and magnitude)
  - Alive bonus

- **Gymnasium Environments**
  - `Isaac-UUV-BlueROV-v0`: Basic hover task
  - `Isaac-UUV-BlueROV-Current-v0`: Hover with ocean currents

### Dependencies
- Isaac Lab framework
- BlueROV2 USD model (included in `assets/BlueROV/`, originally from MarineGym)

---

## File Structure

```
isaaclab_tasks/direct/uuv/
├── __init__.py              # Module exports and Gym registration
├── CHANGELOG.md             # This file
├── uuv_env.py               # Main environment class
├── uuv_env_cfg.py           # Environment configuration
├── hydrodynamics_model.py   # Fossen model implementation
├── bluerov_cfg.py           # BlueROV2 robot configuration
└── agents/
    ├── __init__.py          # Agent config exports
    ├── rsl_rl_ppo_cfg.py    # RSL-RL PPO configurations
    ├── rl_games_ppo_cfg.yaml # RL-Games configuration
    └── skrl_ppo_cfg.yaml    # SKRL configuration
```

## Usage

```bash
# Test environment
./isaaclab.sh -p scripts/environments/random_agent.py --task Isaac-UUV-BlueROV-v0

# Train with RSL-RL
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-UUV-BlueROV-Train-v0

# Train with domain randomization
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-UUV-BlueROV-Eval-v0
```

## References

- Fossen, T.I. (2011). Handbook of Marine Craft Hydrodynamics and Motion Control
- MarineGym: GPU-Accelerated Underwater Vehicle Simulation (IROS 2025)
- BlueROV2 specifications: https://bluerobotics.com/store/rov/bluerov2/
