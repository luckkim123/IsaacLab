# Changelog

All notable changes to the UUV (Underwater Vehicle) environment module.

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
