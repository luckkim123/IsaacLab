# Changelog

All notable changes to the UUV (Underwater Vehicle) environment module.

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
- BlueROV2 USD model from MarineGym (`/workspace/marinegym/marinegym/robots/assets/usd/BlueROV/BlueROV.usd`)

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
