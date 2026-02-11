<div align="center">

# Hero Agent ALBC

**Thruster-free underwater vehicle attitude control via buoyancy manipulation in Isaac Lab.**

[![License][license-img]][license-url]
[![Python][python-img]][python-url]
[![Isaac Sim][isaacsim-img]][isaacsim-url]
[![PyTorch][pytorch-img]][pytorch-url]

[Getting Started](#getting-started) | [Environments](#available-environments) | [Contributing](#contributing)

</div>

## Overview

Hero Agent ALBC (Active Linear Buoyancy Controller) is a reinforcement learning environment for NVIDIA Isaac Lab that trains an underwater vehicle to stabilize its attitude without thrusters. Instead of conventional propulsion, the robot repositions a buoyancy element through a 2-link revolute arm (l1 = l2 = 0.233 m), generating restoring torques from buoyancy forces alone.

The package implements a multi-phase training pipeline combining classical Time Delay Control (TDC) with HORA/RMA-style online adaptation. An encoder compresses 24D privileged hydrodynamic information into a 6D latent; a temporal convolution adaptation module then reconstructs this latent from proprioception history alone, enabling sim-to-real transfer without access to simulator-internal parameters.

Domain randomization spans 15+ physical parameters (hydrodynamics, ocean currents, payloads, sensor noise) to bridge the sim-to-real gap.

## Key Features

- **Thruster-Free Control**: Attitude stabilization using only buoyancy manipulation through a 2-DOF revolute arm, eliminating thruster noise and energy consumption
- **Multi-Phase Training Pipeline**: Pure RL (PPO), classical TDC, encoder-TDC hybrid (HORA/RMA), and supervised adaptation -- each registered as a Gymnasium environment
- **Sim-to-Real Transfer**: Domain randomization across 15+ physical parameters with HORA encoder for online adaptation via proprioception history
- **Deployment Export**: JIT-scriptable module exports TorchScript/ONNX models with baked-in gain scaling for direct C++ TDC controller integration
- **GPU-Accelerated Simulation**: Runs 4096+ parallel environments on a single GPU via Isaac Lab and PhysX, with Fossen-model 6-DOF hydrodynamics

## Getting Started

### Prerequisites

- NVIDIA Isaac Sim 5.1.0+
- Isaac Lab (installed via `isaaclab.sh --install`)
- Python 3.10+
- NVIDIA GPU with 8GB+ VRAM

### Installation

This module is part of `isaaclab_tasks`. After installing Isaac Lab, it is available automatically:

```bash
cd /path/to/isaaclab
./isaaclab.sh --install
```

### Quick Start

Train a base RL policy for attitude stabilization:

```bash
# Train with domain randomization + ocean current + payload (4096 envs, 600 iters)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-HeroAgent-Base-v0 \
    --num_envs 4096 --max_iterations 600 \
    --headless --logger wandb

# Evaluate a trained checkpoint
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-HeroAgent-Base-v0
```

## Usage

### Available Environments

| Task ID | Obs / Priv / Act | Description |
|:---|:---:|:---|
| `Isaac-HeroAgent-v0` | 13 / -- / 2 | Debug (no DR, no ocean current) |
| `Isaac-HeroAgent-Base-v0` | 13 / -- / 2 | Base training (DR + ocean current + payload) |
| `Isaac-HeroAgent-Encoder-Base-v0` | 13 / 24 / 2 | HORA Phase 1 encoder training |
| `Isaac-HeroAgent-TDC-v0` | 13 / -- / 2 | Classical TDC control (no RL actions) |
| `Isaac-HeroAgent-Encoder-TDC-v0` | 13 / 24 / 4 | Encoder-TDC (RL adaptive gains + M_hat) |
| `Isaac-HeroAgent-Adapt-TDC-v0` | 13 / 24 / 4 | Phase 2 adaptation (proprio history) |

<details>
<summary><strong>Observation and Action Spaces</strong></summary>

**Policy Observations (13D)**:
```
[0:3]   roll, pitch, yaw (Euler angles)
[3:6]   angular velocity in body frame (p, q, r)
[6:9]   attitude error (target - current, wrapped to [-pi, pi])
[9:11]  joint positions (normalized to [-1, 1])
[11:13] previous actions
```

**Privileged Observations (24D)** -- visible only during training:
```
[0:10]  main body  (volume, r_cg(3), r_cb(3), inertia(3))
[10:20] buoy body  (volume, r_cg(3), r_cb(3), inertia(3))
[20:24] payload    (mass, cog_offset(3))
```

**Actions**:
- Base RL (2D): joint velocity commands [-1, 1]
- Encoder-TDC (4D): [Kp_roll, Kp_pitch, Kd_roll, Kd_pitch] via sigmoid scaling

</details>

<details>
<summary><strong>Multi-Phase Training Pipeline</strong></summary>

```
Phase 1: Encoder-TDC Teacher          Phase 2: Adaptation           Phase 3: Deploy

privileged (24D) --> Encoder --> z(6D)  proprio_hist (30x12D)         adapt_tconv
                     z[3:5] --> M_hat   --> adapt_tconv --> z_hat     + actor + TDC cfg
policy_obs (13D) + z --> Actor --> 4D   L2 loss: ||z_hat - z_gt||    --> TorchScript/ONNX
                         |                                            --> C++ TDC
                         v
                    Kp, Kd --> TDC --> joint targets
```

#### Phase 1: Encoder-TDC Teacher Training

Train the encoder to compress privileged hydrodynamic information (24D) into a 6D latent, while the actor learns adaptive TDC gains:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-HeroAgent-Encoder-TDC-v0 \
    --num_envs 4096 --max_iterations 600
```

#### Phase 2: Adaptation Module Training

Train a temporal convolution network to estimate the encoder latent from proprioception history only (no privileged info):

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/hero_agent/workflows/train_adaptation.py \
    --task Isaac-HeroAgent-Adapt-TDC-v0 \
    --phase1_checkpoint logs/rsl_rl/<encoder_tdc_run>/model_600.pt \
    --num_envs 4096
```

#### Phase 3: Deploy Export

Evaluate the complete pipeline and export TorchScript/ONNX models for C++ TDC controller integration:

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/hero_agent/workflows/play_phase3.py \
    --task Isaac-HeroAgent-Adapt-TDC-v0 \
    --checkpoint logs/rsl_rl/<adapt_run>/model_final.pt \
    --export-jit --export-onnx --headless
```

</details>

<details>
<summary><strong>Benchmarking</strong></summary>

Compare controllers across standardized scenarios (nominal / easy / hard / extreme) with controlled DR intensity, ocean current, and payload:

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/hero_agent/workflows/benchmark.py \
    --entries "Isaac-HeroAgent-TDC-v0:" \
              "Isaac-HeroAgent-Base-v0:logs/rsl_rl/<base_run>/model_600.pt" \
    --scenarios nominal hard \
    --num_episodes 50 --num_envs 256 \
    --output_dir benchmarks/run1 --headless
```

Entry format: `TASK_ID:CHECKPOINT_PATH` (empty path for TDC-v0 which needs no checkpoint). Results are saved as CSVs with per-episode and aggregated metrics. Optional `--logger wandb` for WandB logging.

| Scenario | DR Intensity | Ocean Current | Payload | Purpose |
|:---|:---:|:---:|:---:|:---|
| nominal | None | None | 0.5 kg fixed | Baseline stability |
| easy | +/-10% hydro | 0.1 m/s | 0-0.5 kg | Mild perturbation |
| hard | +/-30-50% hydro | 0.3 m/s | 0-1.0 kg | Training distribution |
| extreme | +/-60-80% hydro | 0.5 m/s | 0-2.0 kg | Out-of-distribution |

</details>

<details>
<summary><strong>Control Timing</strong></summary>

| Layer | Rate | Interval |
|:---|:---:|:---:|
| Physics (PhysX) | 200 Hz | 0.005 s |
| Policy (decimation=1) | 200 Hz | 0.005 s |
| TDC control (control_decimation=4) | 50 Hz | 0.02 s |
| Episode length | -- | 15 s (3000 steps) |

Actuators use `ImplicitActuatorCfg` (PhysX internal continuous PD), with stiffness and damping domain-randomized per episode.

</details>

<details>
<summary><strong>Adding a Custom Controller</strong></summary>

1. Create your controller in `controllers/` (e.g., `controllers/my_controller.py`)
2. Subclass `HeroAgentEnv` and override `_pre_physics_step()` to compute joint targets
3. Create a config inheriting from `HeroAgentEnvCfg` with controller-specific parameters
4. Register with `gym.register()` in `__init__.py`

Key integration points:
- **State**: `self._robot.data.root_quat_w`, `root_ang_vel_b`, `joint_pos`
- **Control**: `self._robot.set_joint_position_target(targets, joint_ids=self._albc_joint_ids)`
- **Kinematics**: `ALBCKinematics.inverse(ee_pos)` / `.forward(joint_angles)`

See `tdc_env.py` for a complete example.

</details>

## Project Structure

```
hero_agent/
├── __init__.py           # Gymnasium environment registration (6 tasks)
├── config.py             # All environment configuration classes
├── config_benchmark.py   # Benchmark scenario presets (nominal/easy/hard/extreme)
├── base_env.py           # Base RL environment (HeroAgentEnv)
├── tdc_env.py            # Classical TDC controller environment
├── encoder_tdc_env.py    # Encoder-TDC integration environment
├── adapt_tdc_env.py      # Phase 2 adaptation environment
├── controllers/          # TDC controller + 2-link arm kinematics (IK/FK)
├── encoder/              # HORA encoder networks + adaptation module
├── agents/               # RSL-RL PPO runner configurations
├── runners/              # Custom training runners (encoder, adaptation)
├── workflows/            # Multi-phase training, benchmarking + deploy export
├── deploy/               # JIT/ONNX export for C++ integration
├── mdp/                  # Observations, rewards, domain randomization events
├── utils/                # Debug visualization + episode logging
└── docs/                 # Architecture, TDC theory, dynamics, training, DR, sim-to-real
```

## Contributing

Contributions are welcome. This module follows Isaac Lab's coding conventions (Ruff formatting, 120-char line length, Google-style docstrings). See the Isaac Lab [contributing guidelines][contributing-url] for details.

## License

[BSD-3-Clause][license-url]

## Citation

```bibtex
@inproceedings{heroagent2026,
  title     = {Thruster-Free Underwater Vehicle Attitude Stabilization via Buoyancy Manipulation with Online Adaptation},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2026},
}
```

## Acknowledgements

Built on [Isaac Lab][isaaclab-url] (NVIDIA) for GPU-accelerated simulation, [RSL-RL][rslrl-url] for PPO training, and the [HORA][hora-url] framework for online robust adaptation.

<!-- ============================================ -->
<!-- REFERENCE LINKS                              -->
<!-- ============================================ -->
[license-img]: https://img.shields.io/badge/License-BSD--3--Clause-blue.svg
[license-url]: https://github.com/isaac-sim/IsaacLab/blob/main/LICENSE
[python-img]: https://img.shields.io/badge/Python-3.10%2B-blue.svg
[python-url]: https://www.python.org/downloads/
[isaacsim-img]: https://img.shields.io/badge/Isaac_Sim-5.1.0-76b900.svg
[isaacsim-url]: https://developer.nvidia.com/isaac-sim
[pytorch-img]: https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg
[pytorch-url]: https://pytorch.org/
[contributing-url]: https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTING.md
[isaaclab-url]: https://github.com/isaac-sim/IsaacLab
[rslrl-url]: https://github.com/leggedrobotics/rsl_rl
[hora-url]: https://github.com/HaozhiQi/hora
