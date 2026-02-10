# Hero Agent ALBC - Architecture Reference

## Overview

Hero Agent is an underwater vehicle (UUV) that controls its attitude (roll/pitch) **without thrusters**.
Instead, it uses 2 revolute joints (joint1, joint2) to reposition a buoyancy element (buoy).
The buoy's buoyancy force creates restoring torques that stabilize the vehicle.

```
[Main Body]  ── joint1 ── [Link1] ── joint2 ── [Link2 / Buoy]
 (9.18 kg)                (0.233m)              (0.233m, 0.93kg)
 Buoyancy: 80.9N                                Buoyancy: 26.2N
 Net: slightly negative                         Net: strongly positive (+17.1N)
```

System net buoyancy: ~+3N (slight positive buoyancy for passive stability).

---

## Directory Structure

```
hero_agent/
├── __init__.py                 # Gym environment registration
├── hero_agent_env.py           # Main environment class (HeroAgentEnv)
├── hero_agent_env_cfg.py       # Configuration classes
├── agents/                     # RL agent configs (PPO runner, network arch)
│   └── rsl_rl_ppo_cfg.py
├── controllers/
│   ├── __init__.py
│   └── kinematics.py           # ALBCKinematics: 2-link IK/FK/Jacobian
├── encoder/                    # HORA encoder module (RL-specific)
│   ├── actor_critic_encoder.py
│   └── encoder_runner.py
├── mdp/
│   ├── observations.py         # compute_policy_obs (13D), compute_privileged_obs (22D)
│   ├── rewards.py              # Potential-based reward manager (RL-specific)
│   └── events.py               # Domain randomization & reset functions
└── utils/
    ├── debug_vis.py            # Debug visualization (CoM, CoB, frames)
    └── logging.py              # Episode metric logging (RL-specific)
```

---

## Simulation Loop

### Timing Parameters

| Parameter | Value | Meaning |
|:---|:---|:---|
| `sim.dt` | 0.01s | Physics timestep (100 Hz) |
| `decimation` | 1 | Policy acts every physics step (100 Hz) |
| `control_decimation` | 1 | Joint targets updated every physics step |
| `episode_length_s` | 15.0s | Episode duration (1500 steps) |

### Step Execution Flow

```
step(actions)                              # Called at 100 Hz
│
├── _pre_physics_step(actions)             # Once per step
│   ├── Clamp actions to [-1, 1]
│   ├── Check control_decimation counter
│   └── Integrate velocity to position:
│       target += dt * max_joint_velocity * action
│       target = clamp(target, joint_limits)
│
├── [Physics Loop] x decimation (=1)       # Per physics substep
│   ├── _apply_action()
│   │   ├── robot.set_joint_position_target()     → PhysX PD servo
│   │   ├── hydro.compute_forces()                → Main body hydrodynamics
│   │   ├── _add_payload_wrench()                 → Optional payload forces
│   │   ├── permanent_wrench_composer.set(main)   → Apply to main body
│   │   ├── buoy_hydro.compute_forces()           → Buoy hydrodynamics
│   │   └── permanent_wrench_composer.set(buoy)   → Apply to buoy
│   ├── scene.write_data_to_sim()
│   ├── sim.step()                                → PhysX physics solve
│   └── scene.update(dt=0.01)                     → Read back state
│
├── _get_dones()                           # Safety termination
│   ├── Height bounds: min_height(-10) < z < max_height(10)
│   └── Distance: xy_distance < max_distance(10)
│
├── _get_rewards()                         # RL reward (potential-based)
├── _reset_idx(terminated_env_ids)         # Reset terminated environments
└── _get_observations()                    # Return obs dict
```

### Key Design Decisions

1. **Velocity-to-position integration**: Actions are velocity commands [-1, 1], integrated
   to position targets each step. This mimics real servo behavior where you command speed,
   not instantaneous position.

2. **Permanent wrench composer**: Hydrodynamic forces use `permanent_wrench_composer`
   (persistent forces), not instantaneous impulses. Forces are re-applied every substep.

3. **Dual hydrodynamics**: Main body and buoy have **separate** HydrodynamicsModel instances
   because they are different rigid bodies with different volumes, masses, and CoB/CoG.

---

## Robot Physical Parameters

### Main Body

| Parameter | Value | Note |
|:---|:---|:---|
| Mass | 9.18 kg | PhysX handles gravity |
| Volume | 0.00827 m^3 | Buoyancy = 80.9 N |
| Added mass (roll, pitch) | 0.04, 0.05 kg*m^2 | |
| Rigid body inertia (roll, pitch) | 0.0994, 0.0994 kg*m^2 | |
| Linear damping | (2.0, 4.0, 4.0, 0.1, 0.1, 0.1) | |
| Quadratic damping | (26.0, 26.0, 10.7, 1.5, 1.5, 0.01) | |

### Buoy

| Parameter | Value | Note |
|:---|:---|:---|
| Mass | 0.93 kg | |
| Volume | 0.00268 m^3 | Buoyancy = 26.2 N |
| F_bu (buoyancy force) | ~26.24 N | Key for Lambda matrix |

### ALBC Arm

| Parameter | Value | Note |
|:---|:---|:---|
| Link 1 length (L1) | 0.233 m | |
| Link 2 length (L2) | 0.233 m | Equal link lengths |
| Height offset | 0.230 m | Constant Z offset |
| Workspace (reach) | 0 ~ 0.466 m | L1 + L2 |
| Joint limits | from USD | ~+/- 2*pi rad |

### Joint PD Servo

| Parameter | Default | Note |
|:---|:---|:---|
| Stiffness (Kp) | 500.0 | From USD actuator config |
| Damping (Kd) | 1.0 | From USD actuator config |
| max_joint_velocity | 2*pi rad/s | Velocity command scale |
| Overridable at runtime | Yes | via `albc_joint_stiffness`, `albc_joint_damping` in cfg |

---

## Hydrodynamics Model

**File**: `isaaclab_tasks/models/hydrodynamics.py`

Implements Fossen model 6-DOF hydrodynamic forces:

```
Total wrench = -(Coriolis + Damping + Added_mass) + Buoyancy
```

### Components

| Component | Formula | Note |
|:---|:---|:---|
| Damping | D_l * v + D_q * |v| * v | Linear + quadratic (diagonal) |
| Coriolis | C_RB(v_abs) + C_A(v_rel) | Full Fossen formulation |
| Added mass | M_A * v_dot | Optional, uses PhysX acceleration |
| Buoyancy | rho * V * g * up_body | Force at CoB, creates restoring moment |

### Important Notes

- **Only buoyancy is applied** (not weight). PhysX handles gravity via `disable_gravity=False`.
- `buoyancy_force` property returns **scalar** `(num_envs,)`, not 3D vector.
- Ocean current is subtracted from body velocity to get relative velocity for drag.
- `compute_forces()` returns forces and torques in **body frame**.

---

## ALBC Kinematics

**File**: `controllers/kinematics.py`

2-link planar arm operating in XY plane:

```
Forward kinematics:
  x = L1*cos(g1) + L2*cos(g1+g2)
  y = L1*sin(g1) + L2*sin(g1+g2)

Inverse kinematics (analytical, elbow-up):
  cos(g2) = (r^2 - L1^2 - L2^2) / (2*L1*L2)
  g2 = atan2(sqrt(1 - cos^2(g2)), cos(g2))
  g1 = atan2(y, x) - atan2(L2*sin(g2), L1 + L2*cos(g2))

Jacobian (2x2):
  J = [[-L1*sin(g1) - L2*sin(g1+g2), -L2*sin(g1+g2)],
       [ L1*cos(g1) + L2*cos(g1+g2),  L2*cos(g1+g2)]]
```

### Key Methods

- `inverse(target_position) -> joint_angles`: IK with workspace clamping
- `forward(joint_angles) -> ee_position`: Standard 2-link FK
- `jacobian(joint_angles) -> J`: 2x2 velocity mapping matrix

---

## Observation Space (13D)

| Index | Content | Source |
|:---|:---|:---|
| 0:3 | roll, pitch, yaw | `euler_xyz_from_quat(root_quat_w)` |
| 3:6 | Angular velocity (body frame) | `root_ang_vel_b` |
| 6:9 | Attitude error (target - current) | `compute_attitude_error()` |
| 9:11 | Joint positions (normalized [-1,1]) | `joint_pos[:, albc_ids]` |
| 11:13 | Previous actions | `_prev_actions_obs` |

---

## Reset Sequence

Order matters. Called in `_reset_idx()`:

```
1. Logging           → log episode metrics, reset reward manager
2. Component reset   → robot.reset(), super()._reset_idx()
3. Action buffers    → zero out actions, prev_actions, targets
4. Hydro reset       → reset velocity/acceleration state
5. DR (if enabled)   → randomize hydro params, payload, ocean current
6. Task reset        → set target attitude, reset potentials
7. Joint reset       → set initial joint positions (default or random)
8. Pose reset        → set initial robot pose (default or random)
9. PD gains          → write stiffness/damping if overridden
10. Potential init   → set initial potential = current error (prevents spurious reward)
```

**Critical ordering**: Joint positions must be set before root pose.
Potentials must be initialized after pose reset.

---

## Environment Variants

| Environment ID | Config Class | DR | Current | Payload | state_space |
|:---|:---|:---|:---|:---|:---|
| `Isaac-HeroAgent-v0` | `HeroAgentEnvCfg` | No | No | No | 0 |
| `Isaac-HeroAgent-Base-v0` | `HeroAgentTrainEnvCfg` | Yes | Yes (0.2m/s) | Yes | 0 |
| `Isaac-HeroAgent-Encoder-Base-v0` | `HeroAgentEncoderTrainEnvCfg` | Yes | Yes | Yes | 22 |

All variants use the same `HeroAgentEnv` class with different configs.

---

## Domain Randomization

**File**: `mdp/events.py`

Available randomization functions (all reusable, not RL-specific):

| Function | What it randomizes |
|:---|:---|
| `randomize_hydrodynamics()` | Added mass, damping, volume, CoB/CoG offsets, inertia |
| `randomize_ocean_current()` | 6-DOF current velocity |
| `randomize_robot_pose()` | Position (xyz) and orientation (rpy) |
| `randomize_joint_positions()` | Initial joint angles within range |
| `randomize_payload()` | Payload mass and attachment offset |
| `reset_joint_positions_default()` | Set joints to zero |
| `reset_robot_pose_default()` | Set pose to origin at initial_height |

---

## RL-Specific Components (not needed for pure control)

These components exist only for reinforcement learning and can be ignored
when implementing a standalone controller:

| Component | File | Purpose |
|:---|:---|:---|
| RewardManager | `mdp/rewards.py` | Potential-based reward shaping |
| ActorCriticEncoder | `encoder/actor_critic_encoder.py` | HORA neural network |
| EncoderRunner | `encoder/encoder_runner.py` | Training loop extension |
| PPO configs | `agents/rsl_rl_ppo_cfg.py` | PPO hyperparameters |
| Episode logging | `utils/logging.py` | RL episode metrics |
| Privileged obs | `mdp/observations.py` (partial) | Teacher signal for encoder |

---

## How to Add a New Controller

To add a controller that bypasses RL and directly computes actions:

1. **Create controller** in `controllers/` (e.g., `controllers/my_controller.py`)
2. **Create environment subclass** that overrides `_pre_physics_step()`:
   - Read robot state (attitude, angular velocity, joint positions)
   - Compute desired action through your controller
   - Set joint position targets (same as existing `_apply_action()` flow)
3. **Create config** inheriting from `HeroAgentEnvCfg` with controller-specific params
4. **Register** in `__init__.py` with `gym.register()`
5. **Run** with a standalone script that creates the env and calls `step()` in a loop,
   or use the existing `play.py` with a wrapper that replaces the RL policy

### Integration Points

- **State access**: `self._robot.data.root_quat_w`, `root_ang_vel_b`, `joint_pos`
- **Joint control**: `self._robot.set_joint_position_target(targets, joint_ids=self._albc_joint_ids)`
- **Attitude error**: `self.compute_attitude_error(quat)` returns (target - current) wrapped to [-pi, pi]
- **Kinematics**: `ALBCKinematics.inverse(ee_pos)` and `.forward(joint_angles)`
- **Buoy buoyancy**: `self._buoy_hydro.buoyancy_force` returns scalar (num_envs,)
