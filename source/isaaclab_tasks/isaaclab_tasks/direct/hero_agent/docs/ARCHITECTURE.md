# Hero Agent ALBC - Architecture Reference

> **Status**: 2026-02-11 | **Source**: `config.py`, `base_env.py`, `tdc_env.py`, `__init__.py`
>
> Hero Agent ALBC 환경의 전체 아키텍처 레퍼런스.

---

## Overview

Hero Agent는 추진기 없이 자세를 제어하는 수중 로봇(UUV)이다.
2개의 회전 관절(joint1, joint2)로 부력체(buoy)를 재위치시키면,
부력체의 부력이 복원 토크를 생성하여 roll/pitch를 안정화한다.

```
[Main Body]  -- joint1 -- [Link1] -- joint2 -- [Link2 / Buoy]
 (9.18 kg)                (0.233m)              (0.233m, 0.93kg)
 Buoyancy: 80.9N                                Buoyancy: 26.2N
 Net: slightly negative                         Net: strongly positive (+17.1N)
```

System net buoyancy: ~+3N (약한 양의 부력 -> 수동 안정성).

---

## Directory Structure

```
hero_agent/
├── __init__.py                 # Gym environment registration (6 environments)
├── base_env.py                 # Base RL environment (HeroAgentEnv)
├── tdc_env.py                  # TDC controller environment (HeroAgentTDCEnv)
├── encoder_tdc_env.py          # Encoder-TDC env (HeroAgentEncoderTDCEnv)
├── adapt_tdc_env.py            # Phase 2 adaptation env (HeroAgentAdaptTDCEnv)
├── config.py                   # All configuration classes
├── agents/                     # RL agent configs
│   └── rsl_rl_ppo_cfg.py       # PPO runner + network config (4 variants)
├── controllers/
│   ├── tdc.py                  # TDC controller + TDCControllerCfg
│   └── kinematics.py           # ALBCKinematics: 2-link DLS IK/FK/Jacobian
├── encoder/                    # HORA encoder networks
│   ├── actor_critic_encoder.py # ActorCriticEncoder, EncoderTDC, EncoderTDCAdapt
│   ├── adaptation.py           # ProprioAdaptTConv
│   └── normalization.py        # EmpiricalNormalization
├── runners/                    # Training runners
│   ├── encoder_runner.py       # Phase 1 PPO (EncoderRunner)
│   └── adapt_runner.py         # Phase 2 supervised (AdaptRunner)
├── workflows/                  # Phase 2/3 entry-points
│   ├── _config_resolver.py     # Shared config resolution
│   ├── _policy_factory.py      # Shared policy build + checkpoint loading
│   ├── train_adaptation.py     # Phase 2 training entry-point
│   └── play_phase3.py          # Phase 3 evaluation + deploy
├── deploy/                     # Phase 3 deployment export
│   ├── deploy_module.py        # JIT-scriptable deploy module
│   └── deploy_exporter.py      # JIT/ONNX/JSON export
├── mdp/
│   ├── observations.py         # Policy obs (13D) + privileged obs (24D)
│   ├── rewards.py              # RewardManager + reward terms
│   └── events.py               # Domain randomization + reset functions
├── utils/
│   ├── debug_vis.py            # Debug visualization (CoM, CoB, frames)
│   ├── logging.py              # Episode metric + DR logging
│   └── env_utils.py            # unwrap_env, connect_encoder_to_env
└── docs/                       # Documentation (this directory)
```

---

## Simulation Loop

### Timing Parameters

| Parameter | Value | Meaning |
|:---|:---|:---|
| `sim.dt` | 0.005s (200 Hz) | Physics timestep |
| `decimation` | 1 | Step = 1 physics step (200 Hz env step) |
| `step_dt` | 0.005s | `sim.dt * decimation` |
| `control_decimation` | 4 (TDC) / 1 (Base RL) | Joint targets update rate |
| `control_dt` | 0.02s (TDC) / 0.005s (RL) | `step_dt * control_decimation` |
| `episode_length_s` | 15.0s | Episode duration |
| `max_episode_length` | 3000 | `15.0 / 0.005` steps |

### Step Execution Flow

```
step(actions)                              # Called at 200 Hz (step_dt = 0.005s)
│
├── _pre_physics_step(actions)             # Once per step
│   ├── Clamp actions to [-1, 1]
│   ├── Check control_decimation counter
│   └── [Base RL] Integrate velocity to position:
│       target += dt * max_joint_velocity * action
│       target = clamp(target, joint_limits)
│   └── [TDC] Compute TDC controller output:
│       error -> PD -> TDE -> tau -> IK -> position target
│
├── [Physics Loop] x decimation (=1)       # Per physics substep
│   ├── _apply_action()
│   │   ├── robot.set_joint_position_target()     -> PhysX implicit PD
│   │   ├── hydro.compute_forces()                -> Main body hydrodynamics
│   │   ├── _compute_payload_wrench()             -> Payload weight on gripper
│   │   ├── permanent_wrench_composer.set(main)   -> Apply to main body
│   │   ├── buoy_hydro.compute_forces()           -> Buoy hydrodynamics
│   │   └── permanent_wrench_composer.set(buoy)   -> Apply to buoy
│   ├── scene.write_data_to_sim()
│   ├── sim.step()                                -> PhysX physics solve
│   └── scene.update(dt=0.005)                    -> Read back state
│
├── _get_dones()                           # Safety termination
│   ├── Height bounds: min_height(-10) < z < max_height(10)
│   └── Distance: xy_distance < max_distance(10)
│
├── _get_rewards()                         # Potential-based rewards
├── _reset_idx(terminated_env_ids)         # Reset terminated environments
└── _get_observations()                    # Return obs dict
```

### Key Design Decisions

1. **Velocity-to-position integration** (Base RL): Actions are velocity commands [-1, 1],
   integrated to position targets each step. 실제 서보에서 속도를 명령하는 방식을 모방.

2. **TDC direct position** (TDC envs): TDC controller가 IK로 관절 위치를 직접 계산.
   rate-limiting된 velocity로 position target을 갱신.

3. **Permanent wrench composer**: Hydrodynamic forces는 persistent forces로 적용.
   매 substep마다 재적용.

4. **Dual hydrodynamics**: Main body와 buoy는 별도의 `HydrodynamicsModel` 인스턴스.
   서로 다른 volume, mass, CoB/CoG를 가지기 때문.

5. **Payload on gripper**: Payload wrench는 gripper body (base에 fixed joint로 연결)에 적용.
   PhysX가 fixed joint를 통해 힘을 자동 전파. `merge_fixed_joints: false` 필수.

---

## Robot Physical Parameters

### Main Body

| Parameter | Value |
|:---|:---|
| Mass | 9.18 kg |
| Volume | 0.00827 m^3 (Buoyancy = 80.9 N) |
| Added mass (roll, pitch) | 0.04, 0.05 kg*m^2 |
| Inertia (roll, pitch, yaw) | 0.0994, 0.0994, 0.0372 kg*m^2 |
| Linear damping | (2.0, 4.0, 4.0, 0.1, 0.1, 0.1) |
| Quadratic damping | (26.0, 26.0, 10.7, 1.5, 1.5, 0.01) |
| CoG | (0.0, 0.0, -0.10) m |

### Buoy

| Parameter | Value |
|:---|:---|
| Mass | 0.93 kg |
| Volume | 0.00268 m^3 (Buoyancy = 26.2 N) |
| F_bu (buoyancy force) | ~26.24 N (Lambda 행렬의 핵심 파라미터) |

### ALBC Arm

| Parameter | Value |
|:---|:---|
| Link 1 length (L1) | 0.233 m |
| Link 2 length (L2) | 0.233 m |
| Height offset (h) | 0.180 m |
| Workspace (reach) | 0 ~ 0.466 m |
| max_joint_velocity | 3.0 rad/s |

### Joint PD Servo (ImplicitActuatorCfg)

| Environment | Kp Center | Kd Center | DR Range |
|:---|:---|:---|:---|
| Base RL | 100.0 | 3.0 | +-20% ([80,120], [2.4,3.6]) |
| TDC | 200.0 | 10.0 | +-20% ([160,240], [8,12]) |

Asset default: Kp=100, Kd=3. TDC에서 높은 이유: DLS IK의 small delta position 추종에 높은 강성 필요.

---

## ALBC Kinematics

**File**: `controllers/kinematics.py`

2-link planar arm, XY 평면에서 동작:

```
Forward Kinematics:
  x = L1*cos(g1) + L2*cos(g1+g2)
  y = L1*sin(g1) + L2*sin(g1+g2)

Inverse Kinematics (DLS Jacobian pseudo-inverse):
  J_pinv = J^T * (J * J^T + lambda^2 * I)^{-1}
  delta_q = J_pinv * delta_p
  lambda: Yoshikawa adaptive (manipulability -> damping)
```

DLS IK가 analytical IK를 대체한 이유:
- Singularity 근처에서 smooth한 감쇠 (workspace clamp 불필요)
- Single-step small delta에 충분 (iterative 불필요)
- Yoshikawa adaptive lambda가 manipulability에 따라 damping을 자동 조절

---

## Observation Space

### Policy Observations (13D)

| Index | Content | Source |
|:---|:---|:---|
| 0:3 | roll, pitch, yaw | `euler_xyz_from_quat(root_quat_w)` |
| 3:6 | Angular velocity (body frame) | `root_ang_vel_b` |
| 6:9 | Attitude error (target - current) | `compute_attitude_error()` |
| 9:11 | Joint positions (normalized [-1,1]) | `joint_pos[:, albc_ids]` |
| 11:13 | Previous actions | `_prev_actions_obs` |

### Privileged Observations (24D, encoder envs only)

```
Main body (10D):  [volume(1), CoG(3), CoB(3), inertia(3)]
Buoy body (10D):  [volume(1), CoG(3), CoB(3), inertia(3)]
Payload    (4D):  [mass(1), cog_offset_xyz(3)]
```

### Sensor Noise (DR enabled)

| Obs Dims | Noise | Bias |
|:---|:---|:---|
| euler (0:3) | N(0, 0.01 rad) | U(-0.005, 0.005) per episode |
| angular_vel (3:6) | N(0, 0.02 rad/s) | U(-0.01, 0.01) per episode |
| att_error, joint_pos, prev_actions (6:13) | None | None |

---

## Environment Variants

| Task ID | Config | Env Class | DR | Current | Payload | Obs |
|:---|:---|:---|:---:|:---:|:---:|:---|
| `Isaac-HeroAgent-v0` | `HeroAgentEnvCfg` | `HeroAgentEnv` | OFF | OFF | OFF | 13D |
| `Isaac-HeroAgent-Base-v0` | `HeroAgentTrainEnvCfg` | `HeroAgentEnv` | ON | ON | ON | 13D |
| `Isaac-HeroAgent-Encoder-Base-v0` | `HeroAgentEncoderTrainEnvCfg` | `HeroAgentEnv` | ON | ON | ON | 13D+24D |
| `Isaac-HeroAgent-TDC-v0` | `HeroAgentTDCEnvCfg` | `HeroAgentTDCEnv` | ON | ON | ON | 13D |
| `Isaac-HeroAgent-Encoder-TDC-v0` | `HeroAgentEncoderTDCEnvCfg` | `HeroAgentEncoderTDCEnv` | ON | ON | ON | 13D+24D |
| `Isaac-HeroAgent-Adapt-TDC-v0` | `HeroAgentAdaptTDCEnvCfg` | `HeroAgentAdaptTDCEnv` | ON | ON | ON | 13D+24D+hist |

### Environment Inheritance

```
HeroAgentEnv (base_env.py)                    # Base RL: 2D joint velocity actions
  └── HeroAgentTDCEnv (tdc_env.py)            # TDC: classical controller replaces RL actions
       └── HeroAgentEncoderTDCEnv (encoder_tdc_env.py)  # Encoder-TDC: 4D gain actions
            └── HeroAgentAdaptTDCEnv (adapt_tdc_env.py)  # Phase 2: proprio history buffer
```

### Config Inheritance

```
HeroAgentEnvCfg (debug, no DR)
  └── HeroAgentTrainEnvCfg (DR + current + payload)
       ├── HeroAgentEncoderTrainEnvCfg (+ 24D privileged obs)
       ├── HeroAgentTDCEnvCfg (+ TDC config, TDC joint gains)
       ├── HeroAgentEncoderTDCEnvCfg (+ TDC + encoder + 4D actions)
       └── HeroAgentAdaptTDCEnvCfg (+ proprio history)
```

---

## Runner Types

| Runner | Purpose | Algorithm | Source |
|:---|:---|:---|:---|
| `OnPolicyRunner` (RSL-RL) | Base RL / Encoder training | PPO | RSL-RL default |
| `EncoderRunner` | Phase 1: Encoder-TDC | PPO + z logging | `runners/encoder_runner.py` |
| `AdaptRunner` | Phase 2: Adaptation | Supervised L2 | `runners/adapt_runner.py` |

---

## Hydrodynamics Model

**File**: `isaaclab_tasks/models/hydrodynamics.py`

Fossen model 6-DOF hydrodynamic forces:

```
Total wrench = -(Coriolis + Damping + Added_mass) + Buoyancy
```

| Component | Formula |
|:---|:---|
| Damping | $D_l \cdot v + D_q \cdot |v| \cdot v$ |
| Coriolis | $C_{RB}(v_{abs}) + C_A(v_{rel})$ |
| Added mass | $M_A \cdot \dot{v}$ |
| Buoyancy | $\rho \cdot V \cdot g \cdot \hat{z}_{body}$ |

Important notes:
- Buoyancy만 적용 (weight는 PhysX gravity가 처리)
- `buoyancy_force` property returns **scalar** `(num_envs,)`
- Ocean current는 body velocity에서 빼서 relative velocity 계산
- `compute_forces()`는 body frame에서 forces + torques 반환

---

## Payload Wrench

**File**: `base_env.py` (`_compute_payload_wrench`)

Payload는 gripper body에 적용 (base에 fixed joint로 연결, offset (0, 0.0881, -0.185)).

```python
# Weight force in world frame
F_w = [0, 0, -mass * g]

# Transform to gripper body frame
F_b = R_gripper^T @ F_w

# Torque from CoG offset
tau_b = (attachment_offset + cog_offset) x F_b
```

`merge_fixed_joints: false`가 config.yaml에 설정되어야 gripper body가 독립 rigid body로 존재한다.
PhysX가 fixed joint를 통해 wrench를 main body에 자동 전파한다.

---

## Reset Sequence

`_reset_idx()` 실행 순서 (순서 중요):

```
1. Logging              → episode metrics 수집 및 전송
2. Component reset      → robot.reset(), super()._reset_idx()
3. Action buffers       → actions, prev_actions, targets 초기화
4. Episode decorrelation → 초기: full range, 개별: 10% jitter
5. Hydrodynamics reset  → hydro.reset(), buoy_hydro.reset()
6. Payload reset        → payload mass/offset to defaults
7. DR (if enabled):
   ├── randomize_hydrodynamics()     → 질량, 감쇠, 부력, CoB/CoG, 관성
   ├── randomize_body_mass()         → PhysX set_masses()
   ├── randomize_payload()           → mass + CoG offset randomization
   └── randomize_ocean_current()     → linear velocity + noise
8. Task reset           → target attitude 설정 (fixed or random)
9. Robot state reset:
   ├── randomize_joint_positions()   → joint angles within limits
   └── randomize_robot_pose()        → position + orientation
10. Joint actuator DR   → stiffness, damping, friction
11. Potential init      → _initialize_potentials() (prev = current)
```

Critical ordering:
- Joint positions는 root pose 이전에 설정
- Potentials는 pose reset 이후에 초기화 (가짜 보상 방지)
- Episode decorrelation은 DR 이전에 실행

---

## Domain Randomization Summary

**File**: `mdp/events.py`

10개 카테고리, 30+ 파라미터. 상세 내용은 [DOMAIN_RANDOMIZATION.md](./DOMAIN_RANDOMIZATION.md) 참조.

| Category | Key Parameters |
|:---|:---|
| Initial Pose | position xy +-0.5m, z 4-5m, rpy +-45deg |
| Hydrodynamics | added mass +-30%, damping +-30%/40%, volume +-10% |
| Ocean Current | linear +-0.2m/s + gaussian noise |
| Joint State | full range [-pi, pi] |
| Payload | mass [0, 1.0]kg, CoG offset xyz |
| Joint Gains | Kp/Kd +-20% around center |
| Body Mass | +-10% (PhysX set_masses) |
| Water Density | 995-1025 kg/m^3 |
| Sensor Noise | IMU bias + white noise |
| Joint Friction | static [0, 0.05], viscous [0, 0.3] |

---

## How to Add a New Controller

1. **Create controller** in `controllers/` (e.g., `controllers/my_controller.py`)
2. **Create environment subclass** that overrides `_pre_physics_step()`:
   - Read robot state (attitude, angular velocity, joint positions)
   - Compute desired action through your controller
   - Set joint position targets
3. **Create config** inheriting from `HeroAgentEnvCfg` or `HeroAgentTrainEnvCfg`
4. **Register** in `__init__.py` with `gym.register()`

### Integration Points

- **State access**: `self._robot.data.root_quat_w`, `root_ang_vel_b`, `joint_pos`
- **Joint control**: `self._robot.set_joint_position_target(targets, joint_ids=self._albc_joint_ids)`
- **Attitude error**: `self.compute_attitude_error(quat)` -> (target - current), wrapped to [-pi, pi]
- **Kinematics**: `ALBCKinematics.inverse(ee_pos)`, `.forward(joint_angles)`, `.jacobian(joint_angles)`
- **Buoy buoyancy**: `self._buoy_hydro.buoyancy_force` -> scalar (num_envs,)

---

## Related Documents

- [DOMAIN_RANDOMIZATION.md](./DOMAIN_RANDOMIZATION.md): DR 상세
- [TDC_CONTROL_LAW.md](./TDC_CONTROL_LAW.md): TDC 제어 법칙
- [TRAINING_PIPELINE.md](./TRAINING_PIPELINE.md): 학습 파이프라인
- [REWARD_FUNCTIONS.md](./REWARD_FUNCTIONS.md): 보상함수 분석
- [SIM_TO_REAL.md](./SIM_TO_REAL.md): Sim-to-real gap
- [DYNAMICS_ANALYSIS.md](./DYNAMICS_ANALYSIS.md): 동역학 분석
- [TDC_LITERATURE_SURVEY.md](./TDC_LITERATURE_SURVEY.md): TDC 이론
- [tdc-tuning-history.md](./tdc-tuning-history.md): TDC 디버깅 기록

---

**Created**: 2026-02-09
**Updated**: 2026-02-11 (sim.dt, episode_length, control_decimation, env variants, IK, DR, payload, privileged obs updated to current code)
