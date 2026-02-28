# Training Pipeline

> **Status**: 2026-02-23 | **Source**: `config.py`, `encoder/`, `agents/rsl_rl_ppo_cfg.py`, `adapt_base_env.py`
>
> RMA/HORA 2-phase 학습 파이프라인의 구현 현황.
> Phase 1 (Teacher, base RL) + Phase 2 (Student, base RL) + Deployment 구조.

---

## 1. Pipeline Overview

RMA/HORA의 2-phase 학습 구조를 수중 UVMS 자세 제어에 적용한다:

```
Phase 1: Teacher Training (PPO + Encoder, base RL)
    Privileged Info (28D) --> Encoder --> z (13D)
    Policy Obs (13D) + z --> Actor --> 2D velocity actions

Phase 2: Adaptation Module Training (Supervised, base RL)
    Proprio History (N, 30, 8) --> AdaptTConv --> z_hat (13D)
    Frozen Encoder(Privileged) --> z_gt (13D)
    Loss = ||z_hat - z_gt||^2

Deployment: Real World
    Proprio History --> AdaptTConv --> z_hat --> Frozen Actor --> velocity actions
    (No privileged info, no additional training)
```

각 phase에서 사용하는 환경과 runner:

| Phase | Environment | Config | Runner | Task ID |
|:---|:---|:---|:---|:---|
| 1 (Teacher) | `HeroAgentEnv` | `HeroAgentEncoderTrainEnvCfg` | `EncoderRunner` | `Isaac-HeroAgent-Encoder-Base-v0` |
| 2 (Student) | `HeroAgentAdaptBaseEnv` | `HeroAgentAdaptBaseEnvCfg` | Custom `AdaptRunner` | `Isaac-HeroAgent-Adapt-Base-v0` |
| Deploy | (real robot) | - | - | - |

---

## 2. Phase 1: Encoder-Base Teacher Training

### 2.1 Network Architecture

```
Encoder:  Privileged (28D) --> MLP [256, 128, 64] --> sigmoid --> z (13D) in [0.01, 2.0]
Actor:    cat([policy_obs(13D), z(13D)]) = 26D --> MLP [256, 128, 64] --> 2D velocity actions
Critic:   cat([policy_obs(13D), z(13D)]) = 26D --> MLP [256, 128, 64] --> 1D value
```

Critic은 Actor와 동일한 입력(policy_obs + z)을 받는다 (symmetric).
Privileged info를 직접 받지 않으므로, encoder가 유용한 정보를 z에 압축하도록 강제된다.

구현 클래스: `ActorCriticEncoder` (`encoder/actor_critic_encoder.py`)

### 2.2 I/O Variable Map

#### Privileged Information (28D)

```
Main body hydro (7D):     [volume(1), CoG(3), CoB(3)]
Buoy body hydro (7D):     [volume(1), CoG(3), CoB(3)]
Main body dynamics (4D):  [inertia Ixx/Iyy/Izz(3), body_mass(1)]
Buoy dynamics (4D):       [inertia Ixx/Iyy/Izz(3), body_mass(1)]
Payload (4D):             [mass(1), cog_offset_xyz(3)]
Main added mass surge (1D)
Buoy added mass surge (1D)
```

Hydrostatic + dynamics + surge added mass를 포함. Damping은 제외.

- Damping 제외 근거: 자세 안정화(steady state)에서 damping 효과는 미미하다.
- Surge added mass 포함 근거: UUV added mass는 rigid body inertia에 필적하며 (main body M_a_surge=5.76kg), DR로 변동. Encoder가 effective dynamics를 관찰하려면 필요.

Source: `mdp/observations.py` (`_hydro_privileged_info`, `_added_mass_surge`)

#### Encoder Latent z (13D)

```
z = z_min + sigmoid(MLP(privileged)) * (z_max - z_min)    (z_min=0.01, z_max=2.0)
```

- 13D: general compressed latent (물리적 파라미터에 직접 대응하지 않음)
- sigmoid: z in [0.01, 2.0] bounded (softplus의 dead zone collapse 방지)
- Encoder-Base에서는 z에서 M_hat을 추출하지 않음 (base RL은 velocity 명령만 출력)

Source: `encoder/actor_critic_encoder.py` (`_activate_z`, `_encode`)

#### Policy Observations (13D)

```
euler_angles(3) + angular_velocity(3) + attitude_error(3) + joint_pos(2) + prev_actions(2)
```

Source: `mdp/observations.py` (`compute_policy_obs`)

#### Actor Output (2D)

```
actions: 2D joint velocity commands in [-1, 1]
target += dt * max_joint_velocity * action
```

Base RL pipeline은 직접 관절 속도를 명령한다.

### 2.3 Training Configuration

| Parameter | Value | Source |
|:---|:---|:---|
| Algorithm | PPO (On-Policy) | `rsl_rl_ppo_cfg.py:184-227` |
| Parallel Envs | 4096 (CLI default) | CLI `--num_envs` |
| Learning Rate | 3e-4 (adaptive schedule) | `rsl_rl_ppo_cfg.py:220` |
| Gamma | 0.99 | `rsl_rl_ppo_cfg.py:222` |
| GAE Lambda | 0.95 | `rsl_rl_ppo_cfg.py:223` |
| Clip Param | 0.2 | `rsl_rl_ppo_cfg.py:216` |
| Epochs per Update | 8 | `rsl_rl_ppo_cfg.py:218` |
| Mini-batches | 4 | `rsl_rl_ppo_cfg.py:219` |
| Steps per Env | 32 | `rsl_rl_ppo_cfg.py:196` |
| Max Iterations | 600 | `rsl_rl_ppo_cfg.py:197` |
| Entropy Coef | 0.0 | Disabled (dynamic systems에서 std 폭발 방지) |
| Max Grad Norm | 1.0 | `rsl_rl_ppo_cfg.py:225` |
| Init Noise Std | 1.0 | `rsl_rl_ppo_cfg.py:207` |

### 2.4 Training Command

Phase 1 teacher training은 `Isaac-HeroAgent-Encoder-Base-v0` task를 사용한다.

---

## 3. Phase 2: Adaptation Module Training

### 3.1 Architecture: ProprioAdaptTConv

```
Per-timestep MLP:  8D features --> MLP [32, 32] --> 32D embedding
Temporal Conv:     (N, 30, 32) -->
                   Conv1d(32->32, kernel=9, stride=2, padding=4) --> (N, 15, 32)
                   Conv1d(32->32, kernel=5, stride=1, padding=2) --> (N, 15, 32)
                   Conv1d(32->32, kernel=5, stride=1, padding=2) --> (N, 15, 32)
Flatten + Linear:  (N, 480) --> Linear --> z_hat (13D)
Activation:        sigmoid, z in [0.01, 2.0] (same as encoder)
```

구현 클래스: `ActorCriticEncoderAdapt` (`encoder/adaptation.py`)

### 3.2 Proprioception History (8D per timestep)

```
[roll(1), pitch(1), p(1), q(1), joint_pos_norm(2), prev_actions(2)]
```

Ring buffer: `(num_envs, 30, 8)`, 리셋 시 0으로 초기화.

각 feature의 역할:

| Feature | Dim | Role |
|:---|:---|:---|
| roll, pitch | 2 | Restoring torque의 직접적 결과. $T_b = f(\text{roll, pitch, CoG, CoB, V})$ |
| p, q | 2 | Angular acceleration = $\tau / M_{eff}$의 적분. $M_{eff}$ 추론 가능 |
| joint_pos_norm | 2 | Arm 구성 정보 (작동점, IK 상태) |
| prev_actions | 2 | System identification의 입력 신호 역할 |

HORA와의 차이: HORA (Allegro hand)에서는 joint state = dynamics state이므로 joint 정보만으로 충분하다.
ALBC에서는 arm state $\neq$ body dynamics state이므로, body orientation + angular rates를 포함해야 한다.

Source: `adapt_base_env.py` (`_update_proprio_hist`)

### 3.3 Supervised Training

```
z_hat = adapt_tconv(proprio_hist)      # AdaptTConv forward
z_gt  = frozen_encoder(privileged)     # Frozen Phase 1 encoder
loss  = ||z_hat - z_gt||^2            # L2 loss
```

Phase 1에서 학습된 encoder와 actor는 동결 (gradient 없음).
adapt_tconv만 학습된다.

### 3.4 On-Policy Data Collection

RMA의 핵심 통찰: Ground truth z_gt로만 학습하면 "좋은 궤적만" 데이터에 포함되어,
배포 시 trajectory 이탈에 취약하다.

해결: 무작위 초기화된 adapt module로 on-policy exploration trajectory를 확보한다.
adapt_tconv가 출력한 z_hat으로 실제 policy를 구동하면서 동시에 z_gt와의 오차를 줄여나간다.

### 3.5 Training Configuration

| Parameter | Value | Source |
|:---|:---|:---|
| Optimizer | Adam (adapt_tconv only) | `AdaptRunner` |
| Learning Rate | 3e-4 | `rsl_rl_ppo_cfg.py:269` |
| Max Agent Steps | 100M | `rsl_rl_ppo_cfg.py:272` |
| Save Interval | 10M steps | `rsl_rl_ppo_cfg.py:275` |
| Log Interval | 10 iterations | `rsl_rl_ppo_cfg.py:278` |
| Max Grad Norm | 10.0 | `rsl_rl_ppo_cfg.py:281` |
| History Length | 30 timesteps | `rsl_rl_ppo_cfg.py:90` |
| Feature Dim | 12 per timestep | `rsl_rl_ppo_cfg.py:91` |
| Loss | L2 ($\|z_{hat} - z_{gt}\|^2$) | `AdaptRunner` |

Phase 2의 grad norm (10.0)이 Phase 1 (1.0)보다 완화된 이유:
초기 수렴 속도를 보존하면서, 이상 gradient spike만 차단하기 위함.

### 3.6 AdaptRunner

`AdaptRunner`는 RSL-RL의 `OnPolicyRunner`를 상속하지 않는 custom runner이다.
PPO 대신 supervised L2 loss를 사용하며, Phase 1 체크포인트에서 encoder/actor를 로드하고 동결한다.

로깅: `_WandbTBWriter` adapter로 TensorBoard + WandB 이중 로깅 지원.

---

## 4. Phase 3: Deployment

### 4.1 Data Flow

```
Proprioception History --> AdaptTConv --> z_hat (13D)
    policy_obs + z_hat --> Frozen Actor --> 2D velocity commands
    velocity --> joint position targets
```

Privileged information, encoder 모두 불필요. Proprioception history만으로 동작.

### 4.2 Frequency Separation

| Component | Frequency | Rationale |
|:---|:---|:---|
| Actor | 50 Hz (control_decimation=4) | Joint velocity command per control step |
| AdaptTConv | 50 Hz (current) / 10 Hz (optional) | 환경 변화는 느리므로 저주파 가능 |

RMA 원논문에서 adaptation module을 actor보다 낮은 주파수로 실행하는 이유:
(a) 계산 비용 절감, (b) 환경 파라미터는 상태보다 느리게 변함, (c) end-to-end 단일 네트워크보다 분리 구조가 성능 우수.

수중 환경은 유체 저항으로 동역학 응답이 지상보다 느리므로, 이 분리의 이점이 더 크다.

### 4.3 Policy Export

배포 시 필요한 가중치:

| Component | Source | Frozen? |
|:---|:---|:---|
| Actor MLP | Phase 1 checkpoint | Yes |
| AdaptTConv | Phase 2 checkpoint | Yes |

Encoder는 배포 시 불필요 (adapt module이 대체).

---

## 5. Comparison with Design Notes

| Aspect | Design Notes (07) | Current Implementation | Status |
|:---|:---|:---|:---|
| Encoder input | Privileged (CoM, CoB, mass, inertia) | 28D hydro + dynamics + added_mass + payload | OK |
| Latent dim | TBD (HORA: 8) | 13D | OK |
| Latent activation | softplus | sigmoid, z in [0.01, 2.0] | OK |
| M_hat extraction | Separate MLP f_theta | N/A (Encoder-Base has no M_hat) | N/A |
| Actor input | 4D (error, error_rate) + z | 13D policy_obs + z(13D) = 26D | OK (richer) |
| Actor output | 4D gains (softplus) | 2D velocity commands | OK |
| Adapt input/step | 8D (roll, pitch, p, q, act) | 8D (roll, pitch, p, q, joint_pos_norm, prev_actions) | OK |
| History length | TBD (HORA: 30) | 30 | OK |
| Adapt architecture | MLP + 1D CNN | MLP [32,32] + 3x Conv1d | OK |
| Phase 2 loss | L2 | L2 | OK |

### Key Changes from Design Notes

1. **Privileged 22D -> 28D**: Payload 항목 확장 (2D->4D), dynamics (inertia+body_mass, 8D 추가),
   surge added mass (2D 추가). 최종 28D = hydro(14) + dynamics(8) + payload(4) + added_mass(2).

2. **Sigmoid activation**: 초기 설계의 softplus에서 sigmoid로 변경. Softplus의 dead zone
   collapse 문제를 방지하고, bounded output으로 안정적 학습.

3. **Base RL pipeline**: Encoder-Base (velocity output)를 Phase 1 기본 파이프라인으로 채택.

---

## 6. Known Issues and Design Decisions

### Issue A: Proprio History Content -- Resolved

초기 구현이 arm joint state만 포함하던 문제.
body euler + angular rates 추가로 8D 확장 완료.
Conv1d temporal reduction은 history_len에만 의존하므로 변경 없음.

### Issue B: General Latent vs M_hat Extraction -- Accepted

Encoder-Base는 general 13D latent을 사용하며, M_hat을 직접 슬라이싱하지 않는다.
z는 encoder가 자유롭게 구조화한 compressed representation이다.

### Issue C: Missing Damping in Privileged Info -- Accepted

Surge added mass는 28D에 포함 (2D). Damping은 제외.
자세 안정화(steady state)에서 damping 효과는 미미하다.

### Issue D: Sigmoid Activation -- Accepted

sigmoid로 z in [0.01, 2.0] bounded. Softplus의 dead zone collapse 문제를 해결.
z 값의 분포는 WandB에 z statistics를 로깅하여 모니터링.

### Issue E: Latent Dim 13 -- Accepted

28D privileged info를 13D로 압축. 약 2:1 압축 비율.
실질적 독립 자유도보다 여유 있는 latent space가 encoder 학습을 안정화한다.

### Issue F: Yaw in Policy Obs -- Accepted

ALBC는 yaw 제어 불가하지만, yaw angle이 hydrodynamic force 방향에 영향을 미치므로
context 정보로서 포함. Reward에서는 yaw error를 제외.

---

## Related Documents

- [TDC_CONTROL_LAW.md](./TDC_CONTROL_LAW.md): TDC 제어기 수식 유도
- [DOMAIN_RANDOMIZATION.md](./DOMAIN_RANDOMIZATION.md): DR 설정 및 privileged obs 상세
- [REWARD_FUNCTIONS.md](./REWARD_FUNCTIONS.md): 보상함수 분석 (Gaussian + curriculum)
- [DYNAMICS_ANALYSIS.md](./DYNAMICS_ANALYSIS.md): 적응적 M_hat 필요성의 이론적 근거
- [SAC_MPC_MONITORING.md](./SAC_MPC_MONITORING.md): SAC-MPC 학습 모니터링 가이드 (WandB 대시보드)

---

**Created**: 2026-02-11
**Updated**: 2026-02-28
