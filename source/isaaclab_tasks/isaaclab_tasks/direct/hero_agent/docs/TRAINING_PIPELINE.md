# Training Pipeline

> **Status**: 2026-02-11 | **Source**: `config.py`, `encoder/`, `agents/rsl_rl_ppo_cfg.py`, `adapt_tdc_env.py`
>
> RMA/HORA 2-phase 학습 파이프라인의 구현 현황.
> Phase 1 (Teacher) + Phase 2 (Student) + Deployment 구조.

---

## 1. Pipeline Overview

RMA/HORA의 2-phase 학습 구조를 수중 UVMS TDC 제어에 적용한다:

```
Phase 1: Teacher Training (PPO + Encoder)
    Privileged Info (24D) --> Encoder --> z (6D)
    Policy Obs (13D) + z --> Actor --> 4D gains [Kp_r, Kp_p, Kd_r, Kd_p]
    z[3:5] --> M_hat --> TDC Controller

Phase 2: Adaptation Module Training (Supervised)
    Proprio History (N, 30, 12) --> AdaptTConv --> z_hat (6D)
    Frozen Encoder(Privileged) --> z_gt (6D)
    Loss = ||z_hat - z_gt||^2

Deployment: Real World
    Proprio History --> AdaptTConv --> z_hat --> Actor --> TDC
    (No privileged info, no additional training)
```

각 phase에서 사용하는 환경과 runner:

| Phase | Environment | Config | Runner | Task ID |
|:---|:---|:---|:---|:---|
| 1 (Teacher) | `HeroAgentEncoderTDCEnv` | `HeroAgentEncoderTDCEnvCfg` | RSL-RL `OnPolicyRunner` | `Isaac-HeroAgent-Encoder-TDC-v0` |
| 2 (Student) | `HeroAgentAdaptTDCEnv` | `HeroAgentAdaptTDCEnvCfg` | Custom `AdaptRunner` | `Isaac-HeroAgent-Adapt-TDC-v0` |
| Deploy | (real robot) | - | - | - |

---

## 2. Phase 1: Encoder-TDC Teacher Training

### 2.1 Network Architecture

```
Encoder:  Privileged (24D) --> MLP [64, 32] --> softplus + z_min=0.1 --> z (6D)
Actor:    cat([policy_obs(13D), z(6D)]) = 19D --> MLP [64, 64] --> 4D raw gains
Critic:   cat([policy_obs(13D), z(6D)]) = 19D --> MLP [64, 64] --> 1D value
```

Critic은 Actor와 동일한 입력(policy_obs + z)을 받는다 (symmetric).
Privileged info를 직접 받지 않으므로, encoder가 유용한 정보를 z에 압축하도록 강제된다.

구현 클래스: `ActorCriticEncoderTDC` (`encoder/actor_critic_encoder.py`)

### 2.2 I/O Variable Map

#### Privileged Information (24D)

```
Main body (10D):  [volume(1), CoG(3), CoB(3), inertia(3)]
Buoy body (10D):  [volume(1), CoG(3), CoB(3), inertia(3)]
Payload    (4D):  [mass(1), cog_offset_xyz(3)]
```

Hydrostatic 파라미터만 포함. Added mass와 damping은 제외.

- Added mass 제외 근거: TDC의 TDE 메커니즘이 이전 step의 실제 dynamics를 암묵적으로 포착하므로, encoder가 정확한 added mass를 알 필요가 없다.
- Damping 제외 근거: 자세 안정화(steady state)에서 damping 효과는 미미하다.

Source: `mdp/observations.py` (`_hydro_privileged_info`)

#### Encoder Latent z (6D)

```
z = softplus(MLP(privileged)) + z_min    (z_min = 0.1)
```

- 6D: 6-DOF 관례 [surge, sway, heave, roll, pitch, yaw]에 대응
- softplus: z > z_min 보장 (양수 latent), upper bound 없음
- z[3:5] (roll, pitch) -> M_hat으로 직접 사용 (별도 MLP 없음)

Softplus를 tanh 대신 사용하는 이유: M_hat이 물리적으로 양수여야 하므로, positive latent가 자연스럽다.
HORA는 tanh([-1,1])을 사용하지만, M_hat 추출에는 양수 보장이 더 중요하다.

#### Policy Observations (13D)

```
euler_angles(3) + angular_velocity(3) + attitude_error(3) + joint_pos(2) + prev_actions(2)
```

`prev_actions`는 Kp 2D만 포함 (Kd 제외). 13D obs 호환성을 위한 설계.

#### Actor Output (4D)

```
Raw actions --> Sigmoid scaling:
  Kp = kp_min + sigmoid(raw) * (kp_max - kp_min)    [kp_min=10, kp_max=100]
  Kd = kd_min + sigmoid(raw) * (kd_max - kd_min)    [kd_min=2, kd_max=30]
```

sigmoid(0) = 0.5 → 초기 Kp=55, Kd=16 (합리적 midpoint).

#### M_hat Extraction

```python
z = policy.get_last_z()       # (num_envs, 6), computed during act()
m_hat = z[:, 3:5]             # (num_envs, 2), guaranteed >= z_min=0.1
tdc.update_controller_params(m_hat=m_hat)
```

별도 M_hat network를 두지 않고, encoder latent에서 직접 슬라이싱한다.
이 방식의 trade-off:

- 장점: 추가 파라미터 없음, 단순
- 단점: z[3:5]가 물리적 inertia 값 범위에 직접 묶임 (latent space 유연성 감소)

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

```bash
cd /workspace/isaaclab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-HeroAgent-Encoder-TDC-v0 \
    --num_envs 4096 --max_iterations 600 \
    --headless --logger wandb --log_project_name hero_agent
```

---

## 3. Phase 2: Adaptation Module Training

### 3.1 Architecture: ProprioAdaptTConv

```
Per-timestep MLP:  12D features --> MLP [32, 32] --> 32D embedding
Temporal Conv:     (N, 30, 32) -->
                   Conv1d(32->32, kernel=9, stride=2, padding=4) --> (N, 15, 32)
                   Conv1d(32->32, kernel=5, stride=1, padding=2) --> (N, 15, 32)
                   Conv1d(32->32, kernel=5, stride=1, padding=2) --> (N, 15, 32)
Flatten + Linear:  (N, 480) --> Linear --> z_hat (6D)
Activation:        softplus + z_min (same as encoder)
```

구현 클래스: `ActorCriticEncoderTDCAdapt` (`encoder/actor_critic_encoder.py`)

### 3.2 Proprioception History (12D per timestep)

```
[roll(1), pitch(1), p(1), q(1), joint_pos_norm(2), joint_vel(2), actions(4)]
```

Ring buffer: `(num_envs, 30, 12)`, 리셋 시 0으로 초기화.

각 feature의 역할:

| Feature | Dim | Role |
|:---|:---|:---|
| roll, pitch | 2 | Restoring torque의 직접적 결과. $T_b = f(\text{roll, pitch, CoG, CoB, V})$ |
| p, q | 2 | Angular acceleration = $\tau / M_{eff}$의 적분. $M_{eff}$ 추론 가능 |
| joint_pos_norm | 2 | Arm 구성 정보 (TDC 작동점, IK 상태) |
| joint_vel | 2 | Arm dynamics 상태 |
| actions | 4 | System identification의 입력 신호 역할 |

HORA와의 차이: HORA (Allegro hand)에서는 joint state = dynamics state이므로 joint 정보만으로 충분하다.
ALBC에서는 arm state $\neq$ body dynamics state이므로, body orientation + angular rates를 포함해야 한다.

Source: `adapt_tdc_env.py:66-98` (`_update_proprio_hist`)

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
Proprioception History --> AdaptTConv --> z_hat (6D)
    z_hat[3:5] --> M_hat --> TDC Controller
    policy_obs + z_hat --> Frozen Actor --> 4D gains --> TDC Controller
    TDC.compute() --> p_EE --> IK --> joint position targets
```

Privileged information, encoder 모두 불필요. Proprioception history만으로 동작.

### 4.2 Frequency Separation

| Component | Frequency | Rationale |
|:---|:---|:---|
| TDC Controller | 50 Hz (control_decimation=4) | Fast state feedback |
| Actor | 50 Hz (same as TDC) | Gain adjustment per control step |
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
| Encoder input | Privileged (CoM, CoB, mass, inertia) | 24D hydro + payload | OK |
| Latent dim | TBD (HORA: 8) | 6D | OK |
| Latent activation | softplus | softplus + z_min=0.1 | OK |
| M_hat extraction | Separate MLP f_theta | Direct z[3:5] slice | Acceptable |
| Actor input | 4D (error, error_rate) + z | 13D policy_obs + z = 19D | OK (richer) |
| Actor output | 4D gains (softplus) | 4D gains (sigmoid scaling) | OK |
| Adapt input/step | 8D (roll, pitch, p, q, act) | 12D (+ joint_pos_norm, joint_vel) | Updated |
| History length | TBD (HORA: 30) | 30 | OK |
| Adapt architecture | MLP + 1D CNN | MLP [32,32] + 3x Conv1d | OK |
| Phase 2 loss | L2 | L2 | OK |

### Key Changes from Design Notes

1. **Privileged 22D -> 24D**: Payload 항목이 [mass, attachment_z] (2D)에서 [mass, cog_offset_xyz] (4D)로 확장.
   CoG offset의 3D 랜덤화가 payload torque에 직접 영향을 미치므로, 3축 모두 포함.

2. **Proprio 8D -> 12D**: 초기 설계(arm joint only)에서 body orientation + angular rates를 추가.
   Arm state만으로는 body dynamics 정보가 부족하다는 분석 결과(Issue A) 반영.

3. **M_hat sigmoid 미적용**: 설계에서 제안된 `sigmoid(z) * (max - min)` 대신, softplus로 하한만 보장.
   상한 clamp의 dead gradient 문제는 현재 softplus 방식에서 발생하지 않음.

---

## 6. Known Issues and Design Decisions

### Issue A: Proprio History Content -- Resolved

초기 구현이 arm joint state만 포함하던 문제.
body euler + angular rates 추가로 12D 확장 완료.
Conv1d temporal reduction은 history_len에만 의존하므로 변경 없음.

### Issue B: No Separate M_hat Network -- Accepted

별도 MLP 대신 z[3:5] 직접 슬라이싱. 2D output에 대해 별도 MLP는 과도한 설계.
Phase 1 학습에서 M_hat 수렴이 불량할 경우, clamp -> sigmoid 전환을 고려.

### Issue C: Missing Damping/Added Mass in Privileged Info -- Accepted

TDE 메커니즘이 이전 step의 실제 dynamics를 암묵적으로 포착하므로,
encoder가 정확한 added mass를 알 필요는 없다.
M_hat만 합리적이면 TDE가 나머지를 보상.

### Issue D: Softplus vs Tanh for Latent -- Accepted

softplus의 양수 보장이 M_hat 추출에 자연스럽다.
z 값의 scale drift 가능성은 WandB에 z statistics를 로깅하여 모니터링.

### Issue E: Latent Dim 6 vs HORA 8 -- Accepted

22D privileged info의 상당 부분이 redundant (main/buoy body 상관).
실질적 독립 자유도 6-8개. 6D latent은 적절한 압축 비율.

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
**Updated**: 2026-02-11
