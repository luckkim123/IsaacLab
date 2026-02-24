# Reward Functions

> **Status**: 2026-02-20 | **Source**: `mdp/rewards.py`, `config.py`
>
> Hero Agent ALBC 보상함수의 수학적 분석, 설계 근거, 실측 수치.
> Gaussian kernel tracking 지배 + 소규모 regularization penalty 구조.

---

## Overview

Hero Agent ALBC 환경의 보상은 3개 항의 가중합으로 구성된다:

$$r_t = \underbrace{w_1 \cdot e^{-\phi_t^2 / \sigma^2} \cdot \Delta t}_{\text{tracking}} + \underbrace{w_2 \cdot \|\mathbf{a}_t\|^2 \cdot \Delta t}_{\text{action mag.}} + \underbrace{w_3 \cdot \|\Delta\mathbf{a}_t\|^2}_{\text{action rate}}$$

여기서 $\phi_t = \|\mathbf{e}_t^{rp}\|_2$ (roll/pitch 에러의 L2 norm), $\Delta t$ = step_dt, $\sigma$ = tracking sigma이다.

### Configuration

| Symbol | ALBCRewardCfg (Base RL) | Description |
|:---|:---|:---|
| $w_1$ | 1.5 | `tracking_weight` |
| $\sigma$ | 0.25 rad | `tracking_sigma` |
| $w_2$ | -0.1 | `action_magnitude_weight` |
| $w_3$ | -0.01 | `action_rate_weight` (NOT dt-scaled) |
| end iter | 200 | `curriculum_end_iter` |
| $\Delta t$ | 0.005 | `step_dt` (decimation=1, sim dt=0.005) |

**Source**: `mdp/rewards.py`, `config.py`

### Design Principles

1. **Tracking 지배**: AnymalC/Quadcopter 패턴을 따라, tracking reward가 penalty의 15배 이상을 유지. 미숙한 policy도 양수 reward를 받아 gradient signal이 건전.
2. **Gaussian kernel 정규화**: $e^{-\phi^2/\sigma^2}$ 형태로 [0, 1] 자연 바운딩. 가중치 해석이 직관적.
3. **dt-scaling 규칙**: "순간 상태 품질" 측정 항 -> dt-scaled, action rate는 per-step 차분이므로 NOT dt-scaled.
4. **환경별 분리**: 환경에 따라 action semantics가 다르므로 보상 config을 분리할 수 있다.

### Previous Design (deprecated, 2026-02-20 이전)

기존 5항 구조:
- `tracking_weight=1.5`: $e^{-\phi^2/\sigma^2}$ (Gaussian)
- `linear_error_weight=-1.0`: $-\|\mathbf{e}\|_2$ (Gaussian 보완)
- `angular_velocity_weight=-1.0`: $-\|\omega\|^2$ (진동 억제)
- `action_magnitude_weight=-1.0`: $-\|a\|^2$
- `action_rate_weight=-0.01`: $-\|\Delta a\|^2$

문제점:
- **Penalty 지배**: 3개의 -1.0 penalty가 +1.5 tracking을 구조적으로 압도. error > 17.9도이면 tracking + linear_error만으로 순음수 (angular_velocity, action_magnitude penalty 추가 전에 이미 음수).
- **Episode 길이 비례 penalty 누적**: 정규화 없이 raw sum 로깅 -> 생존이 길어지면 mean_reward가 오히려 하락.
- **noise_std 조기 붕괴**: penalty 지배 구조가 "아무것도 하지 마라"로 해석 -> 탐색 포기.
- AnymalC (penalty/tracking 비율 1:100) 대비 Hero Agent는 1:1.5로 penalty 비중이 ~60배 과다.

---

## Potential: Definition and Computation

### Attitude Error

로봇의 현재 쿼터니언 $\mathbf{q}$에서 오일러 각도 $(\phi_r, \phi_p, \phi_y)$를 추출하고, 목표 자세 $(\phi_r^*, \phi_p^*, \phi_y^*)$와의 차이를 계산한다:

$$\mathbf{e}_t = \text{atan2}\!\big(\sin(\boldsymbol{\phi}^* - \boldsymbol{\phi}_t),\; \cos(\boldsymbol{\phi}^* - \boldsymbol{\phi}_t)\big) \in [-\pi, \pi]^3$$

`atan2(sin, cos)` wrapping으로 각도 차이가 항상 $[-\pi, \pi]$ 범위에 있도록 보장한다.

**Source**: `attitude_task.py` (`compute_error`)

### Target Attitude Randomization

목표 자세 $\boldsymbol{\phi}^*$는 환경 설정에 따라 에피소드마다 랜덤화된다:

| Config | `randomize_target_attitude` | 동작 |
|:---|:---|:---|
| `HeroAgentEnvCfg` (디버그) | `False` | 고정 $(0, 0, 0)$ |
| `HeroAgentTrainEnvCfg` (훈련) | `True` | 에피소드마다 랜덤 |

랜덤화 시 `target_attitude_range = (0.5, 0.5, 0.0)` 범위에서 uniform sampling:

$$\phi_r^* \in [-0.5, +0.5] \text{ rad} \;(\approx \pm28\degree), \quad \phi_p^* \in [-0.5, +0.5] \text{ rad}, \quad \phi_y^* = 0$$

Yaw 목표는 항상 0으로 고정 (range=0.0). 목표가 per-env이므로 `_target_euler`은 `(num_envs, 3)` 텐서이며, attitude error 계산 시 각 환경의 개별 목표를 참조한다.

**Source**: `attitude_task.py` (`reset_targets`), `config.py`

### Potential Value

Attitude error의 **roll, pitch 성분만** L2 norm을 취한 것이 potential이다:

$$\phi_t = \|\mathbf{e}_t^{rp}\|_2 = \sqrt{e_{roll}^2 + e_{pitch}^2}$$

**Yaw를 제외하는 이유**: ALBC는 부력체(buoy)의 위치를 조절하여 roll/pitch 토크를 생성한다. 구조적으로 Z축(yaw) 토크를 만들 수 없으므로, yaw를 보상에 포함하면 해결 불가능한 과제를 부여하는 것이 된다.

**Source**: `attitude_task.py` (`update_potentials`)

### Update Timing

매 스텝 `_get_rewards()` 진입 시 `update_potentials()`가 정확히 1회 호출된다:

```python
def update_potentials(self, quat: torch.Tensor) -> None:
    self._prev_potentials = self._potentials.clone()
    self._attitude_error = self.compute_error(quat)
    self._potentials = torch.linalg.norm(self._attitude_error[:, :2], dim=-1)
```

1. 현재 potential을 `_prev_potentials`에 복사 (상태 추적용)
2. 새로운 attitude error 계산 (로깅에서도 참조)
3. roll/pitch norm으로 새 potential 갱신

### Initialization on Reset

에피소드 리셋 직후 `initialize_potentials()`가 호출된다:

$$\phi_0 = \phi_{-1} = \|\mathbf{e}_0^{rp}\|_2$$

두 값을 동일하게 설정하여 리셋 직후 상태를 일관성 있게 초기화한다.

**Source**: `attitude_task.py` (`initialize_potentials`)

---

## Term 1: Tracking Reward (Gaussian Kernel)

### Formula

$$r_{tracking} = e^{-\phi_t^2 / \sigma^2}, \quad \sigma = 0.25 \text{ rad}$$

**Source**: `mdp/rewards.py` (`tracking_reward`)

```python
def tracking_reward(_robot, env, sigma=0.25, **_kwargs):
    err_sq = env._potentials ** 2
    return torch.exp(-err_sq / (sigma ** 2))
```

### Behavior

| $\phi_t$ (에러) | $e^{-\phi_t^2/\sigma^2}$ | 보상 (dt-scaled) |
|:---|:---|:---|
| 0.0 (완벽) | 1.0000 | 0.0050 |
| 0.1 (~5.7도) | 0.8521 | 0.0043 |
| 0.25 (~14.3도) | 0.3679 (= 1/e) | 0.0018 |
| 0.5 (~28.6도) | 0.0183 | 0.0001 |
| 0.785 (~45도) | 0.0001 | ~0 |

### Design Rationale

1. **Natural [0, 1] bound**: 정규화 없이 자연스럽게 [0, 1] 범위. 가중치의 의미가 직관적.
2. **Sigma 파라미터**: 민감도 조절 가능. $\sigma=0.25$ rad에서 error=0.1 rad이면 보상 0.85 (좋음), error=0.5 rad이면 보상 0.02 (나쁨).
3. **L2 squared norm**: 기존 L1 norm ($e^{-\phi}$) 대비 작은 오차 근처에서 더 큰 gradient 제공 (미세 조정에 유리).
4. **dt-scaled** (`scale_by_dt=True`): 시뮬레이션 주파수 변경 시 에피소드 리턴 일정.

### Comparison: $e^{-\phi^2/\sigma^2}$ vs $e^{-\phi}$

기존 방식 $e^{-\phi}$의 gradient:
$$\frac{d}{d\phi} e^{-\phi} = -e^{-\phi}$$

Gaussian kernel의 gradient:
$$\frac{d}{d\phi} e^{-\phi^2/\sigma^2} = -\frac{2\phi}{\sigma^2} \cdot e^{-\phi^2/\sigma^2}$$

$\phi \to 0$에서:
- $e^{-\phi}$: gradient $\to -1$ (일정)
- $e^{-\phi^2/\sigma^2}$: gradient $\to 0$ (자연 감쇄)

$\phi \approx \sigma$에서:
- $e^{-\phi}$: gradient $\approx -0.78$
- $e^{-\phi^2/\sigma^2}$: gradient $\approx -\frac{2}{\sigma} \cdot e^{-1} \approx -2.94$ (더 강함)

Gaussian은 "목표에 거의 도달한 상태"에서는 gradient가 작아져 불필요한 진동을 줄이고, "중간 오차 영역"에서는 더 강한 gradient로 적극적 교정을 유도한다.

---

## Term 2: Action Magnitude Penalty

### Formula

$$r_{action\_mag} = \sum_{i=1}^{n} a_i^2, \quad n = \text{action\_space}$$

**Source**: `mdp/rewards.py` (`action_magnitude_penalty`)

```python
def action_magnitude_penalty(_robot, actions, **_kwargs):
    return torch.sum(actions ** 2, dim=-1)
```

### Behavior

Base RL 환경에서 action space는 2D (관절 2개), 각 action $\in [-1, 1]$:

| $a_1$ | $a_2$ | $\sum a^2$ | 페널티 (Base, w=-0.1, dt) | Tracking 대비 |
|:---|:---|:---|:---|:---|
| 0.0 | 0.0 | 0.00 | 0.0000 | 0% |
| 0.3 | 0.3 | 0.18 | -0.00009 | 1.2% |
| 0.5 | 0.5 | 0.50 | -0.00025 | 3.3% |
| 1.0 | 1.0 | 2.00 | -0.00100 | 13.3% |

(Tracking 대비 = 완벽 tracking 0.0075/step 기준)

### Design Rationale

1. **Small regularizer**: tracking의 ~1/7 ~ 1/75 수준으로, AnymalC의 penalty/tracking 비율 패턴을 따름.
2. **L2 (제곱) 페널티**: 작은 행동은 거의 무시, 큰 행동만 약하게 억제.
3. **환경별 가중치 분리**:
   - Base RL ($w_2 = -0.1$): 관절 속도 제어의 에너지 효율 유도.
4. **dt-scaled**: 순간 상태 품질 측정.

---

## Term 3: Action Rate Penalty

### Formula

$$r_{action\_rate} = \sum_{i=1}^{n} (a_{t,i} - a_{t-1,i})^2$$

**Source**: `mdp/rewards.py` (`action_rate_penalty`)

```python
def action_rate_penalty(_robot, actions, prev_actions, **_kwargs):
    return torch.sum((actions - prev_actions) ** 2, dim=-1)
```

### Behavior

| $\Delta a_1$ | $\Delta a_2$ | $\sum (\Delta a)^2$ | 페널티 (Base, w=-0.01) |
|:---|:---|:---|:---|
| 0.0 | 0.0 | 0.000 | 0.000 |
| 0.01 | 0.01 | 0.0002 | -0.000002 |
| 0.1 | 0.1 | 0.02 | -0.0002 |
| 0.3 | 0.3 | 0.18 | -0.0018 |
| 0.5 | 0.5 | 0.50 | -0.005 |

### Design Rationale

1. **Smooth control**: 연속 스텝 간 행동 변화를 최소화하여 부드러운 제어 유도. 특히 TDC에서 게인 급변 방지.
2. **NOT dt-scaled**: per-step 차분이므로 dt를 곱하지 않음.
3. **환경별 가중치 분리**:
   - Base RL ($w_3 = -0.01$): 급격한 관절 이동 억제.
4. **Curriculum**: 초기 $w_3 / 10$에서 시작하여 점진적 증가.

---

## Curriculum Strategy

Action rate의 가중치를 학습 초기에는 작게 유지하여 exploration을 보장하고, 점진적으로 증가시켜 smooth한 행동을 유도한다.

### Schedule

$$w(i) = w_{start} + (w_{full} - w_{start}) \cdot \min\!\big(1, \; i / i_{end}\big)$$

| Term | $w_{start}$ | $w_{full}$ | $i_{end}$ |
|:---|:---|:---|:---|
| action_rate (Base RL) | -0.001 | -0.01 | 200 |

### Implementation

`RewardTermCfg`에 `curriculum_start_weight` 필드가 설정된 항만 curriculum이 적용된다.

```python
@configclass
class RewardTermCfg:
    func: Callable
    weight: float           # full weight (curriculum 도달 목표)
    curriculum_start_weight: float | None = None  # 시작 가중치 (None = 상수)
```

`RewardManager._active_weights`는 **초기화 시 `curriculum_start_weight`로 설정**되어, 첫 iteration부터 올바른 시작값을 사용한다.

Runner의 `log()` 메서드에서 매 iteration 호출:

```python
raw_env._reward_manager.update_curriculum(iteration, raw_env.cfg.reward.curriculum_end_iter)
```

**Source**: `mdp/rewards.py` (`update_curriculum`), `runners/encoder_runner.py`

---

## Scale Balance Analysis

### Expected Per-step Balance (15s episode)

#### Base RL (error ~ 15 deg, moderate actions)

| Term | Raw | Weight | dt | Per-step | Share |
|:---|:---|:---|:---|:---|:---|
| tracking | 0.334 | 1.5 | 0.005 | **+0.00251** | **93%** |
| action_magnitude | 0.08 | -0.1 | 0.005 | -0.00004 | 1.5% |
| action_rate | 0.005 | -0.01 | no | -0.00005 | 1.9% |
| **Net** | | | | **+0.00242** | |

Tracking(93%)이 압도적으로 지배. Net 양수 -> value function baseline 추정이 안정.

### Episode Budget (15s, normalized per second)

| Error | Tracking/s | Action penalties/s | Total/s |
|:---|:---|:---|:---|
| 5 deg | +1.33 | -0.018 | **+1.31** |
| 10 deg | +0.92 | -0.018 | **+0.90** |
| 15 deg | +0.50 | -0.018 | **+0.48** |
| 20 deg | +0.21 | -0.018 | **+0.20** |
| 30 deg | +0.02 | -0.018 | **+0.00** |

30도까지 양수 reward 유지. 학습 초기(~30도 error)에도 양수 신호를 받을 수 있다.

### Key Observations

1. **Tracking 지배 구조**: tracking weight(1.5) 대비 action_magnitude(-0.1)는 1/15, action_rate(-0.01)는 1/150. AnymalC의 penalty/tracking 비율(~1:100)과 유사.
2. **Episode reward 정규화**: `_collect_episode_metrics()`에서 `/ max_episode_length_s`로 나누어 episode 길이에 무관한 per-second 평균을 로깅. AnymalC/Quadcopter와 동일한 패턴.

---

## Reward Manager Architecture

### Pipeline

```
_get_rewards() [base_env.py]
    |
    +-- update_potentials()              # prev <- current, recompute current
    |
    +-- RewardManager.compute()           # iterate active terms
            |
            +-- tracking_reward()            --> * active_weight * dt
            +-- action_magnitude_penalty()   --> * active_weight * dt
            +-- action_rate_penalty()        --> * active_weight (no dt)  [curriculum]
            |
            +-- accumulate to _episode_sums
            |
            +-- return total_reward
```

### Zero-Weight Optimization

`RewardManager.__init__`에서 `weight=0.0`인 항은 `_term_cfgs`에 등록되지 않는다. 특정 보상 항을 config에서 0으로 설정하면 자동으로 비활성화된다.

### Environment-Specific Registration

| Environment | Registration | Terms |
|:---|:---|:---|
| Base RL (Base-v0 등) | `base_env._build_reward_terms()` | 3개 active (tracking, action_mag, action_rate) |
| Pure TDC (TDC-v0) | `base_env._build_reward_terms()` (상속) | 3개 active (action은 dummy) |

### Logging Integration

`RewardManager.reset(env_ids)` 호출 시 리셋되는 환경들의 에피소드 합 평균을 반환한다. 이 값은 `base_env._collect_episode_metrics()`를 통해 WandB/TensorBoard에 **per-second 평균으로 정규화**되어 기록된다:

```python
log[f"Episode_Reward/{name}"] = value / self.max_episode_length_s
```

로깅 항목:
- `Episode_Reward/tracking`
- `Episode_Reward/action_magnitude`
- `Episode_Reward/action_rate`

---

## Comparison with Reference Implementations

### Isaac Gym Reference (`references/isaacgym_agent/tasks/heroagent.py`)

```python
# line 770
pose_reward = 8 * torch.exp(-potentials)

# line 766
progress_reward = potentials - prev_potentials    # 부호 주의!

# line 776
total_reward = pose_reward + progress_reward - 2 * actions_cost_scale * actions_cost
```

| 항목 | Isaac Gym | Isaac Lab (현재) |
|:---|:---|:---|
| Tracking | $8 \cdot e^{-\phi}$ | $e^{-\phi^2 / \sigma^2}$ (Gaussian) |
| Action cost | $-2 \cdot \|a\|^2$ | $-0.1 \cdot \|a\|^2 \cdot \Delta t$ (Base) |
| Alive reward | 0.5/step | 없음 |
| Action rate | 없음 | $-0.01 \cdot \|\Delta a\|^2$ (curriculum) |
| dt scaling | 없음 | tracking, action_mag에 적용 |
| Penalty/Tracking ratio | ~1:4 | ~1:15 |

### Cross-Environment Comparison

| Environment | Tracking Weight | Penalty Weights | Ratio |
|:---|:---|:---|:---|
| AnymalC | +1.0 | -0.05, -0.01, -2.5e-5, -2.5e-7 | 100:1 ~ 10000:1 |
| Quadcopter | +15.0 | -0.05, -0.01 | 300:1 |
| Hero Agent | +1.5 | **-0.1, -0.01** | **15:1** |

### Literature Comparison

| Method | Tracking | Penalty Terms | Curriculum |
|:---|:---|:---|:---|
| **RMA** (Kumar 2021, 4족보행) | $e^{-err/\sigma}$ | 10개 (torque, contact, stumble...) | No |
| **HORA** (Qi 2023, 손 조작) | clipped rotation | 5개 (energy, torque, pose) | No |
| **Legged Gym** (ETH) | $e^{-err/\sigma}$ | squared penalties | No |
| **"Learning to Swim"** (Cai 2024) | $e^{-err^2}$ (Gaussian) | energy, drag | No |
| **Hero Agent (현재)** | $e^{-err^2/\sigma^2}$ (Gaussian) | 2개 (action_mag, rate) | **Yes** |

---

## Design Considerations

### Strengths

1. **Tracking 지배 구조**: tracking이 penalty의 15배 이상이므로, 미숙한 policy(~30도 error)도 양수 reward를 받아 건전한 학습 gradient 유지.
2. **dt-invariant**: 적절한 dt 스케일링으로 `decimation` 변경 시 재튜닝 불필요.
3. **Episode-length-independent logging**: `/ max_episode_length_s` 정규화로 생존 시간에 무관한 quality 비교 가능.
4. **간결한 구조**: 3개 항으로 해석 가능성이 높고, reward hacking 위험 감소.

### Known Limitations

1. **Sigma 고정**: `tracking_sigma=0.25`가 모든 상황에 최적인지 검증 필요. per-env 또는 curriculum sigma 고려 가능.
2. **진동 억제 암묵적 의존**: angular velocity penalty 제거 후, 진동 억제는 Gaussian tracking의 "error 증가 = reward 감소" 신호에 전적으로 의존. DR 극단 조건에서 불충분할 가능성.

---

## Related Notes

- [TDC_CONTROL_LAW.md](./TDC_CONTROL_LAW.md): TDC 제어기 구조 및 제어 법칙 유도 (보상과 독립)
- [TRAINING_PIPELINE.md](./TRAINING_PIPELINE.md): 학습 파이프라인 상세
- [DOMAIN_RANDOMIZATION.md](./DOMAIN_RANDOMIZATION.md): Domain Randomization 설정 (보상 robustness에 영향)

---
**Created**: 2026-02-11
**Updated**: 2026-02-20 (Reward redesign: removed linear_error and angular_velocity penalties due to structural penalty domination. Reduced action_magnitude from -1.0 to -0.1. Added episode reward normalization by max_episode_length_s. Active terms: 5 -> 3.)
