# Reward Functions

> **Status**: 2026-02-28 | **Source**: `mdp/rewards.py`, `config.py`
>
> Hero Agent ALBC 보상함수의 수학적 분석, 설계 근거, 실측 수치.
> Gaussian kernel tracking + settling 지배 + 소규모 regularization penalty 구조.

---

## Overview

Hero Agent ALBC 환경의 보상은 6개 항의 가중합으로 구성된다:

$$r_t = \underbrace{w_1 \cdot e^{-\phi_t^2 / \sigma^2} \cdot \Delta t}_{\text{tracking}} + \underbrace{w_2 \cdot \text{settling}(\phi_t) \cdot \Delta t}_{\text{settling}} + \underbrace{w_3 \cdot \text{PBRS}(\phi)}_{\text{progress}} + \underbrace{w_4 \cdot \text{hf}^2 \cdot \Delta t}_{\text{joint osc.}} + \underbrace{w_5 \cdot \|\gamma\|^2 \cdot \Delta t}_{\text{joint angle}} + \underbrace{w_6 \cdot \|\omega\|^2 \cdot \Delta t}_{\text{ang. vel.}}$$

여기서 $\phi_t = \|\mathbf{e}_t^{rp}\|_2$ (roll/pitch error의 L2 norm), $\Delta t$ = step_dt, $\sigma$ = tracking sigma이다.

### Configuration

| Symbol | ALBCRewardCfg (Base RL) | Description |
|:---|:---|:---|
| $w_1$ | 3.0 | `tracking_weight` |
| $\sigma$ | 1.0 rad | `tracking_sigma` (~57.3 deg 1/e point) |
| $w_2$ | 2.0 | `settling_weight` |
| $w_3$ | 0.3 | `progress_weight` (NOT dt-scaled) |
| $w_4$ | -1.0 | `joint_oscillation_weight` |
| $w_5$ | -0.7 | `joint_angle_weight` |
| $w_6$ | -1.5 | `angular_velocity_weight` |
| end iter | 750 | `penalty_curriculum_end_iter` |
| $\Delta t$ | 0.005 | `step_dt` (decimation=1, sim dt=0.005) |

**Source**: `mdp/rewards.py` (`ALBCRewardCfg`), `config.py`

### Design Principles

1. **Positive reward 지배**: tracking(3.0) + settling(2.0) + progress(0.3) = 5.3 positive vs joint_osc(-1.0) + joint_angle(-0.7) + ang_vel(-1.5) = -3.2 max penalty. 미숙한 policy도 양수 reward를 받아 gradient signal이 건전.
2. **Gaussian kernel 정규화**: $e^{-\phi^2/\sigma^2}$ 형태로 [0, 1] 자연 바운딩. 가중치 해석이 직관적.
3. **dt-scaling 규칙**: "순간 상태 품질" 측정 항 -> dt-scaled, progress (PBRS)는 state transition 기반이므로 NOT dt-scaled.
4. **Penalty curriculum**: 모든 음수 가중치 항이 0->1로 선형 증가. 초기 탐색 보장 후 점진적 규제.
5. **PBRS (Ng 1999)**: potential-based reward shaping으로 optimal policy 보존.

---

## Potential: Definition and Computation

### Attitude Error

로봇의 현재 쿼터니언 $\mathbf{q}$에서 오일러 각도 $(\phi_r, \phi_p, \phi_y)$를 추출하고, 목표 자세 $(\phi_r^*, \phi_p^*, \phi_y^*)$와의 차이를 계산한다:

$$\mathbf{e}_t = \text{atan2}\!\big(\sin(\boldsymbol{\phi}^* - \boldsymbol{\phi}_t),\; \cos(\boldsymbol{\phi}^* - \boldsymbol{\phi}_t)\big) \in [-\pi, \pi]^3$$

`atan2(sin, cos)` wrapping으로 각도 차이가 항상 $[-\pi, \pi]$ 범위에 있도록 보장한다.

**Source**: `base_env.py` (`compute_error`)

### Target Attitude Randomization

목표 자세 $\boldsymbol{\phi}^*$는 환경 설정에 따라 에피소드마다 랜덤화된다:

| Config | `randomize_target_attitude` | 동작 |
|:---|:---|:---|
| `HeroAgentEnvCfg` (디버그) | `False` | 고정 $(0, 0, 0)$ |
| `HeroAgentTrainEnvCfg` (훈련) | `True` | 에피소드마다 랜덤 |

랜덤화 시 `target_attitude_range = (0.5, 0.5, 0.0)` 범위에서 uniform sampling:

$$\phi_r^* \in [-0.5, +0.5] \text{ rad} \;(\approx \pm28\degree), \quad \phi_p^* \in [-0.5, +0.5] \text{ rad}, \quad \phi_y^* = 0$$

Yaw 목표는 항상 0으로 고정 (range=0.0).

**Source**: `base_env.py` (`reset_targets`), `config.py`

### Potential Value

Attitude error의 **roll, pitch 성분만** L2 norm을 취한 것이 potential이다:

$$\phi_t = \|\mathbf{e}_t^{rp}\|_2 = \sqrt{e_{roll}^2 + e_{pitch}^2}$$

**Yaw를 제외하는 이유**: ALBC는 부력체(buoy)의 위치를 조절하여 roll/pitch 토크를 생성한다. 구조적으로 Z축(yaw) 토크를 만들 수 없으므로, yaw를 보상에 포함하면 해결 불가능한 과제를 부여하는 것이 된다.

**Source**: `base_env.py` (`update_potentials`)

### Update Timing

매 스텝 `_get_rewards()` 진입 시 `update_potentials()`가 정확히 1회 호출된다:

```python
def update_potentials(self, quat: torch.Tensor) -> None:
    self._prev_potentials = self._potentials.clone()
    self._attitude_error = self.compute_error(quat)
    self._potentials = torch.linalg.norm(self._attitude_error[:, :2], dim=-1)
```

### Initialization on Reset

에피소드 리셋 직후 `initialize_potentials()`가 호출된다:

$$\phi_0 = \phi_{-1} = \|\mathbf{e}_0^{rp}\|_2$$

두 값을 동일하게 설정하여 리셋 직후 PBRS progress가 0을 반환한다.

---

## Individual Reward Terms

### Term 1: Tracking Reward (Gaussian Kernel)

$$r_{tracking} = e^{-\phi_t^2 / \sigma^2}, \quad \sigma = 1.0 \text{ rad}, \quad w = 3.0$$

**Source**: `mdp/rewards.py` (`tracking_reward`)

| $\phi_t$ (에러) | $e^{-\phi_t^2/\sigma^2}$ | dt-scaled | weighted |
|:---|:---|:---|:---|
| 0.0 (완벽) | 1.0000 | 0.00500 | 0.01500 |
| 0.087 (~5도) | 0.9925 | 0.00496 | 0.01489 |
| 0.175 (~10도) | 0.9698 | 0.00485 | 0.01455 |
| 0.349 (~20도) | 0.8853 | 0.00443 | 0.01328 |
| 0.524 (~30도) | 0.7602 | 0.00380 | 0.01141 |
| 0.785 (~45도) | 0.5394 | 0.00270 | 0.00809 |
| 1.0 (= sigma) | 0.3679 | 0.00184 | 0.00552 |

$\sigma = 1.0$ rad은 넓은 kernel width로, 45도 error에서도 의미 있는 gradient(0.54)를 제공한다. 미세 조정은 settling bonus가 담당.

### Term 2: Settling Bonus (Sigmoid)

$$r_{settling} = \sigma(k \cdot (\theta_{thr} - \phi_t)), \quad k = 30, \; \theta_{thr} = 0.10 \text{ rad}, \quad w = 2.0$$

**Source**: `mdp/rewards.py` (`settling_bonus`)

| $\phi_t$ (에러) | settling | dt-scaled | weighted |
|:---|:---|:---|:---|
| 0.0 (완벽) | 0.9526 | 0.00476 | 0.00953 |
| 0.05 (~2.9도) | 0.8176 | 0.00409 | 0.00818 |
| 0.10 (threshold) | 0.5000 | 0.00250 | 0.00500 |
| 0.15 (~8.6도) | 0.1824 | 0.00091 | 0.00182 |
| 0.20 (~11.5도) | 0.0474 | 0.00024 | 0.00047 |

Gaussian tracking이 flat top (gradient -> 0 near zero error)인 영역에서 dense gradient를 제공하여 미세 수렴을 유도한다.

### Term 3: Progress Reward (PBRS)

$$r_{progress} = \phi_{t-1} - \gamma \cdot \phi_t, \quad \gamma = 0.99, \quad w = 0.3$$

**Source**: `mdp/rewards.py` (`progress_reward_pbrs`)

Ng et al. (1999) potential-based reward shaping으로 optimal policy를 보존한다. NOT dt-scaled. $\gamma$는 PPO discount factor와 일치시켜야 한다 (`ALBCRewardCfg.progress_gamma`).

Error 감소 시 양수 보상 제공:
- $\phi_{t-1} = 0.30, \; \phi_t = 0.28$: $r = 0.30 - 0.99 \cdot 0.28 = 0.0228 \to w \cdot r = 0.0068$
- Error 증가 시 음수로 전환, 자연스러운 gradient 제공

Off-policy (SAC) replay buffer에서도 안전하게 사용 가능. 대안으로 `progress_reward` (tanh 기반)가 있으나, PBRS가 이론적 보장이 강하다.

### Term 4: Joint Oscillation Penalty (EMA High-Pass)

$$r_{osc} = \text{mean}((\dot{\gamma} - \text{EMA}(\dot{\gamma}))^2), \quad \alpha_{EMA} = 0.2, \quad w = -1.0$$

**Source**: `mdp/rewards.py` (`joint_oscillation_penalty`)

EMA가 저주파 성분을 추적하고, 차이(고주파 잔차)를 제곱 페널티로 부과한다. 부드러운 움직임은 허용하면서 고주파 진동만 선택적으로 억제한다.

$\alpha = 0.2$는 50Hz 제어 주파수에서 약 1.6Hz cutoff에 해당한다.

### Term 5: Joint Angle Penalty

$$r_{angle} = \text{mean}(\gamma^2), \quad w = -0.7$$

**Source**: `mdp/rewards.py` (`joint_angle_penalty`)

관절 각도가 0으로부터 벗어날수록 quadratic하게 증가하는 페널티. 에너지 효율적인 attitude control을 유도하고, workspace 경계 근처의 비선형 동역학 영역을 회피한다.

### Term 6: Angular Velocity Penalty

$$r_{angvel} = \sum_{i \in \{p, q\}} \omega_i^2, \quad w = -1.5$$

**Source**: `mdp/rewards.py` (`angular_velocity_penalty`)

Roll/pitch 각속도(body frame)의 제곱합. Yaw는 제어 불가능하므로 제외. `sum` (not `mean`) 사용: 축 수가 2로 고정이므로 결과 동일하나, 총 각속도 크기에 비례하는 penalty를 명시적으로 표현.

DR 환경에서 강한 외란 하에 과도한 각속도 진동을 억제한다. 가중치 -1.5는 tracking(3.0)의 절반으로, ang_vel > 0.7 rad/s에서도 tracking gradient를 완전히 억압하지 않도록 조정되었다 (이전 -3.0에서 하향 조정).

---

## Penalty Curriculum

모든 음수 가중치 항에 선형 ramp curriculum이 적용된다:

$$w_{eff}(i) = w_{full} \cdot \min(1, \; i / i_{end})$$

| Parameter | Value |
|:---|:---|
| `penalty_curriculum_end_iter` | 750 |
| Applies to | `joint_oscillation`, `joint_angle`, `angular_velocity` (all negative-weight terms) |
| Scale at iter 0 | 0.0 (penalties disabled) |
| Scale at iter 375 | 0.5 |
| Scale at iter 750 | 1.0 (full penalties) |

### Implementation

`RewardManager.update_curriculum(iteration)` 메서드에서 `penalty_scale`을 갱신한다. `compute()` 내부에서 `weight < 0`인 항에만 scale을 곱한다:

```python
if weight < 0:
    scaled_value = scaled_value * self._penalty_scale
```

Runner의 `log()` 메서드에서 매 iteration 호출:
```python
raw_env._reward_manager.update_curriculum(iteration)
```

초기 학습에서 penalty 없이 자유롭게 탐색하고, 점진적으로 규제를 강화하여 smooth하고 에너지 효율적인 행동으로 수렴한다.

---

## Scale Balance Analysis

### Expected Per-step Balance (15s episode)

#### Base RL (error ~ 15 deg = 0.262 rad, moderate angular velocity ~ 0.5 rad/s)

| Term | Raw | Weight | dt | Per-step | Share |
|:---|:---|:---|:---|:---|:---|
| tracking | 0.934 | 3.0 | 0.005 | **+0.01401** | **55%** |
| settling | 0.012 | 2.0 | 0.005 | +0.00012 | 0.5% |
| progress | ~0.01 | 0.3 | no | +0.00300 | 12% |
| joint_oscillation | ~0.1 | -1.0 | 0.005 | -0.00050 | 2% |
| joint_angle | ~0.05 | -0.7 | 0.005 | -0.00018 | 0.7% |
| angular_velocity | ~0.5 | -1.5 | 0.005 | -0.00375 | 15% |
| **Net** | | | | **+0.01270** | |

Positive 항(tracking+settling+progress=67%)이 지배. Net 양수 -> value function baseline 안정.

### Episode Budget (15s, normalized per second)

| Error | Positive rewards/s | Penalties/s (full curriculum) | Total/s |
|:---|:---|:---|:---|
| 5 deg | +3.97 | -0.50 | **+3.47** |
| 10 deg | +3.76 | -0.50 | **+3.26** |
| 20 deg | +3.06 | -0.50 | **+2.56** |
| 30 deg | +2.39 | -0.50 | **+1.89** |
| 45 deg | +1.60 | -0.50 | **+1.10** |

45도까지도 양수 reward 유지. 학습 초기에도 강건한 양수 gradient signal을 제공한다.

### Key Observations

1. **Positive reward 지배 구조**: tracking(3.0) + settling(2.0) + progress(0.3) = 5.3 총 positive vs max penalty -3.2. 45도 error에서도 양수 유지.
2. **Angular velocity penalty 균형**: -1.5 가중치는 ang_vel=1.0 rad/s에서 -0.0075/step. tracking 0.0055/step과 비교해 억압이 가능하지만, 더 작은 ang_vel에서는 tracking gradient가 우세.
3. **Episode reward 정규화**: `_collect_episode_metrics()`에서 `/ max_episode_length_s`로 나누어 episode 길이에 무관한 per-second 평균을 로깅.

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
            +-- tracking_reward()            --> * weight * dt
            +-- settling_bonus()             --> * weight * dt
            +-- progress_reward_pbrs()       --> * weight (no dt)
            +-- joint_oscillation_penalty()  --> * weight * dt * penalty_scale
            +-- joint_angle_penalty()        --> * weight * dt * penalty_scale
            +-- angular_velocity_penalty()   --> * weight * dt * penalty_scale
            |
            +-- accumulate to _episode_sums
            |
            +-- return total_reward
```

### Zero-Weight Optimization

`RewardManager.__init__`에서 `weight=0.0`인 항은 `_term_cfgs`에 등록되지 않는다. 특정 보상 항을 config에서 0으로 설정하면 자동으로 비활성화된다.

### Environment-Specific Registration

`_build_reward_terms()`에서 config의 각 가중치를 확인하고 0이 아닌 항만 등록한다:

| Environment | Active Terms |
|:---|:---|
| Base RL (Base-v0 등) | 6개 (tracking, settling, progress, joint_osc, joint_angle, ang_vel) |
| Debug (HeroAgent-v0) | 동일 (동일 config) |
| TDC (TDC-v0) | 6개 base + mhat_accuracy, tdc_torque (if weight != 0) |

TDC 환경에서 `tdc_env._build_reward_terms()`가 base를 상속 + 확장한다.

### Logging Integration

`RewardManager.reset(env_ids)` 호출 시 리셋되는 환경들의 에피소드 합 평균을 반환한다. `base_env._collect_episode_metrics()`를 통해 WandB/TensorBoard에 **per-second 평균으로 정규화**되어 기록된다:

```python
log[f"Episode_Reward/{name}"] = value / self.max_episode_length_s
```

로깅 항목:
- `Episode_Reward/tracking`
- `Episode_Reward/settling`
- `Episode_Reward/progress`
- `Episode_Reward/joint_oscillation`
- `Episode_Reward/joint_angle`
- `Episode_Reward/angular_velocity`

---

## Comparison with Reference Implementations

### Isaac Gym Reference (`references/isaacgym_agent/tasks/heroagent.py`)

```python
# line 770
pose_reward = 8 * torch.exp(-potentials)

# line 766
progress_reward = potentials - prev_potentials

# line 776
total_reward = pose_reward + progress_reward - 2 * actions_cost_scale * actions_cost
```

| 항목 | Isaac Gym | Isaac Lab (현재) |
|:---|:---|:---|
| Tracking | $8 \cdot e^{-\phi}$ | $3.0 \cdot e^{-\phi^2 / \sigma^2}$ (Gaussian, $\sigma=1.0$) |
| Progress | $\phi_{t-1} - \phi_t$ (raw delta) | $\phi_{t-1} - \gamma \phi_t$ (PBRS, $\gamma=0.99$) |
| Settling | 없음 | $2.0 \cdot \sigma(30 \cdot (0.1 - \phi))$ |
| Action cost | $-2 \cdot \|a\|^2$ | 없음 (제거됨) |
| Alive reward | 0.5/step | 없음 |
| Joint oscillation | 없음 | $-1.0 \cdot \text{EMA-HP}(\dot\gamma)^2$ |
| Joint angle | 없음 | $-0.7 \cdot \|\gamma\|^2$ |
| Angular velocity | 없음 | $-1.5 \cdot \|\omega_{rp}\|^2$ |
| dt scaling | 없음 | 상태 품질 항에 적용 |
| Curriculum | 없음 | 모든 penalty, 750 iter ramp |

### Cross-Environment Comparison

| Environment | Positive Weights | Penalty Weights | Pos:Neg Ratio |
|:---|:---|:---|:---|
| AnymalC | +1.0 | -0.05, -0.01, -2.5e-5, -2.5e-7 | ~100:1 |
| Quadcopter | +15.0 | -0.05, -0.01 | ~300:1 |
| Hero Agent | **+3.0, +2.0, +0.3** | **-1.0, -0.7, -1.5** | **~1.7:1** |

Hero Agent의 pos:neg 비율이 상대적으로 낮지만, 이는 의도적 설계이다. UUV의 강한 DR(added mass +-50% 등) 환경에서 penalty가 과도한 진동을 적극 억제해야 하며, curriculum으로 초기 탐색을 보장한다.

---

## Design Considerations

### Strengths

1. **Positive reward 지배**: 45도 error에서도 양수 reward를 유지하여 학습 gradient가 건전.
2. **dt-invariant**: 적절한 dt 스케일링으로 `decimation` 변경 시 재튜닝 불필요.
3. **Episode-length-independent logging**: `/ max_episode_length_s` 정규화로 생존 시간에 무관한 quality 비교 가능.
4. **PBRS 이론적 보장**: optimal policy 보존, off-policy safe.
5. **Penalty curriculum**: 초기 탐색 보장 -> 점진적 규제 -> 최종 smooth control.

### Known Limitations

1. **Sigma 고정**: `tracking_sigma=1.0` (고정). 넓은 kernel width로 먼 error에서도 gradient를 유지하나, near-target precision은 settling bonus에 의존.
2. **Angular velocity penalty 조정 필요성**: DR 극단 조건에서 -1.5가 충분한지 미검증. 과도하면 tracking gradient를 억압하고, 부족하면 진동을 방치.
3. **Progress weight 상대적 약함**: 0.3 weight는 tracking(3.0)의 1/10. PBRS의 이론적 장점에도 불구하고 실제 gradient 기여가 작을 수 있음.

---

## Related Notes

- [TDC_CONTROL_LAW.md](./TDC_CONTROL_LAW.md): TDC 제어기 구조 및 제어 법칙 유도 (보상과 독립)
- [TRAINING_PIPELINE.md](./TRAINING_PIPELINE.md): 학습 파이프라인 상세
- [DOMAIN_RANDOMIZATION.md](./DOMAIN_RANDOMIZATION.md): Domain Randomization 설정 (보상 robustness에 영향)

---
**Created**: 2026-02-11
**Updated**: 2026-02-28 (Full rewrite: 3-term system -> 6-term system. Updated all weights, formulas, tables, and comparisons to match current code. Added settling, progress PBRS, joint oscillation, joint angle, angular velocity terms. Removed deprecated action_magnitude and action_rate. angular_velocity_weight -3.0 -> -1.5.)
