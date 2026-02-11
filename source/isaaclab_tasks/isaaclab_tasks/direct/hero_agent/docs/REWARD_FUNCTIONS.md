# Reward Functions

> **Status**: 2026-02-11 | **Source**: `mdp/rewards.py`, `config.py`
>
> Hero Agent ALBC 보상함수의 수학적 분석, 설계 근거, 실측 수치.
> Gaussian kernel 정규화 + multi-term penalty + curriculum 기반 설계.

---

## Overview

Hero Agent ALBC 환경의 보상은 5개 항(Base RL) 또는 6개 항(Encoder-TDC)의 가중합으로 구성된다:

$$r_t = \underbrace{w_1 \cdot e^{-\phi_t^2 / \sigma^2} \cdot \Delta t}_{\text{tracking}} + \underbrace{w_2 \cdot (\phi_{t-1} - \phi_t)}_{\text{progress}} + \underbrace{w_3 \cdot \|\boldsymbol{\omega}_t\|^2 \cdot \Delta t}_{\text{ang. vel.}} + \underbrace{w_4 \cdot \|\mathbf{a}_t\|^2 \cdot \Delta t}_{\text{action mag.}} + \underbrace{w_5 \cdot \|\Delta\mathbf{a}_t\|^2}_{\text{action rate}} + \underbrace{w_6 \cdot \frac{\|\hat{U}\|}{\|M\hat{u}_{pd}\|} \cdot \Delta t}_{\text{TDE residual (Enc-TDC only)}}$$

여기서 $\phi_t = \|\mathbf{e}_t^{rp}\|_2$ (roll/pitch 에러의 L2 norm), $\Delta t$ = step_dt, $\sigma$ = tracking sigma이다.

### Configuration

| Symbol | ALBCRewardCfg (Base RL) | EncoderTDCRewardCfg | Description |
|:---|:---|:---|:---|
| $w_1$ | 1.0 | 1.0 | `tracking_weight` |
| $\sigma$ | 0.25 rad | 0.25 rad | `tracking_sigma` |
| $w_2$ | 5.0 | 5.0 | `progress_weight` |
| $w_3$ | -0.5 | -0.5 | `angular_velocity_weight` |
| $w_4$ | -0.1 | -0.02 | `action_magnitude_weight` |
| $w_5$ | -0.05 | -0.1 | `action_rate_weight` |
| $w_6$ | N/A | -0.05 | `tde_residual_weight` |
| end iter | 200 | 200 | `curriculum_end_iter` |
| $\Delta t$ | 0.005 | 0.005 | `step_dt` (decimation=1, sim dt=0.005) |

**Source**: `mdp/rewards.py`, `config.py`

### Design Principles

1. **Gaussian kernel 정규화**: 양의 보상은 $e^{-\phi^2/\sigma^2}$ 형태로 [0, 1] 자연 바운딩. 가중치 해석이 직관적.
2. **dt-scaling 규칙**: "순간 상태 품질" 측정 항 -> dt-scaled, "텔레스코핑 차이" 항 -> NOT dt-scaled.
3. **환경별 분리**: Base RL과 Encoder-TDC는 action semantics가 다르므로 보상 config 분리.
4. **Curriculum**: penalty 항은 작게 시작해서 점진적 증가 -> 초기 exploration 보장.

### Previous Design (deprecated)

기존 3항 구조:
- `potential_weight=1.0, scale=8.0`: $8 \cdot e^{-\phi}$ (L1 norm 지수)
- `progress_weight=1.0`: $\phi_{t-1} - \phi_t$
- `action_cost_weight=-1.0`: $-\|a\|^2$

문제점:
- potential episode sum (~15)이 progress (~0.5)를 30:1로 압도 -> progress 신호가 value function 추정 노이즈에 묻힘
- action rate penalty 없음 -> 게인 급변으로 TDC 불안정 유발 가능
- angular velocity penalty 없음 -> 목표 도달 후 진동 억제 미흡
- Encoder-TDC와 Base RL이 동일 보상 사용 -> 게인 튜닝 vs 관절 속도 제어의 차이 미반영

---

## Potential: Definition and Computation

### Attitude Error

로봇의 현재 쿼터니언 $\mathbf{q}$에서 오일러 각도 $(\phi_r, \phi_p, \phi_y)$를 추출하고, 목표 자세 $(\phi_r^*, \phi_p^*, \phi_y^*)$와의 차이를 계산한다:

$$\mathbf{e}_t = \text{atan2}\!\big(\sin(\boldsymbol{\phi}^* - \boldsymbol{\phi}_t),\; \cos(\boldsymbol{\phi}^* - \boldsymbol{\phi}_t)\big) \in [-\pi, \pi]^3$$

`atan2(sin, cos)` wrapping으로 각도 차이가 항상 $[-\pi, \pi]$ 범위에 있도록 보장한다.

**Source**: `base_env.py` (`compute_attitude_error`)

### Target Attitude Randomization

목표 자세 $\boldsymbol{\phi}^*$는 환경 설정에 따라 에피소드마다 랜덤화된다:

| Config | `randomize_target_attitude` | 동작 |
|:---|:---|:---|
| `HeroAgentEnvCfg` (디버그) | `False` | 고정 $(0, 0, 0)$ |
| `HeroAgentTrainEnvCfg` (훈련) | `True` | 에피소드마다 랜덤 |

랜덤화 시 `target_attitude_range = (0.3, 0.3, 0.0)` 범위에서 uniform sampling:

$$\phi_r^* \in [-0.3, +0.3] \text{ rad} \;(\approx \pm17\degree), \quad \phi_p^* \in [-0.3, +0.3] \text{ rad}, \quad \phi_y^* = 0$$

Yaw 목표는 항상 0으로 고정 (range=0.0). 목표가 per-env이므로 `_target_euler`은 `(num_envs, 3)` 텐서이며, attitude error 계산 시 각 환경의 개별 목표를 참조한다.

**Source**: `base_env.py` (`_reset_attitude_task`), `config.py`

### Potential Value

Attitude error의 **roll, pitch 성분만** L2 norm을 취한 것이 potential이다:

$$\phi_t = \|\mathbf{e}_t^{rp}\|_2 = \sqrt{e_{roll}^2 + e_{pitch}^2}$$

**Yaw를 제외하는 이유**: ALBC는 부력체(buoy)의 위치를 조절하여 roll/pitch 토크를 생성한다. 구조적으로 Z축(yaw) 토크를 만들 수 없으므로, yaw를 보상에 포함하면 해결 불가능한 과제를 부여하는 것이 된다.

**Source**: `base_env.py` (`_update_potentials`)

### Update Timing

매 스텝 `_get_rewards()` 진입 시 `_update_potentials()`가 정확히 1회 호출된다:

```python
def _update_potentials(self) -> None:
    self._prev_potentials = self._potentials.clone()
    self._attitude_error = self.compute_attitude_error(self._robot.data.root_quat_w)
    self._potentials = torch.linalg.norm(self._attitude_error[:, :2], dim=-1)
```

1. 현재 potential을 `_prev_potentials`에 복사 (progress 보상용)
2. 새로운 attitude error 계산 (로깅에서도 참조)
3. roll/pitch norm으로 새 potential 갱신

### Initialization on Reset

에피소드 리셋 직후 `_initialize_potentials()`가 호출된다:

$$\phi_0 = \phi_{-1} = \|\mathbf{e}_0^{rp}\|_2$$

두 값을 동일하게 설정하여 첫 스텝의 progress 보상이 0이 되도록 한다. 만약 `prev=0`으로 두면:

$$r_{progress,0} = 0 - \phi_0 = -\phi_0 < 0 \quad \leftarrow \text{가짜 페널티}$$

**Source**: `base_env.py` (`_initialize_potentials`)

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

1. **Natural [0, 1] bound**: 정규화 없이 자연스럽게 [0, 1] 범위. 가중치 $w_1=1.0$의 의미가 직관적.
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

## Term 2: Progress Reward

### Formula

$$r_{progress} = \phi_{t-1} - \phi_t$$

**Source**: `mdp/rewards.py` (`progress_reward`)

```python
def progress_reward(_robot, env, **_kwargs):
    return env._prev_potentials - env._potentials
```

### Behavior

| 상황 | $\phi_{t-1}$ | $\phi_t$ | 보상 (w=5.0) |
|:---|:---|:---|:---|
| 에러 감소 (개선) | 0.50 | 0.30 | +1.00 |
| 에러 유지 | 0.50 | 0.50 | 0.00 |
| 에러 증가 (악화) | 0.50 | 0.80 | -1.50 |

### Design Rationale

1. **방향 신호**: "현재 상태가 좋은가"가 아니라 "**나아지고 있는가**"를 측정
2. **원거리 학습**: 목표에서 멀리 있어도 다가가기만 하면 양의 보상 -- 학습 초기에 핵심 신호
3. **가중치 5.0**: tracking과의 스케일 균형 맞춤 (기존 1.0에서 상향). Episode sum $\approx 5 \times (\phi_0 - \phi_T) \approx 2.25$로, tracking sum과 유사한 크기.

### Telescoping Property

에피소드 전체에 걸쳐 합산하면 중간 항이 상쇄된다:

$$\sum_{t=1}^{T} (\phi_{t-1} - \phi_t) = \phi_0 - \phi_T$$

에피소드 리턴은 **초기 에러와 최종 에러의 차이**만으로 결정된다. 이 특성은 시뮬레이션 주파수와 무관하므로 dt 스케일링이 불필요하다 (`scale_by_dt=False`).

### dt-Scaling 비적용 근거

Tracking reward는 "순간값"이므로 주파수를 2배로 올리면 스텝 수가 2배가 되어 에피소드 합도 2배가 된다. 이를 보정하기 위해 $\Delta t$를 곱한다.

Progress reward는 "차분값"이므로 주파수를 올리면 한 스텝당 변화량이 작아지지만 스텝 수가 늘어나, 전체 합(텔레스코핑)은 동일하다. 따라서 $\Delta t$ 보정이 불필요하다.

---

## Term 3: Angular Velocity Penalty

### Formula

$$r_{ang\_vel} = \sum_{i \in \{p, q\}} \omega_i^2$$

여기서 $p, q$는 body frame angular velocity의 roll rate, pitch rate 성분이다.

**Source**: `mdp/rewards.py` (`angular_velocity_penalty`)

```python
def angular_velocity_penalty(_robot, env, **_kwargs):
    ang_vel = env._robot.data.root_ang_vel_b[:, :2]  # [p, q]
    return torch.sum(ang_vel ** 2, dim=-1)
```

### Behavior

| $p$ (roll rate) | $q$ (pitch rate) | $\sum \omega^2$ | 페널티 (w=-0.5, dt-scaled) |
|:---|:---|:---|:---|
| 0.0 | 0.0 | 0.00 | 0.0000 |
| 0.1 | 0.1 | 0.02 | -0.00005 |
| 0.5 | 0.5 | 0.50 | -0.00125 |
| 1.0 | 1.0 | 2.00 | -0.00500 |
| 2.0 | 2.0 | 8.00 | -0.02000 |

### Design Rationale

1. **진동 억제**: 목표 도달 후에도 빠른 회전이 발생하면 페널티. DR로 수중 감쇠가 변동되므로 명시적 페널티 필요.
2. **L2 penalty**: 작은 속도는 허용하되 큰 속도를 강하게 억제.
3. **Curriculum**: 초기 $w_3 = -0.05$ (1/10)에서 시작하여 iteration 200까지 $w_3 = -0.5$로 증가. 초기 탐색을 방해하지 않으면서 점진적으로 smooth한 행동을 유도.
4. **dt-scaled**: 순간 상태 품질 측정이므로 dt-scaled.

---

## Term 4: Action Magnitude Penalty

### Formula

$$r_{action\_mag} = \sum_{i=1}^{n} a_i^2, \quad n = \text{action\_space}$$

**Source**: `mdp/rewards.py` (`action_magnitude_penalty`)

```python
def action_magnitude_penalty(_robot, actions, **_kwargs):
    return torch.sum(actions ** 2, dim=-1)
```

### Behavior

Base RL 환경에서 action space는 2D (관절 2개), 각 action $\in [-1, 1]$:

| $a_1$ | $a_2$ | $\sum a^2$ | 페널티 (Base, w=-0.1, dt) | 페널티 (Enc-TDC, w=-0.02, dt) |
|:---|:---|:---|:---|:---|
| 0.0 | 0.0 | 0.00 | 0.0000 | 0.0000 |
| 0.3 | 0.3 | 0.18 | -0.00009 | -0.00002 |
| 0.5 | 0.5 | 0.50 | -0.00025 | -0.00005 |
| 1.0 | 1.0 | 2.00 | -0.00100 | -0.00020 |

### Design Rationale

1. **L2 (제곱) 페널티**: 작은 행동은 거의 무시, 큰 행동만 강하게 억제.
2. **환경별 가중치 분리**:
   - Base RL ($w_4 = -0.1$): 관절 속도 제어이므로 과도한 속도 억제 필요.
   - Encoder-TDC ($w_4 = -0.02$): sigmoid midpoint가 합리적 기본값이므로 가중치 완화. 과한 페널티는 게인을 midpoint에 고착시킴.
3. **dt-scaled**: 순간 상태 품질 측정.

---

## Term 5: Action Rate Penalty

### Formula

$$r_{action\_rate} = \sum_{i=1}^{n} (a_{t,i} - a_{t-1,i})^2$$

**Source**: `mdp/rewards.py` (`action_rate_penalty`)

```python
def action_rate_penalty(_robot, actions, prev_actions, **_kwargs):
    return torch.sum((actions - prev_actions) ** 2, dim=-1)
```

### Behavior

| $\Delta a_1$ | $\Delta a_2$ | $\sum (\Delta a)^2$ | 페널티 (Base, w=-0.05) | 페널티 (Enc-TDC, w=-0.1) |
|:---|:---|:---|:---|:---|
| 0.0 | 0.0 | 0.000 | 0.000 | 0.000 |
| 0.01 | 0.01 | 0.0002 | -0.00001 | -0.00002 |
| 0.1 | 0.1 | 0.02 | -0.001 | -0.002 |
| 0.3 | 0.3 | 0.18 | -0.009 | -0.018 |
| 0.5 | 0.5 | 0.50 | -0.025 | -0.050 |

### Design Rationale

1. **Smooth control**: 연속 스텝 간 행동 변화를 최소화하여 부드러운 제어 유도. 특히 TDC에서 게인 급변 방지.
2. **NOT dt-scaled**: per-step 차분이므로 $\Delta a \sim \Delta t$ 관계에 의해 자연적으로 frequency-invariant.
3. **환경별 가중치 분리**:
   - Base RL ($w_5 = -0.05$): 관절 속도 변화는 물리적으로 감쇠됨.
   - Encoder-TDC ($w_5 = -0.1$): 게인 안정성이 TDC 성능에 직결 (M_hat * u_pd 항 안정성).
4. **Curriculum**: 초기 $w_5 / 10$에서 시작하여 점진적 증가.

---

## Term 6: TDE Residual Penalty (Encoder-TDC Only)

### Formula

$$r_{tde} = \frac{\|\hat{U}\|}{\|M\hat{u}_{pd}\| + \epsilon}, \quad \epsilon = 10^{-6}$$

여기서:
- $\hat{U}$: TDE 보상 토크 (dynamics 불확실성 추정치)
- $M\hat{u}_{pd}$: PD 제어 토크 (설계 관성 * PD 출력)

**Source**: `mdp/rewards.py` (`tde_residual_penalty`)

```python
def tde_residual_penalty(_robot, env, **_kwargs):
    u_hat_norm = env._tdc.u_hat.norm(dim=-1)
    pd_norm = env._tdc.pd_torque.norm(dim=-1) + 1e-6
    return u_hat_norm / pd_norm
```

### Behavior

| $\|\hat{U}\| / \|M\hat{u}_{pd}\|$ | 의미 | 페널티 (w=-0.05, dt-scaled) |
|:---|:---|:---|
| 0.0 | 완벽한 M_hat (보상 불필요) | 0.0000 |
| 0.3 | 양호 (작은 보상 필요) | -0.000075 |
| 0.5 | 경계 (M_hat 약간 부정확) | -0.000125 |
| 1.0 | 나쁨 (PD와 보상 토크 동등) | -0.000250 |
| 3.0 | 매우 나쁨 (게인 부적절) | -0.000750 |

### Design Rationale

1. **M_hat 정확도 유도**: M_hat이 정확하면 TDE 보상 토크 $\hat{U}$가 작아짐. 이 비율을 최소화하면 간접적으로 정확한 관성 추정을 유도.
2. **게인 적절성**: 과도한 게인은 큰 PD 토크를 만들어 비율을 낮추지만, action magnitude penalty가 이를 견제. 두 항의 균형으로 적절한 게인 탐색.
3. **RL-TDC 논문의 안정성 조건의 soft proxy**: 논문의 binary 안정성 조건 $\|[I - M^{-1}\bar{M}] \cdot \hat{M}_t \cdot \hat{M}_{t-L}^{-1}\| < 1$을 직접 구현하지 않는 대신, TDE residual ratio가 같은 의도의 continuous 대리 지표 역할.

### RL-TDC Stability Condition 미사용 근거

1. $M_{true}$ 계산에 rigid_body_inertia + added_mass + payload 등 여러 소스 결합 필요 (복잡)
2. Binary 조건(안정/불안정)은 sparse reward -> 학습 어려움
3. DLS IK가 이미 singularity를 자연 처리
4. TDE residual ratio가 동일 의도의 soft proxy

---

## Curriculum Strategy

Penalty 항의 가중치를 학습 초기에는 작게 유지하여 exploration을 보장하고, 점진적으로 증가시켜 smooth/efficient한 행동을 유도한다.

### Schedule

$$w(i) = w_{start} + (w_{full} - w_{start}) \cdot \min\!\big(1, \; i / i_{end}\big)$$

| Term | $w_{start}$ | $w_{full}$ | $i_{end}$ |
|:---|:---|:---|:---|
| angular_velocity (Base/Enc-TDC) | -0.05 | -0.5 | 200 |
| action_rate (Base RL) | -0.005 | -0.05 | 200 |
| action_rate (Enc-TDC) | -0.01 | -0.1 | 200 |

### Implementation

`RewardTermCfg`에 `curriculum_start_weight` 필드가 추가되었다. 이 값이 설정된 항만 curriculum이 적용된다.

```python
@configclass
class RewardTermCfg:
    func: Callable
    weight: float           # full weight (curriculum 도달 목표)
    curriculum_start_weight: float | None = None  # 시작 가중치 (None = 상수)
```

`RewardManager._active_weights`는 **초기화 시 `curriculum_start_weight`로 설정**되어, 첫 iteration부터 올바른 시작값을 사용한다:

```python
self._active_weights = [
    cfg.curriculum_start_weight if cfg.curriculum_start_weight is not None else cfg.weight
    for cfg in self._term_cfgs
]
```

Runner의 `log()` 메서드에서 매 iteration 호출:

```python
raw_env._reward_manager.update_curriculum(iteration, raw_env.cfg.reward.curriculum_end_iter)
```

**Source**: `mdp/rewards.py` (`update_curriculum`), `runners/encoder_runner.py`

### Weight Progression (angular_velocity)

| Iteration | Progress | $w_3$ |
|:---|:---|:---|
| 0 | 0% | -0.050 |
| 50 | 25% | -0.163 |
| 100 | 50% | -0.275 |
| 150 | 75% | -0.388 |
| 200 | 100% | -0.500 |
| 300+ | capped | -0.500 |

---

## Scale Balance Analysis

### After Convergence ($\phi \approx 0.05$, $\omega \approx 0.05$, $|a| \approx 0.3$, $\Delta a \approx 0.02$)

| Term | Per-step value | Weight | dt | Episode Sum (3000 steps) |
|:---|:---|:---|:---|:---|
| tracking | 0.96 | 1.0 | 0.005 | **14.4** |
| progress | net ~0 | 5.0 | 1.0 | **~2.25** |
| angular_velocity | 0.005 | -0.5 | 0.005 | -0.04 |
| action_magnitude | 0.18 | -0.1 | 0.005 | -0.27 |
| action_rate | 0.0008 | -0.05 | 1.0 | -0.12 |

**Total**: ~16.2 (tracking 지배적 -- 수렴 상태에서 정상)

### Early Training ($\phi \approx 0.5$, $\omega \approx 1.0$, $|a| \approx 0.7$, $\Delta a \approx 0.3$)

| Term | Per-step value | Weight (curriculum) | dt | Episode Sum |
|:---|:---|:---|:---|:---|
| tracking | 0.018 | 1.0 | 0.005 | 0.27 |
| progress | variable | 5.0 | 1.0 | **~2.25** |
| angular_velocity | 2.0 | -0.05 (cur.) | 0.005 | -1.50 |
| action_magnitude | 0.98 | -0.1 | 0.005 | -1.47 |
| action_rate | 0.18 | -0.005 (cur.) | 1.0 | -2.70 |

**Total**: ~-3.15 (음수 -> 에이전트가 개선해야 양의 return 획득)

### Key Observations

1. **Tracking:Progress 비율**: 수렴 시 6:1 (기존 30:1~200:1에서 대폭 개선)
2. **학습 초기**: progress가 주도적 신호, penalty는 curriculum으로 완화
3. **수렴 후**: tracking이 자세 유지를 보상, angular velocity가 진동 억제
4. **Action cost**: 전 과정에서 약한 regularization. Encoder-TDC에서는 더 약하게 설정 (sigmoid midpoint 허용)

---

## Reward Manager Architecture

### Pipeline

```
_get_rewards() [base_env.py]
    |
    +-- _update_potentials()              # prev <- current, recompute current
    |
    +-- RewardManager.compute()           # iterate active terms
            |
            +-- tracking_reward()            --> * active_weight * dt
            +-- progress_reward()            --> * active_weight (no dt)
            +-- angular_velocity_penalty()   --> * active_weight * dt  [curriculum]
            +-- action_magnitude_penalty()   --> * active_weight * dt
            +-- action_rate_penalty()        --> * active_weight (no dt)  [curriculum]
            +-- [tde_residual_penalty()]     --> * active_weight * dt  [Enc-TDC only]
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
| Base RL (Base-v0 등) | `base_env._init_task_and_rewards()` | 5개 (tracking, progress, ang_vel, action_mag, action_rate) |
| Encoder-TDC (Encoder-TDC-v0) | `encoder_tdc_env._init_task_and_rewards()` | 6개 (+ tde_residual) |
| Pure TDC (TDC-v0) | `base_env._init_task_and_rewards()` (상속) | 5개 (action은 dummy) |

### Logging Integration

`RewardManager.reset(env_ids)` 호출 시 리셋되는 환경들의 에피소드 합 평균을 반환한다. 이 값은 `base_env._collect_episode_metrics()`를 통해 WandB/TensorBoard에 기록된다:

- `Episode_Reward/tracking`
- `Episode_Reward/progress`
- `Episode_Reward/angular_velocity`
- `Episode_Reward/action_magnitude`
- `Episode_Reward/action_rate`
- `Episode_Reward/tde_residual` (Encoder-TDC only)

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
| Progress 부호 | $\phi_t - \phi_{t-1}$ | $\phi_{t-1} - \phi_t$ (**반대**) |
| Progress 가중치 | 1.0 | 5.0 |
| Action cost 가중치 | $-2.0$ | $-0.1 \times \Delta t$ (Base) |
| Alive reward | 0.5/step | 없음 |
| Angular velocity penalty | 없음 | $-0.5$ (curriculum) |
| Action rate penalty | 없음 | $-0.05$ (Base), $-0.1$ (Enc-TDC) |
| dt 스케일링 | 없음 | tracking, ang_vel, action_mag에 적용 |
| Curriculum | 없음 | ang_vel + action_rate |

Isaac Gym에서 progress 부호가 반대인 이유: potential 정의 자체가 다르거나, 코드 line 767의 주석에서 보듯 개발 과정에서 부호 혼동이 있었던 것으로 보인다. Isaac Lab 구현이 정확한 방향(에러 감소 = 양의 보상)이다.

### Literature Comparison

| Method | Tracking | Penalty Terms | Curriculum |
|:---|:---|:---|:---|
| **RMA** (Kumar 2021, 4족보행) | $e^{-err/\sigma}$ | 10개 (torque, contact, stumble...) | No |
| **HORA** (Qi 2023, 손 조작) | clipped rotation | 5개 (energy, torque, pose) | No |
| **Legged Gym** (ETH) | $e^{-err/\sigma}$ | squared penalties | No |
| **"Learning to Swim"** (Cai 2024) | $e^{-err^2}$ (Gaussian) | energy, drag | No |
| **RL-TDC** (적응 게인 TDC) | $e^{-err}$ | stability mask (binary) | No |
| **Hero Agent (현재)** | $e^{-err^2/\sigma^2}$ (Gaussian) | 4-5개 (ang_vel, action_mag, rate, TDE) | **Yes** |

---

## Design Considerations

### Strengths

1. **Gaussian + Progress 시너지**: Gaussian은 목표 근처 유지 (gradient 감쇄로 진동 최소), progress는 개선 방향 제공 (원거리 학습). 함께 사용하면 local minima를 피하면서도 안정적으로 수렴.
2. **dt-invariant**: 적절한 dt 스케일링과 텔레스코핑으로 `decimation` 변경 시 재튜닝 불필요.
3. **Multi-term penalty**: angular velocity (진동), action rate (급변), action magnitude (효율)가 각각 다른 실패 모드를 억제.
4. **Curriculum**: 학습 초기 탐색을 보장하면서 점진적으로 quality constraint 강화.
5. **환경별 분리**: Base RL (관절 속도 제어)과 Encoder-TDC (게인 튜닝)의 action semantics 차이 반영.
6. **TDE residual proxy**: binary 안정성 조건 대신 continuous ratio로 동일 목표 달성 (학습 친화적).

### Known Limitations

1. **Progress clipping 미사용**: DR 환경에서 에러 급증 시 큰 음수 보상 가능. 필요시 `clamp(min=-threshold)` 고려.
2. **Sigma 고정**: `tracking_sigma=0.25`가 모든 상황에 최적인지 검증 필요. per-env 또는 curriculum sigma 고려 가능.
3. **TDE residual warm-up**: TDC 초기화 직후 첫 스텝은 `is_initialized=False`라 U_hat=0이므로 비율이 0. 실질적 신호는 2번째 스텝부터.

---

## Related Notes

- [TDC_CONTROL_LAW.md](./TDC_CONTROL_LAW.md): TDC 제어기 구조 및 제어 법칙 유도 (보상과 독립)
- [TRAINING_PIPELINE.md](./TRAINING_PIPELINE.md): 초기 보상 설계안 (Gaussian 형태 일치, stability -> TDE residual로 대체), Encoder-TDC action space와 게인 범위
- [DOMAIN_RANDOMIZATION.md](./DOMAIN_RANDOMIZATION.md): Domain Randomization 설정 (보상 robustness에 영향)

---
**Created**: 2026-02-11
**Updated**: 2026-02-11 (Consolidated from research note 11)
