# TDC Controller Tuning History

> **Status**: 2026-02-11 | **Source**: `controllers/tdc.py`, `controllers/kinematics.py`
>
> Hero Agent ALBC의 TDC 디버깅 과정 전체 기록.
> 초기 분석(2026-02-05) -> 근본 원인 분석(2026-02-06) -> 10회 시도(2026-02-09) -> 최종 안정화.

---

## Background: Initial Failure Analysis (2026-02-05~06)

### Gain Sweep (2026-02-05)

36개 gain 조합 (Kp in {1,5,10,20,30,50}, Kd in {0.1,0.5,1,2,5,10})으로 sweep을 수행한 결과:

- **31/36 조합이 초기 에러보다 악화**
- Best: Kp=50, Kd=0.5 -> 51.2도 (6.1도 개선에 불과)
- **Workspace 만성 포화**: WS utilization 2.5-3.0x (모든 config에서)

Three failure regimes:

| Kp Range | Roll Error | Pitch Error | Behavior |
|:---|:---|:---|:---|
| 1~20 | 16~19 deg (개선) | 39~42 deg (악화) | PD 약함; TDE가 torque budget 지배 |
| 30 | 27~37 deg | 20~36 deg | Regime 전환; TDE lag 증폭 |
| 50 | 44~47 deg (악화) | 6~17 deg (개선) | 한 축이 workspace 독점 |

### TDE Term Isolation (2026-02-06)

TDE의 4개 term을 개별 활성화하여 발산 원인을 추적:

| Term | Expression | 단독 테스트 결과 |
|:---|:---|:---|
| term1 (Lambda*p_EE delayed) | $\Lambda_{t-L} \cdot p_{EE,t-L}$ | **발산** (PD 대비 6.6배) |
| term2 (-M_hat*nu_dot) | $-\hat{M} \cdot \dot{\nu}_{t-L}$ | **발산** (max 20 Nm spike) |
| PD | $\hat{M} \cdot (K_p e + K_d \dot{e})$ | 9.97도 수렴 (안정) |
| term4 (Delta_T_b) | $T_{b,t-L} - T_{b,t}$ | 9.97도 수렴 (안전) |

핵심 발견: M_hat 값(0.02~1.0 범위)은 TDE ON 시 모든 값에서 발산 (56~67도).
문제는 M_hat 부정확이 아니라, TDE 출력의 절대 크기가 actuator authority를 초과하는 것.

### Root Cause Analysis (2026-02-06)

세 가지 발산 메커니즘이 규명됨:

1. **Lambda 시변성**: Lambda가 자세(roll, pitch)에 의존하므로, term1 $\Lambda_{t-L} \cdot p_{EE,t-L}$이 Lambda 변화에 의해 양의 피드백 루프를 형성.

2. **유한차분 증폭**: $\dot{\nu} = (\nu_t - \nu_{t-1})/dt$에서 $1/dt = 100$ 배 증폭. M_hat * nu_dot가 actuator authority (12.2 Nm)를 초과.

3. **$H_t \approx H_{t-L}$ 가정 위반**: TDE의 근본 가정 (불확실성이 한 step 동안 거의 불변)이 이 시스템에서 구조적으로 성립하지 않음. Arm-body coupling으로 H가 빠르게 변동.

세 원인이 독립적이 아니라 서로를 강화하는 양의 피드백 루프를 형성.

---

## System Specifications (Tuning Time)

| Parameter | Value (2026-02-09) | Current (2026-02-11) |
|:---|:---|:---|
| Actuator | 2-link planar arm | Same |
| Link lengths | l1=l2=0.233m | Same |
| Buoyancy force | F_bu = 26.24 N | Same |
| Max torque | F_bu * r_max = 12.2 Nm | Same |
| Physics dt | 0.005s (200 Hz) | Same |
| Control rate | 200 Hz (dec=1) | 50 Hz (dec=1, ctrl_dec=4) |
| Joint PD | Kp=500, Kd=10 | Kp=200, Kd=10 (DR +-20%) |
| IK method | Analytical + workspace clamp | DLS IK (Yoshikawa adaptive) |

---

## Attempt History (2026-02-09)

### Attempt 1: H_hat EMA Filter (alpha=0.05)

Config: h_hat_filter_alpha=0.05, 100Hz, Kp=10, Kd=5

Result: 50도 stagnation (수렴 실패)

Root cause: H_hat 필터에 T_b가 포함되어 있어, $\Delta T_b = T_{b,\text{prev}} - T_b$ 취소가 bias를 가짐.

### Attempt 2: T_b Separation (U_hat Only Filtered)

Config: U_hat만 필터링, T_b는 delta_T_b로 직접 취소

Result: 동일한 50도 stagnation

Root cause: 필터가 순수 uncertainty에도 여전히 bias를 유발.

### Attempt 3: Filter Off (alpha=1.0) + 100Hz

Config: h_hat_filter_alpha=1.0, 100Hz

Result: 발산 (M*nu_dot ~ 29 Nm)

Root cause: 100Hz에서 nu_dot finite difference가 매우 noisy.

### Attempt 4: 200Hz + Aggressive PD

Config: decimation=1 (200Hz), Kp=40, Kd=12, alpha=1.0

Result: 발산 (M*nu_dot ~ 39 Nm)

Root cause: 200Hz에서도 TDE가 PD의 25배 크기.

### Attempt 5-6: Anti-Windup TDE

Config: p_EE_prev = clamped commanded (FK actual 대신)

Result: 여전히 발산 (roll -54도)

Root cause: Saturation gap에 의한 positive feedback은 차단했으나, TDE 크기 문제는 미해결.

### Attempt 7: nu_dot EMA Filter (alpha=0.05)

Config: nu_dot_ema_alpha=0.05 (cutoff ~1.7Hz), h_hat_filter_alpha=1.0

Result: 진동 유지 (roll -44 ~ +41도), 발산 -> 진동으로 개선

Analysis: M*nu_dot가 2.5-6.7 Nm으로 감소했지만, Lambda*pEE 항 (8-12 Nm)이 여전히 지배.

### Attempt 8: Combined Tuning

Config: M_hat=0.5, Kp=15, Kd=8, tde_gain=0.3

Result: 악화 (roll -48 ~ +49도)

Root cause: M_hat을 키우면 M_hat*u_pd도 커지지만 M_hat*nu_dot도 비례 증가. 파라미터 튜닝의 한계 확인.

### Attempt 9: Saturated TDE (BREAKTHROUGH)

Config: M_hat=(0.15,0.15), Kp=40, Kd=12, **tde_saturation=5.0 Nm**

Result: Roll +-13도, pitch 2.5도까지 수렴 (10초)

Analysis: TDE 출력이 5 Nm으로 제한되어 PD가 주 제어력. Anti-windup + nu_dot filter + saturation의 조합이 안정성 확보.

### Attempt 10: PD 감소 (Kp=20, Kd=8)

Config: Saturated TDE 유지, PD 감소

Result: 악화 (pitch 수렴 실패, 15도 근처 유지)

Root cause: PD가 주 제어력이므로 PD를 낮추면 자세 교정 능력 직접 감소. Kp=40/Kd=12 복원.

---

## Post-Tuning Evolution (2026-02-09 -> 2026-02-11)

Attempt 9의 TDE saturation이 초기 돌파구였으나, 이후 구조 변경으로 saturation이 불필요해짐:

| Date | Change | Impact |
|:---|:---|:---|
| 2026-02-09 | TDE saturation=5.0 Nm 도입 | 첫 안정화 (13도 잔여 진동) |
| 2026-02-09 | Sign fix: Lambda/T_b 부호 반전 | 물리적으로 올바른 방향 |
| 2026-02-09 | Workspace clamp 제거 | DLS damping이 singularity를 자연 처리 |
| 2026-02-09 | DLS IK (Yoshikawa adaptive) 도입 | Analytical IK 대체, smooth near singularity |
| 2026-02-09 | **TDE saturation 제거** | DLS IK가 p_EE를 자연스럽게 제한하여 불필요해짐 |
| 2026-02-10 | control_decimation=4 (50Hz TDC) | C++ reference와 매칭 |
| 2026-02-10 | Joint PD Kp=200, Kd=10 | TDC position tracking에 적합 |
| 2026-02-10 | Payload on gripper body | Fixed joint, PhysX force propagation |
| 2026-02-11 | Privileged obs 22D -> 24D | Payload cog_offset 3D 확장 |

---

## Current Configuration (2026-02-11)

```python
# controllers/tdc.py — TDCControllerCfg
m_hat = (0.15, 0.16)           # kg*m^2 (default, per-env via encoder)
kp = 40.0                      # TDC PD proportional gain
kd = 12.0                      # TDC PD derivative gain
dls_lambda_damping = 0.01      # DLS regularization for Lambda_inv
ik_dls_lambda = 0.15            # DLS IK lambda (Yoshikawa adaptive)
nu_dot_ema_alpha = 0.05        # angular accel filter (cutoff ~1.7Hz)
h = 0.18                       # ALBC mechanism height offset (m)
max_joint_velocity = 3.0       # rad/s, rate limiting
# tde_saturation: REMOVED (DLS IK naturally limits p_EE)
# workspace_radius: REMOVED (DLS damping handles singularity)
# h_hat_filter_alpha: REMOVED (U_hat not filtered)
```

```python
# config.py — HeroAgentTDCEnvCfg
decimation = 1                  # step_dt = physics_dt = 0.005s
control_decimation = 4          # TDC dt = 0.02s (50Hz)
joint_stiffness_range = (160.0, 240.0)   # DR around 200
joint_damping_range = (8.0, 12.0)        # DR around 10
```

---

## Lessons Learned

### 1. Actuator Authority가 TDC 적용 가능성을 결정

TDC 문헌은 "M_hat이 true inertia에 가까우면 TDE가 수렴한다"고 하지만,
실제로는 TDE 출력이 actuator authority를 초과하면 발산한다.
이 시스템에서 max torque = 12.2 Nm이지만 TDE가 30-50 Nm을 요구했다.

### 2. 파라미터 튜닝의 한계

M_hat, Kp/Kd, tde_gain, filter alpha 등 8개 조합을 시도했지만 모두 실패.
구조적 변경 (saturation, 이후 DLS IK)이 필요했다.

### 3. PD가 주 제어력이어야 함

Saturated TDE에서 TDE는 보조 역할. PD가 자세 교정의 주력이므로 PD를 낮추면 성능 악화.

### 4. Filter 위치가 중요

- H_hat 전체 필터 -> delta_T_b bias -> stagnation
- U_hat만 필터 -> 여전히 TDE 크기 문제
- **nu_dot 필터 -> T_b에 영향 없이 noise 감소** (올바른 위치)

Never filter H_hat or U_hat containing T_b.

### 5. DLS IK가 TDE Saturation을 대체

DLS Jacobian pseudo-inverse가 singularity 근처에서 p_EE를 자연스럽게 감쇠시킨다.
이로 인해 workspace clamp와 TDE saturation이 모두 불필요해졌다.
DLS IK 도입은 TDC 안정화의 핵심 구조적 해결책이었다.

### 6. Anti-windup 패턴

`update_ee_position(FK(rate-limited-actual))`: EE position을 항상 실제 관절 위치의 FK에서 갱신.
rate-limited desired와 actual 사이의 차이가 축적되지 않도록 보장.

---

## File References

| File | Description |
|:---|:---|
| `controllers/tdc.py` | TDC controller core (TDE, anti-windup, DLS Lambda) |
| `controllers/kinematics.py` | ALBC kinematics (FK/IK, DLS Jacobian pseudo-inverse) |
| `tdc_env.py` | TDC environment (classical control, rate limiting) |
| `config.py` | Configuration (HeroAgentTDCEnvCfg, TDCControllerCfg) |

---

**Created**: 2026-02-05 (initial gain sweep analysis)
**Updated**: 2026-02-11 (consolidated from 3 debug logs, updated to current configuration)
