# TDC Controller Tuning History

**Date**: 2026-02-09
**Status**: Saturated TDE + aggressive PD (Kp=40, Kd=12)로 안정화 성공

---

## Executive Summary

Hero Agent ALBC의 TDC (Time Delay Control) 제어기가 시뮬레이션에서 반복적으로 발산하는 문제를 10회에 걸친 체계적 디버깅을 통해 해결한 기록이다. 핵심 해결책은 **TDE Saturation** — TDE 출력의 벡터 norm을 actuator authority 이내로 제한하는 것이다.

### 최종 결과

- 초기 자세 15도 tilt에서 **10초 이내에 ~2.5도까지 수렴**
- 발산 없음 (이전: 50도 이상 발산)
- 잔여 진동 ~13도 존재 (arm-body coupling)

### 핵심 발견

1. **TDE의 M_hat*nu_dot 항이 actuator authority (12.2 Nm)를 초과**하는 것이 발산의 근본 원인
2. 파라미터 튜닝 (M_hat, Kp/Kd, tde_gain, filter alpha)으로는 해결 불가
3. **TDE Saturation** (norm clamp)으로 TDE를 actuator authority 내에 제한해야 안정화
4. PD gains를 낮추면 오히려 악화 — 이 시스템에서 PD가 주 제어력

---

## System Specifications

| Parameter | Value | Description |
|:---|:---|:---|
| Actuator | 2-link planar arm | l1=l2=0.233m, r_max=0.466m |
| Buoyancy force | F_bu = 26.24 N | Buoy hydrodynamics |
| Max torque | F_bu * r_max = 12.2 Nm | Actuator authority limit |
| Physics dt | 0.005s (200 Hz) | Isaac Lab simulation |
| Control rate | 200 Hz (decimation=1) | TDC runs every physics step |
| Joint PD | Kp=500, Kd=10 | PhysX PD drive |
| Initial tilt | 15 deg roll, 15 deg pitch | Test condition |

---

## Attempt History

### Attempt 1: H_hat EMA Filter (alpha=0.05)

**Config**: h_hat_filter_alpha=0.05, 100Hz, Kp=10, Kd=5

**Result**: 50도 stagnation (수렴 실패)

**Root cause**: H_hat 필터에 T_b가 포함되어 있어, delta_T_b = T_b_prev - T_b 취소가 bias를 가짐. 필터가 T_b_prev의 오래된 값을 유지하므로 정적 오프셋 발생.

---

### Attempt 2: T_b Separation (U_hat Only Filtered)

**Config**: U_hat만 필터링, T_b는 delta_T_b로 직접 취소

**Result**: 동일한 50도 stagnation

**Root cause**: 필터가 순수 uncertainty에도 여전히 bias를 유발. 근본적으로 필터 접근 자체가 문제.

---

### Attempt 3: Filter Off (alpha=1.0) + 100Hz

**Config**: h_hat_filter_alpha=1.0 (필터 비활성화), 100Hz

**Result**: 발산 (M*nu_dot ~ 29 Nm)

**Root cause**: 100Hz에서 nu_dot finite difference가 매우 noisy. 필터 없이는 noise가 TDE를 통해 증폭.

---

### Attempt 4: 200Hz + Aggressive PD

**Config**: decimation=1 (200Hz), Kp=40, Kd=12, alpha=1.0

**Result**: 발산 (M*nu_dot ~ 39 Nm, pEE_raw_r > 1.0)

**Root cause**: 200Hz로 올려도 nu_dot가 여전히 크고, TDE가 PD의 25배 크기. Workspace saturation → TDE positive feedback.

---

### Attempt 5: Anti-Windup TDE

**Config**: p_EE_prev = clamped commanded (FK actual 대신), workspace_radius=0.46

**Implementation**: TDE history에 FK(actual) 대신 clamped p_EE를 저장하여 workspace saturation시 positive feedback loop 차단.

**Result**: 여전히 발산 (roll -54도)

**Root cause**: Anti-windup이 saturation gap에 의한 positive feedback은 차단했으나, TDE 자체의 크기 문제는 해결하지 못함.

---

### Attempt 6: 동일 (확인)

동일 설정 재확인. 발산 패턴 동일.

---

### Attempt 7: nu_dot EMA Filter (alpha=0.05)

**Config**: nu_dot_ema_alpha=0.05 (cutoff ~1.7Hz), h_hat_filter_alpha=1.0

**Key insight**: H_hat 필터와 달리 nu_dot 필터는 angular acceleration 추정치만 필터링하므로 delta_T_b 취소에 영향 없음.

**Result**: 진동 유지 (roll -44 ~ +41도), 발산 → 진동으로 개선

**Analysis**: M*nu_dot가 2.5-6.7 Nm으로 크게 감소했지만, Lambda*pEE 항 (8-12 Nm)이 여전히 TDE를 지배. 수렴하지 못함.

---

### Attempt 8: Combined (M_hat=0.5, Kp=15, Kd=8, tde_gain=0.3)

**Config**: M_hat 증가 + PD 감소 + TDE gain 감소 동시 적용

**Result**: 악화 (roll -48 ~ +49도, M*nu_dot up to 28.7 Nm)

**Root cause**: M_hat을 키우면 M_hat*u_pd도 커지지만 M_hat*nu_dot도 비례 증가. TDE 잔차가 줄어들지 않음. 파라미터 튜닝의 한계 확인.

---

### Attempt 9: Saturated TDE (BREAKTHROUGH)

**Config**: M_hat=(0.15,0.15), Kp=40, Kd=12, tde_gain=1.0, **tde_saturation=5.0 Nm**

**Implementation**:
```python
tde_term = self.tde_gain * (U_hat + tde_delta_T_b)
if self.tde_saturation > 0.0:
    tde_norm = torch.norm(tde_term, dim=-1, keepdim=True)
    tde_scale = torch.clamp(self.tde_saturation / (tde_norm + 1e-8), max=1.0)
    tde_term = tde_term * tde_scale
tau_tdc = tde_term + m_hat_u_pd
```

**Result**: 성공! Roll ±13도, pitch 2.5도까지 수렴 (10초)

**Analysis**:
- TDE 출력이 5 Nm으로 제한되어 PD가 주 제어력
- Workspace saturation이 거의 발생하지 않음 (pEE_raw_r < 0.78)
- Anti-windup + nu_dot filter + saturation의 조합이 안정성 확보

---

### Attempt 10: PD 감소 (Kp=20, Kd=8)

**Config**: Saturated TDE 유지, Kp=20, Kd=8로 감소

**Result**: 악화 (pitch 수렴 실패, 15도 근처 유지)

**Root cause**: 이 시스템에서 PD가 주 제어력이므로 PD를 낮추면 자세 교정 능력 직접 감소. Kp=40/Kd=12로 복원.

---

## Final Configuration

```python
# hero_agent_env_cfg.py — HeroAgentTDCEnvCfg
tdc_m_hat = (0.15, 0.15)         # kg*m^2, close to true inertia
tdc_kp = 40.0                     # omega_n = 16.3 rad/s
tdc_kd = 12.0                     # zeta = 2.45 (overdamped)
tdc_dls_damping = 0.01            # DLS regularization
tdc_h = 0.230                     # buoyancy height offset (m)
tdc_workspace_radius = 0.46       # EE clamp radius (m)
tdc_nu_dot_ema_alpha = 0.05       # angular accel filter (cutoff ~1.7Hz)
tdc_tde_gain = 1.0                # full TDE (saturation limits magnitude)
tdc_h_hat_filter_alpha = 1.0      # no U_hat filter
tdc_tde_saturation = 5.0          # max TDE norm (Nm), ~40% actuator authority
tdc_log_interval = 200            # console log every N steps
decimation = 1                    # 200Hz (every physics step)
control_decimation = 1            # target updates every step
```

---

## Key Implementation Details

### 1. Anti-Windup TDE (`tdc.py`)

TDE history에 FK(actual) 대신 workspace-clamped commanded p_EE를 저장:

```python
self._p_EE_prev = p_EE.clone()  # clamped commanded, not FK(actual)
```

FK(actual)을 쓰면 workspace saturation gap이 TDE에서 "unmodeled dynamics"로 인식되어 positive feedback loop 발생.

### 2. T_b Separation (`tdc.py`)

U_hat (pure uncertainty)만 필터링하고, delta_T_b는 필터 밖에서 직접 계산:

```python
U_hat_raw = tde_lambda_p - tde_m_nu_dot          # T_b 미포함
U_hat = beta * U_hat_raw + (1-beta) * U_hat_prev  # 필터
tde_term = tde_gain * (U_hat + tde_delta_T_b)      # delta_T_b는 정확히 취소
```

H_hat 전체를 필터링하면 delta_T_b cancellation에 bias 발생.

### 3. TDE Saturation (`tdc.py`)

TDE 출력의 벡터 norm을 제한 (방향 보존):

```python
if self.tde_saturation > 0.0:
    tde_norm = torch.norm(tde_term, dim=-1, keepdim=True)
    tde_scale = torch.clamp(self.tde_saturation / (tde_norm + 1e-8), max=1.0)
    tde_term = tde_term * tde_scale
```

Per-axis clamp이 아닌 vector scaling으로 TDE의 방향 정보를 보존.

### 4. nu_dot EMA Filter (`tdc.py`)

Angular acceleration finite difference에 저주파 통과 필터 적용:

```python
nu_dot_raw = (nu - self._nu_prev) / self.dt
nu_dot = alpha * nu_dot_raw + (1-alpha) * self._nu_dot_filtered
```

alpha=0.05 → cutoff ~1.7Hz. H_hat 필터와 달리 delta_T_b에 영향 없음.

---

## Lessons Learned

### 1. Actuator Authority가 TDC 적용 가능성을 결정

TDC 문헌은 "M_hat이 true inertia에 가까우면 TDE가 수렴한다"고 하지만, 실제로는 **TDE 출력이 actuator authority를 초과하면 발산**한다. 이 시스템에서 max torque = 12.2 Nm이지만 TDE가 30-50 Nm을 요구.

### 2. 파라미터 튜닝의 한계

M_hat, Kp/Kd, tde_gain, filter alpha 등 8개 조합을 시도했지만 모두 실패. 구조적 변경 (saturation)이 필요했다.

### 3. PD가 주 제어력이어야 함

Saturated TDE에서 TDE는 보조 역할 (최대 5 Nm). PD가 자세 교정의 주력이므로 PD를 낮추면 성능 악화.

### 4. 필터 위치가 중요

- H_hat 전체 필터 → delta_T_b bias → stagnation
- U_hat만 필터 → 여전히 TDE 크기 문제
- nu_dot 필터 → T_b에 영향 없이 noise 감소 (올바른 위치)

### 5. Anti-windup은 필요하지만 충분하지 않음

Workspace saturation에 의한 positive feedback을 차단하지만, TDE 크기 자체를 제한하지 않음. Saturation과 함께 사용해야 함.

---

## Remaining Issues

1. **잔여 진동 (~13도)**: arm-body coupling에 의한 반작용 토크. PD가 aggressive할수록 arm 이동이 빠르고 반작용이 큼. 현재 Kp=40/Kd=12에서 trade-off.

2. **높은 각속도 (p, q up to 7 rad/s)**: 간헐적으로 큰 angular velocity가 발생. 이것이 TDE의 M*nu_dot를 키움.

3. **수렴 후 steady-state error**: t=15s에서 roll=2.4, pitch=2.5 — 0도로 완전 수렴하지 못함.

---

## File References

| File | Description |
|:---|:---|
| `controllers/tdc.py` | TDC controller core (anti-windup, T_b separation, saturation) |
| `hero_agent_tdc_env.py` | TDC environment (overrides RL actions with TDC output) |
| `hero_agent_env_cfg.py` | Configuration (HeroAgentTDCEnvCfg with all parameters) |
| `test/test_tdc_controller.py` | 14 unit tests (Lambda, T_b, control loop, reset, IK) |
| `controllers/kinematics.py` | ALBC kinematics (FK/IK for 2-link arm) |
