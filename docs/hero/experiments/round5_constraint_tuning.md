# Round 5: Constraint Budget Tuning

Period: 2026-04-17.
Constraint budget 조절로 SS error 감소 시도. rp_vel_settling budget tightening과
lin_vel/yaw settling constraint 활성화.

---

## Hypothesis

Round 4 constraint audit: rp_vel_settling cost_return 6.0-6.6 vs budget 20.0 = **33% utilization** (slack).
lin_vel_settling과 yaw_settling은 `_FULL_DOF_CONSTRAINT_TERMS`에 등록되지 않아 Round 4에서 미사용.

1. rp_vel_settling budget tightening (0.20->0.08)으로 utilization 80%까지 올리면 attitude SS 개선
2. lin_vel/yaw settling 활성화로 velocity/yaw SS 개선

## Baseline

Round 2 PerDimEnt control (`perdiment_kl06`).

## Experimental Setup

| Run | Task | Config | Target |
|-----|------|--------|--------|
| R5-RpVel (`r5_rpvel_b008`) | Isaac-FullDOF-R5-RpVel-v0 | rp_vel budget 0.20->0.08 | Attitude SS |
| R5-VelSettling (`r5_velsettling_th010`) | Isaac-FullDOF-R5-VelSettling-v0 | +lin_vel_settling, +yaw_settling (threshold=0.10, budget=0.015) | Velocity/Yaw SS |

Constraint 활성화 설계:
- settling_threshold: 0.04 (original) -> 0.10 (= reward sigma). 0.04는 Control hard DR SS (0.06-0.07)
  이하로, chicken-egg problem 발생
- Budget 0.015 (3x original 0.005): active region 2.5x 확대에 비례 완화

## Results

### R5-RpVel (budget tightening)

| Metric | Control | R5-RpVel | Change |
|--------|---------|----------|--------|
| rp_vel utilization | 30.8% | 63.3% | +32.5pp |
| **Roll SS (hard)** | 1.68 deg | **1.90 deg** | **+13%** |
| Pitch SS (hard) | 1.38 deg | 1.47 deg | +7% |
| Roll rise time | - | - | **+29%** |
| Reward | 145.1 | 150.5 | +3.7% |
| lin_vel reward | - | - | **+27%** |
| **vy SS** | 0.059 | **0.044** | **-24%** (sigma-ratio 0.70) |

**Root cause**: rp_vel_settling = "suppress |p|+|q| when near target" -> **over-damped control**.
Policy가 ~5 deg에 도달 후 정지, 마지막 residual correction을 차단.
Rise time +29% + roll sigma ratio 0.94 (mean shift, not distribution widening)로 확인.

**Unexpected win**: vy SS -24% (sigma-ratio 0.70, robust). Angular stabilization이 horizontal
thrust를 linear motion에 더 정밀하게 할당 (TAM indirect effect).

### R5-VelSettling (settling activation)

| Metric | Control | R5-VelSettling | Change |
|--------|---------|---------------|--------|
| **Yaw SS** | 0.025 | **0.308 rad/s** | **+1117%** |
| yaw reward | - | **-0.002** | Policy abandoned yaw |
| yaw US_env_mean | 0% | 40% (all DR) | Target undershoot |
| yaw sigma-ratio | - | **5.78x** | Massive env spread |
| Reward | 148 | 103 | -30% |

**Round 3 Settling과 동일한 catastrophic failure 재현.**
threshold 0.04->0.10 (2.5x), budget 0.005->0.015 (3x) 완화에도 동일 perverse incentive:
- err=0.3에서 reward ~0.01 (exp-tail), yaw_settling cost = 0 (gate off)
- **Target 회피 >> target 도달** in policy reward calculus

### 6-Way Hard-DR SS Comparison

| Axis | Control | L1 | Tanh | Arctan | R5-RpVel | R5-VelSettling |
|------|---------|------|------|--------|----------|---------------|
| roll | 1.68 | 1.91 | 1.53 | **1.42** | 1.90 | 2.00 |
| pitch | 1.38 | 1.47 | 1.46 | 1.44 | 1.47 | 1.57 |
| vy | 0.059 | **0.044** | 0.045 | 0.051 | **0.044** | 0.049 |
| yaw | 0.025 | **0.019** | **0.021** | 0.028 | 0.036 | **0.308** |

Pattern: Arctan만 roll SS 개선. L1/Tanh가 vy/yaw SS 최고. Pitch는 전 run에서 +5-14% (구조적 한계).
**단일 intervention으로 전 axis 개선 불가.**

### Observation Structure Audit

`mdp/observations.py:40`: `compute_policy_obs`가 command(6D) + body state(9D) + arm(5D) +
thruster(6D) = 26D proprioception만 반환. **Error, integral error, accumulated bias 없음.**
Policy가 SS bias를 직접 관측 불가 -- command와 state로 추론해야 함.
Hwangbo 2017 (quadrotor) 선례: integral error를 obs에 추가하면 SS offset 제거.

## Conclusions

- **Settling-constraint approach를 구조적 dead end로 최종 선언.**
  Round 3 + R5-RpVel + R5-VelSettling = 3회 시도 전 실패.
  rp_vel_settling은 over-damp, lin/yaw settling은 perverse incentive.
  Binary gate `(err < thr)*|dv|`는 parameter tuning으로 수정 불가
- **Per-env std가 판단 도구로 유효**: sigma-ratio가 robust improvement (0.70)과
  catastrophic regression (5.78)을 정량적으로 구분
- **Observation structure가 SS error의 근본 제약**: 6개 run 중 어떤 reward/constraint
  intervention도 전 axis SS를 동시에 개선 못함. Policy가 SS bias를 관측할 수 없기 때문

## Impact on Next Round

Round 6: Axis-specific shape calibration (5-way 데이터 기반 최적 shape per axis).
Round 7: Integral error in observation (Hwangbo 2017 패턴, 구조적 해결).
