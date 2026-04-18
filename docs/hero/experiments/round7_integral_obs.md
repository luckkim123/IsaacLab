# Round 7: Integral Observation + EpsSmooth

Period: 2026-04-17 ~ 2026-04-18.
SS error의 두 가지 접근: parameter tuning (wider eps) vs observation structure (integral error).
R7-Integral이 50-67% SS 감소를 달성하여 reward shape tuning을 초월.

---

## Hypothesis

1. **EpsSmooth**: tanh_eps 0.10->0.20 (wider saturation). Off-equilibrium zone에서
   moderate velocity error (0.1-0.2 m/s)가 saturation regime에 진입하여 roll penalty 완화.
   + k_s -0.1->-0.2 (smoothness 2x, OS 감소)

2. **Integral**: Hwangbo 2017 (quadrotor, RA-L) 패턴. Leaky integrator를
   policy observation에 추가 (81D->84D). "Just arrived" vs "stuck at 1 deg offset for
   100 steps"를 구분하는 cumulative error signal 제공.
   ```
   I_t = leak * I_{t-1} + err_t   (leak=0.99, clamp=+-2.0)
   ```

## Baseline

R6-VelTanh (`r6_veltanh_c03`).

## Experimental Setup

| Run | Task | Config | Change |
|-----|------|--------|--------|
| R7-EpsSmooth (`2026-04-17_22-37-26_r7_epssmooth`) | Isaac-FullDOF-R7-EpsSmooth-v0 | tanh_eps 0.10->0.20, k_s -0.1->-0.2 | Config only |
| R7-Integral (`2026-04-17_22-41-51_r7_integral`) | Isaac-FullDOF-R7-Integral-v0 | 3D leaky integrator [roll_err, pitch_err, vy_err], 81D->84D obs | New obs dim |

Both: 5000 iters. R7-Integral은 새 obs dim -> checkpoint 호환 불가, fresh training 필요.
Two launch failures fixed: (1) 84D noise model vs 81D vectors, (2) policy config inheritance.

## Results

### R7-EpsSmooth: FAILED

| Metric | R6-VelTanh | R7-EpsSmooth | Change |
|--------|-----------|-------------|--------|
| Roll SS (none) | 1.05 | 0.87 | -17% |
| Roll OS | 15.8% | 9.7% | -39% |
| **Pitch SS (hard)** | - | **+90%** | Degraded |
| **Yaw SS** | 0.011 | **0.031** | **3x worse** |
| **vx OS** | - | **+34%** | Degraded |
| Entropy | -0.50 | -1.66 | Collapsed further |
| Reward | 147 | 137 | -7% |

**Root cause**: Wider eps가 ALL velocity gradient를 indiscriminately 약화.
Roll은 개선됐지만 pitch, yaw, vx가 악화. Round 4와 동일한 교훈 -- blunt instrument.

### R7-Integral: SUCCESS for SS Error

| Axis | R6-VelTanh | R7-Integral | Change | Cohen's d |
|------|-----------|-------------|--------|-----------|
| att_norm SS (none) | 1.37 deg | 0.49 deg | **-64%** | -1.70 (large) |
| att_norm SS (hard) | 2.40 deg | 0.98 deg | **-59%** | -0.84 (large) |
| vy SS | 0.037 | 0.016 | **-56%** | - |
| yaw SS | 0.011 | **0.021** | **+94%** | Integral 미적용 |
| Reward | 147 | **227** | **+54%** | - |
| DORAEMON success | - | 98.6% | - |
| Smoothness | - | -0.073 | - |

**Integral이 적용된 모든 channel에서 50-67% SS 감소, 전부 large effect size.**
yaw SS가 악화된 이유: integral을 yaw channel에 적용하지 않았기 때문 (3D: roll, pitch, vy만).

### Overshoot Increase (predicted by PI control theory)

| Axis | R6-VelTanh | R7-Integral | Change |
|------|-----------|-------------|--------|
| Roll OS | 15.8% | 22.2% | +41% |
| Pitch OS | 11.0% | 19.7% | +80% |
| Yaw OS | 46.9% | 32.5% | -31% (indirect improvement) |

### Overshoot Root Cause Analysis

Per-segment trajectory + integral windup 분석:
- **Integral windup은 primary cause가 아님**: WINDUP vs BRAKE cases에서 동일 OS (18.9% vs 19.5%).
  |I|↔OS correlation r=-0.37 (weak negative)
- **True cause**: Policy가 "slow start + integral-driven push" 패턴 학습.
  Initial response 0.82x slower than R6, but peak 0.3s later
- **OS는 DR-invariant**: 18.8-21.6% across all DR levels -> learned behavior, not DR sensitivity

## Conclusions

- **R7-Integral을 새 baseline으로 확정.** 전 channel에서 large effect size SS 감소.
  Reward shape tuning (Rounds 3-6)을 통째로 초월하는 구조적 해결
- **R7-EpsSmooth 폐기.** Wider eps = blunt instrument, indiscriminate gradient weakening
- **Overshoot 증가는 known PI control tradeoff.** Integral windup이 아닌 learned behavior.
  Error gating 또는 faster leak으로 address 가능
- **6D integral 필요**: R7은 3D만 적용. yaw SS 악화(+94%)는 integral 미적용 때문.
  R8에서 6D (roll, pitch, vx, vy, vz, yaw_rate)로 확장

## Impact on Next Round

R8: 6D integral baseline + 두 가지 OS 감소 전략:
1. Error-gated conditional integration: `|err| < reward_sigma` 일 때만 적분.
   Simulation에서 ~9x integral reduction at target crossing
2. Faster leak (0.99->0.95, tau 2.0s->0.39s): 간단하지만 SS 개선을 일부 희생
